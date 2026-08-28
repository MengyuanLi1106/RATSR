import argparse
import copy
import os
from functools import partial

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


DEFAULT_TEST_SCALES = [3]
DEFAULT_SAVE_ROOT = 'output'


def get_device(gpu='0'):
    if str(gpu).lower() in ('cpu', '-1', 'none'):
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to_device(batch, device, non_blocking=True):
    return {
        k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def resolve_project_paths(obj, project_root):
    path_keys = {'root_path', 'split_file', 'resume', 'pre_train'}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in path_keys:
                if isinstance(value, str) and value and not os.path.isabs(value):
                    obj[key] = os.path.normpath(os.path.join(project_root, value))
                elif isinstance(value, list):
                    obj[key] = [
                        os.path.normpath(os.path.join(project_root, item))
                        if isinstance(item, str) and item and not os.path.isabs(item)
                        else item
                        for item in value
                    ]
                else:
                    resolve_project_paths(value, project_root)
            else:
                resolve_project_paths(value, project_root)
    elif isinstance(obj, list):
        for value in obj:
            resolve_project_paths(value, project_root)
    return obj


def normalize_eval_bsize(eval_bsize):
    if eval_bsize is None:
        return None
    if isinstance(eval_bsize, str):
        value = eval_bsize.strip()
        if value.lower() in ('none', 'null', '~', ''):
            return None
        return int(value)
    return int(eval_bsize)


def make_data_loader(spec, tag='', device=None):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    num_workers = spec.get('num_workers', 8)
    loader_kwargs = {
        'batch_size': spec.get('batch_size', 1),
        'shuffle': (tag == 'train'),
        'num_workers': num_workers,
        'pin_memory': device is not None and device.type == 'cuda',
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    return DataLoader(dataset, **loader_kwargs)


def batched_predict(model, inp, coord, cell, eval_bsize=None, cell_factor=1):
    eval_bsize = normalize_eval_bsize(eval_bsize)
    if eval_bsize is None:
        return model(inp, coord, cell * cell_factor)

    module = model.module if hasattr(model, 'module') else model
    if not (hasattr(module, 'gen_feat') and hasattr(module, 'query_rgb')):
        return model(inp, coord, cell * cell_factor)

    b, h, w, c = coord.shape
    coord_flat = coord.contiguous().view(b, h * w, c)
    module.gen_feat(inp)

    preds = []
    for left in range(0, h * w, eval_bsize):
        right = min(left + eval_bsize, h * w)
        coord_chunk = coord_flat[:, left:right, :].view(b, right - left, 1, c)
        pred_chunk = module.query_rgb(coord_chunk, cell * cell_factor)
        preds.append(pred_chunk.squeeze(-1))

    return torch.cat(preds, dim=2).view(b, -1, h, w).contiguous()


def _metric_fns(eval_type):
    if eval_type is None:
        return utils.calc_psnr, utils.calc_ssim
    if eval_type.startswith('mine'):
        return (
            partial(utils.calc_psnr, dataset='mine'),
            partial(utils.calc_ssim, dataset='mine'),
        )
    if eval_type.startswith('VGTSR'):
        return (
            partial(utils.calc_psnr, dataset='VGTSR'),
            partial(utils.calc_ssim, dataset='VGTSR'),
        )
    for dataset in ('div2k', 'benchmark', 'flir'):
        if eval_type.startswith(dataset):
            scale = int(eval_type.split('-')[1])
            return (
                partial(utils.calc_psnr, dataset=dataset, scale=scale),
                partial(utils.calc_ssim, dataset=dataset, scale=scale),
            )
    raise NotImplementedError


def _data_norm_tensors(data_norm, device):
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]},
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    return inp_sub, inp_div, gt_sub, gt_div


def _cell_factor(batch, scale_max, mcell):
    if not mcell:
        return 1
    scale = batch['coord'].shape[1] / batch['inp'].shape[-2]
    return max(scale / scale_max, 1)


def _metric_to_float(value):
    if hasattr(value, 'item'):
        value = value.item()
    return float(value)


def _format_scale_value(scale):
    scale = float(scale)
    if scale.is_integer():
        return str(int(scale))
    return f'{scale:g}'


def _join_output_path(root, *parts):
    if root.startswith('/'):
        return '/'.join([root.rstrip('/\\'), *[str(part).strip('/\\') for part in parts]])
    return os.path.join(root, *parts)


def make_scale_output_dir(save_root, scale):
    output_dir = _join_output_path(save_root, 'x' + _format_scale_value(scale))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def make_result_image_path(output_dir, image_name, psnr_value):
    stem, ext = os.path.splitext(os.path.basename(image_name))
    if not ext:
        ext = '.png'
    return _join_output_path(output_dir, f'{stem}_{psnr_value:.5f}dB{ext}')


def _nested_dataset_files(dataset):
    visited = set()
    while dataset is not None and id(dataset) not in visited:
        visited.add(id(dataset))
        files = getattr(dataset, 'files', None)
        if files is not None:
            return files
        dataset = getattr(dataset, 'dataset', None)
    return None


def _image_name_from_file_entry(file_entry):
    if isinstance(file_entry, str):
        return os.path.basename(file_entry)
    return None


def batch_image_names(loader, batch_start, batch_size):
    files = _nested_dataset_files(loader.dataset)
    names = []
    for offset in range(batch_size):
        index = batch_start + offset
        image_name = None
        if files is not None and index < len(files):
            image_name = _image_name_from_file_entry(files[index])
        names.append(image_name or f'sample_{index:06d}.png')
    return names


def _unique_output_path(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    suffix = 1
    while True:
        candidate = f'{stem}_{suffix}{ext}'
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def _save_tensor_image(image, output_path):
    output_path = _unique_output_path(output_path)
    array = image.detach().clamp(0, 1).mul(255).round().byte()
    array = array.permute(1, 2, 0).cpu().numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(output_path)


def _per_image_metric_values(metric_fn, pred_img, gt_img):
    values = []
    for idx in range(pred_img.shape[0]):
        value = metric_fn(pred_img[idx: idx + 1], gt_img[idx: idx + 1])
        values.append(_metric_to_float(value))
    return values


def save_result_images(pred_img, output_dir, image_names, psnr_values):
    os.makedirs(output_dir, exist_ok=True)
    for image, image_name, psnr_value in zip(pred_img, image_names, psnr_values):
        output_path = make_result_image_path(output_dir, image_name, psnr_value)
        _save_tensor_image(image, output_path)


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              scale_max=4, verbose=False, mcell=False):
    model.eval()
    device = next(model.parameters()).device
    norm_tensors = _data_norm_tensors(data_norm, device)
    psnr_fn, _ = _metric_fns(eval_type)
    psnr_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        batch = move_to_device(batch, device)
        inp_sub, inp_div, gt_sub, gt_div = norm_tensors
        inp = (batch['inp'] - inp_sub) / inp_div
        cell_factor = _cell_factor(batch, scale_max, mcell)

        with torch.no_grad():
            pred = batched_predict(
                model, inp, batch['coord'], batch['cell'],
                eval_bsize=eval_bsize, cell_factor=cell_factor)
            pred = (pred * gt_div + gt_sub).clamp(0, 1)
            psnr_value = psnr_fn(pred, batch['gt'])

        psnr_res.add(_metric_to_float(psnr_value), inp.shape[0])
        if verbose:
            pbar.set_description('val {:.4f}'.format(psnr_res.item()))

    return psnr_res.item()


def eval_psnr_ssim(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
                   scale_max=4, verbose=False, mcell=False, save_dir=None):
    model.eval()
    device = next(model.parameters()).device
    norm_tensors = _data_norm_tensors(data_norm, device)
    psnr_fn, ssim_fn = _metric_fns(eval_type)
    psnr_res = utils.Averager()
    ssim_res = utils.Averager()
    batch_start = 0

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        batch = move_to_device(batch, device)
        inp_sub, inp_div, gt_sub, gt_div = norm_tensors
        inp = (batch['inp'] - inp_sub) / inp_div
        cell_factor = _cell_factor(batch, scale_max, mcell)

        with torch.no_grad():
            pred = batched_predict(
                model, inp, batch['coord'], batch['cell'],
                eval_bsize=eval_bsize, cell_factor=cell_factor)
            pred = (pred * gt_div + gt_sub).clamp(0, 1)
            psnr_value = psnr_fn(pred, batch['gt'])
            ssim_value = ssim_fn(pred, batch['gt'])
            image_psnr_values = _per_image_metric_values(psnr_fn, pred, batch['gt'])

        psnr_res.add(_metric_to_float(psnr_value), inp.shape[0])
        ssim_res.add(_metric_to_float(ssim_value), inp.shape[0])
        if save_dir is not None:
            image_names = batch_image_names(loader, batch_start, pred.shape[0])
            save_result_images(pred, save_dir, image_names, image_psnr_values)
        batch_start += pred.shape[0]
        if verbose:
            pbar.set_description(
                f'val PSNR: {psnr_res.item():.4f}, SSIM: {ssim_res.item():.4f}')

    return psnr_res.item(), ssim_res.item()


def set_wrapper_scale(spec, scale):
    scaled_spec = copy.deepcopy(spec)
    wrapper_args = scaled_spec.setdefault('wrapper', {}).setdefault('args', {})
    wrapper_args['scale_min'] = float(scale)
    wrapper_args['scale_max'] = float(scale)
    wrapper_args.pop('inp_size', None)
    return scaled_spec


def _scale_from_suffix(suffix):
    return float(suffix[1:].replace('p', '.'))


def iter_test_specs(config):
    if config.get('test_dataset_x2') is not None:
        yield 'test', copy.deepcopy(config['test_dataset_x2']), None
        return

    test_datasets = config.get('test_datasets_x2')
    if isinstance(test_datasets, dict):
        for name, spec in test_datasets.items():
            yield name, copy.deepcopy(spec), None
        return
    if isinstance(test_datasets, list):
        for spec in test_datasets:
            name = spec.get('name', 'test')
            yield name, copy.deepcopy(spec), None
        return

    legacy_scales = ('x1p5', 'x2', 'x2p5', 'x3', 'x3p5', 'x4')
    for suffix in legacy_scales:
        spec = config.get('val_dataset_' + suffix)
        if spec is not None:
            yield 'val_' + suffix, copy.deepcopy(spec), [_scale_from_suffix(suffix)]


def make_root_test_spec(root_path, scale):
    return {
        'dataset': {
            'name': 'image-folder',
            'args': {'root_path': root_path},
        },
        'wrapper': {
            'name': 'sr-implicit-downsampled-fast',
            'args': {'scale_min': float(scale), 'scale_max': float(scale)},
        },
        'batch_size': 1,
    }


def default_data_norm():
    return {
        'inp': {'sub': [0.5], 'div': [0.5]},
        'gt': {'sub': [0.5], 'div': [0.5]},
    }


def resolve_config_path(config_path):
    if config_path is None or os.path.isabs(config_path):
        return config_path
    return os.path.join(os.path.dirname(__file__), config_path)


def load_eval_model(model_path, config, device):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_spec = checkpoint['model']
        if model_spec.get('name') == 'idw':
            model_spec = copy.deepcopy(model_spec)
            model_spec['name'] = 'ratsr'
        return models.make(model_spec, load_sd=True).to(device)

    model_spec = copy.deepcopy(config.get('model'))
    if model_spec is None:
        raise ValueError('A state_dict-only checkpoint requires model settings in the config file.')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'sd' in checkpoint:
        state_dict = checkpoint['sd']
    else:
        state_dict = checkpoint

    model = models.make(model_spec).to(device)
    model.load_state_dict(state_dict)
    return model


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test.yaml')
    parser.add_argument('--model', default='checkpoints/best.pth')
    parser.add_argument('--scale', nargs='+', type=float, default=None)
    parser.add_argument('--root_path', default=None)
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--eval_type', default=None)
    parser.add_argument('--eval_bsize', default=None)
    parser.add_argument('--scale_max', type=float, default=4)
    parser.add_argument('--mcell', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_root', default=DEFAULT_SAVE_ROOT)
    parser.add_argument('--no_save_images', action='store_true')
    args = parser.parse_args(args)

    if str(args.gpu).lower() not in ('cpu', '-1', 'none'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = get_device(args.gpu)

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_config_path(args.config)
    config = {}
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        resolve_project_paths(config, project_root)

    model_path = args.model if os.path.isabs(args.model) else os.path.join(project_root, args.model)
    model = load_eval_model(model_path, config, device)
    save_root = args.save_root if os.path.isabs(args.save_root) else os.path.join(project_root, args.save_root)

    data_norm = config.get('data_norm', default_data_norm())
    eval_type = args.eval_type if args.eval_type is not None else config.get('eval_type', 'mine')
    eval_bsize = normalize_eval_bsize(
        args.eval_bsize if args.eval_bsize is not None else config.get('eval_bsize'))

    if args.root_path:
        for scale in (args.scale or DEFAULT_TEST_SCALES):
            loader = make_data_loader(make_root_test_spec(args.root_path, scale), tag='val', device=device)
            save_dir = None if args.no_save_images else make_scale_output_dir(save_root, scale)
            psnr, ssim = eval_psnr_ssim(
                loader, model, data_norm=data_norm, eval_type=eval_type,
                eval_bsize=eval_bsize, scale_max=args.scale_max,
                mcell=args.mcell, verbose=args.verbose, save_dir=save_dir)
            print(f'test{scale}x: psnr={psnr:.2f}, ssim={ssim:.4f}')
        return

    test_specs = list(iter_test_specs(config))
    if not test_specs:
        raise ValueError('No test_dataset_x2, test_datasets_x2, or val_dataset_x* found in config.')

    for name, base_spec, spec_scales in test_specs:
        scales = args.scale or spec_scales or DEFAULT_TEST_SCALES
        for scale in scales:
            loader = make_data_loader(set_wrapper_scale(base_spec, scale), tag='val', device=device)
            save_dir = None if args.no_save_images else make_scale_output_dir(save_root, scale)
            psnr, ssim = eval_psnr_ssim(
                loader, model, data_norm=data_norm, eval_type=eval_type,
                eval_bsize=eval_bsize, scale_max=args.scale_max,
                mcell=args.mcell, verbose=args.verbose, save_dir=save_dir)
            print(f'{name}{scale}x: psnr={psnr:.2f}, ssim={ssim:.4f}')


if __name__ == '__main__':
    main()
