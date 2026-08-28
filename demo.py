import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import utils
from test import (
    batched_predict,
    default_data_norm,
    get_device,
    load_eval_model,
    normalize_eval_bsize,
    resolve_config_path,
    resolve_project_paths,
)


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def image_to_tensor(image):
    array = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_image(tensor):
    array = tensor.detach().clamp(0, 1).mul(255).round().byte()
    array = array.squeeze(0).cpu().numpy()
    return Image.fromarray(array)


def make_lr_image(gt_image, scale):
    width, height = gt_image.size
    lr_width = max(1, int(width // scale))
    lr_height = max(1, int(height // scale))
    hr_width = int(round(lr_width * scale))
    hr_height = int(round(lr_height * scale))

    gt_image = gt_image.crop((0, 0, hr_width, hr_height))
    lr_image = gt_image.resize((lr_width, lr_height), Image.BICUBIC)
    return lr_image, gt_image


def make_coord_cell(height, width, device):
    coord = utils.make_coord([height, width], flatten=False).unsqueeze(0).to(device)
    cell = torch.tensor([[2 / height, 2 / width]], dtype=torch.float32, device=device)
    return coord, cell


def load_config(config_path, project_root):
    config = {}
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        resolve_project_paths(config, project_root)
    return config


def iter_images(input_dir):
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(IMG_EXTENSIONS):
            yield filename, os.path.join(input_dir, filename)


def save_image(image, output_dir, stem, scale, suffix):
    scale_text = f'{scale:g}'
    output_path = os.path.join(output_dir, f'{stem}_x{scale_text}_{suffix}.png')
    image.save(output_path)
    return output_path


def run_demo(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(project_root, args.input_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(project_root, args.output_dir)
    config_path = resolve_config_path(args.config)
    model_path = args.model if os.path.isabs(args.model) else os.path.join(project_root, args.model)

    if str(args.gpu).lower() not in ('cpu', '-1', 'none'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = get_device(args.gpu)

    config = load_config(config_path, project_root)
    model = load_eval_model(model_path, config, device)
    model.eval()

    data_norm = config.get('data_norm', default_data_norm())
    inp_norm = data_norm['inp']
    gt_norm = data_norm['gt']
    inp_sub = torch.FloatTensor(inp_norm['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(inp_norm['div']).view(1, -1, 1, 1).to(device)
    gt_sub = torch.FloatTensor(gt_norm['sub']).view(1, -1, 1, 1).to(device)
    gt_div = torch.FloatTensor(gt_norm['div']).view(1, -1, 1, 1).to(device)
    eval_bsize = normalize_eval_bsize(args.eval_bsize)

    os.makedirs(output_dir, exist_ok=True)
    image_items = list(iter_images(input_dir))
    if not image_items:
        raise ValueError(f'No images found in {input_dir}')

    with torch.no_grad():
        for filename, image_path in tqdm(image_items, desc='RATSR demo'):
            stem = os.path.splitext(filename)[0]
            gt_image = Image.open(image_path).convert('L')

            for scale in args.scale:
                lr_image, cropped_gt_image = make_lr_image(gt_image, scale)
                lr_tensor = image_to_tensor(lr_image).unsqueeze(0).to(device)
                inp = (lr_tensor - inp_sub) / inp_div

                hr_height, hr_width = cropped_gt_image.size[1], cropped_gt_image.size[0]
                coord, cell = make_coord_cell(hr_height, hr_width, device)
                pred = batched_predict(model, inp, coord, cell, eval_bsize=eval_bsize)
                pred = (pred * gt_div + gt_sub).clamp(0, 1)

                save_image(tensor_to_image(pred[0]), output_dir, stem, scale, 'RATSR')

    print(f'Processed {len(image_items)} image(s). Results saved to: {output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/demo/GT')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--config', default='configs/test.yaml')
    parser.add_argument('--model', default='checkpoints/best.pth')
    parser.add_argument('--scale', nargs='+', type=float, default=[2.0])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--eval_bsize', default=None)
    args = parser.parse_args()
    run_demo(args)


if __name__ == '__main__':
    main()
