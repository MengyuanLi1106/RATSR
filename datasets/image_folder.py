import json
import os
import pickle
from PIL import Image

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _normalize_per_root_arg(value, n_roots):
    if isinstance(value, (list, tuple)):
        if len(value) != n_roots:
            raise ValueError('Per-root arguments must match root_path length.')
        return list(value)
    return [value] * n_roots


def _load_l_image(file):
    return transforms.ToTensor()(Image.open(file).convert('L'))


def _load_l_array(file):
    img = imageio.imread(file)
    if img.ndim == 3:
        img = np.asarray(Image.fromarray(img).convert('L'))
    img = np.ascontiguousarray(img[None, :, :])
    return torch.from_numpy(img).float() / 255


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', index_start=None, index_end=None):
        self.repeat = repeat
        self.cache = cache

        root_paths = root_path if isinstance(root_path, (list, tuple)) else [root_path]
        split_files = _normalize_per_root_arg(split_file, len(root_paths))
        split_keys = _normalize_per_root_arg(split_key, len(root_paths))
        first_ks = _normalize_per_root_arg(first_k, len(root_paths))
        index_starts = _normalize_per_root_arg(index_start, len(root_paths))
        index_ends = _normalize_per_root_arg(index_end, len(root_paths))

        files = []
        for root, cur_split_file, cur_split_key, cur_first_k, cur_start, cur_end in zip(
                root_paths, split_files, split_keys, first_ks, index_starts, index_ends):
            if cur_split_file is None:
                filenames = sorted([
                    filename for filename in os.listdir(root)
                    if filename.lower().endswith(IMG_EXTENSIONS)
                ])
            else:
                with open(cur_split_file, 'r') as f:
                    filenames = json.load(f)[cur_split_key]

            if cur_first_k is not None:
                filenames = filenames[:int(cur_first_k)]
            if cur_start is not None or cur_end is not None:
                start = 0 if cur_start is None else int(cur_start) - 1
                end = None if cur_end is None else int(cur_end)
                filenames = filenames[start:end]

            files.extend(os.path.join(root, filename) for filename in filenames)

        if cache == 'none':
            self.files = files
        elif cache == 'bin':
            self.files = []
            for file in files:
                bin_root = os.path.join(
                    os.path.dirname(os.path.dirname(file)),
                    '_bin_' + os.path.basename(os.path.dirname(file)))
                os.makedirs(bin_root, exist_ok=True)
                bin_file = os.path.join(
                    bin_root, os.path.splitext(os.path.basename(file))[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(_load_l_array(file), f)
                self.files.append(bin_file)
        elif cache == 'in_memory':
            self.files = [_load_l_image(file) for file in files]
        else:
            raise ValueError('Unsupported cache mode: {}'.format(cache))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.cache == 'none':
            return _load_l_image(x)
        if self.cache == 'bin':
            with open(x, 'rb') as f:
                return pickle.load(f)
        return x
