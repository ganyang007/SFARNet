from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class InfraredDataset(Dataset):
    """Paired 8-bit grayscale PNG images and binary masks; no implicit split.

    root/train/images/name.png matches root/train/masks/name.png.
    Images must already share the desired training resolution; no resizing is
    performed, avoiding silent changes to tiny target dimensions.
    """
    def __init__(self, root, split):
        folder = Path(root) / split
        self.pairs = [(p, folder / 'masks' / p.name) for p in sorted((folder/'images').glob('*.png'))]
        if not self.pairs:
            raise ValueError(f'No PNG images found in {folder / "images"}')
        for _, mask in self.pairs:
            if not mask.is_file():
                raise FileNotFoundError(mask)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image:
            if image.mode != 'L':
                raise ValueError(f'{image_path}: expected 8-bit grayscale L mode')
            image = np.array(image, dtype=np.float32) / 255
        with Image.open(mask_path) as mask:
            if mask.mode not in ('1', 'L'):
                raise ValueError(f'{mask_path}: expected binary/grayscale mask')
            mask = (np.array(mask) > 0).astype(np.float32)
        if image.shape != mask.shape:
            raise ValueError(f'Image/mask size mismatch: {image_path.name}')
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None])
