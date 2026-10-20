"""Evaluate a local reference checkpoint; reports pixel IoU/nIoU, not Pd/Fa."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader
from build import build_model
from utils.dataset import InfraredDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-dir', help='Optional new directory for binary prediction masks')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must lie in [0,1]')
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = InfraredDataset(args.data_root, args.split)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=False)
    intersection, union, image_iou = 0, 0, 0.
    with torch.inference_mode():
        for index, (image, mask) in enumerate(DataLoader(dataset, batch_size=1)):
            probability = model(image.to(args.device))
            if not torch.isfinite(probability).all():
                raise FloatingPointError('Nonfinite prediction')
            prediction = probability >= args.threshold
            target = mask.to(args.device) > 0.5
            a = (prediction & target).sum().item()
            b = (prediction | target).sum().item()
            intersection += a
            union += b
            image_iou += a / b if b else 1.
            if save_dir:
                binary = prediction[0, 0].cpu().numpy().astype('uint8') * 255
                Image.fromarray(binary).save(save_dir / dataset.pairs[index][0].name)
    print(json.dumps(dict(images=len(dataset), threshold=args.threshold,
                          IoU=intersection/union if union else 1.,
                          nIoU=image_iou/len(dataset)), indent=2))


if __name__ == '__main__':
    main()
