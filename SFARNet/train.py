"""Minimal reference trainer; defaults are not the paper's verified protocol."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from build import build_model
from utils.dataset import InfraredDataset
from utils.loss import deep_supervision_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON defaults; explicit flags override them')
    parser.add_argument('--data-root')
    parser.add_argument('--output', default='runs/reference')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text())
        allowed = {a.dest for a in parser._actions} - {'help', 'config'}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error('Config must be an object containing known option names')
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.data_root:
        parser.error('--data-root or data_root in --config is required')
    if min(args.epochs, args.batch_size, args.base_channels, args.lr) <= 0:
        parser.error('epochs, batch size, base channels and lr must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    train = DataLoader(InfraredDataset(args.data_root, 'train'), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(InfraredDataset(args.data_root, 'val'), batch_size=1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    model = build_model(vars(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = -1.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.
        for image, mask in train:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = deep_supervision_loss(model(image, return_aux=True), mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; inspect inputs and adaptive state')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * image.shape[0]
        model.eval()
        intersection, union = 0, 0
        with torch.inference_mode():
            for image, mask in val:
                prediction = model(image.to(device)) >= 0.5
                target = mask.to(device) > 0.5
                intersection += (prediction & target).sum().item()
                union += (prediction | target).sum().item()
        iou = intersection / union if union else 1.
        metrics = dict(epoch=epoch, train_loss=total_loss/len(train.dataset), val_iou=iou)
        print(json.dumps(metrics), flush=True)
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics)+'\n')
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          config=vars(args), **metrics)
        torch.save(checkpoint, output/'last.pt')
        if iou > best:
            best = iou
            torch.save(checkpoint, output/'best.pt')


if __name__ == '__main__':
    main()
