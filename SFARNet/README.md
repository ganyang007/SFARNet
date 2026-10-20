# SFARNet

**Spatial–Frequency Adaptive Reconstruction Network for Infrared Small Target Detection**

PyTorch reference implementation of SFARNet, with structure-guided feature
enhancement, adaptive spatial–frequency skip connections and semantic-guided
wavelet reconstruction.

> **Release status:** manuscript-based reconstruction, not the original
> experiment code. Synthetic checks have passed; real-dataset training and
> reproduction of the paper's results remain pending. No pretrained weights
> are provided. Implementation assumptions are recorded in
> [Implementation notes](docs/IMPLEMENTATION_NOTES.md).

## Model

The network uses five feature scales with default channel widths
**16 / 32 / 64 / 128 / 256** and three components:

- **SGEM:** Structure-Guided Energy Modulation Module.
- **FSCAM:** Frequency–Spatial Collaborative Adaptation Module.
- **CAWR:** Cross-Layer Adaptive Wavelet Reconstruction Module.

Five individual prediction maps and one fused prediction are supervised with
equally weighted SoftIoU losses. Inference returns the highest-resolution
prediction **M0**.

## Files

| Path | Description |
| --- | --- |
| `model/SFARNet.py` | Network definition and encoder–decoder connections |
| `build/` | Model construction helper |
| `configs/` | Editable JSON training configuration |
| `train.py` | Training, validation IoU and checkpoint saving |
| `model/modules/` | SGEM, FSCAM, CAWR and shared building blocks |
| `test.py` | Checkpoint evaluation and binary mask export |
| `utils/` | Paired image/mask loader and SoftIoU loss |
| `demo.py` | Synthetic module demonstration |
| `tests/` | Module and network checks |
| `docs/IMPLEMENTATION_NOTES.md` | Assumptions and reproduction limits |
| `VALIDATION.md` | Validation record |

## Requirements

Python 3.10+ and PyTorch 2.1+. Install the appropriate PyTorch build for your
device, then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run from the repository root:

```bash
python demo.py
python -m unittest discover -s tests -v
```

```python
import torch
from model import SFARNet

model = SFARNet(base_channels=16).eval()
image = torch.randn(1, 1, 32, 32)
with torch.inference_mode():
    probability = model(image)
    mask = probability >= 0.5
```

The model is randomly initialized until trained or loaded from a checkpoint.
This example checks interfaces; its output is not a meaningful detection.

## Dataset preparation

Provide your own data and explicit train/validation split. No datasets or
automatic benchmark splits are bundled. Pair files by identical filename:

| Images | Masks |
| --- | --- |
| `data/train/images/0001.png` | `data/train/masks/0001.png` |
| `data/val/images/0002.png` | `data/val/masks/0002.png` |

Images must be 8-bit grayscale PNGs; masks must be binary or grayscale PNGs
with zero background and positive target pixels. Images are scaled to [0,1].
No resizing or augmentation is performed. For batch sizes above one, training
images must have identical dimensions. Use disjoint splits; do not use test
images for checkpoint selection.

## Training

```bash
python train.py --config configs/SFARNet.json --data-root data --output runs/experiment_01 --device cuda
```

Use `--device cpu` if CUDA is unavailable. The output directory must not already
exist. Training saves `config.json`, `metrics.jsonl`, `last.pt` and `best.pt`.
The best checkpoint is selected by validation dataset-level pixel IoU at 0.5.

These defaults are a starting configuration, not the verified paper protocol.
LAA currently uses a sequential reference implementation. Large feature maps
can be slow and memory intensive; validate feasibility before a full run.

## Testing

Prepare `data/test/images/` and `data/test/masks/` with matching filenames.

```bash
python test.py --checkpoint runs/experiment_01/best.pt --data-root data --device cuda --save-dir predictions/experiment_01
```

This reports dataset-level pixel IoU and image-averaged nIoU as fractions,
with an empty prediction/empty target IoU defined as 1. Target-level Pd and
Fa are not implemented. Confirm metric conventions against the paper before
comparing results. The optional prediction directory must not already exist.

## Load a trained checkpoint

```python
import torch
from model import SFARNet

checkpoint = torch.load('runs/experiment_01/best.pt', map_location='cpu', weights_only=True)
model = SFARNet(base_channels=checkpoint['config']['base_channels'])
model.load_state_dict(checkpoint['model'])
model.eval()
```

## Results

This package does not yet provide reproduced benchmark results or pretrained
weights. The manuscript's parameter count, FLOPs and accuracy metrics should
not be presented as measurements of this implementation.

See [Validation](VALIDATION.md) for the checks actually completed.

## Acknowledgment

Repository organization follows the `build`, `configs`, `model`, `utils`,
`train.py` and `test.py` layout of [CSAFNet](https://github.com/LingchuanK/CSAFNet).
Its implementation was not copied. Methodological references and assumptions
are discussed in the accompanying manuscript and implementation notes.

## License and citation

License selection and publication metadata are pending confirmation by the
repository owner. No publication venue, DOI or acceptance status is claimed.
