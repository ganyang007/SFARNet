# Validation record

Environment: Python 3.12, PyTorch 2.14.1+cpu, CPU; 2026-10-20.

`python -m unittest discover -s tests -v`: 4 tests passed.
- Haar round-trip and input gradients on even, odd and 1x1 inputs.
- SGEM/FSCAM/CAWR output shapes, finite outputs and input/parameter gradients.
- FSCAM repeated-call state reset, unchanged state_dict, batch independence under inference_mode.
- Analytic linear-state update gradient agrees with autograd.

`python demo.py`: synthetic linked SGEM/FSCAM/CAWR forward and backward completed.
Shapes: shallow (1,16,16,16), deep (1,32,8,8), reconstructed (1,16,16,16).

No real-data training, accuracy evaluation, CUDA/mixed-precision testing,
full-network complexity profiling or runtime benchmark has been performed.

## CSAFNet-style package update

- `python -m unittest discover -s tests -v`: 6 tests passed after reorganization.
- Added full-network shape, six-head supervision, finite-gradient, optimizer-step and inference checks on odd-sized input with base_channels=2.
- A one-epoch command-line run on one synthetic 16x16 training image completed; config, metrics, last.pt and best.pt were saved.
- `test.py` successfully loaded best.pt, evaluated a synthetic image and exported a binary PNG. Synthetic IoU/nIoU were 0; these are plumbing checks, not benchmark results.
- Synthetic smoke data and checkpoints are intentionally excluded from the release.
- The default 16-channel configuration has not undergone real-data training or full-resolution performance profiling.
