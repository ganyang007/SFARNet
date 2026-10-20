# SFARNet — manuscript-based module reference implementation

PyTorch reference implementations of **SGEM**, **FSCAM** and **CAWR**, based on
the supplied manuscript *SFARNet: Spatial–Frequency Adaptive Reconstruction
Network for Infrared Small Target Detection*.

**Status:** newly generated research prototype. This is not the original
implementation used to produce the manuscript's experiments. It has not been
trained or evaluated on SIRST, NUDT-SIRST or IRSTD-1K. No pretrained weights are
included. The manuscript's 8.63M parameters, 15.68 GFLOPs and detection results
must not be attributed to this package. This release includes modules, an assembled reference network and a minimal
trainer. It is not a complete benchmark reproduction.

## Quick start

Use Python 3.10 or later and install PyTorch appropriate for your CPU/CUDA setup.

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

Run these commands from this repository's root. Tests use synthetic tensors.

```python
import torch
from model.modules import SGEM, FSCAM, CAWR

shallow = torch.randn(1, 16, 16, 16)
deep = torch.randn(1, 32, 8, 8)
encoded = SGEM(16)(shallow)       # [1, 32, 16, 16]
skip = FSCAM(16)(shallow)         # [1, 16, 16, 16]
decoded = CAWR(16)(skip, deep)    # [1, 16, 16, 16]
```

## Components

| Module | Implementation | Input and output |
| --- | --- | --- |
| SGEM | Local depthwise structure, pooled context, energy importance, channel selection, residual and parallel convolutions | C channels to 2C by default |
| FSCAM | Scharr gradients, real/imaginary Fourier mapping, spatial attention, local K/Q projections, linear TTT | C channels to C |
| CAWR | Haar decomposition, grouped deep projection, shared subband channel weighting, inverse transform, residual refinement | shallow C plus deep 2C to C |

FSCAM uses raster-order spatial tokens and eight-token updates with step size
0.001. Its internal state is initialized from learned parameters independently
for each sample on every call. Updates use an analytic gradient and work inside
`torch.inference_mode()`. State is never carried over between images. Training
retains gradients through the update sequence.

## Reconstruction assumptions requiring author confirmation

The supplied DOCX embeds equations as MathType objects. Prose was available,
but the exact formula contents were not decoded. The following choices are
explicit implementation assumptions; inspect them against the equations,
architecture figures and original experiments before presenting this as an
author-verified implementation:

1. SGEM uses a 3x3 structural depthwise convolution, 1x1 projection and pooling
   factor 2. The dynamic resize is an independently implemented learned-offset
   bilinear sampler, **not the original DySample implementation**. SimAM-style
   importance uses channel statistics with sigmoid reciprocal-energy weighting;
   exact leave-one-out energy is not used.
2. FSCAM maps packed real/imaginary rFFT coefficients with a learned 1x1
   convolution. LAA hidden dimension defaults to C and is configurable. The
   manuscript's embedded hidden-dimension formula still needs verification.
   Linear state uses identity initialization, no bias and the objective
   `sum((K @ W - V)^2) / (2 * number_of_tokens)`. Chunk queries are evaluated
   after each chunk update; the final partial chunk is also processed.
3. CAWR assumes `deep_channels=2*C`, group count C, band order LL/LH/HL/HH,
   sigmoid channel weights, initial shallow scale 1, and fusion
   `weight * (shallow_scale * shallow_coefficients + deep_projection)`.
   Global context pooling is implemented as spatial average pooling.
4. Residual refinement uses two 3x3 convolutions and SiLU without batch norm.
   Odd sizes use replicated right/bottom padding and inverse-output cropping;
   the deep feature must already match the resulting coefficient resolution.

The simple Python loop in LAA prioritizes inspectability. It is not a fused TTT
kernel. Large feature maps create many sequential steps and a large training
graph. Runtime, peak memory, stability on real data, CUDA and mixed precision
must be evaluated before practical deployment.

## Validation and publication

See `VALIDATION.md` for checks actually run. Tests cover Haar inversion and
gradients, forward/backward shapes, finite gradients, state reset, batch
independence and the analytic update gradient. They do not establish detection
accuracy, reproduce parameter counts, or prove exact manuscript equivalence.

Before a final paper-code release, confirm the assumptions, verify the network
configuration, dataset splits, preprocessing and evaluation protocol, then run
experiments and supply checkpoints. The provided trainer uses six equally
weighted SoftIoU losses, Adam, gradient clipping at 1 and threshold 0.5 for
validation IoU. Its defaults are implementation choices, not verified paper
settings. Encoder downsampling uses average pooling; inputs are padded to
multiples of 16. The test entry reports pixel IoU/nIoU; Pd and Fa are not implemented.
No dataset, private manuscript or experimental images are bundled.

No license has been selected yet; public visibility alone does not grant an
open-source license. The repository owner should select an appropriate license
before inviting reuse. Paper publication metadata and author details should
be added only when confirmed.

## 中文说明

这是依据论文文字新生成的三个模块参考代码，并非产生论文表格结果的原始代码。
公式以嵌入对象存储，其中未确认的维度、融合形式与算子选型已在上方逐项列出。
当前提供基础训练入口，但没有预训练权重或实测检测指标。发布时建议保留上述说明，
待核对公式并重新训练验证后，再更新为正式的论文配套实现。
