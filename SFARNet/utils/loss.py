import torch


def soft_iou_loss(probability, target, eps=1e-6):
    """Per-image soft IoU, averaged across the batch; inputs are probabilities."""
    if probability.shape != target.shape:
        raise ValueError('Prediction and target shapes must match')
    axes = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(axes)
    union = (probability + target - probability * target).sum(axes)
    return (1 - (intersection + eps) / (union + eps)).mean()


def deep_supervision_loss(predictions, target):
    if len(predictions) != 6:
        raise ValueError('Expected five individual maps plus one fused map')
    return torch.stack([soft_iou_loss(p, target) for p in predictions]).sum()
