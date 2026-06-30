"""
metrics.py
----------
Metrics and loss functions for the 2D UNet trainer.

All functions expect model outputs as **probabilities** in [0, 1]
(i.e. after softmax/sigmoid), not raw logits.

Inputs are assumed to be (B, C, H, W) tensors unless noted otherwise.
Default axes=[2,3] reduces over H and W, keeping B and C.
"""

import torch


# ---------------------------------------------------------------------------
# Confusion-matrix primitives
# ---------------------------------------------------------------------------

def true_positives(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes percentage of true positives along the given axes."""
    tp = y_true * y_pred
    if weight is not None:
        tp = weight * tp
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
    tp_per = torch.sum(tp, axis=axes) / counts
    return tp_per


def true_negatives(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes percentage of true negatives along the given axes."""
    tn = (1 - y_pred) * (1 - y_true)
    if weight is not None:
        tn = weight * tn
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
    tn_per = torch.sum(tn, axis=axes) / counts
    return tn_per


def false_positives(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes percentage of false positives along the given axes."""
    fp = (1 - y_true) * y_pred
    if weight is not None:
        fp = weight * fp
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
    fp_per = torch.sum(fp, axis=axes) / counts
    return fp_per


def false_negatives(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes percentage of false negatives along the given axes."""
    fn = (1 - y_pred) * y_true
    if weight is not None:
        fn = weight * fn
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
    fn_per = torch.sum(fn, axis=axes) / counts
    return fn_per


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def crossentropy_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """
    Computes the crossentropy loss along the given axes and then
    takes the mean across the remaining axes.
    """
    epsilon = 1e-6
    if weight is not None:
        ce = weight * (y_true * torch.log(y_pred + epsilon)
                       + (1 - y_true) * torch.log(1 - y_pred + epsilon))
        counts = torch.sum(weight, axis=axes)
    else:
        ce = (y_true * torch.log(y_pred + epsilon)
              + (1 - y_true) * torch.log(1 - y_pred + epsilon))
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
    ce = - torch.sum(ce, axis=axes) / counts
    return torch.mean(ce)


def dice_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes the Dice loss."""
    return 1 - dice(y_pred, y_true, weight, axes)


def iou_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes the intersection over union (Jaccard index) loss."""
    return 1 - iou(y_pred, y_true, weight, axes)


def mcc_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes the Matthews correlation coefficient (MCC) loss."""
    return 1 - mcc(y_pred, y_true, weight, axes)


def dice_ce_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes the combined Dice and crossentropy loss."""
    return dice_loss(y_pred, y_true, weight, axes) + \
           crossentropy_loss(y_pred, y_true, weight, axes)


def iou_ce_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes the combined IoU (Jaccard) and crossentropy loss."""
    return iou_loss(y_pred, y_true, weight, axes) + \
           crossentropy_loss(y_pred, y_true, weight, axes)


def mcc_ce_loss(y_pred, y_true, weight=None, axes=[2, 3]):
    """Computes the combined MCC and crossentropy loss."""
    return mcc_loss(y_pred, y_true, weight, axes) + \
           crossentropy_loss(y_pred, y_true, weight, axes)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def dice(y_pred, y_true, weight=None, axes=[2, 3]):
    """
    Computes the Dice score along the given axes and then
    takes the mean across the remaining axes.
    """
    epsilon = 1e-12
    tp = true_positives(y_pred, y_true, weight, axes)
    fp = false_positives(y_pred, y_true, weight, axes)
    fn = false_negatives(y_pred, y_true, weight, axes)
    num = 2 * tp
    den = 2 * tp + fp + fn
    dice_score = (num + epsilon) / (den + epsilon)
    return torch.mean(dice_score)


def iou(y_pred, y_true, weight=None, axes=[2, 3]):
    """
    Computes the intersection over union (Jaccard index) along the given axes
    and then takes the mean across the remaining axes.
    """
    epsilon = 1e-12
    tp = true_positives(y_pred, y_true, weight, axes)
    fp = false_positives(y_pred, y_true, weight, axes)
    fn = false_negatives(y_pred, y_true, weight, axes)
    num = tp
    den = tp + fp + fn
    iou_score = (num + epsilon) / (den + epsilon)
    return torch.mean(iou_score)


def mcc(y_pred, y_true, weight=None, axes=[2, 3]):
    """
    Computes the Matthews correlation coefficient (MCC) along the given axes
    and then takes the mean across the remaining axes.
    """
    epsilon = 1e-6
    tp = true_positives(y_pred, y_true, weight, axes)
    tn = true_negatives(y_pred, y_true, weight, axes)
    fp = false_positives(y_pred, y_true, weight, axes)
    fn = false_negatives(y_pred, y_true, weight, axes)
    num = (tp * tn) - (fp * fn)
    # Clamp before sqrt to avoid infinite gradients when denominator is near zero
    den = torch.sqrt(torch.clamp((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), min=epsilon))
    mcc_score = num / (den + epsilon)
    return torch.mean(mcc_score)


# ---------------------------------------------------------------------------
# Loss dispatcher
# ---------------------------------------------------------------------------

_BASE_LOSSES = {
    'dice': dice_loss,
    'ce':   crossentropy_loss,
    'iou':  iou_loss,
    'mcc':  mcc_loss,
}

_COMBINED_PAIRS = {
    'dice_ce': ('dice', 'ce'),
    'iou_ce':  ('iou',  'ce'),
    'mcc_ce':  ('mcc',  'ce'),
}

AVAILABLE_LOSSES = sorted(list(_BASE_LOSSES) + list(_COMBINED_PAIRS))


def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                 weight: torch.Tensor = None,
                 loss: str = 'dice_ce',
                 weight_a: float = 0.5,
                 weight_b: float = 0.5) -> torch.Tensor:
    """
    Unified loss dispatcher.

    Single-component losses (dice, ce, iou, mcc) ignore weight_a / weight_b.
    Combined losses (dice_ce, iou_ce, mcc_ce) blend two base losses:
        weight_a * loss_A(pred, target, weight) + weight_b * loss_B(pred, target, weight)
    """
    if loss in _BASE_LOSSES:
        return _BASE_LOSSES[loss](pred, target, weight)
    if loss in _COMBINED_PAIRS:
        a_name, b_name = _COMBINED_PAIRS[loss]
        return (weight_a * _BASE_LOSSES[a_name](pred, target, weight) +
                weight_b * _BASE_LOSSES[b_name](pred, target, weight))
    raise ValueError(f"Unknown loss '{loss}'. Available: {AVAILABLE_LOSSES}")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dice(pred_probs: torch.Tensor, target: torch.Tensor,
                 threshold: float = 0.5, eps: float = 1e-6) -> dict:
    """
    Returns binary Dice for foreground and boundary channels separately.
    Expects pred_probs as probabilities in [0, 1] (model already applies softmax).
    """
    pred = (pred_probs > threshold).float()
    metrics = {}
    for i, name in enumerate(['foreground', 'boundary']):
        p = pred[:, i]
        t = target[:, i]
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        metrics[f'dice_{name}'] = ((2 * inter + eps) / (denom + eps)).item()
    return metrics
