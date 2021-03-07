import torch
import torch.autograd as autograd
import torch.nn as nn

from .pixelwise_loss import l1_loss
from ..registry import LOSSES

_reduction_modes = ['none', 'mean', 'sum']


@LOSSES.register_module()
class MaskL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, gt_loss_weight=1.0, mask_loss_weight=1.0,
                 reduction='mean', sample_wise=False):
        super(MaskL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.mask_loss_weight = mask_loss_weight
        self.gt_loss_weight = gt_loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, mask_pred, mask_gt, is_disc, weight=None):
        """Forward Function.

        Args:
            mask_pred (Tensor): of shape (N, 1, H, W). Predicted tensor.
            mask_gt (Tensor): of shape (N, 1, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        assert (mask_pred >= 0).all() and (mask_pred <= 1).all()
        mask_target = (mask_gt.clone().detach() if is_disc
                       else torch.zeros_like(mask_gt))
        mask_loss = l1_loss(mask_pred * mask_gt,
                            mask_target * mask_gt,
                            reduction=self.reduction)
        gt_loss = l1_loss(mask_pred * (1 - mask_gt),
                          mask_target * (1 - mask_gt),
                          reduction=self.reduction)

        return mask_loss * self.mask_loss_weight + gt_loss * self.gt_loss_weight
