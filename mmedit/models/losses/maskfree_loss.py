import torch
import torch.autograd as autograd
import torch.nn as nn

from .pixelwise_loss import l1_loss, L1Loss
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


@LOSSES.register_module()
class SoftL1Loss(L1Loss):

    def __init__(self, *args, **kwargs):
        super(SoftL1Loss, self).__init__(*args, **kwargs)

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # import matplotlib.pyplot as plt
        # import numpy as np
        assert weight is not None

        unmask_weight = 1.0 - weight
        final_weight = torch.zeros_like(unmask_weight)
        for ketnel_size in (31, 63, 127, 255):
            final_weight += nn.AvgPool2d((ketnel_size, ketnel_size),
                                         1, ketnel_size // 2)(unmask_weight) / 4
        final_weight = final_weight * weight
        max_weight = final_weight.view(weight.size(0), -1).max(1)[0]
        assert max_weight.shape[0] == final_weight.shape[0]
        norm_final_weight = final_weight / max_weight
        # def to_img(arr):
        #     return np.array(arr.cpu() * 255).astype(np.uint8)
        #
        # weight_img = to_img(weight[0, 0])
        # final_weight_img = to_img(final_weight[0, 0])
        # norm_final_weight_img = to_img(norm_final_weight[0, 0])
        # num_cols = 3
        # num_rows = 1
        # fig, axs = plt.subplots(
        #     num_rows,
        #     num_cols,
        #     squeeze=False,
        #     figsize=(num_cols * 7, num_rows * 6))
        # axs[0, 0].imshow(weight_img, cmap='gray')
        # axs[0, 1].imshow(final_weight_img, cmap='gray')
        # axs[0, 2].imshow(norm_final_weight_img, cmap='gray')
        # plt.show()
        assert norm_final_weight.shape == weight.shape
        return self.loss_weight * l1_loss(
            pred,
            target,
            norm_final_weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
