import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class NaiveEncoder(nn.Module):
    """Encoder with partial conv.

    About the details for this architecture, pls see:
    Image Inpainting for Irregular Holes Using Partial Convolutions

    Args:
        in_channels (int): The number of input channels. Default: 3.
        num_layers (int): The number of convolutional layers. Default 7.
        conv_cfg (dict): Config for convolution module. Default:
            {'type': 'PConv', 'multi_channel': True}.
        norm_cfg (dict): Config for norm layer. Default:
            {'type': 'BN'}.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effective on Batch Norm
            and its variants only.
    """

    def __init__(self,
                 in_channels=3,
                 num_layers=7,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False):
        super(NaiveEncoder, self).__init__()
        self.num_layers = num_layers
        self.norm_eval = norm_eval

        self.enc1 = ConvModule(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'))

        self.enc2 = ConvModule(
            64,
            128,
            kernel_size=5,
            stride=2,
            padding=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

        self.enc3 = ConvModule(
            128,
            256,
            kernel_size=5,
            stride=2,
            padding=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

        self.enc4 = ConvModule(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

        for i in range(4, num_layers):
            name = f'enc{i + 1}'
            self.add_module(
                name,
                ConvModule(
                    512,
                    512,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU')))
            self.out_channels = 512

    def init_weights(self, pretrained):
        pass

    def train(self, mode=True):
        super(NaiveEncoder, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        """Forward function for partial conv encoder.

        Args:
            x (torch.Tensor): Masked image with shape (n, c, h, w).
            mask (torch.Tensor): Mask tensor with shape (n, c, h, w).

        Returns:
            dict: Contains the results and middle level features in this \
                module. `hidden_feats` contain the middle feature maps and \
                `hidden_masks` store updated masks.
        """
        # dict for hidden layers of main information flow
        hidden_feats = {}

        hidden_feats['h0'] = x
        h_key_prev = 'h0'

        for i in range(1, self.num_layers + 1):
            l_key = f'enc{i}'
            h_key = f'h{i}'
            hidden_feats[h_key] = getattr(self, l_key)(hidden_feats[h_key_prev])
            h_key_prev = h_key
        outputs = dict(
            out=hidden_feats[f'h{self.num_layers}'],
            hidden_feats=hidden_feats)

        return outputs
