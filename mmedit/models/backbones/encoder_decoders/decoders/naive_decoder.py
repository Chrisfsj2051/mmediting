import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class NaiveDecoder(nn.Module):
    """Decoder with partial conv.

    About the details for this architecture, pls see:
    Image Inpainting for Irregular Holes Using Partial Convolutions

    Args:
        num_layers (int): The number of convolutional layers. Default: 7.
        interpolation (str): The upsample mode. Default: 'nearest'.
        conv_cfg (dict): Config for convolution module. Default:
            {'type': 'PConv', 'multi_channel': True}.
        norm_cfg (dict): Config for norm layer. Default:
            {'type': 'BN'}.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=7,
                 interpolation='nearest',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN')):
        super(NaiveDecoder, self).__init__()
        self.num_layers = num_layers
        self.interpolation = interpolation
        assert in_channels == 512

        for i in range(4, num_layers):
            name = f'dec{i+1}'
            self.add_module(
                name,
                ConvModule(
                    512 + 512,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2)))

        self.dec4 = ConvModule(
            512 + 256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

        self.dec3 = ConvModule(
            256 + 128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

        self.dec2 = ConvModule(
            128 + 64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

        self.dec1 = ConvModule(
            64 + 3,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def init_weights(self):
        pass

    def forward(self, input_dict):
        """Forward Function.

        Args:
            input_dict (dict | torch.Tensor): Input dict with middle features
                or torch.Tensor.

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h, w).
        """
        hidden_feats = input_dict['hidden_feats']
        h_key = 'h{:d}'.format(self.num_layers)
        h = hidden_feats[h_key]

        for i in range(self.num_layers, 0, -1):
            enc_h_key = f'h{i-1}'
            dec_l_key = f'dec{i}'

            h = F.interpolate(h, scale_factor=2, mode=self.interpolation)
            h = torch.cat([h, hidden_feats[enc_h_key]], dim=1)

            h = getattr(self, dec_l_key)(h)

        return h
