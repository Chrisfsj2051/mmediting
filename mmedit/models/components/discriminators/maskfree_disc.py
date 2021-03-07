import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import load_checkpoint

from mmedit.models import build_component, SimpleEncoderDecoder
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class MaskFreeDiscriminator(nn.Module):

    def __init__(self, img_disc_cfg, mask_disc_cfg):
        super(MaskFreeDiscriminator, self).__init__()
        self.img_disc = build_component(img_disc_cfg)
        self.mask_disc = build_component(mask_disc_cfg)

    def forward(self, x):
        img_pred = self.img_disc(x)
        mask_pred = self.mask_disc(x).sigmoid()
        return img_pred, mask_pred

    def init_weights(self, pretrained=None):
        pass

@COMPONENTS.register_module()
class MaskDiscriminator(SimpleEncoderDecoder):
    pass