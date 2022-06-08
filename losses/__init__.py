from .FMLoss import FMLoss
from .GANLoss import VanillaLoss, LSGANLoss, HingeLoss
from .KLDLoss import KLDLoss
from .VGGLoss import VGGLoss
from .FocalLoss import FocalLoss

__all__ = ['FMLoss', 'VanillaLoss', 'LSGANLoss',
           'HingeLoss', 'VGGLoss', 'KLDLoss', 'FocalLoss']
