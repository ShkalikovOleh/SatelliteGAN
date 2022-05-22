from .FMLoss import FMLoss
from .GANLoss import VanillaLoss, LSGANLoss, HingeLoss
from .KLDLoss import KLDLoss
from .VGGLoss import VGGLoss

__all__ = ['FMLoss', 'VanillaLoss', 'LSGANLoss',
           'HingeLoss', 'VGGLoss', 'KLDLoss']
