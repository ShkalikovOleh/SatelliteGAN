import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning import LightningModule

from .pix2pix_generator import Pix2PixGenerator
from .discriminator import MultiscalDiscriminator

from losses import *


class Pix2Pix(LightningModule):

    def __init__(self, in_channels, out_channels, lr_gen=2*10**-4, lr_disc=2.10**-4,
                 lambda_fm=10, lambda_l1=100):
        super().__init__()

        self.save_hyperparameters()

        self.gen = Pix2PixGenerator(in_channels, out_channels)
        self.disc = MultiscalDiscriminator(in_channels + out_channels)

        self.gan_criterion = VanillaLoss()
        self.fm_criterion = FMLoss()
        self.vgg_criterion = VGGLoss()
        self.l1_criterion = nn.L1Loss()

    def forward(self, mask):
        return self.gen(mask)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, masks = batch
        fakes = self(masks)

        if optimizer_idx == 0:  # disc training
            real_disc_out = self.disc(torch.cat([masks, images], dim=1))
            fake_disc_out = self.disc(torch.cat([masks, fakes.detach()], dim=1))
            loss = self._calc_disc_loss(real_disc_out, fake_disc_out)

        elif optimizer_idx == 1:  # gen training
            fake_disc_out = self.disc(torch.cat([masks, images], dim=1))
            loss = self._calc_gen_loss(fake_disc_out, images, fakes)

        return loss

    def _calc_disc_loss(self, real_disc_out, fake_disc_out):
        loss = self.gan_criterion(real_disc_out, True, for_disc=True) * 0.5
        loss += self.gan_criterion(fake_disc_out, False, for_disc=True) * 0.5

        self.log(f'loss/disc_GAN', loss, on_step=False, on_epoch=True)

        return loss

    def _calc_gen_loss(self, disc_out, real_images, fake_images):
        gan_loss = self.gan_criterion(disc_out, True, for_disc=False)
        l1_loss = self.l1_criterion(fake_images, real_images)
        vgg_loss = self.vgg_criterion(real_images[:, :3], fake_images[:, :3])
        fm_loss = self.fm_criterion(real_images, fake_images)

        self.log(f'loss/gen_GAN', gan_loss, on_step=False, on_epoch=True)
        self.log(f'loss/gen_L1', l1_loss, on_step=False, on_epoch=True)
        self.log(f'loss/gen_FM', fm_loss, on_step=False, on_epoch=True)
        self.log(f'loss/gen_VGG', vgg_loss, on_step=False, on_epoch=True)

        return gan_loss + self.hparams.lambda_l1 * l1_loss + self.hparams.lambda_fm * (fm_loss + vgg_loss)

    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(),
                       self.hparams.lr_gen, betas=(0.5, 0.999))
        disc_opt = Adam(self.disc.parameters(),
                        self.hparams.lr_disc, betas=(0.5, 0.999))

        return disc_opt, gen_opt
