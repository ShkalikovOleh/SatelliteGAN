import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp


class UNet(LightningModule):

    def __init__(self, in_channels, n_classes, encoder='resnet34', lr=2*10**-4):
        super().__init__()

        self.save_hyperparameters()

        self.model = smp.Unet(encoder_name=encoder,
                              in_channels=in_channels,
                              classes=n_classes)

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)

        loss = self.dice_loss(preds, torch.argmax(masks, dim=1))

        self.log(f'loss/train', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), self.hparams.lr)
        return opt
