from torch.optim import Adam
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp
from torchmetrics.classification import ConfusionMatrix, CohenKappa, JaccardIndex, Accuracy
from torchmetrics import MetricCollection


class UNet(LightningModule):

    def __init__(self, in_channels, n_classes, encoder='resnet34', lr=2*10**-4, gamma=2):
        super().__init__()

        self.save_hyperparameters()

        self.model = smp.Unet(encoder_name=encoder,
                              in_channels=in_channels,
                              classes=n_classes)

        self.loss = smp.losses.FocalLoss(mode='multiclass', gamma=gamma)

        metrics = MetricCollection([Accuracy(num_classes=n_classes),
                                    JaccardIndex(n_classes),
                                    CohenKappa(n_classes)])
        self.train_metric = metrics.clone(prefix='train/')
        self.val_metric = metrics.clone(prefix='val/')

        self.train_conf_matrix = ConfusionMatrix(n_classes)
        self.val_conf_matrix = ConfusionMatrix(n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        target = masks.argmax(dim=1)

        loss = self.loss(preds, target)

        self.log('train/loss', loss, on_step=False, on_epoch=True)

        metric_value = self.train_metric(preds, target)
        self.log_dict(metric_value, on_epoch=True, on_step=False)

        self.train_conf_matrix(preds, target)

        return loss

    def training_epoch_end(self, outs):
        conf_matrix = self.train_conf_matrix.confmat

        UA = conf_matrix.diag() / conf_matrix.sum(0)
        PA = conf_matrix.diag() / conf_matrix.sum(1)

        for i in range(UA.shape[0]):
            self.log(f'train/UA_{i}', UA[i])
            self.log(f'train/PA_{i}', PA[i])

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        target = masks.argmax(dim=1)

        metric_value = self.val_metric(preds, target)
        self.log_dict(metric_value, on_epoch=True, on_step=False)

        self.val_conf_matrix(preds, target)

    def validation_epoch_end(self, outputs):
        conf_matrix = self.val_conf_matrix.confmat

        UA = conf_matrix.diag() / conf_matrix.sum(0)
        PA = conf_matrix.diag() / conf_matrix.sum(1)

        for i in range(UA.shape[0]):
            self.log(f'val/UA_{i}', UA[i])
            self.log(f'val/PA_{i}', PA[i])

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), self.hparams.lr)
        return opt
