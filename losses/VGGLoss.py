import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class VGGLoss(nn.Module):

    def __init__(self, out_features_idxs=[1, 6, 11, 20, 29],
                 weights=[1./32, 1./16, 1./8, 1./4, 1]):
        super().__init__()
        assert len(weights) == len(out_features_idxs)

        self.features = vgg19(pretrained=True).features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.weight = weights
        self.out_idxs = out_features_idxs

    def forward(self, real, fake):
        loss = 0.
        x = fake
        y = real

        for i, (f, w) in enumerate(zip(self.features, self.weight)):
            x = f(x)
            y = f(y)
            if i in self.out_idxs:
                loss += w * F.l1_loss(x, y.detach())

        return loss
