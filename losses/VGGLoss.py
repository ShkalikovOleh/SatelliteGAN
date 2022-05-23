import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class VGGLoss(nn.Module):

    def __init__(self, out_features_idxs=[1, 6, 11, 20, 29],
                 weights=[1./32, 1./16, 1./8, 1./4, 1]):
        super().__init__()
        assert len(weights) == len(out_features_idxs)

        features = vgg19(pretrained=True).features.eval()
        for param in features.parameters():
            param.requires_grad = False

        blocks = []
        start = 0
        for end in out_features_idxs:
            blocks.append(nn.Sequential(*features[start:end + 1]))
            start = end + 1

        self.blocks = nn.ModuleList(blocks)

        self.weight = weights

    def forward(self, real, fake):
        loss = 0.
        x = fake
        y = real

        for f, w in zip(self.blocks, self.weight):
            x = f(x)
            y = f(y)
            loss += w * F.l1_loss(x, y.detach())

        return loss
