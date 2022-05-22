import torch.nn as nn
import torch.nn.functional as F


class FMLoss(nn.Module):

    def forward(self, real_feat, fake_feat):
        mult = 4. / (len(fake_feat) * len(fake_feat[0]))

        loss = 0.
        for real_outs, fake_outs in zip(real_feat, fake_feat):
            for real_f, fake_f in zip(real_outs[:-1], fake_outs[:-1]):
                loss += F.l1_loss(fake_f, real_f)

        return mult * loss
