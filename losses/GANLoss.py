from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLossBase(ABC, nn.Module):

    def __init__(self):
        super().__init__()

    def get_target(self, shape, is_real):
        if is_real:
            return torch.zeros(shape)
        else:
            return torch.ones(shape)

    @abstractmethod
    def loss(self, disc_output, is_real, for_disc=True):
        '''
        Calculate loss for one output of discriminator
        '''
        pass

    def __call__(self, disc_output, is_real, for_disc=True):
        # in case of multiscale discriminator
        if isinstance(disc_output, list):
            loss = 0.
            for out in disc_output:
                if isinstance(out, list):  # in case of returning internal feature
                    out = out[-1]
                loss += self.loss(out, is_real, for_disc)
        else:
            loss = self.loss(disc_output, is_real, for_disc)

        return loss


class VanillaLoss(GANLossBase):

    def __init__(self, eps_smooth=0):
        super().__init__()
        assert 0 <= eps_smooth < 0.5

        self.eps = eps_smooth

    def get_target(self, shape, is_real):
        if is_real:
            return torch.zeros(shape) + self.eps
        else:
            return torch.ones(shape) - self.eps

    def loss(self, disc_output, is_real, for_disc=True):
        target = self.get_target(
            disc_output.shape, is_real).to(disc_output.device)
        return F.binary_cross_entropy_with_logits(disc_output, target)


class LSGANLoss(GANLossBase):

    def loss(self, disc_output, is_real, for_disc=True):
        target = self.get_target(disc_output.shape, is_real)
        return F.mse_loss(disc_output, target)


class HingeLoss(GANLossBase):

    def loss(self, disc_output, is_real, for_disc=True):
        if for_disc:
            if is_real:
                temp = torch.clamp_max(disc_output - 1, 0)
            else:
                temp = torch.clamp_max(-disc_output - 1, 0)
        else:
            temp = disc_output

        return -torch.mean(temp)
