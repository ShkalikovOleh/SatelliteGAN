from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGAN(nn.Module):

    def __init__(self, in_channels, n_layers=5, ret_feat=True):
        super().__init__()

        assert n_layers >= 2

        self.ret_feat = ret_feat

        conv_blocks = [self._make_conv_block(in_channels, 64)]

        prev_channels = 64
        for i in range(1, n_layers - 1):
            out_channels = min(2**i * 64, 512)
            stride = 2 if i < n_layers - 2 else 1

            conv_blocks.append(
                self._make_conv_block(prev_channels, out_channels, stride=stride))

            prev_channels = out_channels

        conv_blocks.append(
            nn.Conv2d(prev_channels, 1, kernel_size=4, stride=1, padding=1))

        self.conv_blocks = nn.ModuleList(conv_blocks)

    def _make_conv_block(self, in_channels, out_channels, norm=True, stride=2):
        blocks = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=4,
                            stride=stride, padding=1)]

        if norm:
            blocks.append(nn.BatchNorm2d(out_channels))

        blocks.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outs = []
        for layer in self.conv_blocks:
            x = layer(x)
            outs.append(x)

        if self.ret_feat:
            return outs
        else:
            return outs[-1]


class MultiscalDiscriminator(nn.Module):

    def __init__(self, in_channels, n_disc=3, ret_feat=True):
        super().__init__()

        assert n_disc > 1

        self.discs = nn.ModuleList(
            [PatchGAN(in_channels, ret_feat=ret_feat) for _ in range(n_disc)]
        )

        self.pooling = nn.AvgPool2d(kernel_size=3, stride=2,
                                    padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outs = []
        for i, disc in enumerate(self.discs):
            y = disc(x)
            outs.append(y)

            if i < len(self.discs) - 1:
                x = self.pooling(x)

        return outs
