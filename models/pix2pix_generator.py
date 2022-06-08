import torch
import torch.nn as nn


class Pix2PixGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_blocks = nn.ModuleList([
            self._make_down_block(in_channels, 64, norm=False),
            self._make_down_block(64, 128),
            self._make_down_block(128, 256),
            self._make_down_block(256, 512),
            self._make_down_block(512, 512),
            self._make_down_block(512, 512),
            self._make_down_block(512, 512),
        ])

        self.bottleneck = self._make_down_block(512, 512, norm=False)

        self.up_blocks = nn.ModuleList([
            self._make_up_block(512, 512, dropout=True),
            self._make_up_block(1024, 512, dropout=True),
            self._make_up_block(1024, 512, dropout=True),
            self._make_up_block(1024, 512),
            self._make_up_block(1024, 256),
            self._make_up_block(512, 128),
            self._make_up_block(256, 64),
        ])

        self.last_block = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels,
                               kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()
        )

    def _make_down_block(self, in_channels, out_channels, norm=True):
        blocks = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=4, stride=2,
                            padding=1, bias=False)]

        if norm:
            blocks.append(nn.BatchNorm2d(out_channels))

        blocks.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*blocks)

    def _make_up_block(self, in_channels, out_channels, norm=True, dropout=False):
        blocks = [nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=4, stride=2,
                                     padding=1, bias=False)]

        if norm:
            blocks.append(nn.BatchNorm2d(out_channels))

        if dropout:
            blocks.append(nn.Dropout2d())

        blocks.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*blocks)

    def forward(self, x):
        skip_connections = []
        for layer in self.down_blocks:
            x = layer(x)
            skip_connections.append(x)
        skip_connections.reverse()

        x = self.bottleneck(x)

        for skip, layer in zip(skip_connections, self.up_blocks):
            x = layer(x)
            x = torch.cat([x, skip], dim=1)

        return self.last_block(x)
