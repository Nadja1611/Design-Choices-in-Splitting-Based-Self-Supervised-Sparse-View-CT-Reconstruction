# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:41:00 2023

@author: nadja
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Function needed when defining the UNet encoding and decoding parts
def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
    ]

    return nn.Sequential(*list_of_operations)


class DownConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(DownConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# Class for encoding part of the UNet.
# This is the part of the UNet which goes down.
class encoding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.convs_and_relus1 = double_conv_and_ReLU(
            in_channels=self.in_channels,
            out_channels=32
        )

        self.down1 = DownConvolution(
            in_channels=32,
            out_channels=32
        )

        self.convs_and_relus2 = double_conv_and_ReLU(
            in_channels=32,
            out_channels=64
        )

        self.down2 = DownConvolution(
            in_channels=64,
            out_channels=64
        )

        self.convs_and_relus3 = double_conv_and_ReLU(
            in_channels=64,
            out_channels=128
        )

    def forward(self, g):
        g_start = g
        encoding_features = []

        g = self.convs_and_relus1(g)
        encoding_features.append(g)   # skip connection with 32 channels

        g = self.down1(g)

        g = self.convs_and_relus2(g)
        encoding_features.append(g)   # skip connection with 64 channels

        g = self.down2(g)

        g = self.convs_and_relus3(g)

        return g, encoding_features, g_start


# Class for decoding part of the UNet.
# This is the part of the UNet which goes back up.
class decoding(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.out_channels = out_channels

        self.transpose1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2),
            stride=2,
            padding=0
        )

        # After transpose1:
        # g has 64 channels.
        # skip2 has 64 channels.
        # concat gives 128 channels.
        self.convs_and_relus1 = double_conv_and_ReLU(
            in_channels=128,
            out_channels=64
        )

        self.transpose2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(2, 2),
            stride=2,
            padding=0
        )

        # After transpose2:
        # g has 32 channels.
        # skip1 has 32 channels.
        # concat gives 64 channels.
        self.convs_and_relus2 = double_conv_and_ReLU(
            in_channels=64,
            out_channels=32
        )

        self.final_conv = nn.Conv2d(
            in_channels=32,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            padding=1
        )

    @staticmethod
    def match_spatial_size(x, reference):
        """
        Resize x so that its height and width match reference.

        This does nothing for sizes that already match, for example 336 x 336.
        It makes the model safe for image/sinogram sizes where downsampling and
        upsampling do not perfectly invert each other.
        """
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return x

    def forward(self, g, encoding_features, g_start):
        skip1 = encoding_features[0]   # feature map from first encoder block: 32 channels
        skip2 = encoding_features[1]   # feature map from second encoder block: 64 channels

        g = self.transpose1(g)
        g = self.match_spatial_size(g, skip2)
        g = torch.cat([g, skip2], dim=1)
        g = self.convs_and_relus1(g)

        g = self.transpose2(g)
        g = self.match_spatial_size(g, skip1)
        g = torch.cat([g, skip1], dim=1)
        g = self.convs_and_relus2(g)

        g = self.final_conv(g)
        g = torch.sigmoid(g)

        # Optional residual connection.
        # Only enable this if your target is residual denoising/interpolation
        # and the input/output shapes and channels are identical.
        #
        # g = g_start + g

        return g


# Class for the UNet model itself
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)

    def forward(self, g):
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)

        return g


# Class for the UNet interpolation model
class UNet_interpolation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)

    def forward(self, g):
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)

        return g