"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import einops
# from denoising_diffusion_pytorch.denoising_diffusion_pytorch import LinearAttention, PreNorm, Residual


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out = self.block(x)
        return out


def Upsample(dim_in, dim_out):
    """
    Better Deconvolution without the checkerboard problem [1].
    [1] https://distill.pub/2016/deconv-checkerboard/
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim_in, dim_out, 3, padding = 1)
    )


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            Upsample(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 d: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 bn: bool = True,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        self.encoder = nn.Sequential(
            VQVAEEncBlock(num_channels, d),
            *[VQVAEEncBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            *[nn.Sequential(ResBlock(d, d, bn=bn), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
        )

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.zeros(1).int())
        self.register_buffer('H_prime', torch.zeros(1).int())
        self.register_buffer('W_prime', torch.zeros(1).int())

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return (B, C, H, W') where W' <= W
        """
        out = self.encoder(x)
        if not self.is_num_tokens_updated:
            self.H_prime += out.shape[2]
            self.W_prime += out.shape[3]
            self.num_tokens += self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 d: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 img_size: int,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        pos_emb_dim = d
        self.pos_emb = torch.nn.parameter.Parameter(torch.randn(pos_emb_dim, img_size // downsample_rate, img_size // downsample_rate))

        self.decoder = nn.Sequential(
            nn.Conv2d(d+pos_emb_dim, d, kernel_size=3, padding=1),
            *[nn.Sequential(ResBlock(d, d), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
            *[VQVAEDecBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            Upsample(d, num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        :param x: output from the encoder (B, C, H, W')
        :return  (B, C, H, W)
        """
        pos_emb = einops.repeat(self.pos_emb, 'd h w -> b d h w', b=x.shape[0])
        x = torch.cat((x, pos_emb), dim=1)

        out = self.decoder(x)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 2, 4, 128)  # (batch, channels, height, width)

    encoder = VQVAEEncoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder = VQVAEDecoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder.upsample_size = torch.IntTensor(np.array(x.shape[2:]))

    z = encoder(x)
    x_recons = decoder(z)

    print('x.shape:', x.shape)
    print('z.shape:', z.shape)
    print('x_recons.shape:', x_recons.shape)
