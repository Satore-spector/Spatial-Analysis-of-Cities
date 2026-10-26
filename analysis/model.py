"""The neural network: a small convolutional autoencoder.

An autoencoder has two halves. The encoder squeezes an image (here a stack of
city layers) into a short list of numbers, the "embedding". The decoder tries
to rebuild the image from just those numbers. Training makes the rebuild as
close as possible, which forces the embedding to keep what matters most about
the image's layout. No labels are used anywhere.

Rotations and flips: north should not matter, so training shows each image in
a random one of its 8 rotations/mirror images, and the final embedding of an
image is the average over all 8. Two cities that are rotated copies of each
other therefore get exactly the same embedding.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """The k-th of the 8 rotations/flips of a batch of images (k = 0..7)."""
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch: int, size: int, latent: int = 16, base: int = 32):
        super().__init__()
        n_down = int(math.log2(size)) - 2          # shrink down to a 4 x 4 map
        widths = [min(base * 2 ** i, 256) for i in range(n_down)]
        enc, prev = [], in_ch
        for w in widths:
            enc += [nn.Conv2d(prev, w, 4, stride=2, padding=1), nn.BatchNorm2d(w), nn.LeakyReLU(0.2)]
            prev = w
        self.encoder = nn.Sequential(*enc, nn.Flatten(), nn.Linear(prev * 16, latent))

        dec = []
        rev = widths[::-1]
        for w_in, w_out in zip(rev, rev[1:] + [base]):
            dec += [nn.ConvTranspose2d(w_in, w_out, 4, stride=2, padding=1), nn.BatchNorm2d(w_out),
                    nn.LeakyReLU(0.2)]
        self.unflatten = nn.Sequential(nn.Linear(latent, rev[0] * 16), nn.LeakyReLU(0.2))
        self.decoder = nn.Sequential(*dec, nn.Conv2d(base, in_ch, 3, padding=1), nn.Softplus())
        self._top = rev[0]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.unflatten(z).view(z.shape[0], self._top, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))

    @torch.no_grad()
    def embed_invariant(self, x):
        """Embedding averaged over all 8 rotations/flips (rotation-invariant)."""
        return torch.stack([self.encode(d4(x, k)) for k in range(8)]).mean(0)
