# models/discriminator.py

import torch
import torch.nn as nn

class NLayerDiscriminator(nn.Module):
    """Базовый PatchGAN дискриминатор"""
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        input_nc: число каналов на входе (input + target)
        ndf: число фильтров в первом слое
        n_layers: глубина дискриминатора
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4  # размер ядра свертки
        padw = 1  # паддинг для сохранения размера
        sequence = [
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            ),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Строим слои глубже
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw)
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Последний слой без stride
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw)
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Последний выходной слой
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            )
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)