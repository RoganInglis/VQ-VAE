import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm


class PixelCNN(nn.Module):
    def __init__(self, n_layers=1, layer_sizes=(256,), activation=nn.ReLU(), use_bias=True, device=None, dtype=None):
        super(PixelCNN, self).__init__()

        # TODO

        assert len(layer_sizes) == 1 or len(layer_sizes) == n_layers, 'Layer sizes must be iterable with len 1 or n_layers'

        if len(layer_sizes) == 1 and n_layers > 1:
            layer_sizes = [layer_sizes[0]]*n_layers

        layers = [nn.Flatten()]

        layers.extend([
            Dense(size, activation=activation, use_bias=use_bias, device=device, dtype=dtype)
            for size in layer_sizes
        ])


        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)