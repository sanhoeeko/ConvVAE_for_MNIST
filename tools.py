import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class ConvBlock:
    def __init__(self, *image_size):
        self.in_size = image_size[1:]
        self.in_channels = image_size[0]
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None

    @property
    def out_size(self) -> tuple:
        def conv_size_formula(x) -> int:
            return (x - self.kernel_size + 2 * self.padding) // self.stride + 1

        return conv_size_formula(self.in_size[0]), conv_size_formula(self.in_size[1])

    def layer(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def nextLayer(self, out_channels, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_size = self.out_size
        if self.out_channels is not None:
            self.in_channels = self.out_channels
        self.out_channels = out_channels
        return self.layer()


class InvConvBlock:
    def __init__(self, *image_size):
        self.in_size = image_size[1:]
        self.in_channels = image_size[0]
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None

    @property
    def full_size(self):
        return self.in_channels, *self.in_size

    @property
    def out_size(self) -> tuple:
        # padding = input_padding; output_padding is always set to 0.
        def conv_size_formula(x) -> int:
            return (x - 1) * self.stride - 2 * self.padding + self.kernel_size

        return conv_size_formula(self.in_size[0]), conv_size_formula(self.in_size[1])

    def layer(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def nextLayer(self, out_channels, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_size = self.out_size
        if self.out_channels is not None:
            self.in_channels = self.out_channels
        self.out_channels = out_channels
        return self.layer()


class MyDataset(Dataset):
    def __init__(self, data):  # data: iterable
        super().__init__()
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def vec(self):
        return np.hstack(self.data).reshape(-1)

    def histogram(self):
        plt.hist(self.vec(), bins=100)
        plt.show()

    def linearNormalize(self, mean, std):
        vec = self.vec()
        current_mean = vec.mean()
        current_std = vec.std()
        a = std / current_std
        b = mean - a * current_mean

        def f(x):
            return a * x + b

        self.data = list(map(f, self.data))
        return self

    def minmaxNormalize(self, min_v, max_v):
        vec = self.vec()
        current_min = vec.min()
        current_max = vec.max()
        a = (max_v - min_v) / (current_max - current_min)
        b = (current_max * min_v - current_min * max_v) / (current_max - current_min)

        def f(x):
            return a * x + b

        self.data = list(map(f, self.data))
        return self


def MyMNIST(root='./data', train=True, download=True):
    transform = transforms.Compose([transforms.ToTensor(), ])
    dataset = torchvision.datasets.MNIST(root, train, transform, download=download)
    dataset = [inputs for inputs, _ in dataset]
    return MyDataset(dataset)


def to2d(tensor: torch.Tensor):
    # (batch_size, 1, w, h) -> (w * batch_size, h)
    x = tensor.to('cpu').numpy()
    lst = list(map(lambda x: x[0, :, :], x))
    return np.hstack(lst)
