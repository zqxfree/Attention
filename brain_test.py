from math import sqrt, log
from collections.abc import Iterable
import numpy as np
import pandas as pd
from numpy import inf, pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from brain import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 100
learning_rate = 0.001

cifar10_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])


transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root=r'E:/PycharmProjects/dataset',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=False)
#
# test_dataset = torchvision.datasets.MNIST(root='E:/PycharmProjects/dataset',
#                                           train=False,
#                                           transform=transforms.ToTensor(),
#                                           download=False)
train_dataset = torchvision.datasets.CIFAR10(root=r'E:/PycharmProjects/dataset',
                                             train=True,
                                             transform=transforms.ToTensor(), # cifar10_transform,
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root=r'E:/PycharmProjects/dataset',
                                            train=False,
                                            transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class ImageView(nn.Module):
    def __init__(self, image_size, reverse=False):
        super(ImageView, self).__init__()
        self.image_size = image_size
        self.reverse = reverse

    def forward(self, x):
        a, b = 28 // self.image_size, self.image_size
        if self.reverse:
            x = x.view(x.shape[:-2] + (a, a, b, b)).transpose(-2, -3).contiguous()
            return x.view(x.shape[:-4] + (a*b, a*b))
        x = x.view(x.shape[:-2] + (a, b, a, b)).transpose(-2, -3).contiguous()
        return x.view(x.shape[:-4] + (a*a, b*b))


class Roll(nn.Module):
    def __init__(self, image_size):
        super(Roll, self).__init__()
        self.image_size = image_size

    def forward(self, x):
        return x.roll((self.image_size, self.image_size), dims=(-1, -2))


class GAP(nn.Module):
    def __init__(self, dims, keepdim=False, mean=True):
        super(GAP, self).__init__()
        self.dims = dims
        self.keepdim= keepdim
        self.mean = mean

    def forward(self, x):
        dims = self.dims if '__getitem__' in dir(self.dims) else list(range(self.dims, x.dim()))
        return x.mean(dims, keepdim=self.keepdim) if self.mean else x.sum(dims, keepdim=self.keepdim)


class CyclicPad(nn.Module):
    def __init__(self, pad_size):
        super(CyclicPad, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        x = torch.cat((x[:, :, :, -self.pad_size:], x, x[:, :, :, :self.pad_size]), 3)
        x = torch.cat((x[:, :, -self.pad_size:, :], x, x[:, :, :self.pad_size, :]), 2)
        return x


class RandomRelu(nn.Module):
    def __init__(self, c):
        super(RandomRelu, self).__init__()
        self.alpha = nn.Parameter(torch.randint(0, 2, (c, 1, 1), dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(c, 1, 1))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.relu(x) + self.beta * x


# padding_mode in ('zeros', 'reflect', 'replicate', 'circular')
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class MLPLayer(nn.Module):
    def __init__(self, n):
        super(MLPLayer, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(n, n),
            nn.LayerNorm(n, elementwise_affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, n, hidden_size, label_num):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n, hidden_size),
            MLPLayer(hidden_size),
            MLPLayer(hidden_size),
            MLPLayer(hidden_size),
            MLPLayer(hidden_size),
            MLPLayer(hidden_size),
            MLPLayer(hidden_size),
            MLPLayer(hidden_size),
            nn.Linear(hidden_size, label_num)
        )

    def forward(self, x):
        return self.layer(x)


# model = ConvNet(10).to(device)
# model = SparseConv2d(2, 1, 4, 4, (28,28), 10)
# model = ConvRNN([28, 28], [1, 16, 32], [[3]*2, [3]*2], [[1]*2, [1]*2], 10)
# model = DeepSwapConv(64, 49, [1, 16, 64], [8, 32], [[9, 9]] * 2, 10, True, (7, 3))
# model = DeepTransformConv([28, 28], (1, 16, 32), 5, 10, True, False, (7, 3))
# model = nn.Sequential(
#     nn.Conv2d(1, 16, 5),
#     nn.ReLU6(),
#     PMAXPool2d(),
#     nn.Conv2d(16, 32, 3),
#     nn.ReLU6(),
#     nn.Conv2d(32, 64, 3, padding=1),
#     nn.BatchNorm2d(64),
#     nn.ReLU(),
#     PMAXPool2d(),
#     nn.Flatten(1),
#     nn.Linear(64*5*5, 10)
# )
from collections import  OrderedDict
# model1 = nn.Sequential(
#     nn.Conv2d(1, 32, 5, padding=2),
#     nn.ReLU6(),
#     nn.Conv2d(32, 64, 5, padding=2),
#     nn.BatchNorm2d(64),
#     nn.ReLU6(),
#     PMAXPool2d(p=0.5),
#     ResNetLayer(64, 64, 3, p=1., random_method='rand'),
#     ResNetLayer(64, 64, 3, p=1., random_method='rand'),
#     nn.ReLU6(),
#     PMAXPool2d(p=0.5),
#     nn.Flatten(1),
#     nn.Linear(64 * 12 * 16, 10)
# )

entropy_model = AutoEncoder()

model2 = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    PatchMerge(),
    ConvLayer(64),
    ConvLayer(64),
    ConvLayer(64),
    ConvLayer(64),
    ConvLayer(64),
    nn.Flatten(1),
    nn.Linear(64 * 14 * 14, 10)
)

model3 = nn.Sequential(
    # nn.Unfold(7, stride=3),
    # SequenceEncoding(32, 32, need_grad=False),
    # Transpose(1, 2),
    # Unsqueeze(1),
    SplitConv(3, 16, 3),
    SplitConv(16, 32, 3),
    SplitConv(32, 64, 3),
    SplitConv(64, 128, 3),
    PMAXPool2d(p=0.9),
    SplitConv(128, 128, 3),
    SplitConv(128, 128, 3),
    SplitConv(128, 128, 3),
    PMAXPool2d(p=0.9),
    nn.Flatten(1),
    nn.Linear(128 * 8 * 8, 10)
)



class Patching(nn.Module):
    def __init__(self):
        super(Patching, self).__init__()
        self.patch = nn.Sequential(
            nn.Unfold(4, stride=4),
            Transpose(1, 2),
            SequenceEncoding(49, 16, need_grad=False),
            Unsqueeze(1),
        )

    def forward(self, x):
        return self.patch(x[0]), x[1]


class FirstConv(nn.Module):
    def __init__(self):
        super(FirstConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            # nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return self.layer(x[0]), x[1]

size = 16

model4 = nn.Sequential(
    # nn.Unfold(7, stride=3),
    # Transpose(1, 2),
    # SequenceEncoding(64, 49, need_grad=False),
    # Unsqueeze(1),
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU6(),
    nn.MaxPool2d(2),
    LightAttention(32, (16, 16), 8),
    ResNetLayer((32, 32), 3, (16, 16)),
    LightAttention(32, (16, 16), 8),
    ResNetLayer((32, 32), 3, (16, 16)),
    nn.MaxPool2d(2),
    nn.Flatten(1),
    nn.Linear(32 * (size // 2) * (size // 2), 10)
)

criterion = nn.CrossEntropyLoss()
a = protolearn(model4, train_loader, test_loader, criterion, lr=0.001, num_epochs=100, thres_train_accuracy=1,
               thres_test_accuracy=1.,
               train_over_thres=5, test_over_thres=0,num_classes=10, image_patch_kernels=None, validate_test_split=0.,
               move_data_device=True, device=None)
print(a[2])
print(a[3])

