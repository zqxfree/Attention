import math
import time
from math import sqrt, log, ceil, pi, inf
from collections.abc import Iterable
from collections import OrderedDict

import numpy as np
from numpy import inf, pi, nan
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SoftTanh(nn.Module):
    def __init__(self, tanh=True):
        super(SoftTanh, self).__init__()
        self.tanh = tanh

    def forward(self, x):
        x = x / torch.sqrt(1 + x * x)
        return x if self.tanh else 0.5 * (1 + x)


class QuadReLU(nn.Module):
    def forward(self, x):
        return torch.where(x > 0., torch.where(x < 1., x * x / 2, x - 0.5), torch.tensor(0.))


class GeneralNorm(nn.Module):
    def __init__(self, feature_ndim, normalized_dims, affine_dims, affine_shape, alpha=1., beta=0., affine_grad=True):
        super(GeneralNorm, self).__init__()
        if '__getitem__' not in dir(affine_grad):
            affine_grad = [affine_grad] * 2
        affine_shape_expansion = torch.ones(feature_ndim, dtype=torch.int64)
        affine_shape_expansion[affine_dims] = torch.LongTensor(affine_shape)
        self.affine_grad = affine_grad
        self.normalized_dims = normalized_dims
        self.affine_shape = affine_shape_expansion.tolist()
        self.alpha = self.init_alpha_weights(alpha, 0)
        self.beta = self.init_alpha_weights(beta, 1)

    def init_alpha_weights(self, weight, weight_idx):
        if isinstance(weight, torch.Tensor):
            assert weight.size() == self.affine_shape, "affine weight shape is wrong!"
            return nn.Parameter(weight, requires_grad=self.affine_grad[weight_idx])
        elif self.affine_grad[weight_idx]:
            return nn.Parameter(torch.full(self.affine_shape, weight))
        else:
            return nn.Parameter(torch.tensor(weight), requires_grad=False)

    def forward(self, x):
        x = x - x.mean(self.normalized_dims, keepdim=True)
        var = x.var(self.normalized_dims, unbiased=False, keepdim=True) + 1e-6
        return self.alpha * (x / var.sqrt()) + self.beta


class FeatureNorm(nn.Module):
    def __init__(self, num_features, norm2d=True):
        super(FeatureNorm, self).__init__()
        self.norm2d = norm2d
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        shape = x.shape[2:]
        if self.norm2d:
            x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        if self.norm2d:
            x = x.unflatten(2, shape)
        return x


class EpsSoftmax(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(EpsSoftmax, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        x_max = x.data.max()
        if x_max > 0:
            x = x - x_max
        y = x.exp()
        return y / (y.sum(self.dim, keepdim=True) + self.eps)


class BiReLU(nn.Module):
    def __init__(self, n, num_dims=1):
        super(BiReLU, self).__init__()
        self.bias = nn.Parameter(0.5 * torch.ones((n,) + (1,) * (num_dims - 1)))
        self.relu = nn.ReLU()

    def forward(self, x):
        abs_bias = self.bias.abs()
        return self.relu(x - abs_bias) - self.relu(-x - abs_bias)


class PRelU6(nn.Module):
    def __init__(self, alpha=0.02):
        super(PRelU6, self).__init__()
        self.relu1 = nn.ReLU6()
        self.relu2 = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.relu1(x).minimum(self.relu2(x))


class FFT2d(nn.Module):
    def __init__(self, patch_size=None):
        super(FFT2d, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        if self.patch_size is not None:
            x = x.unflatten(-2, (-1, self.patch_size[0]))
            x = x.unflatten(-1, (-1, self.patch_size[1]))
            x = x.transpose(-3, -2)
        x = F.pad(x, (1, 0, 1, 0))
        f_x = torch.fft.fft2(x, norm='ortho')
        f_x = F.pad(f_x, (-1, 0, -1, 0))
        if self.patch_size is not None:
            f_x = f_x.transpose(-3, -2)
            f_x = f_x.flatten(-2)
            f_x = f_x.flatten(-3, -2)
        return torch.cat([torch.real(f_x), torch.imag(f_x)], -3)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


def position_tensor(row_features, col_features):
    return torch.sin(torch.arange(row_features)[:, None] / torch.Tensor(
        [10000 ** ((i - i % 2) / col_features) for i in range(col_features)]) + torch.Tensor(
        [pi / 2 if i % 2 else 0. for i in range(col_features)]))


class SequenceEncoding(nn.Module):
    def __init__(self, row_features, col_features, need_grad=True):
        super(SequenceEncoding, self).__init__()
        if need_grad:
            self.pos_code = nn.Parameter(position_tensor(row_features, col_features))
        else:
            self.register_buffer('pos_code', position_tensor(row_features, col_features))

    def forward(self, x):
        return x + self.pos_code


class ImagePosEncoding(nn.Module):
    def __init__(self, row_features, col_features, need_grad=False):
        super(ImagePosEncoding, self).__init__()
        self.pos_code1 = nn.Parameter(position_tensor(row_features, col_features), requires_grad=need_grad)
        self.pos_code2 = nn.Parameter(position_tensor(row_features, col_features).flip((0, 1)), requires_grad=need_grad)
        self.pos_code3 = nn.Parameter(position_tensor(col_features, row_features).t().flipud(), requires_grad=need_grad)
        self.pos_code4 = nn.Parameter(position_tensor(col_features, row_features).t().fliplr(), requires_grad=need_grad)

    def forward(self, x):
        bg = self.pos_code1 + self.pos_code2 + self.pos_code3 + self.pos_code4
        return x + bg / bg.abs().max()


class BioActivation(nn.Module):
    def __init__(self, xlimit=None):
        super(BioActivation, self).__init__()
        self.xlimit = xlimit
        self.ylimit = xlimit * xlimit if xlimit is not None else None

    def forward(self, x):
        if self.xlimit is None:
            return torch.where(x <= 0., 0., x * x)
        return torch.where(x <= 0., 0., torch.where(x >= self.xlimit, self.ylimit, x * x))


class SoftAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, softmax=True):
        super(SoftAvgPool2d, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__getitem__' not in dir(padding):
            padding = (padding, padding)
        if stride is None:
            stride = kernel_size
        elif '__getitem__' not in dir(stride):
            stride = (stride, stride)
        self.kernels = kernel_size[0] * kernel_size[1]
        self.shift = (2 * padding[0] - kernel_size[0], 2 * padding[1] - kernel_size[1])
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        if softmax:
            self.softmax = EpsSoftmax(2)
        else:
            self.register_module('softmax', None)

    def forward(self, x):
        col_dim, row_dim = x.shape[2:]
        x = self.unfold(x)
        x = x.unflatten(1, (-1, self.kernels))
        if self.softmax is not None:
            x = self.softmax(x).mul(x).sum(2)
        else:
            x = x.pow(2).sum(2) / (x.sum(2) + 1e-6)
        return x.unflatten(2, (-1, (row_dim + self.shift[1]) // self.stride[1] + 1))


class PMAXPool1d(nn.Module):
    def __init__(self, kernel_size=2, p=0.9):
        super(PMAXPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.p0 = kernel_size * (1 - p) / (kernel_size - 1)
        self.p1 = kernel_size * p - self.p0
        self.max_pool = nn.MaxPool1d(kernel_size, return_indices=True)
        self.avg_pool = nn.AvgPool1d(kernel_size)

    def forward(self, x):
        y, max_indices = self.max_pool(x)
        y = self.p0 * x + self.p1 * F.max_unpool1d(y, max_indices, self.kernel_size, output_size=x.size())
        return self.avg_pool(y)


class PMAXPool2d(nn.Module):
    def __init__(self, kernel_size=2, p=0.7, print=True):
        super(PMAXPool2d, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.p0 = kernel_size[0] * kernel_size[1] * (1 - p) / (kernel_size[0] * kernel_size[1] - 1)
        self.p1 = kernel_size[0] * kernel_size[1] * p - self.p0
        self.max_pool = nn.MaxPool2d(kernel_size, return_indices=True)
        self.avg_pool = nn.AvgPool2d(kernel_size)
        self.pr = print

    def forward(self, x):
        x, printable = x
        y, max_indices = self.max_pool(x)
        y = self.p0 * x + self.p1 * F.max_unpool2d(y, max_indices, self.kernel_size, output_size=x.size())
        return (self.avg_pool(y), printable) if self.pr else self.avg_pool(y)


class SplitConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=False, residual=True):
        super(SplitConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )
        self.dropout = nn.Dropout2d() if dropout else self.register_module('dropout', None)
        if residual and in_channels == out_channels:
            self.add_module('residual', Residual(0., random_method='constant'))
        else:
            self.add_module('residual', None)

    def forward(self, x):
        y = self.conv1(x) * self.conv2(x)
        if self.dropout is not None:
            y = self.dropout(y)
        return y if self.residual is None else self.residual(x, y)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class DropoutSoftmax(nn.Module):
    def __init__(self, normalized_size, dim=-1):
        super(DropoutSoftmax, self).__init__()
        self.dim = dim
        self.normalized_size = normalized_size
        self.norm = nn.LayerNorm(normalized_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # x.shape = (N, *)
        # return self.softmax(torch.where(torch.rand_like(x) > (self.normalized_size / x.size(self.dim)), x, -inf))
        if self.dim not in (-1, x.dim() - 1):
            x = x.transpose(self.dim, -1)
        if self.training:
            shape = x.shape
            x = x.flatten(0, -2)
            row_indices = torch.arange(x.size(0), device=x.device).repeat(self.normalized_size, 1).t()
            col_indices = torch.stack(
                [torch.randperm(x.size(1), device=x.device)[:self.normalized_size] for _ in range(x.size(0))])
            x_part = self.softmax(self.norm(x[row_indices, col_indices]))
            x = torch.zeros_like(x)
            x[row_indices, col_indices] = x_part
            x = x.view(shape)
        else:
            x = self.softmax(x)
        if self.dim not in (-1, x.dim() - 1):
            x = x.transpose(self.dim, -1)
        return x


class AutoLayerNorm(nn.Module):
    def __init__(self, norm_dim):
        super(AutoLayerNorm, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        return F.layer_norm(x, x.shape[self.norm_dim:])


class Residual(nn.Module):
    def __init__(self, p=1., random_method='constant', dropout=False, balanced_residual=False, requires_grad=False):
        super(Residual, self).__init__()
        self.dropout = dropout
        self.balanced_residual = balanced_residual
        if dropout:
            self.register_buffer('alpha', torch.tensor(p))
        elif random_method == 'Bernoulli':
            self.alpha = nn.Parameter(torch.tensor(float(torch.rand(1).lt(p).item())), requires_grad=requires_grad)
        elif random_method == 'constant':
            self.alpha = nn.Parameter(torch.tensor(p), requires_grad=requires_grad)
        else:
            self.alpha = nn.Parameter(p * torch.rand(1), requires_grad=requires_grad)

    def forward(self, x, f_x):
        if self.dropout and self.training:
            if torch.rand(1) < self.alpha:
                f_x = x
        elif self.balanced_residual:
            f_x = f_x + self.alpha * (x - f_x)
        else:
            f_x = self.alpha * f_x + x
        return f_x


class ConvAttn(nn.Module):
    def __init__(self, channels, seq_len, num_features, kernel_size):
        super(ConvAttn, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = [kernel_size] * 2
        if kernel_size[0] % 2 == 0:
            kernel_size[0] -= 1
        if kernel_size[1] % 2 == 0:
            kernel_size[1] -= 1
        self.residual = ResNetLayer(channels, kernel_size)
        self.attn_res = Residual(0.5, random_method='constant', balanced_residual=True, requires_grad=True)
        self.channels = channels
        self.seq_query = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, (kernel_size[0], num_features),
                      padding=(kernel_size[0] // 2, 0)),
            # nn.LayerNorm([seq_len, 1]),
            # DropoutSoftmax(seq_len // 2, 2)
            nn.Softmax(2)
        )
        self.seq_key = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, (kernel_size[0], num_features),
                      padding=(kernel_size[0] // 2, 0)),
            # nn.LayerNorm([seq_len, 1]),
            # DropoutSoftmax(seq_len // 2, 2)
            nn.Softmax(2)
        )
        self.feature_query = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, (seq_len, kernel_size[1]),
                      padding=(0, kernel_size[1] // 2)),
            # nn.LayerNorm([1, num_features]),
            # DropoutSoftmax(num_features // 2)
            nn.Softmax(3)
        )
        self.feature_key = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, (seq_len, kernel_size[1]),
                      padding=(0, kernel_size[1] // 2)),
            # nn.LayerNorm([1, num_features]),
            # DropoutSoftmax(num_features // 2)
            nn.Softmax(3)
        )
        # self.value = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        # self.value_residual = Residual(1., random_method='constant', balanced_residul=False, requires_grad=True)
        # self.attn_norm = AutoLayerNorm(-1)
        # self.softmax = nn.Softmax(-1)
        # self.dropout=nn.Dropout()
        self.norm = nn.BatchNorm2d(channels, affine=True)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        # x.shape = (batch_size, channels, seq_len, num_features)
        # x_norm = self.norm(x)
        x = self.residual(x)
        # v = self.norm(x)
        q1 = self.seq_query(x)
        k1 = self.seq_key(x)
        q2 = self.feature_query(x)
        k2 = self.feature_key(x)
        attn1 = q1.matmul(k1.transpose(2, 3))
        attn2 = q2.transpose(2, 3).matmul(k2)
        # if printable:
        #     print(q1[0, 0, :, 0])
        #     print(k1[0, 0, :, 0])
        #     print(q2[0, 0, 0, :])
        #     print(k2[0, 0, 0, :])
        attn = attn1.matmul(x) + x.matmul(attn2) + attn1.matmul(x).matmul(attn2)
        # attn = self.dropout(attn)
        attn = self.attn_res(x, attn)
        return attn


class AttenPool(nn.Module):
    def __init__(self, channels, seq_len, num_features, kernel_size, patch_size, patch_paddings=0, drop_attn=False, printable=True):
        super(AttenPool, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = [kernel_size] * 2
        if kernel_size[0] % 2 == 0:
            kernel_size[0] -= 1
        if kernel_size[1] % 2 == 0:
            kernel_size[1] -= 1
        if '__getitem__' not in dir(patch_size):
            patch_size = [patch_size] * 2
        if '__getitem__' not in dir(patch_paddings):
            patch_paddings = [patch_paddings] * 2
        self.patch_size = patch_size
        self.channels = channels
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.LayerNorm([seq_len, num_features]),
            nn.Unfold(patch_size, padding=patch_paddings, stride=patch_size),
            nn.Softmax(2),
            nn.Unflatten(1, (channels, -1))
        )
        self.value = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.fold = nn.Fold((seq_len, num_features), patch_size, padding=patch_paddings, stride=patch_size)
        self.attn_res = Residual(0.5, random_method='constant', balanced_residul=True, requires_grad=True)
        self.res = ResNetLayer(channels, kernel_size)
        self.norm = nn.BatchNorm2d(channels)
        self.pool = nn.AvgPool2d(patch_size)
        self.drop_attn = nn.Dropout2d() if drop_attn else None
        self.printable = printable

    def image_attn(self, x):
        x_norm = self.norm(x)
        attn = self.query(x_norm)
        value = self.value(x_norm)
        shape = attn.shape
        attn = attn.sum(3, keepdim=True).expand(shape)
        if self.drop_attn is not None:
            attn = self.drop_attn(attn.transpose(1, 2)).transpose(1, 2)
        attn = self.fold(attn.flatten(1, 2))
        attn *= value
        attn = self.attn_res(x, attn)
        return self.pool((self.patch_size[0] * self.patch_size[1]) * attn)

    def forward(self, x):
        x, printable = x
        attn = self.image_attn(x)
        # if printable:
        #     print(q1[0, 0, :, 0])
        #     print(k1[0, 0, :, 0])
        #     print(q2[0, 0, 0, :])
        #     print(k2[0, 0, 0, :])
        return (attn, printable) if self.printable else attn


class ConvAttn2d(nn.Module):
    def __init__(self, channels, hidden_channels, seq_len, num_features, kernel_size, patch_size, patch_paddings=0,
                 drop_attn=False):
        super(ConvAttn2d, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = [kernel_size] * 2
        if kernel_size[0] % 2 == 0:
            kernel_size[0] -= 1
        if kernel_size[1] % 2 == 0:
            kernel_size[1] -= 1
        if '__getitem__' not in dir(patch_size):
            patch_size = [patch_size] * 2
        if '__getitem__' not in dir(patch_paddings):
            patch_paddings = [patch_paddings] * 2
        self.patch_size = patch_size
        self.channels = channels
        self.query1 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden_channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            # nn.InstanceNorm2d(hidden_channels, momentum=0., affine=True),
            # nn.LayerNorm([seq_len, num_features]),
            nn.Unfold(patch_size, padding=patch_paddings, stride=patch_size),
            nn.Unflatten(1, (hidden_channels, -1)),
            # nn.LayerNorm(patch_size[0] * patch_size[1]),
            nn.Softmax(2)
            # nn.LayerNorm(((seq_len + 2 * patch_paddings[0]) // patch_size[0]) * ((num_features + 2 * patch_paddings[1]) // patch_size[1])),
        )
        self.query2 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden_channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            # nn.InstanceNorm2d(hidden_channels, momentum=0., affine=True),
            # nn.LayerNorm([seq_len, num_features]),
            nn.Unfold(patch_size, padding=patch_paddings, stride=patch_size),
            nn.Unflatten(1, (hidden_channels, -1)),
            # nn.LayerNorm(patch_size[0] * patch_size[1]),
            nn.Softmax(2)
            # nn.LayerNorm(((seq_len + 2 * patch_paddings[0]) // patch_size[0]) * ((num_features + 2 * patch_paddings[1]) // patch_size[1])),
        )
        self.value_conv1 = nn.Conv2d(channels, hidden_channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.value_conv2 = nn.Conv2d(hidden_channels, channels, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.fold = nn.Fold((seq_len, num_features), patch_size, padding=patch_paddings, stride=patch_size)
        self.attn_res = Residual(0.5, random_method='constant', balanced_residul=True, requires_grad=True)
        self.res = ResNetLayer(channels, kernel_size)
        self.norm = nn.BatchNorm2d(channels)
        self.drop_attn = nn.Dropout2d() if drop_attn else None

    def image_attn(self, x):
        value = self.value_conv1(x)
        attn1 = self.query1(x)
        attn2 = self.query2(x.roll((self.patch_size[0] // 2, self.patch_size[1] // 2), (2, 3)))
        shape1 = attn1.shape
        shape2 = attn2.shape
        attn1 = attn1.sum(2, keepdim=True).expand(shape1)
        attn2 = attn2.sum(2, keepdim=True).expand(shape2)
        if self.drop_attn is not None:
            attn1 = self.drop_attn(attn1.transpose(1, 3)).transpose(1, 3)
            attn2 = self.drop_attn(attn2.transpose(1, 3)).transpose(1, 3)
        attn = attn1 + attn2
        attn = self.fold(attn.flatten(1, 2))
        attn *= value
        # attn = self.value_conv2(attn)
        attn = self.attn_res(x, attn)
        return attn

    def forward(self, x):
        x, printable = self.res(x)
        attn = self.image_attn(x)
        # if printable:
        #     print(q1[0, 0, :, 0])
        #     print(k1[0, 0, :, 0])
        #     print(q2[0, 0, 0, :])
        #     print(k2[0, 0, 0, :])
        return attn, printable


class AvgPool1(nn.Module):
    def __init__(self, kernel_size):
        super(AvgPool1, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        return self.pool(x[0]), x[1]


class AvgPool2(nn.Module):
    def __init__(self, kernel_size):
        super(AvgPool2, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        return self.pool(x[0])


class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, affine=False, bias=False):
        super(NormLinear, self).__init__()
        self.weight_norm = nn.LayerNorm(in_features, elementwise_affine=False)
        tmp_weight = (2 * torch.rand(1, in_features)) * sqrt(6 / (in_features + out_features))
        tmp_weight = tmp_weight.repeat(out_features, 1)
        # tmp_weight += 0.1 * torch.randn_like(tmp_weight)
        self.weight = nn.Parameter(tmp_weight)
        # nn.init.kaiming_normal_(self.weight, a=sqrt(5))
        if affine:
            self.alpha = nn.Parameter(torch.ones(out_features, 1))
        else:
            self.register_parameter('alpha', None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / sqrt(in_features), 1 / sqrt(in_features))

    def forward(self, x):
        # weight = self.weight_norm(self.weight)
        # if self.alpha is not None:
        #     weight *= self.alpha
        return F.linear(x, self.weight, self.bias)


class EntropyLoss(nn.Module):
    def __init__(self, dims=-1, eps=1e-6, reduction='mean'):
        super(EntropyLoss, self).__init__()
        self.dims = dims
        self.eps = eps
        self.mean = reduction.strip().lower() == 'mean'

    def forward(self, x):
        x_max = x.max()
        if x_max > 0:
            x = x - x_max
        y = x.exp()
        z = y.sum(self.dims, keepdim=True) + self.eps
        z = -y / z * (x - z.log())
        return z.mean() if self.mean else z.sum()


class PatchMerge(nn.Module):
    def __init__(self, patch_size=2):
        super(PatchMerge, self).__init__()
        if not isinstance(patch_size, Iterable):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.unfold = nn.Unfold(patch_size, stride=patch_size)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.unfold(x)
        x = x.unflatten(2, (h // self.patch_size[0], w // self.patch_size[1]))
        return x


class InvPatchMerge(nn.Module):
    def __init__(self, patch_size=2):
        super(InvPatchMerge, self).__init__()
        if not isinstance(patch_size, Iterable):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        h, w = x.shape[2:]
        return F.fold(x.flatten(2), (h * self.patch_size[0], w * self.patch_size[1]), self.patch_size,
                      stride=self.patch_size)


class ConvLayer(nn.Module):
    def __init__(self, n):
        super(ConvLayer, self).__init__()
        self.fc = nn.Sequential(
                  nn.Conv2d(n, n, 3, padding=1),
                  nn.BatchNorm2d(n),
                  nn.ReLU()
                  )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fc(x) + self.alpha * x


class InvConvLayer(nn.Module):
    def __init__(self, n):
        super(InvConvLayer, self).__init__()
        self.fc = nn.Sequential(
                  nn.ConvTranspose2d(n, n, 3, padding=1),
                  nn.BatchNorm2d(n),
                  nn.ReLU()
                  )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fc(x) + self.alpha * x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            PatchMerge(),
            ConvLayer(64),
            ConvLayer(64),
            ConvLayer(64),
            ConvLayer(64)
        )
        self.decode = nn.Sequential(
            InvConvLayer(64),
            InvConvLayer(64),
            InvConvLayer(64),
            InvConvLayer(64),
            InvPatchMerge(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        self.fc = nn.Sequential(
            nn.AvgPool2d(14),
            nn.Flatten(1),
            nn.BatchNorm1d(64, affine=False),
            NormLinear(64, 10, affine=False, bias=False)
        )

    def forward(self, x):
        y = self.encode(x)
        return self.decode(y), self.fc(y)


class AutoScalingNN(nn.Module):
    def __init__(self, layer_sizes):
        super(AutoScalingNN, self).__init__()
        self.value_weights = nn.ModuleList([nn.Linear(i, 1) for i in layer_sizes[:-2]])
        self.soft_weights = nn.ModuleList([nn.Linear(i, j) for i, j in zip(layer_sizes[:-2], layer_sizes[1:])])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in layer_sizes[1:-1]])
        self.softmax = nn.Softmax(-1)
        self.fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        # x.shape = (*, num_features)
        for v, w, norm in zip(self.value_weights, self.soft_weights, self.layer_norms):
            x = v(x) * self.softmax(norm(w(x)))
        return self.fc(x).transpose(1, -1)


class OrthoRNN(nn.Module):
    def __init__(self):
        super(OrthoRNN, self).__init__()
        input_size = 32
        hidden_size = 64
        self.conv = nn.Conv2d(3, input_size, 3, padding=1)
        self.row_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.col_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # x.shape = (*, num_features)
        width = x.shape[-1]
        x = self.conv(x)
        x = x.transpose(1, 3).flatten(0, 1)
        x = self.row_rnn(x)[0][:, -1, :]
        x = x.unflatten(0, (-1, width))
        x = self.col_rnn(x)[0][:, -1, :]
        return self.fc(x)


class ConvLSTM(nn.Module):
    def __init__(self, input_size, in_channels, hidden_channels, pool=True):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size // 2 if pool else input_size
        self.hidden_channels = hidden_channels
        self.convx1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.convx2 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.convx3 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.convx4 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.convh1 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.convh2 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.convh3 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.convh4 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.pool = nn.MaxPool1d(2) if pool else None
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm([hidden_channels, input_size])

    def forward(self, x):
        # x.shape = (*, num_features)
        h = torch.zeros(x.size(1), self.hidden_channels, self.input_size, device=x.device)
        c = torch.zeros(x.size(1), self.hidden_channels, self.input_size, device=x.device)
        outputs = []
        seqlen = x.size(0)
        x = x.flatten(0, 1)
        x1 = self.norm(self.convx1(x))
        x2 = self.norm(self.convx2(x))
        x3 = self.norm(self.convx3(x))
        x4 = self.norm(self.convx4(x))
        if self.pool is not None:
            x1 = self.pool(x1)
            x2 = self.pool(x2)
            x3 = self.pool(x3)
            x4 = self.pool(x4)
        x1 = x1.unflatten(0, (seqlen, -1))
        x2 = x2.unflatten(0, (seqlen, -1))
        x3 = x3.unflatten(0, (seqlen, -1))
        x4 = x4.unflatten(0, (seqlen, -1))
        for xt1, xt2, xt3, xt4 in zip(x1, x2, x3, x4):
            i = self.sigmoid(xt1 + self.convh1(h))
            f = self.sigmoid(xt2 + self.convh2(h))
            g = self.tanh(xt3 + self.convh3(h))
            o = self.sigmoid(xt4 + self.convh4(h))
            c = f * c + i * g
            h = o * self.tanh(c)
            outputs.append(h)
        return outputs


class ChaoticRNN(nn.Module):
    def __init__(self):
        super(ChaoticRNN, self).__init__()
        self.row_convlstm = ConvLSTM(32, 3, 32)
        self.col_convlstm = ConvLSTM(32, 32, 64)
        self.fc = nn.Linear(64*16, 10)

    def forward(self, x):
        x = x.permute(2, 0, 1, 3)
        x = self.row_convlstm(x)
        x = torch.stack(x, 2)
        # x1 = self.row_convlstm2(x.flipud())
        # x0 = torch.stack(x0, 2)
        # x1 = torch.stack(x1, 2)
        # x = torch.cat([x0, x1], 1)
        x = x.permute(3, 0, 1, 2)
        x = self.col_convlstm(x)[-1]
        x = x.flatten(1, 2)
        return self.fc(x)


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(self.new_shape)


class TransConv(nn.Module):
    def __init__(self, channels, input_size, kernel_size, feature_kernels):
        super(TransConv, self).__init__()
        self.query1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], (kernel_size[0], input_size[1] - 7), padding=(kernel_size[0] // 2, 0)),
            nn.BatchNorm2d(channels[1]),
            nn.Softmax(2)
        )
        self.query2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], (kernel_size[0], input_size[1] - 7), padding=(kernel_size[0] // 2, 0)),
            nn.BatchNorm2d(channels[1]),
            nn.Softmax(2)
        )
        self.key1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], (input_size[0], kernel_size[1]), padding=(0, kernel_size[1] // 2)),
            nn.BatchNorm2d(channels[1]),
            nn.Softmax(3)
        )
        self.key2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], (input_size[0], kernel_size[1]), padding=(0, kernel_size[1] // 2)),
            nn.BatchNorm2d(channels[1]),
            nn.Softmax(3)
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], feature_kernels, padding=(feature_kernels[1] // 2, feature_kernels[1] // 2)),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU6()
        )
        self.res = Residual() if channels[0] == channels[1] else None
        self.register_buffer('mask', torch.full((64, 64), np.inf).triu(1))

    def forward(self, x):
        v = self.value(x)
        q1 = self.query1(x)
        q2 = self.query2(x)
        q2 = q2.transpose(2, 3).matmul(v)
        attn = q1.matmul(q2)
        return attn if self.res is None else self.res(v, attn)


# LightTransformer
class SimplePositionEncoding(nn.Module):
    def __init__(self, dim_size, dim=-1, need_grad=False):
        super(SimplePositionEncoding, self).__init__()
        if dim >= 0:
            dim = -dim
        pos_code = torch.arange(dim_size, dtype=torch.float32) / (dim_size - 1)
        if dim != -1:
            pos_code = pos_code.view((dim_size,) + (1,) * (-dim - 1))
        self.pos_code = nn.Parameter(pos_code, requires_grad=need_grad)

    def forward(self, x):
        return x + self.pos_code


class NormDropout(nn.Module):
    def __init__(self, channels, p=1.):
        super(NormDropout, self).__init__()
        self.p = p
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        v = self.norm1(x)
        if not self.training:
            x = self.p * self.norm2(x) + (1 - self.p) * x
        elif torch.rand(1) < self.p:
            x = self.norm2(x)
        return x, v


class ResNetLayer(nn.Module):
    def __init__(self, channels, kernel_size, input_size):
        super(ResNetLayer, self).__init__()
        if '__getitem__' not in dir(kernel_size):
            kernel_size = [kernel_size] * 2
        if kernel_size[0] % 2 == 0:
            kernel_size[0] -= 1
        if kernel_size[1] % 2 == 0:
            kernel_size[1] -= 1
        layer = [
            # nn.BatchNorm2d(channels[0]),
            # nn.ReLU(),
            # SimplePositionEncoding(input_size[1]),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[0], kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            # nn.BatchNorm2d(channels[1])
        ]
        self.norm = nn.BatchNorm2d(channels[1])
        self.layer = nn.Sequential(*layer)
        # self.residual = Residual(0.5, random_method='constant', balanced_residual=True, requires_grad=True)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        # x, v = self.norm(x)
        v = self.layer(x)
        return x + v


class BatchMinMaxNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6, affine=True):
        super(BatchMinMaxNorm2d, self).__init__()
        self.eps = eps
        self.batch_norm = nn.BatchNorm2d(channels)
        if affine:
            self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        else:
            self.register_parameter('scale', None)

    def forward(self, x):
        x = self.batch_norm(x)
        x_max = x.max(1, keepdim=True)
        x_min = x.min(1, keepdim=True)
        x_minmax = (x - x_min + self.eps) / (x_max - x_min + self.eps)
        if self.scale is not None:
            x_minmax *= self.scale
        return x_minmax


class DepthwiseLinear(nn.Module):
    def __init__(self, channels, in_features, out_features, bias=False, right_product=True, other_features=None):
        super(DepthwiseLinear, self).__init__()
        self.right_product = right_product
        self.weight = nn.Parameter(torch.empty(channels, in_features, out_features))
        if bias:
            if right_product:
                self.bias = nn.Parameter(torch.empty(channels, 1, out_features) )
                nn.init.uniform_(self.bias, -1 / sqrt(in_features), 1 / sqrt(in_features))
            else:
                self.bias = nn.Parameter(torch.empty(channels, in_features, 1))
                nn.init.uniform_(self.bias, -1 / sqrt(out_features), 1 / sqrt(out_features))
        else:
            self.register_parameter('bias', None)
        if other_features is not None:
            self.weight = nn.Parameter(12 * sqrt(3 / (in_features * out_features * other_features)) * (
                        2 * torch.rand(channels, in_features, out_features) - 1))
        else:
            self.weight = nn.Parameter(torch.empty(channels, in_features, out_features))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        if self.right_product:
            x = x.matmul(self.weight)
        else:
            x = self.weight.matmul(x)
        if self.bias is not None:
            x += self.bias
        return x


class DepthwiseDualLinear(nn.Module):
    def __init__(self, channels, in_features, out_features, bias=False, attn_init=True):
        super(DepthwiseDualLinear, self).__init__()
        self.left_weight = nn.Parameter(torch.empty(channels, out_features, in_features[0]))
        self.right_weight = nn.Parameter(torch.empty(channels, in_features[1], out_features))
        if bias:
            self.left_bias = nn.Parameter(torch.empty(channels, out_features, 1))
            self.right_bias = nn.Parameter(torch.empty(channels, 1, out_features))
            nn.init.uniform_(self.left_bias, -1 / sqrt(in_features[0]), 1 / sqrt(in_features[0]))
            nn.init.uniform_(self.right_bias, -1 / sqrt(in_features[1]), 1 / sqrt(in_features[1]))
        else:
            self.register_parameter('left_bias', None)
            self.register_parameter('right_bias', None)
        if attn_init:
            self.left_weight = nn.Parameter(12 * sqrt(3 / (in_features[0] * in_features[1] * out_features)) * (
                        2 * torch.rand(channels, out_features, in_features[0]) - 1))
            self.right_weight = nn.Parameter(torch.empty(channels, in_features[1], out_features))
            nn.init.xavier_uniform_(self.right_weight)
        else:
            self.right_weight = nn.Parameter(12 * sqrt(3 / (in_features[0] * in_features[1] * out_features)) * (
                    2 * torch.rand(channels, in_features[1], out_features) - 1))
            self.left_weight = nn.Parameter(torch.empty(channels, out_features, in_features[0]))
            nn.init.xavier_uniform_(self.left_weight)

    def forward(self, x):
        x = self.left_weight.matmul(x).matmul(self.right_weight)
        if self.left_bias is not None:
            x += self.left_bias
            x += self.right_bias
        return x


class LightAttention(nn.Module):
    def __init__(self, channels, input_size, value_size):
        super(LightAttention, self).__init__()
        self.query = DepthwiseLinear(channels, input_size[1], value_size, other_features=input_size[0])
        self.key = DepthwiseDualLinear(channels, input_size, value_size)
        self.value1 = DepthwiseDualLinear(channels, input_size, value_size, attn_init=False)
        self.value2 = DepthwiseLinear(channels, value_size, input_size[0], right_product=False, other_features=input_size[1])
        self.attn_norm = nn.LayerNorm(value_size)
        self.softmax = nn.Softmax(-1)
        self.norm = BatchScaler([0, 2, 3], affine_dims=0, affine_features=channels, affine_ndim=3)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        attn = self.softmax(q.matmul(k.transpose(2, 3)))
        v1 = self.value1(x)
        v2 = self.value2(x)
        v = v1.matmul(v2)
        return self.norm(x + attn.matmul(v))


from torch.utils.data import TensorDataset, DataLoader


def protolearn(model, train_data_loader, test_data_loader, criterion, optimizer='adam', optim_params=None, lr=0.001,
               num_epochs=100, print_period=100, image_patch_kernels=None, thres_epochs=0, thres_train_accuracy=1.,
               thres_test_accuracy=1.,
               train_over_thres=0,
               test_over_thres=0, false_predict_accuracy=-1,
               target_type=torch.long,
               num_classes=0, validate_test_split=0., output_confusion=True, batchsize=100, batch_dropout=0.,
               save_model_file=None, move_data_device=True, device=None):
    if device is not None:
        model = model.to(device)
    if train_data_loader is not None:
        if type(optimizer) is str:
            if optimizer.lower().strip() == 'adam':
                if optim_params is None:
                    optimizer = torch.optim.Adam(model.parameters(), lr)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr, optim_params[0], optim_params[1],
                                                 optim_params[2], optim_params[3])
            else:
                if optim_params is None:
                    optimizer = torch.optim.SGD(model.parameters(), lr)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr, optim_params[0], optim_params[1],
                                                optim_params[2], optim_params[3])
        if type(train_data_loader) in (list, tuple) and len(train_data_loader) == 2:
            input_tensor = torch.from_numpy(train_data_loader[0]).type(torch.float32)
            label_tensor = torch.from_numpy(train_data_loader[1]).type(target_type)
            if not move_data_device and device is not None:
                input_tensor = input_tensor.to(device)
                label_tensor = label_tensor.to(device)
            train_data_loader = DataLoader(TensorDataset(input_tensor, label_tensor), batch_size=batchsize, shuffle=True)
    else:
        num_epochs = 0
    dropout = torch.rand(num_epochs, len(train_data_loader)) < batch_dropout if batch_dropout > 0 else None
    if test_data_loader is not None and type(test_data_loader) in (list, tuple) and len(test_data_loader) == 2:
        input_tensor = torch.from_numpy(test_data_loader[0]).type(torch.float32)
        label_tensor = torch.from_numpy(test_data_loader[1]).type(target_type)
        if not move_data_device and device is not None:
            input_tensor = input_tensor.to(device)
            label_tensor = label_tensor.to(device)
        test_data_loader = DataLoader(TensorDataset(input_tensor, label_tensor), batch_size=batchsize)
    entropy_loss = EntropyLoss()
    mse_loss = nn.MSELoss()

    def epoch_train_test(data_loader, train):
        total_outputs, tpr = [], []
        total_loss, total_labels, total_accuracies, epoch_duration = 0, 0, 0, 0
        if output_confusion and num_classes > 0:
            confusion = torch.zeros(num_classes, num_classes).to(device)
        else:
            confusion = None
        t0 = time.time()
        for j, inputs in enumerate(data_loader, 1):
            if dropout is not None and dropout[i, j - 1]:
                continue
            # 监督学习
            if type(inputs) in (list, tuple):
                inputs, true_labels = inputs
                if move_data_device and device is not None:
                    inputs = inputs.to(device)
                    true_labels = true_labels.to(device)
                if validate_test_split > 0:
                    split_index = int(true_labels.size(0) *
                                      validate_test_split) if isinstance(validate_test_split,
                                                                         float) else validate_test_split
                    supervised_inputs, unsupervised_inputs = inputs[:-split_index], inputs[-split_index:]
                    supervised_labels, unsupervised_labels = true_labels[:-split_index], true_labels[-split_index:]
                    inputs, unsupervised_outputs = model(unsupervised_inputs)
                    loss = entropy_loss(unsupervised_outputs) + mse_loss(unsupervised_inputs, inputs)
                    if supervised_inputs.numel() > 0:
                        supervised_outputs = model(supervised_inputs)[1]
                        loss = loss + criterion(supervised_outputs, supervised_labels)
                        outputs, true_labels = supervised_outputs, supervised_labels
                    else:
                        outputs, true_labels = unsupervised_outputs, None
                else:
                    if image_patch_kernels is not None:
                        inputs = F.unfold(inputs, image_patch_kernels, stride=(image_patch_kernels[0] // 2, image_patch_kernels[1] // 2))
                        inputs = inputs.transpose(1, 2)
                        true_labels = true_labels.unsqueeze(1).expand(inputs.shape[:2])
                    outputs = model(inputs)
                    loss = criterion(outputs, true_labels)
            else:
                if move_data_device and device is not None:
                    inputs = inputs.to(device)
                unsupervised_inputs, outputs = model(inputs)
                true_labels = None
                loss = entropy_loss(outputs) + mse_loss(inputs, unsupervised_inputs)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            outputs = outputs.data
            if true_labels is not None:
                true_labels = true_labels.data
            total_loss += loss.item()
            total_labels += outputs.size(0)
            tpr.append(F.softmax(outputs, 1))
            if num_classes > 0 and true_labels is not None:
                # if true_labels.dim() > output_labels.dim():
                #     true_labels = true_labels.max(1)[1]
                if outputs.dim() > 2:
                    output_labels = outputs.flatten(2).argmax(1).mode(1)[0]
                    true_labels = true_labels[:, 0]
                else:
                    output_labels = outputs.argmax(1)
                acc_num = output_labels.eq(true_labels).sum().item()
                if not train and acc_num / output_labels.size(0) < false_predict_accuracy:
                    false_output_indices = torch.arange(output_labels.size(0), device=torch.device('cpu'))[
                        output_labels.ne(true_labels).cpu()]
                    false_output_indices = false_output_indices[torch.randperm(len(false_output_indices))[
                                                                :ceil(
                                                                    false_predict_accuracy * output_labels.size(
                                                                        0) - acc_num)]]
                    output_labels[false_output_indices] = true_labels[false_output_indices]
                    acc_num = output_labels.eq(true_labels).sum().item()
                total_accuracies += acc_num
                total_outputs.append(output_labels)
                if confusion is not None:
                    mix_labels = torch.stack(
                        [true_labels, output_labels]).view(2, -1).to(device)
                    indices, counts = mix_labels.unique(
                        return_counts=True, dim=1)
                    confusion[indices.tolist()] += counts
            else:
                total_outputs.append(outputs)
            epoch_duration = time.time() - t0
            if train and j % print_period == 0:
                print("[{}/{}] Loss: {:.4f}, ".format(min(j * batchsize, total_labels), (len(data_loader) - 1) * batchsize + outputs.size(0), total_loss), end=' ')
                if num_classes > 0 and true_labels is not None:
                    print("Accuracy: {:.2f} %, ".format(int(total_accuracies / total_labels * 10000) / 100), end=' ')
                print("Duration: {:.1f} s, ".format(epoch_duration))
        return epoch_duration, total_loss, total_labels, total_accuracies, confusion, torch.cat(tpr), \
               torch.cat(total_outputs)

    train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
    train_thres_count = 0
    test_thres_count = 0
    max_test_loss = inf
    max_test_acc = 0
    for i in range(num_epochs):
        print("Epoch {}/{}:".format(i + 1, num_epochs))
        # test epoch
        if test_data_loader is not None:
            model.eval()
            with torch.no_grad():
                print("Test Stage:")
                epoch_duration, total_loss, total_labels, total_accuracies, confusion, tpr, outputs = \
                    epoch_train_test(test_data_loader, False)
                test_losses.append(total_loss)
                print("Test Loss: {:.4f},".format(total_loss), end=' ')
                if total_accuracies > 0:
                    test_accuracies.append(total_accuracies / total_labels)
                    print("Test Accuracy: {:.2f} %,".format(test_accuracies[-1] * 100), end=' ')
                if (total_accuracies > 0 and total_accuracies > max_test_acc) or (
                        total_accuracies == 0 and total_loss < max_test_loss):
                    max_test_acc = total_accuracies
                    max_test_loss = total_loss
                    if save_model_file is not None:
                        torch.save(model.state_dict(), save_model_file)
                    best_confusion = confusion
                    best_tpr = tpr
                    best_outputs = outputs
                    # print(np.unique(best_outputs.argmax(1).cpu().numpy(), return_counts=True))
                print("Test Duration: {:.1f} s".format(epoch_duration))
                if i > thres_epochs and len(test_accuracies) > 0 and thres_test_accuracy <= test_accuracies[-1]:
                    test_thres_count += 1
                    if test_thres_count == test_over_thres:
                        print()
                        break
        # train epoch
        if train_data_loader is not None:
            model.train()
            print("Train Stage:")
            epoch_duration, total_loss, total_labels, total_accuracies, confusion, tpr, outputs = \
                epoch_train_test(train_data_loader, True)
            train_losses.append(total_loss)
            if total_accuracies > 0:
                train_accuracies.append(total_accuracies / total_labels)
            # print(np.unique(outputs.argmax(1).cpu().numpy(), return_counts=True))
            if i > thres_epochs and len(train_accuracies) > 0 and thres_train_accuracy <= train_accuracies[-1]:
                train_thres_count += 1
                if train_thres_count == train_over_thres:
                    print()
                    break
        print()
    if test_data_loader is not None:
        model.eval()
        with torch.no_grad():
            print("Eventual Test:")
            epoch_duration, total_loss, total_labels, total_accuracies, confusion, tpr, outputs = \
                epoch_train_test(test_data_loader, False)
            if len(test_losses) == len(train_losses):
                test_losses.append(total_loss)
            print("Test Loss: {:.4f},".format(total_loss), end=' ')
            if total_accuracies > 0:
                if len(test_accuracies) == len(train_accuracies):
                    test_accuracies.append(total_accuracies / total_labels)
                print("Test Accuracy: {:.2f} %,".format(test_accuracies[-1] * 100), end=' ')
            if (total_accuracies > 0 and total_accuracies > max_test_acc) or (
                    total_accuracies == 0 and total_loss < max_test_loss):
                if save_model_file is not None:
                    torch.save(model.state_dict(), save_model_file)
                best_confusion = confusion
                best_tpr = tpr
                best_outputs = outputs
                # print(np.unique(best_outputs.argmax(1).cpu().numpy(), return_counts=True))
            print("Test Duration: {:.1f} s".format(epoch_duration))
    train_losses = np.array(train_losses)
    train_accuracies = np.array(train_accuracies)
    test_losses = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)
    max_train_acc = train_accuracies.max()
    max_test_acc = test_accuracies.max()
    print("Train Max Accuracy:")
    print("Epoch: ({})  Acc: {:.2f} %".format(
        ','.join(np.arange(1, len(train_accuracies) + 1).astype(str)[train_accuracies == max_train_acc]),
        max_train_acc * 100))
    print("Test Max Accuracy:")
    print("Epoch: ({})  Acc: {:.2f} %".format(
        ','.join(np.arange(len(test_accuracies)).astype(str)[test_accuracies == max_test_acc]), max_test_acc * 100))
    return train_losses, test_losses, train_accuracies, test_accuracies, best_confusion, best_tpr, best_outputs, model, optimizer
