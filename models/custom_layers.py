from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class CustomConvFunction(Function):
    """
    Кастомная функция свертки для 2D:
    """
    @staticmethod
    def forward(
            ctx,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            stride: int = 1,
            padding: int = 0
    ) -> torch.Tensor:
        """
        Прямой проход кастомной свертки.
        :param ctx: контекст для сохранения данных для backward.
        :param input: входной тензор формы (N, C_in, H, W).
        :param weight: тензор свертки формы (C_out, C_in, kH, kW).
        :param bias: тензор смещений длины C_out или None. По умолчанию None.
        :param stride: шаг свертки. По умолчанию 1.
        :param padding: размер паддинга. По умолчанию 0.
        :return: результат свертки формы (N, C_out, H_out, W_out).
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        return F.conv2d(input, weight, bias, stride, padding)

    @staticmethod
    def backward(
            ctx,
            *grad_outputs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        """
        Обратный проход: вычисление градиентов по input, weight и bias.
        :param ctx: контекст с сохраненными данными.
        :param grad_outputs: градиент по выходу свертки.
        :return: кортеж (grad_input, grad_weight, grad_bias, None, None).
        """
        grad_output = grad_outputs[0]

        input, weight, bias = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, stride, padding
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, stride, padding
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


class CustomConv2d(nn.Module):
    """
    Кастомный свёрточный слой 2D с кастомным ядром и смещением.

    Реализует операцию свёртки вручную через CustomConvFunction, используя
    параметры веса и смещения как nn.Parameter.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True
    ) -> None:
        """
        Инициализация параметров свёрточного слоя.
        :param in_channels: число входных каналов.
        :param out_channels: число выходных каналов.
        :param kernel_size: размер свёрточного ядра (kernel_size × kernel_size).
        :param stride: шаг свёртки.
        :param padding: количество нулевых пикселей, добавляемых по границам.
        :param bias: добавлять ли смещение после свёртки. По умолчанию True.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride, self.padding = stride, padding

    def forward(self, x):
        return CustomConvFunction.apply(x, self.weight, self.bias, self.stride, self.padding)


class SpatialAttention(nn.Module):
    """
    Модуль пространственного внимания (Spatial Attention).
    """
    def __init__(self, kernel_size: int = 7) -> None:
        """
        Инициализация свёртки для вычисления карты внимания.
        :param kernel_size: размер свёрточного ядра для карты внимания. По умолчанию 7.
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход SpatialAttention.
        :param x: входной тензор формы (B, C, H, W).
        :return: масштабированный входной тензор той же формы (B, C, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class NormalizedSwishFunction(Function):
    """
    Кастомная активация Normalized Swish с нормировкой на максимум.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход Normalized Swish.
        :param ctx: контекст для сохранения данных для backward.
        :param x: входной тензор.
        :return: y = x * sigmoid(x), нормированный на максимальное значение y.
        """
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y / y.max()

    @staticmethod
    def backward(
            ctx,
            *grad_output
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        """
        Обратное распространение для Normalized Swish.
        :param ctx: контекст с сохраненными данными.
        :param grad_output: градиент по выходу.
        :return: градиент по входу, нормированный на максимальное значение.
        """
        x, = ctx.saved_tensors
        sig = torch.sigmoid(x)
        grad = grad_output[0] * (sig + x * sig * (1 - sig))
        return grad / grad.max()


class NormalizedSwish(nn.Module):
    """
    Модуль активации Normalized Swish, обёртка для NormalizedSwishFunction.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применение Normalized Swish к входному тензору.
        :param x:
        :return:
        """
        return NormalizedSwishFunction.apply(x)


class LPPooling2d(nn.Module):
    """
    LP-пулинг 2D с настраиваемым порядком нормы.
    """
    def __init__(
            self,
            norm_type: int = 3,
            kernel_size: int = 2,
            stride: int = 2
    ) -> None:
        """
        Функция инициализации LP-пулинга 2D с настраиваемым порядком нормы.
        :param norm_type: порядок нормы p.
        :param kernel_size: размер окна пулинга.
        :param stride: шаг окна пулинга.
        """
        super().__init__()
        self.p = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход LP-пулинга.
        :param x: входной тензор (B, C, H, W).
        :return: результат пулинга.
        """
        return F.lp_pool2d(x, norm_type=self.p, kernel_size=self.kernel_size, stride=self.stride)


class BasicResBlock(nn.Module):
    """
    Простой остаточный блок (Basic ResNet Block) без изменения размерности.
    """
    def __init__(self, channels: int) -> None:
        """
        Функция инициализации простого остаточного блока.
        :param channels: число каналов во входном и выходном тензоре блока.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход BasicResBlock.
        :param x: входной тензор (B, C, H, W).
        :return: выходной тензор (B, C, H, W).
        """
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class BottleneckResBlock(nn.Module):
    """
    Бутылочный остаточный блок (Bottleneck ResNet Block) с уменьшением и восстановлением каналов.
    """
    def __init__(self, in_channels: int, bottleneck_channels: int) -> None:
        """
        Функция инициализации бутылочного остаточного блока
        :param in_channels: число входных каналов.
        :param bottleneck_channels: число каналов в узком (бутылочном) слое.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, 1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход BottleneckResBlock.
        :param x: входной тензор (B, in_channels, H, W).
        :return: выходной тензор той же формы.
        """
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)


class WideResBlock(nn.Module):
    """
    Широкий остаточный блок (Wide ResNet Block), расширяющий число каналов.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Функция инициализации широкого остаточного блока.
        :param in_channels: число каналов во входном тензоре.
        :param out_channels: число каналов в выходном тензоре.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход WideResBlock с изменением каналов через skip-связь.
        :param x: входной тензор (B, in_channels, H, W).
        :return: выходной тензор (B, out_channels, H, W).
        """
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
