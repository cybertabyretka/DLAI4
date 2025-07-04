import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.custom_layers import CustomConv2d, SpatialAttention, NormalizedSwish, LPPooling2d, BasicResBlock, \
    BottleneckResBlock, WideResBlock
from utils.datasets_utils import get_mnist_loaders
from utils.model_utils import count_params
from utils.training_utils import train_epoch, eval_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 32, 32, device=device, requires_grad=True)

    # 3.1 Реализация кастомных слоев
    print(f"{'-' * 70}\n3.1 Реализация кастомных слоев\n{'-' * 70}")
    custom_layers = [
        ("CustomConv2d", CustomConv2d(3, 8, 3, padding=1)),
        ("SpatialAttention", SpatialAttention()),
        ("NormalizedSwish", NormalizedSwish()),
        ("LPPooling2d", LPPooling2d(3,2))
    ]
    std_layers = [
        ("Conv2d", nn.Conv2d(3, 8, 3, padding=1)),
        ("Identity", nn.Identity()),
        ("SiLU", nn.SiLU()),
        ("AvgPool2d", nn.AvgPool2d(2))
    ]
    bench_results = []
    for (cname, cl), (sname, sl) in zip(custom_layers, std_layers):
        cl, sl = cl.to(device), sl.to(device)
        torch.cuda.synchronize() if device.type=='cuda' else None
        t0 = time.time()
        y = cl(x)
        loss = y.mean()
        loss.backward(retain_graph=True)
        torch.cuda.synchronize() if device.type=='cuda' else None
        t_custom = time.time() - t0
        x.grad.zero_()
        t0 = time.time()
        y2 = sl(x)
        loss2 = y2.mean() if isinstance(y2, torch.Tensor) else y2
        if isinstance(loss2, torch.Tensor): loss2.backward()
        torch.cuda.synchronize() if device.type=='cuda' else None
        t_std = time.time() - t0
        bench_results.append({
            "layer": cname,
            "custom_time": t_custom,
            "std_time": t_std,
            "params_custom": count_params(cl),
            "params_std": count_params(sl)
        })
    pd.DataFrame(bench_results).to_csv("results/custom/custom_layers_benchmark.csv", index=False)

    # 3.2 Эксперименты с Residual блоками
    print(f"{'-' * 70}\n3.2 Эксперименты с Residual блоками\n{'-' * 70}")


    class BlockModel(nn.Module):
        """
        Класс сверточной модели, состоящей из стем-слоя (начальной свертки), произвольного блока, глобального адаптивного пула и выходного полносвязного слоя.
        """
        def __init__(self, block: nn.Module, in_channels: int = 1) -> None:
            """
            Функция инициализации класса свёрточной модели.
            :param block: Модуль (блок сверток или любая другая структура), принимающий входной тензор с размером каналов, равным выходу stem-слоя (по умолчанию 16 каналов).
            :param in_channels: Количество каналов во входном изображении. По умолчанию 1
            """
            super().__init__()
            self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
            self.block = block
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, 28, 28)
                dummy = self.stem(dummy)
                dummy = self.block(dummy)
                c = dummy.shape[1]

            self.fc = nn.Linear(c, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Прямой проход через модель.
            :param x: Входной тензор
            :return: Выходной тензор
            """
            x = self.stem(x)
            x = self.block(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    trainloader, testloader = get_mnist_loaders(batch_size=1024)
    blocks = [
        BasicResBlock(16),
        BottleneckResBlock(16, 16),
        WideResBlock(16, 4)
    ]

    exp_results = []
    for blk in blocks:
        name = blk.__class__.__name__
        model = BlockModel(blk).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        acc_history = []
        result_time = 0
        for epoch in range(10):
            _, _, time = train_epoch(model, trainloader, criterion, optimizer, device)
            _, acc, _ = eval_model(model, testloader, criterion, device)
            logging.info(f'{epoch + 1}/10 | test Acc: {acc}')
            acc_history.append(acc)
            result_time += time
        exp_results.append({
            "block": name,
            "epochs": 10,
            "params": count_params(blk),
            "mean_acc": np.mean(acc_history),
            "std_acc": np.std(acc_history),
            "max_acc": np.max(acc_history),
            "learn_time": result_time
        })
    pd.DataFrame(exp_results).to_csv("results/residual/resblock_experiments.csv", index=False)
