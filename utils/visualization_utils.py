import os
from typing import Optional, List, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_feature_maps(
        feature_map: torch.Tensor,
        save_path: Optional[str] = None,
        max_channels: int = 8
) -> None:
    """
    Строит карту признаков.
    :param feature_map: Карта признаков
    :param save_path: Путь для созранения графика
    :param max_channels: Максимальное количество каналов для отображения
    :return: None
    """
    fmap = feature_map.detach().cpu().numpy()
    batch_idx = 0
    channels = fmap.shape[1]
    num_channels_to_plot = min(channels, max_channels)

    fig, axes = plt.subplots(1, num_channels_to_plot, figsize=(3*num_channels_to_plot, 3))

    for i in range(num_channels_to_plot):
        ax = axes[i]
        ax.imshow(fmap[batch_idx, i, :, :], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Channel {i}")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_learning_curve(
        train_hist: Sequence[tuple],
        test_hist: Sequence[tuple],
        metric: str = 'accuracy',
        title: Optional[str] = None,
        save_path: Optional[str] = None
) -> None:
    """
    Строит кривую обучения.
    :param train_hist: Историю обучения
    :param test_hist: Историю тестирования
    :param metric: Метрика
    :param title: Название графика
    :param save_path: Путь для сохранения графика
    :return: None
    """
    epochs = np.arange(1, len(train_hist) + 1)
    if metric == 'loss':
        train_vals = [x[0] for x in train_hist]
        test_vals = [x[0] for x in test_hist]
    else:
        train_vals = [x[1] for x in train_hist]
        test_vals = [x[1] for x in test_hist]
    plt.figure()
    plt.plot(epochs, train_vals, label='train')
    plt.plot(epochs, test_vals, label='test')
    ylabel = metric.capitalize()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
        cm: np.ndarray,
        classes: Optional[List[str]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None
) -> None:
    """
    Строит матрицу ошибок.
    :param cm: Матрица ошибок
    :param classes: Список названий классов для подписей осей.
    :param normalize: Нормировать ли матрицу по строкам (вывод в процентах).
    :param title: Заголовок графика.
    :param save_path: Путь для сохранения графика. Если None, график не сохраняется, а закрывается.
    :return: None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes)) if classes is not None else []
    if classes is not None:
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_grad_flow(grad_history: List[Dict[str, float]], save_path: str) -> None:
    """
    Строит график истории градиентов.
    :param grad_history: История градиентов.
    :param save_path: Путь для сохранения графика.
    :return: None
    """
    layer_names = list(grad_history[0].keys())
    epochs = range(1, len(grad_history) + 1)

    plt.figure(figsize=(8, 6))
    for name in layer_names:
        norms = [epoch_stats[name] for epoch_stats in grad_history]
        plt.plot(epochs, norms, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Grad Norm")
    plt.title("Gradient Norm Flow")
    plt.legend(loc="upper right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
