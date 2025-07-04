from typing import Optional, Any, Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    """
    Кастомная обёртка MNIST датасета.
    """
    def __init__(self, train: bool = True, transform: Optional[Any] = None) -> None:
        """
        Функция инициализации класса-обёртки MNIST датасета
        :param train: использовать часть датасета для обучения или нет. По умолчанию True
        :param transform: Функция предобработки датасета. По умолчанию None
        """
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self) -> int:
        """
        Возвращает размер датасета.
        :return: Количество объектов в датасете
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Вовзращает объект из датасета по индексу.
        :param idx: Индекс объекта
        :return: Изображение, его класс
        """
        return self.dataset[idx]


class CIFARDataset(Dataset):
    """
    Кастомная обёртка CIFAR-10 датасета.
    """
    def __init__(self, train: bool = True, transform: Optional[Any] = None) -> None:
        """
        Инициализация обёртки CIFAR-10 датасета.
        :param train: Использовать ли тренировочную часть датасета. По умолчанию True.
        :param transform: Функция предобработки изображений. По умолчанию None.
        """
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self) -> int:
        """
        Возвращает размер датасета.
        :return: Количество объектов в датасете.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Возвращает элемент датасета по индексу.
        :param idx: Индекс элемента.
        :return: Кортеж (изображение, класс).
        """
        return self.dataset[idx]


def get_mnist_loaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Возвращает DataLoader'ы для тренировочной и тестовой выборок MNIST.

    :param batch_size: Размер батча. По умолчанию 64.
    :return: Кортеж (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar_loaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Возвращает DataLoader'ы для тренировочной и тестовой выборок CIFAR-10.

    :param batch_size: Размер батча. По умолчанию 64.
    :return: Кортеж (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFARDataset(train=True, transform=transform)
    test_dataset = CIFARDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
