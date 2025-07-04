import logging
import time
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from models.cnn_models import GenericCNN, GenericCNNWithResidual
from models.fc_models import FCNet
from utils.datasets_utils import get_cifar_loaders, get_mnist_loaders
from utils.model_utils import count_params
from utils.training_utils import train_epoch, eval_model
from utils.visualization_utils import plot_learning_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MNIST_EPOCHS = 10
CIFAR_EPOCHS = 10

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def run_mnist_experiment(device: Union[torch.device, str]) -> None:
    """
    Запускает серию экспериментов по классификации изображений из набора данных MNIST с использованием полносвязной сети, сверточной сети и остаточной сверточной сети.
    :param device: Девайс, на котором будет работать эксперимент.
    :return: None
    """
    trainloader, testloader = get_mnist_loaders(batch_size=1024)

    experiments = {
        'FC': FCNet(28*28, [512, 256, 128], 10),
        'CNN': GenericCNN(1, (28, 28), 10, conv_channels=(32, 64), linear_units=128),
        'ResCNN': GenericCNNWithResidual(1, (28, 28), 10, channels=(32, 64), pool_out_size=(7, 7), dropout=0.25)
    }

    criterion = nn.CrossEntropyLoss()
    results = []

    for name, model in experiments.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_hist, test_hist = [], []

        best_train_acc = 0.0
        best_test_acc = 0.0
        total_time = 0.0

        for epoch in range(MNIST_EPOCHS):
            start = time.perf_counter()
            tr_loss, tr_acc, _ = train_epoch(model, trainloader, criterion, optimizer, device)
            epoch_time = time.perf_counter() - start
            total_time += epoch_time
            te_loss, te_acc, _ = eval_model(model, testloader, criterion, device)

            best_train_acc = max(best_train_acc, tr_acc)
            best_test_acc = max(best_test_acc, te_acc)

            train_hist.append((tr_loss, tr_acc))
            test_hist.append((te_loss, te_acc))

            logging.info(f"{name} MNIST | Epoch {epoch+1}/{MNIST_EPOCHS}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}, Epoch Time={epoch_time:.2f}s")

        plot_learning_curve(train_hist, test_hist, metric='accuracy',
                            title=f"{name} Accuracy (MNIST)", save_path=f"plots/mnist_comparison/mnist_{name}_acc.png")
        plot_learning_curve(train_hist, test_hist, metric='loss',
                            title=f"{name} Loss (MNIST)", save_path=f"plots/mnist_comparison/mnist_{name}_loss.png")

        params = count_params(model)

        start_inf = time.time()
        _ = eval_model(model, testloader, criterion, device)
        inference_time = time.time() - start_inf

        results.append({
            "model": name,
            "best_train_acc": best_train_acc,
            "best_test_acc": best_test_acc,
            "total_train_time_s": total_time,
            "inference_time_s": inference_time,
            "params": params,
        })

    df = pd.DataFrame(results)
    df.to_csv('results/mnist_comparison/mnist_results.csv', index=False)
    logging.info("MNIST results saved to results/mnist_comparison/mnist_results.csv")


def run_cifar10_experiment(device: Union[torch.device, str]) -> None:
    """
    Запускает серию экспериментов по классификации изображений из набора данных CIFAR10 с использованием полносвязной сети, сверточной сети и остаточной сверточной сети.
    :param device: Девайс, на котором будет работать эксперимент.
    :return: None
    """
    trainloader, testloader = get_cifar_loaders(batch_size=1024)

    experiments = {
        'FC': FCNet(3*32*32, [2048, 1024, 512, 256], 10),
        'ResCNN': GenericCNNWithResidual(3, (32, 32), 10, channels=(32, 64), pool_out_size=(4, 4), dropout=0.0),
        'RegResCNN': GenericCNNWithResidual(3, (32, 32), 10, channels=(32, 64), pool_out_size=(4, 4), dropout=0.5)
    }

    criterion = nn.CrossEntropyLoss()
    results = []

    for name, model in experiments.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        train_hist, test_hist = [], []

        best_train_acc = 0.0
        best_test_acc = 0.0
        total_time = 0.0
        grad_norm_acc = 0.0
        grad_count = 0

        for epoch in range(CIFAR_EPOCHS):
            start = time.perf_counter()
            tr_loss, tr_acc, _ = train_epoch(model, trainloader, criterion, optimizer, device)
            norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if norms:
                grad_norm_acc += sum(norms) / len(norms)
                grad_count += 1
            te_loss, te_acc, _ = eval_model(model, testloader, criterion, device)
            epoch_time = time.perf_counter() - start
            total_time += epoch_time

            best_train_acc = max(best_train_acc, tr_acc)
            best_test_acc = max(best_test_acc, te_acc)

            train_hist.append((tr_loss, tr_acc))
            test_hist.append((te_loss, te_acc))

            logging.info(f"{name} CIFAR10 | Epoch {epoch+1}/{CIFAR_EPOCHS}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}, Epoch Time={epoch_time:.2f}s")

        all_preds, all_targs = [], []
        model.eval()
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device)
                preds = model(inputs).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_targs.extend(targets.numpy())
        cm = confusion_matrix(all_targs, all_preds)
        plot_confusion_matrix(cm, classes=CIFAR10_CLASSES, normalize=True,
                              title=f"{name} Confusion Matrix", save_path=f"plots/cifar_comparison/cifar10_{name}_cm.png")

        plot_learning_curve(train_hist, test_hist, metric='accuracy',
                            title=f"{name} Accuracy (CIFAR10)", save_path=f"plots/cifar_comparison/cifar10_{name}_acc.png")
        plot_learning_curve(train_hist, test_hist, metric='loss',
                            title=f"{name} Loss (CIFAR10)", save_path=f"plots/cifar_comparison/cifar10_{name}_loss.png")

        params = count_params(model)
        avg_grad_norm = grad_norm_acc / grad_count if grad_count else 0.0
        results.append({
            'dataset': 'CIFAR10',
            'model': name,
            'best_train_acc': best_train_acc,
            'best_test_acc': best_test_acc,
            'avg_grad_norm': avg_grad_norm,
            'total_time_s': total_time,
            'params': params
        })

    df = pd.DataFrame(results)
    df.to_csv('results/cifar_comparison/cifar10_results.csv', index=False)
    logging.info("CIFAR10 results saved to results/cifar_comparison/cifar10_results.csv")


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 1.1 Comparison on MNIST
    print(f"{'-'*70}\n1.1 Comparison on MNIST\n{'-'*70}")
    run_mnist_experiment(device)
    # 1.2 Comparison on CIFAR-10
    print(f"{'-'*70}\n1.2 Comparison on CIFAR-10\n{'-'*70}")
    run_cifar10_experiment(device)
