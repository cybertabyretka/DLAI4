import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_models import KernelSizeCNN, MixedKernelCNN, DepthCNN, ResNetLike
from utils.comparison_utils import compute_receptive_field
from utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import count_params
from utils.training_utils import train_epoch, eval_model
from utils.visualization_utils import plot_learning_curve, plot_feature_maps, plot_grad_flow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def measure_gradients_per_layer(model, loader, device):
    stats = {}
    criterion = nn.CrossEntropyLoss()
    model.train()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            stats[name] = p.grad.norm().item()
    model.zero_grad()
    return stats


def run_kernel_size_experiments(
    trainloader, testloader,
    plots_path, results_path,
    dataset_name, in_channels,
    target_params=50000,
    device='cuda'
):
    results = []
    kernels = [3, 5, 7]
    # 1) Подбираем для каждого ядра число каналов C так, чтобы кол‑во параметров == target_params
    channel_settings = {}
    for k in kernels:
        for C in range(1, 512):
            model = KernelSizeCNN(in_channels, num_classes=10, kernel_size=k, base_channels=C, ref_kernel_size=3)
            if count_params(model) == target_params:
                channel_settings[k] = C
                break
        else:
            # если не нашли — сообщение и дефолт
            print(f"WARNING: couldn't match params for kernel={k}, defaulting C=32")
            channel_settings[k] = 32

    # 2) Прогоняем все три KernelSizeCNN
    for k in kernels:
        C = channel_settings[k]
        model = KernelSizeCNN(in_channels, num_classes=10,
                              kernel_size=k,
                              base_channels=C,
                              ref_kernel_size=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        train_hist, test_hist = [], []
        start_time = time.time()
        for epoch in range(5):
            t_loss, t_acc, _ = train_epoch(model, trainloader, criterion, optimizer, device)
            v_loss, v_acc, _ = eval_model(model, testloader, criterion, device)
            train_hist.append((t_loss, t_acc))
            test_hist.append((v_loss, v_acc))
        training_time = time.time() - start_time

        grad_stats = measure_gradients_per_layer(model, trainloader, device)
        rf = compute_receptive_field([model.conv1, model.conv2])

        results.append({
            "model":      f"Kernel_{k}x{k}",
            "test_acc":   test_hist[-1][1],
            "train_time_s": training_time,
            "grad_stats": grad_stats,
            "receptive_field": rf,
            "params":     count_params(model),
        })

        # фичармапы и кривые
        inputs, _ = next(iter(testloader))
        inputs = inputs[:5].to(device)
        fmap1 = model.relu(model.conv1(inputs))
        plot_feature_maps(fmap1, save_path=f"{plots_path}/kernel_{k}x{k}_first_layer.png")
        plot_learning_curve(train_hist, test_hist, metric='accuracy',
                            title=f"Kernel {k}x{k} Acc",
                            save_path=f"{plots_path}/kernel_{k}x{k}_acc.png")
        plot_learning_curve(train_hist, test_hist, metric='loss',
                            title=f"Kernel {k}x{k} Loss",
                            save_path=f"{plots_path}/kernel_{k}x{k}_loss.png")

    # 3) MixedKernelCNN: снова ищем C по target_params
    mixed_C = None
    for C in range(1, 512):
        m = MixedKernelCNN(in_channels, num_classes=10, channels=C)
        if count_params(m) == target_params:
            mixed_C = C
            break
    if mixed_C is None:
        print("WARNING: couldn't match params for MixedKernel, defaulting C=32")
        mixed_C = 32

    # Прогоним MixedKernel
    model = MixedKernelCNN(in_channels, num_classes=10, channels=mixed_C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_hist, test_hist = [], []
    start_time = time.time()
    for epoch in range(5):
        t_loss, t_acc, _ = train_epoch(model, trainloader, criterion, optimizer, device)
        v_loss, v_acc, _ = eval_model(model, testloader, criterion, device)
        train_hist.append((t_loss, t_acc))
        test_hist.append((v_loss, v_acc))
    training_time = time.time() - start_time

    grad_stats = measure_gradients_per_layer(model, trainloader, device)
    rf = compute_receptive_field([model.conv1, model.conv2])

    results.append({
        "model":      "Mixed 1x1+3x3",
        "test_acc":   test_hist[-1][1],
        "train_time_s": training_time,
        "grad_stats": grad_stats,
        "receptive_field": rf,
        "params":     count_params(model),
    })

    # фичармапы и кривые для смешанной
    inputs, _ = next(iter(testloader))
    inputs = inputs[:5].to(device)
    fmap1 = model.relu(model.conv1(inputs))
    plot_feature_maps(fmap1, save_path=f"{plots_path}/kernel_mixed_first_layer.png")
    plot_learning_curve(train_hist, test_hist, metric='accuracy',
                        title="Mixed Acc",
                        save_path=f"{plots_path}/mixed_acc.png")
    plot_learning_curve(train_hist, test_hist, metric='loss',
                        title="Mixed Loss",
                        save_path=f"{plots_path}/mixed_loss.png")

    # Сохраняем все результаты
    df = pd.DataFrame(results)
    df.to_csv(f"{results_path}/{dataset_name}_kernel_experiments.csv", index=False)


def run_depth_experiments(trainloader, testloader, plots_path, results_path, dataset_name, in_channels):
    results = []
    configs = {"Shallow_2": 2, "Medium_4": 4, "Deep_6": 6}
    for name, depth in configs.items():
        result_time = 0
        model = DepthCNN(in_channels, 10, num_layers=depth).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train_hist, test_hist = [], []
        grad_history = []
        for epoch in range(5):
            t_loss, t_acc, epoch_time = train_epoch(model, trainloader, criterion, optimizer, device)
            test_loss, test_acc, _ = eval_model(model, testloader, criterion, device)
            train_hist.append((t_loss, t_acc))
            test_hist.append((test_loss, test_acc))
            grad_history.append(measure_gradients_per_layer(model, trainloader, device))
            result_time += epoch_time
        results.append({
            "model": name,
            "acc": test_hist[-1][1],
            "grad_history": grad_history,
            'learn_time_s': result_time,
            "params": count_params(model)
        })
        inputs, _ = next(iter(testloader))
        inputs = inputs[:5].to(device)
        fmap1 = model.layers[0](inputs)
        plot_feature_maps(fmap1, save_path=f"{plots_path}/{name}_first_layer.png")
        plot_learning_curve(train_hist, test_hist, metric='accuracy', title=f"{name} Acc", save_path=f"{plots_path}/{name}_acc.png")
        plot_grad_flow(grad_history, save_path=f"{plots_path}/{name}_grad_flow.png")

    model = ResNetLike(in_channels, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_hist, test_hist, grad_history = [], [], []
    result_time = 0
    for epoch in range(5):
        t_loss, t_acc, epoch_time = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc, _ = eval_model(model, testloader, criterion, device)
        train_hist.append((t_loss, t_acc))
        test_hist.append((test_loss, test_acc))
        grad_history.append(measure_gradients_per_layer(model, trainloader, device))
        result_time += epoch_time
    results.append({
        "model": "ResNetLike",
        "acc": test_hist[-1][1],
        "grad_history": grad_history,
        'learn_time_s': result_time,
        "params": count_params(model)
    })
    inputs, _ = next(iter(testloader))
    inputs = inputs[:5].to(device)
    fmap1 = model.stem[0](inputs)
    plot_feature_maps(fmap1, save_path=f"{plots_path}/ResNetLike_first_layer.png")
    plot_learning_curve(train_hist, test_hist, metric='accuracy', title="ResNetLike Acc", save_path=f"{plots_path}/ResNetLike_acc.png")
    plot_grad_flow(grad_history, save_path=f"{plots_path}/ResNetLike_grad_flow.png")

    pd.DataFrame(results).to_csv(f"{results_path}/{dataset_name}_depth_experiments.csv", index=False)


if __name__ == "__main__":
    trainloader_mnist, testloader_mnist = get_mnist_loaders(batch_size=1024)
    trainloader_cifar, testloader_cifar = get_cifar_loaders(batch_size=1024)
    run_kernel_size_experiments(trainloader_mnist, testloader_mnist, 'plots/mnist_architecture_analysis', 'results/architecture_analysis', 'mnist', 1)
    run_kernel_size_experiments(trainloader_cifar, testloader_cifar, 'plots/cifar_architecture_analysis', 'results/architecture_analysis', 'cifar', 3)
    run_depth_experiments(trainloader_mnist, testloader_mnist, 'plots/mnist_architecture_analysis', 'results/architecture_analysis', 'mnist', 1)
    run_depth_experiments(trainloader_cifar, testloader_cifar, 'plots/cifar_architecture_analysis', 'results/architecture_analysis', 'cifar', 3)
