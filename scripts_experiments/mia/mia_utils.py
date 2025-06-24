import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.datasets import SVHN
from torchvision import transforms
from sklearn.datasets import load_iris
import torch
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from torchvision.models import resnet18
from torchvision.models import vgg11
import numpy as np
import os
from tqdm import tqdm
import functools
import scipy.stats
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

import multiprocessing as mp
import os
from pathlib import Path
from torch.utils.data import Subset

def train_loop(model, train_loader, criterion, optimizer, sched, device, epochs, privacy_engine=None):
    """
    Generic training loop for PyTorch models, compatible with Opacus-privatized models.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on ('cpu' or 'cuda').
        epochs: Number of epochs to train.
        privacy_engine: Optional Opacus PrivacyEngine (for DP training).
    """
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        if sched is not None:
            sched.step()
        avg_loss = running_loss / len(train_loader)
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.8f}")
    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"(DP) ε = {epsilon:.2f}")

def load_dataset(name, root='./data', train=True, transform=None, download=True):
    """
    Loads a standard dataset from torchvision (for vision) or torchtext/tabular (for tabular).

    Args:
        name (str): Name of the dataset ('MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'IMDB', etc.).
        batch_size (int): Batch size for DataLoader.
        root (str): Root directory for dataset.
        train (bool): Whether to load the training set.
        transform: Transformations to apply (for vision datasets).
        download (bool): Whether to download the dataset if not present.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """

    # Vision datasets
    if name.lower() == 'mnist':
        if transform is None:
            transform = transforms.ToTensor()
        dataset = MNIST(root=root, train=train, transform=transform, download=download)
    elif name.lower() == 'mnist_4':
        if transform is None:
            transform = transforms.ToTensor()
        full_dataset = MNIST(root=root, train=True, transform=transform, download=download)
        # Get indices for class 0 and class 1, 2, 3
        targets = np.array(full_dataset.targets)
        idx_0 = np.where(targets == 0)[0][:250]
        idx_1 = np.where(targets == 1)[0][:250]
        idx_2 = np.where(targets == 2)[0][:250]
        idx_3 = np.where(targets == 3)[0][:250]
        selected_idx = np.concatenate([idx_0, idx_1, idx_2, idx_3])
        # Subset the dataset
        dataset = Subset(full_dataset, selected_idx)
        # Overwrite targets attribute for compatibility
        dataset.targets = torch.tensor(targets[selected_idx])
    elif name.lower() == 'cifar10':
        if transform is None:
            transform = transforms.ToTensor()
        dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
    elif name.lower() == 'fashionmnist':
        if transform is None:
            transform = transforms.ToTensor()
        dataset = FashionMNIST(root=root, train=train, transform=transform, download=download)
    elif name.lower() == 'svhn':
        if transform is None:
            transform = transforms.ToTensor()
        split = 'train' if train else 'test'
        dataset = SVHN(root=root, split=split, transform=transform, download=download)
    # Tabular datasets (example: Iris, Wine, BreastCancer from sklearn)
    elif name.lower() == 'iris':
        iris = load_iris()
        X = torch.tensor(iris.data, dtype=torch.float32)
        y = torch.tensor(iris.target, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'wine':
        wine = load_wine()
        X = torch.tensor(wine.data, dtype=torch.float32)
        y = torch.tensor(wine.target, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'breast_cancer':
        bc = load_breast_cancer()
        X = torch.tensor(bc.data, dtype=torch.float32)
        y = torch.tensor(bc.target, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    return dataset

import torch.nn as nn

def load_model(dataset_name, model_name=None, num_classes=None):
    """
    Loads a basic neural network classifier for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('MNIST', 'CIFAR10', etc.).
        model_name (str, optional): Name of the model architecture to load. If None, selects a default.
        num_classes (int, optional): Number of output classes. If None, inferred from dataset.

    Returns:
        nn.Module: PyTorch model.
    """

    dataset_name = dataset_name.lower()
    # Infer number of classes if not provided
    if num_classes is None:
        if dataset_name in ['mnist', 'fashionmnist']:
            num_classes = 10
        elif dataset_name == 'cifar10':
            num_classes = 10
        elif dataset_name == 'svhn':
            num_classes = 10
        elif dataset_name == 'iris':
            num_classes = 3
        elif dataset_name == 'wine':
            num_classes = 3
        elif dataset_name in ['breast_cancer', 'mnist_4']:
            num_classes = 4 #2
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Default model selection
    if model_name is None:
        if dataset_name in ['mnist', 'fashionmnist', 'mnist_4']:
            model_name = 'mlp'
        elif dataset_name in ['cifar10', 'svhn']:
            model_name = 'simple_cnn'
        elif dataset_name in ['iris', 'wine', 'breast_cancer']:
            model_name = 'mlp'
        else:
            raise ValueError(f"No default model for dataset: {dataset_name}")

    # Model definitions
    if model_name.lower() == 'mlp':
        # For tabular or flattened image data
        if dataset_name in ['mnist', 'fashionmnist', 'mnist_4']:
            input_dim = 28 * 28
        elif dataset_name == 'iris':
            input_dim = 4
        elif dataset_name == 'wine':
            input_dim = 13
        elif dataset_name == 'breast_cancer':
            input_dim = 30
        else:
            raise ValueError(f"MLP input dim unknown for dataset: {dataset_name}")

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    elif model_name.lower() == 'simple_cnn':
        # For small images (CIFAR10, SVHN)
        if dataset_name in ['cifar10', 'svhn']:
            in_channels = 3
        elif dataset_name in ['mnist', 'fashionmnist']:
            in_channels = 1
        else:
            raise ValueError(f"Simple CNN input channels unknown for dataset: {dataset_name}")

        model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if in_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif model_name.lower() == 'resnet18':
        # Use torchvision's ResNet18 for small images
        if dataset_name in ['cifar10', 'svhn']:
            in_channels = 3
        elif dataset_name in ['mnist', 'fashionmnist']:
            in_channels = 1
        else:
            raise ValueError(f"Small ResNet not supported for dataset: {dataset_name}")

        model = resnet18(num_classes=num_classes)
        # Adjust input conv layer if needed
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name.lower() == 'small_vgg' or model_name.lower() == 'vgg':
        # Use torchvision's VGG for small images
        if dataset_name in ['cifar10', 'svhn']:
            in_channels = 3
        elif dataset_name in ['mnist', 'fashionmnist']:
            in_channels = 1
        else:
            raise ValueError(f"VGG not supported for dataset: {dataset_name}")

        model = vgg11(num_classes=num_classes)
        # Adjust input conv layer if needed
        if in_channels != 3:
            model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

def save_logits(args, model, train_dl, DEVICE, savedir, path):
    """
    Computes and saves logits for all data in the dataloader.

    Args:
        model: PyTorch model.
        dataloader: DataLoader providing the data.
        device: Device to run the model on.
        save_path: Path to save the logits (as a .pt file).
    """
    model.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in train_dl:
            x = x.to(DEVICE)
            outputs = model(x)
            logits.append(outputs.detach().cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)

    np.save(os.path.join(savedir, path, "logits.npy"), logits_n)

def load_one(input_tuple):
    """
    This loads a logits and converts it to a scored prediction.
    """
    path, dataset_name = input_tuple
    opredictions = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]

    # Be exceptionally careful.
    # Numerically stable everything, as described in the paper.
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    labels = load_dataset(dataset_name, train=True).targets.numpy()

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(os.path.join(path, "scores.npy"), logit)    


def load_stats(args, savedir):

    with mp.get_context("spawn").Pool(8) as p:
        p.map(load_one, [(os.path.join(savedir, x), args.dataset) for x in os.listdir(savedir)])


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data(args, savedir):
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for path in os.listdir(savedir):
        scores.append(np.load(os.path.join(savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(savedir, path, "keep.npy")))
    scores = np.array(scores)
    keep = np.array(keep)

    return keep, scores


def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers


def do_plot(fn, keep, scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep[-ntest:], scores[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < 0.001)[0][-1]]

    print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)

def do_plot2(fn, keep, scores, keep_target, scores_target, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep_target[-ntest:], scores_target[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < 0.001)[0][-1]]

    print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr(args, keep, scores, savedir):
    os.makedirs(savedir, exist_ok=True)
    plt.figure(figsize=(4, 3))

    do_plot(generate_ours, keep, scores, 1, "Ours (online)\n", metric="auc")

    do_plot(functools.partial(generate_ours, fix_variance=True), keep, scores, 1, "Ours (online, fixed variance)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline), keep, scores, 1, "Ours (offline)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, 1, "Ours (offline, fixed variance)\n", metric="auc")

    do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{savedir}/fprtpr.png")


def fig_fpr_tpr_target(args, keep, scores, keep_target, scores_target, savedir, name="fprtpr_target"):
    os.makedirs(savedir, exist_ok=True)
    plt.figure(figsize=(4, 3))

    do_plot2(generate_ours, keep, scores, keep_target, scores_target, 1, "Ours (online)\n", metric="auc")

    do_plot2(functools.partial(generate_ours, fix_variance=True), keep, scores, keep_target, scores_target, 1, "Ours (online, fixed variance)\n", metric="auc")

    do_plot2(functools.partial(generate_ours_offline), keep, scores, keep_target, scores_target, 1, "Ours (offline)\n", metric="auc")

    do_plot2(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, keep_target, scores_target, 1, "Ours (offline, fixed variance)\n", metric="auc")

    do_plot2(generate_global, keep, scores, keep_target, scores_target, 1, "Global threshold\n", metric="auc")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{savedir}/{name}.png")