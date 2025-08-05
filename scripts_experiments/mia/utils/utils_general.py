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
from sympy.stats import Normal, cdf, quantile
from opacus_new.accountants import RDPAccountant

import multiprocessing as mp
import os
from pathlib import Path
from torch.utils.data import Subset
from scipy.stats import norm
import seaborn as sns
import numpy as np
import random
import time
from opacus_new import PrivacyEngine
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_linnerud
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import argparse

def extend_dict(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = v
    return dict1

def get_budget_to_index_from_seeds(savedir, seeds):
    budget_to_index_list = []
    for seed in seeds:
        bti_path = os.path.join(savedir, str(seed), "bti.npy")
        with open(bti_path, "rb") as f:
            budget_to_index = np.load(f, allow_pickle=True).item()
        budget_to_index_list.append(budget_to_index)
        # Check if all dicts in budget_to_index_list contain the same values
        first_bti = budget_to_index_list[0]
        for bti in budget_to_index_list[1:]:
            assert first_bti.keys() == bti.keys(), "Target models need to have the same pp_budgets (keys mismatch)"
            for key in first_bti.keys():
                assert np.array_equal(first_bti[key], bti[key]), f"Target models need to have the same pp_budgets for key {key}"
    return first_bti

# Construct default_path from dataset, budgets, and portions
def get_dataset_size(dataset_name, train):
    dummy_train_ds = load_dataset(dataset_name, train=train)
    return len(dummy_train_ds)

def generate_pp_budgets(seed, size, portions, budgets):
    # Assign privacy budgets to each sample in the dataset
    assert abs(sum(portions) - 1.0) < 1e-6, "Portions must sum to 1."
    pp_budgets = np.zeros(size)
    start = 0
    for portion, budget in zip(portions, budgets):
        end = start + int(round(portion * size))
        if end > size:
            end = size
        pp_budgets[start:end] = budget
        start = end
    # If rounding caused a mismatch, fill the rest with the last budget
    if start < size:
        pp_budgets[start:] = budgets[-1]

    # Randomly shuffle pp_budgets to avoid any ordering bias
    np.random.seed(seed)  # Set seed
    shuffled_indices = np.random.permutation(size)
    pp_budgets = pp_budgets[shuffled_indices]
    # Log indices for each unique budget
    budget_to_index = {}
    unique_budgets = np.unique(pp_budgets)
    for budget in unique_budgets:
        indices = np.where(pp_budgets == budget)[0]
        budget_to_index[str(budget)] = indices
    return pp_budgets, budget_to_index

def format_list(lst):
    return "_".join(str(x).replace('.', '').replace('-', 'm') for x in lst)


def load_yaml_args(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    # Convert None strings to None
    for k, v in config.items():
        if isinstance(v, str) and v.lower() == 'null':
            config[k] = None
    return config

def parse_args_with_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_yaml', type=str, help='YAML file with experiment parameters', default='./scripts_experiments/mia/exp_yaml/adult.yaml')
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_yaml_args(args.exp_yaml)
    for k, v in vars(args).items():
        if k == "exp_yaml":
            continue
        if v is not None:
            if k in config and config[k] != v:
                print(f"Overwriting config value for '{k}': {config[k]} -> {v}")
            config[k] = v
    config["target_delta"] = float(config["target_delta"])
    return argparse.Namespace(**config)


def train_loop(params, model, train_loader, criterion, optimizer, sched, device, epochs, privacy_engine=None):
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
    disable_inner = params.disable_inner if hasattr(params, 'disable_inner') else params.get('disable_inner', False)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if sched is not None:
            sched.step()
        avg_loss = running_loss / len(train_loader)
        if not disable_inner:
            tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.15f}")
    return model

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
        idx_0 = np.where(targets == 0)[0][:1000]
        idx_1 = np.where(targets == 1)[0][:1000]
        idx_2 = np.where(targets == 2)[0][:1000]
        idx_3 = np.where(targets == 3)[0][:1000]
        selected_idx = np.concatenate([idx_0, idx_1, idx_2, idx_3])
        # Subset the dataset
        dataset = Subset(full_dataset, selected_idx)
        # Overwrite targets attribute for compatibility
        dataset.targets = torch.tensor(targets[selected_idx])
    elif name.lower() == 'mnist_2':
        if transform is None:
            transform = transforms.ToTensor()
        full_dataset = MNIST(root=root, train=True, transform=transform, download=download)
        # Get indices for class 0 and class 1
        targets = np.array(full_dataset.targets)
        idx_0 = np.where(targets == 0)[0][:100]
        idx_1 = np.where(targets == 1)[0][:100]
        selected_idx = np.concatenate([idx_0, idx_1])
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
    elif name.lower() == 'digits':
        digits = load_digits()
        X = torch.tensor(digits.data, dtype=torch.float32)
        y = torch.tensor(digits.target, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'california_housing':
        housing = fetch_california_housing()
        X = torch.tensor(housing.data, dtype=torch.float32)
        y = torch.tensor(housing.target, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'diabetes':
        diabetes = load_diabetes()
        X = torch.tensor(diabetes.data, dtype=torch.float32)
        y = torch.tensor(diabetes.target, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'linnerud':
        linnerud = load_linnerud()
        X = torch.tensor(linnerud.data, dtype=torch.float32)
        y = torch.tensor(linnerud.target, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'adult':
        # Load with as_frame=True for easier preprocessing
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        df = adult.frame

        # Drop rows with missing values (marked as '?')
        df = df.replace('?', np.nan).dropna()

        # Separate features and target
        X = df.drop('class', axis=1)
        y = (df['class'] == ' >50K').astype(np.int64).values

        # One-hot encode categorical features, standardize numerical features
        X = pd.get_dummies(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Fixed train/test split (e.g., 80/20, stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Select train or test split
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'german_credit':
        german = fetch_openml(name='credit-g', version=1, as_frame=False)
        X = torch.tensor(german.data.astype(np.float32))
        y = torch.tensor(german.target.astype(np.int64))
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'bank_marketing':
        bank = fetch_openml(name='BankMarketing', version=1, as_frame=False)
        X = torch.tensor(bank.data.astype(np.float32))
        y = torch.tensor((bank.target == 'yes').astype(np.int64))
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'credit_card_default':
        cc = fetch_openml(name='credit-g', version=1, as_frame=False)
        X = torch.tensor(cc.data.astype(np.float32))
        y = torch.tensor(cc.target.astype(np.int64))
        dataset = torch.utils.data.TensorDataset(X, y)
    elif name.lower() == 'phishing':
        phishing = fetch_openml(name='Phishing', version=1, as_frame=False)
        X = torch.tensor(phishing.data.astype(np.float32))
        y = torch.tensor(phishing.target.astype(np.int64))
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
        elif dataset_name in ['mnist_4']:
            num_classes = 4 #2
        elif dataset_name in ['breast_cancer', 'mnist_2', "adult"]:
            num_classes = 2
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Default model selection
    if model_name is None:
        if dataset_name in ['mnist', 'fashionmnist', 'mnist_4', 'mnist_2']:
            model_name = 'mlp'
        elif dataset_name in ['cifar10', 'svhn']:
            model_name = 'simple_cnn'
        elif dataset_name in ['iris', 'wine', 'breast_cancer']:
            model_name = 'mlp'
        elif dataset_name in ["adult"]:
            model_name = 'mlp'
        else:
            raise ValueError(f"No default model for dataset: {dataset_name}")

    # Model definitions
    if model_name.lower() == 'mlp':
        # For tabular or flattened image data
        if dataset_name in ['mnist', 'fashionmnist', 'mnist_4', 'mnist_2']:
            input_dim = 28 * 28
        elif dataset_name == 'iris':
            input_dim = 4
        elif dataset_name == 'wine':
            input_dim = 13
        elif dataset_name == 'breast_cancer':
            input_dim = 30
        elif dataset_name in ["adult"]:
            input_dim = 105
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
    with torch.no_grad():
        for i in range(args.n_queries):
            logits = []
            for x, _ in train_dl:
                x = x.to(DEVICE)
                outputs = model(x)
                logits.append(outputs.detach().cpu().numpy())
            logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)

    np.save(os.path.join(savedir, path, "logits.npy"), logits_n)
    # Explicit cleanup
    del logits_n, logits, outputs, x
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def make_private(model, train_loader, optimizer, pp_budgets, args, adapt_weights_to_budgets=True, nm=None, sr=None):
    # modulevalidator = ModuleValidator()
    # model = modulevalidator.fix_and_validate(model)

    privacy_engine = PrivacyEngine(accountant=args.accountant,
                                   individualize=args.individualize,
                                   weights=args.weights,
                                   pp_budgets=pp_budgets)
    if adapt_weights_to_budgets:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private_with_epsilon(module=model,
                                       optimizer=optimizer,
                                       data_loader=train_loader,
                                       target_epsilon=min(pp_budgets),
                                       target_delta=args.target_delta,
                                       epochs=args.epochs,
                                       max_grad_norm=args.max_grad_norm,
                                       optimal=True,
                                       max_alpha=10_000,
                                       numeric=True,
                                       precision = 0.00001)
    else:
        unique_budgets = np.unique(pp_budgets)
        budget_to_weight = {}
        for i, budget in enumerate(unique_budgets):
            budget_to_weight[budget] = sr[i]
        pp_weights = np.array([budget_to_weight[budget] for budget in pp_budgets])
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private(module=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          noise_multiplier=nm[0],
                          max_grad_norm=args.max_grad_norm,
                          pp_weights=pp_weights)
    # print(np.unique(pp_budgets))
    # print(privacy_engine.weights)
    return private_model, private_optimizer, private_loader, privacy_engine