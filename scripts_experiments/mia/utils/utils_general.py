"""Utility functions for differential privacy experiments with PyTorch.

This module provides utilities for:
- Loading and preprocessing various datasets (vision, tabular, and medical)
- Defining neural network models for different dataset types
- Privacy-preserving model training using Opacus
- Data loading and batch processing
"""

import gc
import os
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm

# Torchvision imports
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from torchvision.models import resnet18, vgg11

# Scikit-learn imports
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    LabelBinarizer,
    MinMaxScaler,
    LabelEncoder,
)

# Medical datasets
import medmnist
from medmnist import INFO

# Kaggle integration
import kagglehub

# Privacy
from opacus_new import PrivacyEngine

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASETS_INFO = {
    # Vision datasets - MNIST variants and CIFAR
    "mnist": {"classes": 10, "input_features": 28 * 28, "model_type": "mlp"},
    "mnist_4": {"classes": 4, "input_features": 28 * 28, "model_type": "mlp"},
    "mnist_2": {"classes": 2, "input_features": 28 * 28, "model_type": "mlp"},
    "mnist_4_small": {"classes": 4, "input_features": 28 * 28, "model_type": "mlp"},
    "cifar10": {"classes": 10, "input_features": 3, "model_type": "simple_cnn", "num_spatial": 32},
    "fashionmnist": {"classes": 10, "input_features": 1, "model_type": "simple_cnn", "num_spatial": 32},
    "svhn": {"classes": 10, "input_features": 1, "model_type": "simple_cnn", "num_spatial": 32},
    
    # Tabular datasets from sklearn and OpenML
    "breast_cancer": {"classes": 2, "input_features": 30, "model_type": "mlp"},
    "adult": {"classes": 2, "input_features": 105, "model_type": "mlp"},
    "german_credit": {"classes": 2, "input_features": 59, "model_type": "mlp"},
    "bank_marketing": {"classes": 2, "input_features": 16, "model_type": "mlp"},
    "credit_card_default": {"classes": 2, "input_features": 59, "model_type": "mlp"},
    "uci_isolet": {"classes": 26, "input_features": 617, "model_type": "mlp"},
    
    # Kaggle datasets
    "kaggle_credit": {"classes": 2, "input_features": 25, "model_type": "mlp"},
    "kaggle_cervical": {"classes": 2, "input_features": 32, "model_type": "mlp"},
    "compas": {"classes": 2, "input_features": 24, "model_type": "mlp"},
}

# Add MedMNIST datasets from the INFO registry
for dataset_name, info in INFO.items():
    n_classes = len(info["label"].keys())
    n_channels = info["n_channels"]
    
    # DermaMNIST uses ResNet9, others use simple CNN
    model_type = "resnet9" if dataset_name == "dermamnist" else "simple_cnn"
    
    DATASETS_INFO[dataset_name] = {
        "classes": n_classes,
        "input_features": n_channels,
        "model_type": model_type,
        "num_spatial": 28,
    }


# ============================================================================
# Utility Functions - Data Processing
# ============================================================================

def to_padded_array(list_of_lists: List) -> np.ndarray:
    """Convert a list of lists with variable lengths to a padded numpy array.
    
    Args:
        list_of_lists: List of lists or scalar values with potentially different lengths.
        
    Returns:
        Padded numpy array with dtype=object.
    """
    lengths = []
    for lst in list_of_lists:
        if lst is None or isinstance(lst, int):
            lengths.append(1)
        else:
            lengths.append(len(lst))
    
    max_len = max(lengths)
    padded = []
    for lst in list_of_lists:
        if lst is None or isinstance(lst, int):
            padded.append([lst] + [0] * (max_len - 1))
        else:
            padded.append(lst + [0] * (max_len - len(lst)))
    
    return np.array(padded, dtype=object)

def assign_pp_values(pp_budgets: np.ndarray, values: List) -> torch.Tensor:
    """Assign values to samples based on their privacy budgets.
    
    Maps each unique privacy budget to a corresponding value and assigns
    those values to all samples with that budget.
    
    Args:
        pp_budgets: Privacy budget (epsilon) for each data point.
        values: List of values to assign, indexed by unique budgets in sorted order.
        
    Returns:
        Tensor of assigned values, one per data point.
    """
    pp_values = np.zeros(len(pp_budgets))
    for i, budget in enumerate(np.sort(np.unique(pp_budgets))):
        pp_values[pp_budgets == budget] = values[i]
    return torch.Tensor(pp_values)


def reduce_class_samples(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    max_samples_per_class: Optional[int],
) -> Tuple[torch.utils.data.TensorDataset, torch.Tensor, torch.Tensor]:
    """Limit the number of samples per class in a dataset.
    
    Args:
        X_tensor: Feature tensor.
        y_tensor: Label tensor.
        max_samples_per_class: Maximum number of samples to keep per class.
                              If None, returns original tensors unchanged.
    
    Returns:
        Tuple of (TensorDataset, X_tensor_selected, y_tensor_selected).
    """
    if max_samples_per_class is None:
        return torch.utils.data.TensorDataset(X_tensor, y_tensor), X_tensor, y_tensor

    selected_indices = []
    y_np = y_tensor.cpu().numpy()

    for unique_class in torch.unique(y_tensor):
        unique_class = unique_class.item()
        cls_indices = (y_np == unique_class).nonzero()[0]

        # Select at most max_samples_per_class indices per class
        if len(cls_indices) > max_samples_per_class:
            chosen = cls_indices[:max_samples_per_class].tolist()
        else:
            chosen = cls_indices.tolist()

        selected_indices.extend(chosen)

    selected_indices = torch.tensor(sorted(selected_indices), dtype=torch.long)
    X_selected = X_tensor[selected_indices]
    y_selected = y_tensor[selected_indices]
    
    dataset = torch.utils.data.TensorDataset(X_selected, y_selected)
    return dataset, X_selected, y_selected


def generate_string(
    list1: List,
    list2: List,
    sep1: str = ": ",
    sep2: str = ",",
) -> str:
    """Generate a formatted string from two parallel lists.
    
    Args:
        list1: Keys or first elements.
        list2: Values or second elements.
        sep1: Separator between key and value. Default: ": ".
        sep2: Separator between pairs. Default: ",".
        
    Returns:
        Formatted string like "key1: val1,key2: val2".
        
    Raises:
        ValueError: If lists have different lengths.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length.")
    return sep2.join(f"{k}{sep1}{v}" for k, v in zip(list1, list2))


def extend_dict(dict1: Dict, dict2: Dict) -> Dict:
    """Extend dict1 with key-value pairs from dict2 if keys don't exist.
    
    Args:
        dict1: Dictionary to extend (modified in-place).
        dict2: Dictionary with items to add.
        
    Returns:
        Updated dict1.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = v
    return dict1

def get_budget_to_index_from_seeds(savedir: str, seeds: List[int]) -> Dict:
    """Load and verify budget-to-index mappings across multiple training seeds.
    
    Ensures all seeds use identical privacy budget assignments.
    
    Args:
        savedir: Directory containing seed subdirectories with "bti.npy" files.
        seeds: List of seed values corresponding to subdirectories.
        
    Returns:
        Budget-to-index mapping dictionary.
        
    Raises:
        AssertionError: If seeds have inconsistent budget assignments.
    """
    budget_to_index_list = []
    for seed in seeds:
        bti_path = os.path.join(savedir, str(seed), "bti.npy")
        with open(bti_path, "rb") as f:
            budget_to_index = np.load(f, allow_pickle=True).item()
        budget_to_index_list.append(budget_to_index)
    
    # Verify all seeds use identical budgets
    first_bti = budget_to_index_list[0]
    for bti in budget_to_index_list[1:]:
        assert first_bti.keys() == bti.keys(), "Target models have inconsistent pp_budgets (keys mismatch)"
        for key in first_bti.keys():
            assert np.array_equal(first_bti[key], bti[key]), (
                f"Target models have inconsistent pp_budgets for key {key}"
            )
    
    return first_bti


def get_dataset_size(
    dataset_name: str,
    train: bool,
    num_max_per_class_samples: Optional[int],
) -> int:
    """Get the total number of samples in a dataset.
    
    Args:
        dataset_name: Name of the dataset to load.
        train: Whether to load training or test split.
        num_max_per_class_samples: Maximum samples per class (if applicable).
        
    Returns:
        Total number of samples in the dataset.
    """
    dataset = load_dataset(
        dataset_name,
        train=train,
        num_max_samples=num_max_per_class_samples,
    )
    return len(dataset)


def generate_pp_budgets(
    seed: int,
    size: int,
    portions: List[float],
    budgets: List[float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Assign privacy budgets to samples with specified proportions.
    
    Divides the dataset into groups with different privacy budgets,
    respecting the given proportions, and shuffles to avoid ordering bias.
    
    Args:
        seed: Random seed for reproducibility.
        size: Total number of samples.
        portions: Fractions of samples for each budget (must sum to ~1.0).
        budgets: Privacy budgets (epsilon values) for each portion.
        
    Returns:
        Tuple of (pp_budgets, budget_to_index) where:
        - pp_budgets: Privacy budget assigned to each sample.
        - budget_to_index: Mapping from budget values to sample indices.
    """
    assert abs(sum(portions) - 1.0) < 1e-6, "Portions must sum to approximately 1.0"
    
    # Assign budgets to samples based on portions
    pp_budgets = np.zeros(size)
    start = 0
    for portion, budget in zip(portions, budgets):
        end = start + int(round(portion * size))
        if end > size:
            end = size
        pp_budgets[start:end] = budget
        start = end
    
    # Handle rounding errors
    if start < size:
        pp_budgets[start:] = budgets[-1]

    # Shuffle to avoid ordering bias
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(size)
    pp_budgets = pp_budgets[shuffled_indices]
    
    # Create mapping from budget to sample indices
    budget_to_index = {}
    for budget in np.unique(pp_budgets):
        indices = np.where(pp_budgets == budget)[0]
        budget_to_index[str(budget)] = indices
    
    return pp_budgets, budget_to_index


def format_list(lst: List) -> str:
    """Format a list as a string suitable for filenames.
    
    Converts list elements to strings with special formatting:
    - Removes decimal points
    - Replaces negative signs with 'm'
    - Joins with underscores
    
    Args:
        lst: List of values to format.
        
    Returns:
        Formatted string suitable for use in filenames.
    """
    return "_".join(str(x).replace(".", "").replace("-", "m") for x in lst)




# ============================================================================
# Configuration Utilities
# ============================================================================


def load_yaml_args(yaml_path: str) -> Dict[str, Any]:
    """Load experiment configuration from a YAML file.
    
    Converts string 'null' values to None for proper type handling.
    
    Args:
        yaml_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert 'null' strings to None
    for k, v in config.items():
        if isinstance(v, str) and v.lower() == "null":
            config[k] = None
    
    return config


def parse_args_with_yaml() -> argparse.Namespace:
    """Parse command-line arguments and merge with YAML configuration.
    
    Command-line arguments take precedence over YAML config values.
    
    Returns:
        Namespace with merged configuration.
    """
    parser = argparse.ArgumentParser(description="Run differential privacy experiments")
    parser.add_argument(
        "--exp_yaml",
        type=str,
        help="YAML file with experiment parameters",
        default="./scripts_experiments/mia/exp_yaml/temp_2/dermamnist_small.yaml",
    )
    parser.add_argument(
        "--individualize",
        type=str,
        help="Individualization method (sampling or clipping)",
        default="sampling",
    )
    parser.add_argument(
        "--name_ext",
        type=str,
        help="Name extension for the experiment",
        default="temp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=2,
    )

    args = parser.parse_args()
    config = load_yaml_args(args.exp_yaml)
    
    # Merge command-line arguments into config
    for k, v in vars(args).items():
        if k == "exp_yaml" or v is None:
            continue
        if k in config and config[k] != v:
            print(f"Overwriting config value for '{k}': {config[k]} -> {v}")
        config[k] = v
    
    config["target_delta"] = float(config["target_delta"])
    return argparse.Namespace(**config)


# ============================================================================
# Training Utilities
# ============================================================================


def train_loop(
    params: argparse.Namespace,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int,
    pp_max_grad_norms: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Training loop for PyTorch models, compatible with Opacus DP training.
    
    Supports both standard and differentially-private training via per-sample
    gradient norms.
    
    Args:
        params: Configuration namespace (must contain 'disable_inner' if present).
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
                      Batches should have 2 or 3 elements: (inputs, targets[, idx]).
        criterion: Loss function.
        optimizer: Optimizer. For DP training, must support pp_max_grad_norms parameter.
        sched: Learning rate scheduler, or None to skip scheduling.
        device: Device to train on ('cpu' or 'cuda').
        epochs: Number of epochs to train.
        pp_max_grad_norms: Optional per-sample max grad norms for DP training.
                          Should have shape (dataset_size,).
    
    Returns:
        Trained model.
        
    Raises:
        ValueError: If batch size is not 2 or 3.
    """
    model.to(device)
    model.train()
    disable_inner = getattr(params, "disable_inner", False)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for batch in train_loader:
            # Handle variable batch formats
            if len(batch) == 2:
                inputs, targets = batch
                idx = None
            elif len(batch) == 3:
                inputs, targets, idx = batch
            else:
                raise ValueError("Batch must have 2 or 3 elements: (inputs, targets[, idx])")

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Backward pass with optional DP gradient clipping
            if idx is None or pp_max_grad_norms is None:
                optimizer.step()
            else:
                optimizer.step(pp_max_grad_norms=pp_max_grad_norms[idx].to(device))

            # Accumulate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            for label, prediction in zip(targets, predicted):
                class_total[label.item()] += 1
                if prediction.item() == label.item():
                    class_correct[label.item()] += 1

        if sched is not None:
            sched.step()

        # Log progress
        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        if not disable_inner:
            class_acc_str = " ".join(
                f"Class {cls}: {100.0 * class_correct[cls] / class_total[cls]:.2f}%"
                for cls in sorted(class_total.keys())
                if class_total[cls] > 0
            )
            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}% | {class_acc_str}"
            )

    return model




# ============================================================================
# Dataset Loading Utilities
# ============================================================================


def download_kaggle_dataset(dataset_slug: str, root: str) -> str:
    """Download a Kaggle dataset using the Kaggle API.
    
    Args:
        dataset_slug: Kaggle dataset identifier (e.g., "mlg-ulb/creditcardfraud").
        root: Root directory for caching downloaded datasets.
        
    Returns:
        Path to the downloaded dataset directory.
    """
    os.environ["KAGGLEHUB_CACHE"] = root
    dataset_dir = kagglehub.dataset_download(dataset_slug)
    return dataset_dir


def preprocess_tabular(
    df: pd.DataFrame,
    target_col: str,
    bin: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a tabular dataset for machine learning.
    
    Handles:
    - Feature encoding (categorical to numerical)
    - Feature scaling (MinMaxScaler)
    - Target encoding
    - Missing value imputation
    
    Args:
        df: Input DataFrame with features and target.
        target_col: Name of the target column.
        bin: If True, use one-hot encoding for categorical features.
             If False, use label encoding with scaling.
    
    Returns:
        Tuple of (X_processed, y_encoded) where X is (n_samples, n_features)
        and y is (n_samples,) with classes starting from 0.
    """
    # Shuffle dataset
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(df.index)
    df = df.iloc[shuffled_indices].reset_index(drop=True)
    
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # Process features
    X_processed = []
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            if bin:
                # One-hot encoding
                lb = LabelBinarizer()
                transformed = lb.fit_transform(X[col])
                if transformed.shape[1] == 1:  # Binary outcome
                    X_processed.append(transformed.ravel())
                else:  # Multi-class
                    X_processed.append(transformed)
            else:
                # Label encoding + scaling
                le = LabelEncoder()
                scaler = MinMaxScaler()
                transformed = le.fit_transform(X[col]).reshape(-1, 1)
                transformed = scaler.fit_transform(transformed).ravel()
                X_processed.append(transformed)
        else:
            # Numerical features: just scale
            scaler = MinMaxScaler()
            transformed = scaler.fit_transform(X[[col]]).ravel()
            X_processed.append(transformed)

    X_final = np.column_stack(X_processed)

    # Process target
    if isinstance(y[0], str):
        # Try numeric conversion, fall back to label encoding
        try:
            y = np.array([int(x) for x in y])
        except (ValueError, TypeError):
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    if isinstance(y[0], bool) or (isinstance(y[0], np.bool_)):
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel()
    
    # Ensure class labels start from 0
    y = np.array(y, dtype=int) - int(np.min(y))
    
    # Impute missing values with column means
    col_means = np.nanmean(X_final, axis=0)
    nan_mask = np.isnan(X_final)
    X_final[nan_mask] = col_means[np.where(nan_mask)[1]]
    
    return X_final, y

class PrefetchedLoader:
    """DataLoader wrapper that asynchronously prefetches batches to device.
    
    This wrapper moves batches to the target device in a non-blocking manner,
    overlapping data transfer with computation.
    """

    def __init__(self, loader: torch.utils.data.DataLoader, device: torch.device):
        """Initialize the prefetched loader.
        
        Args:
            loader: Base DataLoader to wrap.
            device: Target device for prefetching.
        """
        self.loader = loader
        self.device = device

    def __iter__(self):
        """Iterate over batches with prefetching.
        
        Yields:
            Tuple of (x, y) moved to device asynchronously.
        """
        for x, y in self.loader:
            yield (
                x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
            )

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.loader)

    def __getattr__(self, name: str):
        """Forward unknown attributes to the wrapped loader."""
        return getattr(self.loader, name)


def load_dataset(
    name: str,
    root: str = "./data",
    train: bool = True,
    transform: Optional[torch.nn.Module] = None,
    download: bool = True,
    num_max_samples: Optional[int] = None,
) -> torch.utils.data.TensorDataset:
    """Load a dataset by name, supporting vision, tabular, and medical datasets.
    
    Supports:
    - Vision: MNIST variants, CIFAR-10, FashionMNIST, SVHN
    - Tabular: BreastCancer, Adult, German Credit, ISOLET (sklearn/OpenML)
    - Medical: MedMNIST datasets
    - Kaggle: Cervical Cancer, Credit Risk
    
    Args:
        name: Dataset name (case-insensitive).
        root: Root directory for caching datasets. Default: './data'.
        train: If True, load training split; else test split.
        transform: Optional transforms to apply (vision datasets only).
        download: If True, download dataset if not cached.
        num_max_samples: Maximum samples per class (if specified).
    
    Returns:
        PyTorch TensorDataset with (X, y) pairs.
        
    Raises:
        ValueError: If dataset name is not supported.
        FileNotFoundError: If dataset requires manual download/setup.
    """
    # Vision datasets - MedMNIST 2D datasets
    if name.lower() in INFO:  # check if dataset is in MedMNIST registry
        info = INFO[name.lower()]
        DataClass = getattr(medmnist, info['python_class'])

        if transform is None:
            transform = transforms.ToTensor()

        split = 'train' if train else 'test'
        dataset = DataClass(root=root, split=split, transform=transform, download=download)

        X_tensor = torch.stack([dataset[i][0] for i in range(len(dataset))])
        y_tensor = torch.tensor([dataset[i][1] for i in range(len(dataset))])

        # If labels are one-hot (common in MedMNIST), convert to integer targets
        if y_tensor.ndim > 1 and y_tensor.shape[1] == len(info["label"].keys()):
            y_tensor = y_tensor.argmax(dim=1)
        if y_tensor.ndim > 1:
            y_tensor = y_tensor.squeeze()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'mnist':
        if transform is None:
            transform = transforms.ToTensor()
        dataset = MNIST(root=root, train=train, transform=transform, download=download)
        X_tensor = dataset.test_data / 256.0
        y_tensor = dataset.test_labels
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'mnist_4':
        if transform is None:
            transform = transforms.ToTensor()
        full_dataset = MNIST(root=root, train=train, transform=transform, download=download)
        if train:
            X_tensor = full_dataset.train_data / 256.0
            y_tensor = full_dataset.train_labels
        else:
            X_tensor = full_dataset.test_data / 256.0
            y_tensor = full_dataset.test_labels
        # Get indices for class 0 and class 1, 2, 3
        idx_0 = np.where(y_tensor == 0)[0][:1000]
        idx_1 = np.where(y_tensor == 1)[0][:1000]
        idx_2 = np.where(y_tensor == 2)[0][:1000]
        idx_3 = np.where(y_tensor == 3)[0][:1000]
        selected_idx = np.concatenate([idx_0, idx_1, idx_2, idx_3])
        rng = np.random.default_rng(seed=42)
        rng.shuffle(selected_idx)
        # Subset the dataset
        X_tensor = X_tensor[selected_idx]
        y_tensor = y_tensor[selected_idx]
        # Overwrite targets attribute for compatibility
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'mnist_2':
        if transform is None:
            transform = transforms.ToTensor()
        full_dataset = MNIST(root=root, train=train, transform=transform, download=download)
        if train:
            X_tensor = full_dataset.train_data / 256.0
            y_tensor = full_dataset.train_labels
        else:
            X_tensor = full_dataset.test_data / 256.0
            y_tensor = full_dataset.test_labels
        # Get indices for class 0 and class 1
        idx_0 = np.where(y_tensor == 0)[0][:100]
        idx_1 = np.where(y_tensor == 1)[0][:100]
        selected_idx = np.concatenate([idx_0, idx_1])
        rng = np.random.default_rng(seed=42)
        rng.shuffle(selected_idx)
        # Subset the dataset
        X_tensor = X_tensor[selected_idx]
        y_tensor = y_tensor[selected_idx]
        # Overwrite targets attribute for compatibility
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'mnist_4_small':
        if transform is None:
            transform = transforms.ToTensor()
        full_dataset = MNIST(root=root, train=train, transform=transform, download=download)
        if train:
            X_tensor = full_dataset.train_data / 256.0
            y_tensor = full_dataset.train_labels
        else:
            X_tensor = full_dataset.test_data / 256.0
            y_tensor = full_dataset.test_labels
        # Get indices for class 0 and class 1
        idx_0 = np.where(y_tensor == 0)[0][:50]
        idx_1 = np.where(y_tensor == 1)[0][:50]
        idx_2 = np.where(y_tensor == 2)[0][:50]
        idx_3 = np.where(y_tensor == 3)[0][:50]
        selected_idx = np.concatenate([idx_0, idx_1, idx_2, idx_3])
        rng = np.random.default_rng(seed=42)
        rng.shuffle(selected_idx)
        # Subset the dataset
        X_tensor = X_tensor[selected_idx]
        y_tensor = y_tensor[selected_idx]
        # Overwrite targets attribute for compatibility
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'cifar10':        
        dataset = CIFAR10(root=root, train=train, transform=None, download=download)
        X_tensor = torch.tensor(dataset.data, dtype=torch.float32) / 256.0
        X_tensor = X_tensor.permute(0, 3, 1, 2)
        y_tensor = torch.tensor(dataset.targets, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
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
    elif name.lower() == 'breast_cancer':
        bc = load_breast_cancer()
        X_final = torch.tensor(bc.data, dtype=torch.float32)
        y = torch.tensor(bc.target, dtype=torch.float32)
        test_size = 0.2
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = X_train.float()
            y_tensor = y_train.long()
        else:
            X_tensor = X_test.float()
            y_tensor = y_test.long()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'adult':
        # Load with as_frame=True for easier preprocessing
        adult = fetch_openml(name='adult', version=2, as_frame=True, data_home=root)
        df = adult.frame

        # Drop rows with missing values (marked as '?')
        df = df.replace('?', np.nan).dropna()

        # Separate features and target
        X = df.drop('class', axis=1)
        y = (df['class'] == '>50K').astype(np.int64).values

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
        german = fetch_openml(name='credit-g', version=1, as_frame=True, data_home=root)
        df = german.data
        y = german.target
        df["target"] = y
        target_col = "target"
        test_size = 0.1
        X_final, y = preprocess_tabular(df, target_col)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'credit_card_default':
        cc = fetch_openml(name='credit-g', version=1, as_frame=True, data_home=root)
        df = cc.data
        y = cc.target
        df["target"] = y
        target_col = "target"
        test_size = 0.1
        X_final, y = preprocess_tabular(df, target_col)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == 'phishing':
        phishing = fetch_openml(name='Phishing', version=1, as_frame=False, data_home=root)
        df = phishing.data
        y = phishing.target
        df["target"] = y
        target_col = "target"
        test_size = 0.1
        X_final, y = preprocess_tabular(df, target_col)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == "uci_isolet":
        data = fetch_openml("isolet", version=1, as_frame=True, data_home=root)
        df = data.frame
        target_col = "class"
        test_size = 1500 / 7800
        X_final, y = preprocess_tabular(df, target_col)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    # ===================
    # Kaggle datasets
    # ===================

    elif name.lower() == "kaggle_credit":
        dataset_dir = download_kaggle_dataset("laotse/credit-risk-dataset", root)
        file_path = os.path.join(dataset_dir, "credit_risk_dataset.csv")
        df = pd.read_csv(file_path)
        target_col = "loan_status"
        test_size = 10000 / len(df)
        X_final, y = preprocess_tabular(df, target_col)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long() + 1
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long() + 1
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    elif name.lower() == "kaggle_cervical":
        dataset_dir = download_kaggle_dataset("loveall/cervical-cancer-risk-classification", root)
        file_path = os.path.join(dataset_dir, "kag_risk_factors_cervical_cancer.csv")
        df = pd.read_csv(file_path)
        df = df.replace("?", np.nan).astype(float).apply(lambda col: col.fillna(col.mean()))
        target_col = "Biopsy"
        df = df.drop(columns=["Hinselmann", "Schiller", "Citology"])
        test_size = 150
        X_final, y = preprocess_tabular(df, target_col)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    elif name.lower() == "compas":
        # dataset_dir = download_kaggle_dataset("danofer/compass", root)
        # file_path = os.path.join(dataset_dir, "compas-scores-raw.csv")
        df = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
        drop_cols = [
            "name", "first", "last", "compas_screening_date", "dob", "age_cat",
            "decile_score", "score_text", "v_decile_score", "v_score_text",
            "v_type_of_assessment", "type_of_assessment", "screening_date",
            "assessment_id", "case_id", "is_recid", "r_case_number", "c_case_number",
            "c_offense_date", "c_arrest_date", "c_jail_in", "c_jail_out", "start", "end",
            "in_custody", "out_custody", "days_in_jail", "id", "violent_recid",
            "v_screening_date", "v_assessment_id", "v_charge_desc", "v_charge_degree",
            "r_charge_desc"
        ]

        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # --- Define target ---
        # Typically used: `two_year_recid` (0 or 1)
        df = df[df['two_year_recid'].isin([0, 1])]  # filter invalid entries

        # Rename target to 'target_str' as requested
        df = df.rename(columns={'two_year_recid': 'target_str'})

        # --- Optional: Clean missing values ---
        df = df.dropna()
        df = df.reset_index(drop=True)

        target_col = "target_str"
        test_size = 0.1
        X_final, y = preprocess_tabular(df, target_col, bin=False)

        # Train/test split
        train_idx, test_idx = train_test_split(np.arange(len(X_final)), test_size=test_size, random_state=42)
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if train:
            X_tensor = torch.from_numpy(X_train).float()
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)


    # ===================
    # Other datasets (manual for now)
    # ===================
    elif name.lower() == "loan":
        raise FileNotFoundError("Loan dataset not on Kaggle; please provide manually.")

    elif name.lower() == "maggic":
        raise FileNotFoundError("MAGGIC dataset is proprietary; please provide manually.")
    
    elif name.lower() == "unos":
        raise FileNotFoundError("UNOS dataset requires authorization; please provide manually.")

    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    
    if num_max_samples is not None:
        dataset, X_tensor, y_tensor = reduce_class_samples(X_tensor, y_tensor, num_max_samples)

    return dataset


# ============================================================================
# Neural Network Models
# ============================================================================


class BasicBlock(nn.Module):
    """Basic residual block for ResNet architectures.
    
    Implements a 3x3 convolution block with optional stride for downsampling.
    Uses Group Normalization instead of Batch Normalization for compatibility
    with Opacus privacy-preserving training.
    """
    
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """Initialize basic residual block.
        
        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for the first convolution. Default: 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch, in_planes, height, width).
            
        Returns:
            Output tensor of shape (batch, planes, height//stride, width//stride).
        """
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture with configurable depth.
    
    Adapted for privacy-preserving training with Group Normalization
    instead of Batch Normalization.
    """

    def __init__(
        self,
        block: type,
        num_blocks: List[int],
        num_classes: int = 10,
        num_input_channels: int = 3,
    ):
        """Initialize ResNet.
        
        Args:
            block: Residual block class (e.g., BasicBlock).
            num_blocks: Number of blocks in each layer [layer1, layer2, layer3, layer4].
            num_classes: Number of output classes.
            num_input_channels: Number of input channels.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=False)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a layer of residual blocks.
        
        Args:
            block: Residual block class.
            planes: Number of output channels.
            num_blocks: Number of blocks in this layer.
            stride: Stride for the first block.
            
        Returns:
            Sequential module containing the blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            
        Returns:
            Output logits of shape (batch, num_classes).
        """
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def load_model(
    dataset_name: str,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """Load a neural network model for a specified dataset.
    
    Automatically selects appropriate architecture based on dataset type
    (MLP for tabular, CNN for vision, etc.).
    
    Args:
        dataset_name: Name of the dataset (must exist in DATASETS_INFO).
        model_name: Model architecture name. If None, uses default for dataset.
        num_classes: Number of output classes. If None, inferred from dataset.
    
    Returns:
        Instantiated PyTorch model.
        
    Raises:
        ValueError: If dataset or model name is not supported.
    """
    dataset_name = dataset_name.lower()
    
    # Get dataset info
    num_classes = DATASETS_INFO[dataset_name]["classes"]
    if model_name is None:
        model_name = DATASETS_INFO[dataset_name]["model_type"]
    input_dim = DATASETS_INFO[dataset_name]["input_features"]
    num_spatial = DATASETS_INFO[dataset_name].get("num_spatial", None)

    # Model definitions
    if model_name.lower() == 'mlp':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    elif model_name.lower() == 'large_mlp':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif model_name.lower() == 'simple_cnn':
        if num_spatial == 28:
            embed_dim = 7
        elif num_spatial == 32:
            embed_dim = 8
        else:
            raise Exception
        model = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * embed_dim * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif model_name.lower() == 'resnet9':
        if num_spatial == 28:
            embed_dim = 7
        elif num_spatial == 32:
            embed_dim = 8
        else:
            raise Exception(f"Unsupported spatial size: {num_spatial}")

        model = nn.Sequential(
            # Stage 1
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            # Stage 3 (residual blocks)
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            
            nn.Flatten(),
            nn.Linear(128 * embed_dim * embed_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, num_classes)
        )
        return model

    elif model_name.lower() == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_input_channels=input_dim)

    elif model_name.lower() == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_input_channels=input_dim)

    elif model_name.lower() in ['small_vgg', 'vgg']:
        model = vgg11(num_classes=num_classes)
        # Adjust input conv layer if input channels differ
        if input_dim != 3:
            model.features[0] = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1)
        return model
    
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model


# ============================================================================
# Privacy and Model Management Utilities
# ============================================================================


def save_logits(
    args: argparse.Namespace,
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    DEVICE: torch.device,
    savedir: str,
    path: str,
) -> None:
    """Compute and save model logits for all training samples.
    
    Performs multiple forward passes (specified by args.n_queries) and saves
    the logits for use in membership inference or other analyses.
    
    Args:
        args: Configuration namespace with 'n_queries' attribute.
        model: Trained model to get logits from.
        train_dl: DataLoader providing training data.
        DEVICE: Device to run inference on.
        savedir: Base directory for saving.
        path: Subdirectory path within savedir.
    """
    save_path = os.path.join(savedir, path, "logits.npy")
    if os.path.exists(save_path):
        return  # Already computed

    model.eval()
    logits_n = []
    
    with torch.no_grad():
        for _ in range(args.n_queries):
            logits = []
            for x, _ in train_dl:
                x = x.to(DEVICE)
                outputs = model(x)
                logits.append(outputs.detach().cpu().numpy())
            logits_n.append(np.concatenate(logits))
    
    logits_n = np.stack(logits_n, axis=1)
    np.save(save_path, logits_n)
    
    # Cleanup
    del logits_n, logits, outputs, x
    torch.cuda.empty_cache()
    gc.collect()


def make_private(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    pp_budgets: np.ndarray,
    args: argparse.Namespace,
    adapt_weights_to_budgets: bool = True,
    nm: Optional[np.ndarray] = None,
    sr: Optional[np.ndarray] = None,
    cw: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, Any]:
    """Make a model differentially private using Opacus.
    
    Wraps a model with differential privacy guarantees, supporting:
    - Automatic epsilon/delta-based privacy budgets
    - Per-sample gradient clipping
    - Variable noise multipliers or sampling rates per sample
    
    Args:
        model: PyTorch model to privatize.
        train_loader: Training data loader.
        optimizer: Optimizer for training.
        pp_budgets: Privacy budget (epsilon) per sample.
        args: Configuration namespace with privacy parameters.
        adapt_weights_to_budgets: If True, automatically determine noise/sampling.
                                 If False, use provided nm/sr/cw values.
        nm: Noise multipliers per budget (if adapt_weights_to_budgets=False).
        sr: Sampling rates per budget (if adapt_weights_to_budgets=False).
        cw: Clipping weights per budget (if adapt_weights_to_budgets=False).
    
    Returns:
        Tuple of (private_model, private_optimizer, private_loader, privacy_engine).
    """
    privacy_engine = PrivacyEngine(
        accountant=args.accountant,
        individualize=args.individualize,
        weights=args.weights,
        pp_budgets=pp_budgets,
    )
    
    if adapt_weights_to_budgets:
        # Automatically determine noise/sampling rates based on target epsilon
        private_model, private_optimizer, private_loader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=float(np.min(pp_budgets)),
                target_delta=args.target_delta,
                epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
                optimal=True,
                max_alpha=10_000,
                numeric=True,
                precision=0.00001,
            )
        )
    else:
        # Use provided noise multipliers or sampling rates
        unique_budgets = np.unique(pp_budgets)
        budget_to_weight = {}
        for i, budget in enumerate(unique_budgets):
            if args.individualize == "sampling":
                budget_to_weight[budget] = sr[i]
            else:
                budget_to_weight[budget] = cw[i]
        
        pp_weights = np.array([budget_to_weight[budget] for budget in pp_budgets])
        private_model, private_optimizer, private_loader = (
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=nm[0],
                max_grad_norm=args.max_grad_norm,
                pp_weights=pp_weights,
            )
        )
    
    return private_model, private_optimizer, private_loader, privacy_engine