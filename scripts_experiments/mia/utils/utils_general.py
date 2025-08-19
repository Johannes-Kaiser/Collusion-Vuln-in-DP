import os
import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from torchvision import transforms
from torchvision.models import resnet18, vgg11
from opacus_new import PrivacyEngine
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import kagglehub
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import medmnist
from medmnist import INFO

datasets_info = {
    # Vision datasets
    "mnist": {"classes": 10, "input_features": 28 * 28, "model_type": "mlp"},
    "mnist_4": {"classes": 4, "input_features": 28 * 28, "model_type": "mlp"}, 
    "mnist_2": {"classes": 2, "input_features": 28 * 28, "model_type": "mlp"},
    "mnist_4_small": {"classes": 4, "input_features": 28 * 28, "model_type": "mlp"},
    "cifar10": {"classes": 10, "input_features": 3, "model_type": "simple_cnn", "num_spatial": 32},
    "fashionmnist": {"classes": 10, "input_features": 1, "model_type": "simple_cnn", "num_spatial": 32},
    "svhn": {"classes": 10, "input_features": 1, "model_type": "simple_cnn", "num_spatial": 32},

    # Tabular datasets (sklearn & OpenML)
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


for dataset_name, info in INFO.items():
    n_classes = len(info["label"].keys())
    n_channels = 3 if info["n_channels"] == 3 else 1  # most are grayscale, some RGB
    datasets_info[dataset_name] = {
        "classes": n_classes,
        "input_features": n_channels,   # for CNNs we treat channels as input
        "model_type": "simple_cnn" ,
        "num_spatial": 28     # CNN is more suitable than MLP for images
    }

def to_padded_array(list_of_lists):
    # Find the maximum length of inner lists
    max_len = max(len(lst) for lst in list_of_lists)
    # Pad each list with None to match the maximum length
    padded = [lst + [0] * (max_len - len(lst)) for lst in list_of_lists]
    return np.array(padded, dtype=object)

def assign_pp_values(
    pp_budgets,
    values,
):
    r"""
    Assigns a value to each data point according to the given per-point budgets.
    Args:
        pp_budgets: the privacy budget's epsilon for each data point
        values: list of values to be assigned to all data points
    Returns:
        An array of size equal to the training dataset size that contains one
        value for each data point.
    """
    pp_values = np.zeros(len(pp_budgets))
    for i, budget in enumerate(np.sort(np.unique(pp_budgets))):
        pp_values[pp_budgets == budget] = values[i]
    return torch.Tensor(pp_values)


def generate_string(list1, list2, sep1=": ", sep2=","):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length.")
    return sep2.join(f"{k}{sep1}{v}" for k, v in zip(list1, list2))#


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
    parser.add_argument('--individualize', type=str, help='YAML file with experiment parameters', default='sampling')
    parser.add_argument("--seed", type=int, default=42)
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


def train_loop(params, model, train_loader, criterion, optimizer, sched, device, epochs, pp_max_grad_norms=None):
    """
    Generic training loop for PyTorch models, compatible with Opacus-privatized models.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on ('cpu' or 'cuda').
        epochs: Number of epochs to train.
        pp_max_grad_norms: Optional per-sample max grad norms for Opacus DP training.
    """
    model.to(device)
    model.train()
    disable_inner = params.disable_inner if hasattr(params, 'disable_inner') else params.get('disable_inner', False)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for batch in train_loader:
            if len(batch) == 2:
                inputs, targets = batch
                idx = None
            elif len(batch) == 3:
                inputs, targets, idx = batch
            else:
                raise Exception("Batch must have 2 or 3 elements (inputs, targets, [idx])")

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if idx is None:
                optimizer.step()
            else:
                optimizer.step(pp_max_grad_norms=pp_max_grad_norms[idx].to(device))

            running_loss += loss.item()

            # Compute batch accuracy
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Update per-class accuracy
            for label, prediction in zip(targets, predicted):
                class_total[label.item()] += 1
                if prediction.item() == label.item():
                    class_correct[label.item()] += 1

        if sched is not None:
            sched.step()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        if not disable_inner:
            # Report per-class accuracy
            string = ""
            class_ids = sorted(class_total.keys())
            for cls in class_ids:
                acc = 100.0 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
                string += f"    Class {cls}: Accuracy: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})"
            tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f} - Accuracy: {accuracy:.2f}%  {string}")



    return model


def download_kaggle_dataset(dataset_slug, root):
    """
    Downloads a Kaggle dataset using Kaggle API if not already present.
    dataset_slug example: "mlg-ulb/creditcardfraud"
    """
    os.environ["KAGGLEHUB_CACHE"] = root
    dataset_dir = kagglehub.dataset_download(dataset_slug)
    return dataset_dir

def preprocess_tabular(df, target_col, bin=True):
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(df.index)
    df = df.iloc[shuffled_indices].reset_index(drop=True)
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    X_processed = []
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            if bin:
                lb = LabelBinarizer()
                transformed = lb.fit_transform(X[col])
                if transformed.shape[1] == 1:  # binary
                    X_processed.append(transformed.ravel())
                else:  # multi-class
                    X_processed.append(transformed)
            else:
                le = LabelEncoder()
                scaler = MinMaxScaler()
                transformed = le.fit_transform(X[col]).reshape(-1, 1)
                transformed = scaler.fit_transform(transformed).reshape(-1, 1)
                X_processed.append(transformed)
        else:
            scaler = MinMaxScaler()
            transformed = scaler.fit_transform(X[[col]])
            X_processed.append(transformed.ravel())

    X_final = np.column_stack(X_processed)

    # Encode target if categorical
    if isinstance(y[0], str): 
        try:
            y = [int(x) for x in y]
        except Exception as _:
            le = LabelEncoder()
            y = le.fit_transform(y)
    if isinstance(y[0], bool):
        y_lb = LabelBinarizer()
        y = y_lb.fit_transform(y).ravel()
    y = np.array(y) - min(np.array(y))  # Ensure targets start from 0
    X_final[np.isnan(X_final)] = np.take(np.nanmean(X_final, axis=0), np.where(np.isnan(X_final))[1])
    return X_final, y

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
        # ===================
    # MedMNIST 2D datasets
    # ===================
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
        if transform is None:
            transform = transforms.ToTensor()
        dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
        X_tensor = dataset.test_data
        y_tensor = dataset.test_labels
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
            y_tensor = torch.from_numpy(y_train).long() - 1
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long() - 1

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
            y_tensor = torch.from_numpy(y_train).long()
        else:
            X_tensor = torch.from_numpy(X_test).float()
            y_tensor = torch.from_numpy(y_test).long()
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

    # Preproc
    else:
        raise ValueError(f"Dataset {name} not supported.")

    print(f"highest class {torch.max(y_tensor)} with dist {torch.unique(y_tensor, return_counts=True)}")
    return dataset



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
    num_classes = datasets_info[dataset_name]["classes"] 
    model_name = datasets_info[dataset_name]["model_type"] 
    input_dim = datasets_info[dataset_name]["input_features"]
    num_spatial = datasets_info[dataset_name].get("num_spatial", None)

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
    elif model_name.lower() == 'resnet18':
        # Use torchvision's ResNet18 for small images

        model = resnet18(num_classes=input_dim)
        # Adjust input conv layer if needed
        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name.lower() == 'small_vgg' or model_name.lower() == 'vgg':

        model = vgg11(num_classes=input_dim)
        # Adjust input conv layer if needed
        if input_dim != 3:
            model.features[0] = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1)
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