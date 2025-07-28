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
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np

def parse_extra(parser, configs):
    """Using a parser and a base config, modify the base config according to the parser
    for CLI bash experiments. E.g. python main.py --cf demo_relative.raml --audit.signal "softmax_relative"
    Args:
        parser (Parser): Parser with the basic arguments
        configs (dict): Dict that we want to change according to the parser
    Returns:
        configs: modified input config
    """
    for key in configs:
        # Generate arguments for top-level keys
        arg_name = '--{}'.format(key)
        parser.add_argument(arg_name, dest=key, default=None,
                            help='{} parameter'.format(arg_name))
        for subkey in configs[key]:
            # Generate arguments for second-level keys
            arg_name = '--{}.{}'.format(key, subkey)
            parser.add_argument(arg_name, dest=subkey, default=None,
                                help='{} parameter'.format(arg_name))
            if isinstance(configs[key][subkey], dict):
                for subsubkey in configs[key][subkey]:
                    # Generate arguments for eventual third-level keys
                    arg_name = '--{}.{}.{}'.format(key, subkey, subsubkey)
                    parser.add_argument(arg_name, dest=subsubkey, default=None,
                                        help='{} parameter'.format(arg_name))
    # Parse command-line arguments
    args, unknown_args = parser.parse_known_args()
    # Update configuration dictionary with command-line arguments
    if args:
        for key in configs:
            if args.__dict__.get(key) is not None:
                configs[key] = args.__dict__.get(key)
            for subkey in configs[key]:
                if args.__dict__.get(subkey) is not None:
                    configs[key][subkey] = args.__dict__.get(subkey)
                if isinstance(configs[key][subkey], dict):
                    for subsubkey in configs[key][subkey]:
                        arg_name = '{}.{}.{}'.format(key, subkey, subsubkey)
                        if args.__dict__.get(subsubkey) is not None:
                            configs[key][subkey][subsubkey] = args.__dict__.get(subsubkey)
    return configs


def train_loop(args, model, train_loader, criterion, optimizer, sched, device, epochs, privacy_engine=None):
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
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False, disable=args["train"]["disable_inner"]) as pbar:
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
        if not args["train"]["disable_inner"]:
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
        idx_0 = np.where(targets == 0)[0][:500]
        idx_1 = np.where(targets == 1)[0][:500]
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
        elif dataset_name in ['breast_cancer', 'mnist_2']:
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