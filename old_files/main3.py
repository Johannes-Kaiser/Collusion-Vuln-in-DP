from opacus_new import PrivacyEngine
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from opacus_new.validators.module_validator import ModuleValidator
import torch
from torch.utils.data import Dataset
from opacus_new.accountants import RDPAccountant
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm


def compute_epsilon_delta(noise_multiplier, deltas, iterations, sampling_rate, clipping_norm):
    def compute_epsilon(noise_multiplier, delta, iterations, sampling_rate, clipping_norm):
        accountant = RDPAccountant()
        for _ in range(int(iterations)):
            accountant.step(noise_multiplier=noise_multiplier * 1/clipping_norm, sample_rate=sampling_rate)
        return accountant.get_epsilon(delta)
    epsilons = []
    deltas2 = []
    for delta in tqdm(deltas):
        epsilon = compute_epsilon(noise_multiplier, delta, iterations, sampling_rate, clipping_norm)
        epsilons.append(epsilon)
        deltas2.append(delta)
    return epsilons, deltas2



class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        # Apply transformations if any
        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    @classmethod
    def from_dataset(cls, dataset):
        return cls(data=dataset.data,
                   targets=dataset.targets,
                   transform=dataset.transform,
                   target_transform=dataset.target_transform)



args = SimpleNamespace(
    accountant="rdp",  # Options: "rdp", "gdp", etc.
    individualize="clipping", # "sampling",  # Options: None, "clipping", "sampling"
    weights=None,  # Should be a list or None
    adapt_weights_to_budgets=True,  # Whether to adapt weights to budgets
    target_delta=1e-5,  # Default delta value for DP
    epochs=100,  # Number of training epochs
    max_grad_norm=1.0,  # Clipping norm for DP-SGD
    noise_multiplier=1.0,  # Noise multiplier for DP
)

def make_private(model, train_loader, pp_budgets, args):
    modulevalidator = ModuleValidator()
    model = modulevalidator.fix_and_validate(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    privacy_engine = PrivacyEngine(accountant=args.accountant,
                                   individualize=args.individualize,
                                   weights=args.weights,
                                   pp_budgets=pp_budgets)
    if args.adapt_weights_to_budgets:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private_with_epsilon(module=model,
                                       optimizer=optimizer,
                                       data_loader=train_loader,
                                       target_epsilon=min(pp_budgets),
                                       target_delta=args.target_delta,
                                       epochs=args.epochs,
                                       max_grad_norm=args.max_grad_norm,
                                       optimal=True,
                                       max_alpha=10_000)
    else:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private(module=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          noise_multiplier=args.noise_multiplier,
                          max_grad_norm=args.max_grad_norm)

    if args.individualize == 'clipping':
        return{
            "budgets": list(np.unique(np.array(pp_budgets))),
            "max_grad_norms": privacy_engine.weights,
            "sample_rate": [1 / len(private_loader)] * len(privacy_engine.weights),
            "noise_multiplier": [private_optimizer.noise_multiplier] * len(privacy_engine.weights)
        }
    elif args.individualize == 'sampling':
        return{
            "budgets": list(np.unique(np.array(pp_budgets))),
            "max_grad_norms": [args.max_grad_norm] * len(privacy_engine.weights),
            "sample_rate":privacy_engine.weights,
            "noise_multiplier":[private_optimizer.noise_multiplier] * len(privacy_engine.weights)
        }
    else:
        return



# Create a dummy ResNet18 model
class DummyResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Generate a dummy dataset with 10,000 samples
num_samples = 10_000
num_features = 3  # Number of input channels (e.g., RGB images)
height, width = 32, 32  # Input size

# X_dummy = (255 * torch.randn(num_samples, num_features, height, width)).to(torch.int16)  # Random images
X_dummy = torch.rand(num_samples, num_features, height, width)  # Random floats between 0 and 1

# Scale to integers between 0 and 1
X_dummy = torch.round(255*X_dummy).to(torch.uint8)  # Convert to integers (0 or 1)
X_dummy = X_dummy.permute(0, 2, 3, 1).cpu().numpy()  # Convert from (N, C, H, W) to (N, H, W, C)
y_dummy = torch.randint(0, 10, (num_samples,))  # Random labels (10 classes)

# Create DataLoader
batch_size = 256
dataset = CustomDataset(X_dummy, y_dummy)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model and optimizer
dummy_model = DummyResNet18(num_classes=10)

b1 = 1
b2 = 20
deltas = np.logspace(-12, -2, 400)
epoch_values = np.arange(10, 210, 10)  # Vary epochs from 10 to 200 in steps of 10

epsilon_clipping_b1 = []
epsilon_sampling_b1 = []
epsilon_clipping_b2 = []
epsilon_sampling_b2 = []

for epochs in epoch_values:
    args.epochs = epochs
    pp_budgets = [b1] * int(0.1 * num_samples) + [b2] * int(0.9 * num_samples)
    
    print(f"\nOn clipping for epochs = {epochs}")
    args.individualize = "clipping"
    clipping_data = make_private(dummy_model, train_loader, pp_budgets, args)
    
    print(f"\nOn sampling for epochs = {epochs}")
    args.individualize = "sampling"
    sampling_data = make_private(dummy_model, train_loader, pp_budgets, args)
    
    # Compute epsilon for clipping
    epsilon_clipping_p1, _ = compute_epsilon_delta(
        clipping_data["noise_multiplier"][0],
        deltas,
        args.epochs * num_samples / batch_size,
        clipping_data["sample_rate"][0],
        clipping_data["max_grad_norms"][0],
    )
    epsilon_clipping_p2, _ = compute_epsilon_delta(
        clipping_data["noise_multiplier"][-1],
        deltas,
        args.epochs * num_samples / batch_size,
        clipping_data["sample_rate"][-1],
        clipping_data["max_grad_norms"][-1],
    )
    
    # Compute epsilon for sampling
    epsilon_sampling_p1, _ = compute_epsilon_delta(
        sampling_data["noise_multiplier"][0],
        deltas,
        args.epochs * num_samples / batch_size,
        sampling_data["sample_rate"][0],
        sampling_data["max_grad_norms"][0],
    )
    epsilon_sampling_p2, _ = compute_epsilon_delta(
        sampling_data["noise_multiplier"][-1],
        deltas,
        args.epochs * num_samples / batch_size,
        sampling_data["sample_rate"][-1],
        sampling_data["max_grad_norms"][-1],
    )
    
    epsilon_clipping_b1.append(epsilon_clipping_p1)
    epsilon_sampling_b1.append(epsilon_sampling_p1)
    epsilon_clipping_b2.append(epsilon_clipping_p2)
    epsilon_sampling_b2.append(epsilon_sampling_p2)

# Plot results
plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
cmap = cm.get_cmap('viridis', len(epoch_values))

for i, epochs in enumerate(epoch_values):
    color = cmap(i / len(epoch_values))
    axes[0, 0].loglog(deltas, epsilon_clipping_b1[i], label=f'epochs = {epochs}', color=color)
    axes[0, 1].loglog(deltas, epsilon_sampling_b1[i], label=f'epochs = {epochs}', color=color)
    axes[1, 0].loglog(deltas, epsilon_clipping_b2[i], label=f'epochs = {epochs}', color=color)
    axes[1, 1].loglog(deltas, epsilon_sampling_b2[i], label=f'epochs = {epochs}', color=color)

axes[0, 0].set_title(f'Clipping - epsilon {b1}')
axes[0, 1].set_title(f'Sampling - epsilon {b1}')
axes[1, 0].set_title(f'Clipping - epsilon {b2}')
axes[1, 1].set_title(f'Sampling - epsilon {b2}')

for ax in axes.flat:
    ax.set_xlabel('Delta')
    ax.set_ylabel('Epsilon Values')
    ax.legend()

plt.tight_layout()
plt.savefig('./epoch_variation.png')
plt.show()
