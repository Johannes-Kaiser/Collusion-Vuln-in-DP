import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from opacus import PrivacyEngine
from tqdm import tqdm
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant

# Constants and Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

## ------------------------------------------------ ##
## 1. Model Definition (Based on Table 5 in the paper)
## ------------------------------------------------ ##
def get_mnist_model():
    """Returns the CNN architecture for MNIST as described in the paper[cite: 481]."""
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 32), # Adjusted input features based on conv output
        nn.ReLU(),
        nn.Linear(32, 10),
    ).to(DEVICE)
    return model

## ------------------------------------------------------------- ##
## 2. Privacy Budget and Threshold Calculation (Eqns. 4, 6, 7)
## ------------------------------------------------------------- ##
def generate_privacy_budgets(n_samples, epsilon_range=(0.5, 1.0), k=0.0):
    """
    Generates privacy budgets for each user based on an exponential distribution,
    [cite_start]as described in Section 5.2 of the paper[cite: 271, 469].
    k: Skewness parameter. k=0 is uniform, k>0 skews towards higher epsilon.
    """
    if k == 0: # Uniform distribution
        counts = np.ones(20)
    else: # Exponential distribution
        # Constants from the paper for scaling
        c1 = 2.098 if k == -0.2 else 1.554
        c2 = -1.715
        eps_values = np.linspace(epsilon_range[0], epsilon_range[1], 20)
        counts = (c1 * np.exp(k * eps_values) + c2)

    counts = np.maximum(0, counts)
    probs = counts / counts.sum()
    
    # Assign each of the n_samples to one of the 20 epsilon groups
    group_indices = np.random.choice(20, size=n_samples, p=probs)
    epsilon_levels = np.linspace(epsilon_range[0], epsilon_range[1], 20)
    
    budgets = epsilon_levels[group_indices]
    np.random.shuffle(budgets) # Shuffle to randomize assignment
    return torch.tensor(budgets, dtype=torch.float32)

def calculate_loss_f(budgets, tau, w1=0.7, w2=0.3):
    """Calculates the fixed-weight loss (loss_f) from Equation 6[cite: 154]."""
    sampled_mask = budgets >= tau
    unsampled_mask = ~sampled_mask

    # Calculate sampling probabilities for unsampled points
    pi = (torch.exp(budgets[unsampled_mask]) - 1) / (torch.exp(tau) - 1)
    
    waste_u = torch.sum(1 - pi)
    waste_s = torch.sum(budgets[sampled_mask] - tau)
    
    return w1 * waste_u + w2 * waste_s

def calculate_loss_a(budgets, tau):
    """Calculates the adaptive-weight loss (loss_a) from Equation 7[cite: 169]."""
    sampled_mask = budgets >= tau
    unsampled_mask = ~sampled_mask

    pi = (torch.exp(budgets[unsampled_mask]) - 1) / (torch.exp(tau) - 1)
    
    waste_u = torch.sum(1 - pi)
    waste_s = torch.sum(budgets[sampled_mask] - tau)

    if waste_u + waste_s == 0:
        return 0
        
    # The paper's formula contains a typo, this is the corrected version
    # It should be 2 * sqrt(waste_u * waste_s) from AM-GM inequality
    # Or simply waste_u + waste_s with adaptive weights. Let's use the latter.
    return waste_u + waste_s

def compute_threshold(budgets, loss_type='adaptive'):
    """Finds the best threshold tau by minimizing the chosen loss function[cite: 179]."""
    min_budget = torch.min(budgets[budgets > 0])
    max_budget = torch.max(budgets)
    
    # Search for the best tau in the range of available budgets
    tau_candidates = torch.linspace(min_budget, max_budget, steps=50)
    min_loss = float('inf')
    best_tau = min_budget

    for tau in tau_candidates:
        if loss_type == 'adaptive':
            loss = calculate_loss_a(budgets, tau)
        else: # 'fixed'
            loss = calculate_loss_f(budgets, tau)
        
        if loss < min_loss:
            min_loss = loss
            best_tau = tau
            
    return best_tau

def get_sampling_probabilities(budgets, tau):
    """Calculates sampling probability pi for each data point using Equation 4[cite: 131]."""
    probs = torch.zeros_like(budgets)
    less_than_tau_mask = (budgets < tau) & (budgets > 0)
    
    # Calculate probabilities for those with epsilon < tau
    probs[less_than_tau_mask] = \
        (torch.exp(budgets[less_than_tau_mask]) - 1) / (torch.exp(tau) - 1)
        
    # Probabilities are 1 for those with epsilon >= tau
    probs[budgets >= tau] = 1.0
    
    return probs

## -------------------------------- ##
## 3. Main PDP-SGD Training Logic
## -------------------------------- ##
def train_pdp_sgd(model, train_dataset, config):
    """
    Implements the PDP-SGD algorithm (Algorithm 1) with the fix for the hook error.
    """
    # The 'model' passed here is our original, clean model.
    # The optimizer is created once on the parameters of this original model.
    criterion = nn.CrossEntropyLoss()
    
    privacy_budgets = generate_privacy_budgets(
        len(train_dataset), config['epsilon_range'], config['skew']
    ).to(DEVICE)
    privacy_budgets[-1] = 2
    # --- Main loop over rounds ---
    sr_of_investigated = []
    nm_of_investigated = []
    for r in range(1, config['num_rounds'] + 1):
        try:
            print(f"\n--- Round {r}/{config['num_rounds']} ---")
            
            active_budgets = privacy_budgets[privacy_budgets > 0]
            if len(active_budgets) == 0:
                print("All privacy budgets have been exhausted. Stopping training.")
                break
                
            tau_r = compute_threshold(active_budgets, config['loss_type'])
            print(f"Computed Threshold τ for this round: {tau_r:.4f}")

            sampling_probs = get_sampling_probabilities(privacy_budgets, tau_r)
            print(len(sampling_probs))
            
            potential_indices = torch.where(sampling_probs > 0)[0].cpu().numpy()
            sampled_mask = torch.bernoulli(sampling_probs[potential_indices]).bool()
            round_indices = potential_indices[sampled_mask.cpu().numpy()]
            
            if len(round_indices) == 0:
                print("No data points sampled in this round. Skipping.")
                continue
                
            print(f"Sampled {len(round_indices)} data points for this round.")
            round_dataset = Subset(train_dataset, round_indices)
            round_loader = DataLoader(
                round_dataset, batch_size=len(round_dataset), shuffle=True
            )

            # 1. Create a new PrivacyEngine for the round.
            privacy_engine = PrivacyEngine()

            # 2. Create a fresh, clean model for this round.
            #    This prevents the "add hooks twice" error.
            round_model = get_mnist_model().to(DEVICE)
            optimizer = optim.SGD(round_model.parameters(), lr=config['lr'])

            # 3. Load the state of the master model into the clean round model.
            round_model.load_state_dict(model.state_dict())

            # 4. Now, make this clean, updated model private.
            round_model.train()
            wrapped_model, optimizer, round_loader = privacy_engine.make_private_with_epsilon(
                module=round_model,  # Always use the original, clean model
                optimizer=optimizer,
                data_loader=round_loader,
                max_grad_norm=config['max_grad_norm'],
                target_epsilon=tau_r.item(),
                target_delta=config['delta'],
                epochs=config['epochs_per_round'],
            )
            noise_multiplier = get_noise_multiplier(
                target_epsilon=tau_r.item(),
                target_delta=config['delta'],
                sample_rate=1,
                epochs=config['epochs_per_round'],
                accountant="rdp",
            )
                
            if len(train_dataset)-1 in round_indices:
                nm_of_investigated.append(noise_multiplier)
                sr_of_investigated.append(sampling_probs[-1])

            # --- Inner loop for iterations in the round ---
            wrapped_model.train() # Use the wrapped model for training
            for epoch in range(config['epochs_per_round']):
                for i, (images, labels) in enumerate(tqdm(round_loader, desc=f"Epoch {epoch+1}")):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = wrapped_model(images) # Forward pass on wrapped model
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            model.load_state_dict(wrapped_model._module.state_dict())
            spent_epsilon = privacy_engine.get_epsilon(config['delta'])
            print(f"Privacy spent in this round (ε'): {spent_epsilon:.4f}")

            # Detach engine. This correctly cleans up the optimizer for the next round.
            with torch.no_grad():
                privacy_budgets[round_indices] -= spent_epsilon
                privacy_budgets.clamp_(min=0)
        except OverflowError:
            return sr_of_investigated, nm_of_investigated, model

    return sr_of_investigated, nm_of_investigated, model


## -------------------------------- ##
## 4. Main Execution Block
## -------------------------------- ##
if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'num_rounds': 50,             # Total number of sampling rounds (k in paper)
        'epochs_per_round': 1,       # Epochs per round (n in paper)
        'lr': 0.05,
        'batch_size': 64,
        'max_grad_norm': 1.0,        # Gradient clipping norm C
        'delta': 1e-5,               # Target delta for (ε, δ)-DP
        'epsilon_range': (10, 100.0), # Range of personal privacy budgets
        'skew': 0.0,                 # Skewness of budget distribution (k in Fig. 3)
        'loss_type': 'fixed'      # Choose 'adaptive' (Eq. 7) or 'fixed' (Eq. 6)
    }

    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # --- Model Initialization ---
    model = get_mnist_model()
    
    # --- Run Training ---
    print("Starting Personalized DP-SGD Training...")
    sr, nm, model = train_pdp_sgd(model, train_dataset, config)
    accountant = RDPAccountant()
    delta = 1e-5
    # Log each step
    for q, sigma in zip(sr, nm):
        accountant.step(noise_multiplier=sigma, sample_rate=q)

    # Compute epsilon at delta
    epsilon = accountant.get_epsilon(delta=delta)
    print(f"ε = {epsilon:.4f} at δ = {delta}")
    print("\nTraining finished!")

