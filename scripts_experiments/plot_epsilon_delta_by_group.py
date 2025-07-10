import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from opacus_new.accountants import RDPAccountant

def compute_epsilon_delta(noise_multiplier, deltas, iterations, sampling_rate, clipping_norm):
    def compute_epsilon(noise_multiplier, delta, iterations, sampling_rate, clipping_norm):
        accountant = RDPAccountant()
        for _ in range(int(iterations)):
            accountant.step(noise_multiplier=noise_multiplier * 1/clipping_norm, sample_rate=sampling_rate)
        return accountant.get_epsilon(delta)
    epsilons = []
    deltas2 = []
    for delta in deltas:
        epsilon = compute_epsilon(noise_multiplier, delta, iterations, sampling_rate, clipping_norm)
        epsilons.append(epsilon)
        deltas2.append(delta)
    return np.array(epsilons), np.array(deltas2)

def main():
    # --- User configuration ---
    n_data = 2000
    epochs = 5
    deltas = np.logspace(-8, -1, 100)
    group_budgets = [8, 16, 32, 64]  # Example: 4 groups with different epsilons
    group_portions = [0.05, 0.15, 0.3, 0.5]  # Must sum to 1.0
    noise_multiplier = 1.0
    clipping_norm = 1.0
    sampling_rate = 128 / n_data  # Example batch size 128
    # --------------------------

    assert abs(sum(group_portions) - 1.0) < 1e-6, "Portions must sum to 1.0"
    n_groups = len(group_budgets)
    iterations = epochs * int(n_data * group_portions[0] / (sampling_rate * n_data))  # Approximate steps per group

    # Compute epsilon-delta for each group
    results = []
    for i, (budget, portion) in enumerate(zip(group_budgets, group_portions)):
        # For demo: use same noise_multiplier, sampling_rate, clipping_norm for all groups
        epsilons, deltas_out = compute_epsilon_delta(
            noise_multiplier=noise_multiplier,
            deltas=deltas,
            iterations=epochs * int(n_data * portion / 128),
            sampling_rate=sampling_rate,
            clipping_norm=clipping_norm
        )
        results.append({
            "budget": budget,
            "portion": portion,
            "epsilons": epsilons,
            "deltas": deltas_out
        })

    # Plotting
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), sharey=True)
    if n_groups == 1:
        axes = [axes]
    palette = sns.color_palette("viridis", n_colors=len(group_portions))
    for idx, ax in enumerate(axes):
        for j, res in enumerate(results):
            if idx != j:
                continue
            color = palette[j]
            ax.semilogy(res["epsilons"], res["deltas"], label=f"Portion={res['portion']:.2f}", color=color)
            ax.set_xlabel(r"$\epsilon$")
            ax.set_ylabel(r"$\delta$")
            ax.set_title(f"Group {j+1} (ε={res['budget']})")
            ax.legend()
    plt.suptitle("Delta vs Epsilon for Different Privacy Groups and Portions")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
