import sys
sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")

import os
import json
import numpy as np
from opacus_new.accountants import RDPAccountant
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
})

def format_list(lst):
    return "_".join(str(x).replace('.', '').replace('-', 'm') for x in lst)

# Example definitions
datasets = ["credit_card_default",
           "credit_card_default"]
paths = ["/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/credit_card_default",
             "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/credit_card_default"]
budgets_all =    [[4.0, 20.0],
                [4.0, 20.0]]
portions_list = [
    [0.2, 0.8],
    [0.4, 0.6],
    [0.6, 0.4],
    [0.8, 0.19999999999999996],
]
portions_list_all = [portions_list for _ in datasets]

def plot_experiment(dataset, base_path, budgets, portions_list):
    os.makedirs(f"{base_path}/additional_figures", exist_ok=True)
    all_paths = []
    for portions in portions_list:
        all_paths_portion = []
        path = f"{base_path}/{dataset}_[{format_list(budgets)}]_[{format_list(portions)}]"
        if os.path.isdir(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            for subdir in subdirs:
                all_paths_portion.append(os.path.join(path, subdir, "results"))
        all_paths.append(all_paths_portion)

    path_info = {}
    for all_paths_portion, portions in zip(all_paths, portions_list):
        path_portion_info = {}
        for p in all_paths_portion:
            args_path = os.path.join(p, "args.json")
            with open(args_path, "r") as f:
                args = json.load(f)
                info = {
                "budgets": args.get("budgets"),
                "portions": args.get("portions"),
                "target_delta": args.get("target_delta"),
                }
            # Load numpy arrays
            try:
                info["num_steps_list"] = np.load(os.path.join(p, "num_steps_list.npy"), allow_pickle=True).tolist()
            except FileNotFoundError:
                continue
            info["pg_noise_multiplier_list"] = np.load(os.path.join(p, "pg_noise_multiplier_list.npy"), allow_pickle=True).tolist()
            info["pg_sample_rates_list"] = np.load(os.path.join(p, "pg_sample_rates_list.npy"), allow_pickle=True).tolist()
            info["integrals_all"] = np.load(os.path.join(p, "integrals_all.npy"), allow_pickle=True).tolist()
            info["adv"] = np.load(os.path.join(p, "adv.npy"), allow_pickle=True).tolist()
            path_portion_info[p] = info
        path_info[format_list(portions)] = path_portion_info


    concatenated_integrals_dict = {}
    for portion, runs in path_info.items():
        integrals_all = [info["integrals_all"] for info in runs.values() if "integrals_all" in info]
        if integrals_all:
            keys = integrals_all[0].keys()
            # concatenated = {k: sum((d[k] for d in integrals_all), []) for k in keys}
            concatenated = {
                k: np.mean([np.array(d[k]) for d in integrals_all], axis=0).tolist()
                for k in keys
            }
            concatenated_integrals_dict[portion] = concatenated

    concatenated_adv_dict = {}
    for portion, runs in path_info.items():
        adv_all = [info["adv"] for info in runs.values() if "adv" in info]
        if adv_all:
            keys = adv_all[0].keys()
            # concatenated = {k: sum((d[k] for d in adv_all), []) for k in keys}
            concatenated = {
                k: np.mean([np.array(d[k]) for d in adv_all], axis=0).tolist()
                for k in keys
            }
            concatenated_adv_dict[portion] = concatenated


    # Extract values for each portion
    extracted_info = {}
    for portion, runs in path_info.items():
        budgets_list = []
        portions_list_ = []
        num_steps_list = []
        pg_noise_multiplier_list = []
        pg_sample_rates_list = []
        target_delta_list = []
        for info in runs.values():
            if "budgets" in info:
                budgets_list.append(info["budgets"])
            if "portions" in info:
                portions_list_.append(info["portions"])
            if "num_steps_list" in info:
                num_steps_list.append(info["num_steps_list"])
            if "pg_noise_multiplier_list" in info:
                pg_noise_multiplier_list.append(info["pg_noise_multiplier_list"])
            if "pg_sample_rates_list" in info:
                pg_sample_rates_list.append(info["pg_sample_rates_list"])
            if "target_delta" in info:
                target_delta_list.append(info["target_delta"])
        extracted_info[portion] = {
            "budgets": budgets_list,
            "portions": portions_list_,
            "num_steps_list": num_steps_list,
            "pg_noise_multiplier_list": pg_noise_multiplier_list,
            "pg_sample_rates_list": pg_sample_rates_list,
            "target_delta": target_delta_list
        }

    # Post-process extracted_info to ensure consistency and simplify lists
    for portion, info in extracted_info.items():
        # Check budgets, portions, target_delta
        for key in ["budgets", "portions", "target_delta"]:
            values = info[key]
            if not values:
                continue
            first = values[0]
            if all(v == first for v in values):
                extracted_info[portion][key] = first
            else:
                raise ValueError(f"Inconsistent values for '{key}' in portion '{portion}': {values}")

        # Check num_steps_list, pg_noise_multiplier_list, pg_sample_rates_list
        for key in ["num_steps_list", "pg_noise_multiplier_list", "pg_sample_rates_list"]:
            values = info[key]
            if not values:
                continue
            last = values[-1][-1] if isinstance(values[0], list) and values[0] else values[0]
            if all(isinstance(v, list) and v and all(x == last or x is None for x in v) for v in values):
                extracted_info[portion][key] = last
            else:
                raise ValueError(f"Inconsistent nested values for '{key}' in portion '{portion}': {values}")


    # --- VIOLIN PLOTS ---

    # Prepare keys and counts
    portion_keys = list(concatenated_integrals_dict.keys())
    budget_keys = list(next(iter(concatenated_integrals_dict.values())).keys())
    n_portions = len(portion_keys)
    n_budgets = len(budget_keys)

    # Utility: generate lighter shades of a base color
    def lighten_color(color, amount=0.5):
        """Lighten the given color by multiplying (1-luminance) by the given amount."""
        return tuple(1 - (1 - c) * amount for c in color)

    # Use a single base color and generate lighter shades for each portion
    base_color = mcolors.to_rgb("tab:blue")
    portion_shades = [lighten_color(base_color, 1 - 0.7 * (i / max(n_portions-1, 1))) for i in range(n_portions)]
    portion_colors = {k: portion_shades[i] for i, k in enumerate(portion_keys)}

    # --- VIOLIN PLOT SECTION ---
    # Get colormaps
    # Prepare color maps for consistent portion coloring
    orange_cmap = mpl.cm.get_cmap('Oranges')
    blue_cmap = mpl.cm.get_cmap('Blues')

    orange_cmap = [orange_cmap(x) for x in np.linspace(0, 1, len(portion_keys) + 3)]
    blue_cmap = [blue_cmap(x) for x in np.linspace(0, 1, len(portion_keys) + 3)]


    integral_slices = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    selected_integral_slices = [0.01, 0.05, 0.1]
    n_rows = len(selected_integral_slices) + 1
    n_cols = len(budget_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 1.4 * n_rows))

    for row_idx, integral_slice in enumerate(integral_slices):
        if integral_slice not in selected_integral_slices:
            continue
        for col_idx, budget_key in enumerate(budget_keys):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            print(f"Budget {budget_key} - {integral_slice}- {ax}")

            data = []
            positions = []
            colors_out = []
            colors_in = []
            labels = []
            medians = []

            for p_idx, (portion, portions_val) in enumerate(zip(portion_keys, portions_list)):
                integrals = concatenated_integrals_dict[portion][budget_key]
                integrals = np.array(integrals)[:, row_idx]
                data.append(integrals)
                positions.append(p_idx)

                # Assign consistent color per column
                if col_idx == 0:
                    color_out = orange_cmap[p_idx + 3]
                    color_in = orange_cmap[p_idx + 1]
                else:
                    color_out = blue_cmap[p_idx + 3]
                    color_in = blue_cmap[p_idx + 1]

                colors_out.append(color_out)
                colors_in.append(color_in)
                if col_idx == 0:
                    labels.append(rf"$\mathbf{{{portions_val[0]*100:.0f}\%}}$/{int(portions_val[1]*100)}%")
                else:
                    labels.append(rf"{int(portions_val[0]*100)}%/$\mathbf{{{portions_val[1]*100:.0f}\%}}$")


                medians.append(np.median(integrals))

            for d, color_out, color_in, pos in zip(data, colors_out, colors_in, positions):
                vp = ax.violinplot([d], positions=[pos], showmeans=False, showmedians=True, showextrema=True, widths=0.9)
                linewidth = 0.6
                # Body
                vp['bodies'][0].set_facecolor(color_in)
                vp['bodies'][0].set_edgecolor(color_out)
                vp['bodies'][0].set_alpha(0.7)
                vp['bodies'][0].set_linewidth(linewidth)

                # Median
                if 'cmedians' in vp:
                    vp['cmedians'].set_color(color_out)
                    vp['cmedians'].set_linewidth(linewidth)

                # Bars (whiskers)
                if 'cbars' in vp:
                    vp['cbars'].set_color(color_out)
                    vp['cbars'].set_linewidth(linewidth)

                # Min/max
                if 'cmins' in vp:
                    vp['cmins'].set_color(color_out)
                    vp['cmins'].set_linewidth(linewidth)

                if 'cmaxes' in vp:
                    vp['cmaxes'].set_color(color_out)
                    vp['cmaxes'].set_linewidth(linewidth)



            if row_idx == n_rows - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, fontsize=8)
            else:
                ax.set_xticks([])  # Remove x-ticks completely
                ax.set_xlabel("")  # Optional: clear label
            if col_idx == 0:
                ax.set_ylabel(f"pAUC of MIA \n at {integral_slice}", fontsize=8)
            if row_idx == 0:
                ax.set_title(rf"$\varepsilon_{col_idx}$ = {budget_key}", fontsize=8)

    row_idx = n_rows - 1
    for col_idx, budget_key in enumerate(budget_keys):
        ax = axes[row_idx, col_idx] 
        data = []
        positions = []
        colors_out = []
        colors_in = []
        labels = []
        medians = []
        print(f"Budget {budget_key} - {ax}")
        for p_idx, (portion, portions_val) in enumerate(zip(portion_keys, portions_list)):
                adv = concatenated_adv_dict[portion][budget_key]
                adv = np.array(adv)
                data.append(adv)
                positions.append(p_idx)

                # Assign consistent color per column
                if col_idx == 0:
                    color_out = orange_cmap[p_idx + 3]
                    color_in = orange_cmap[p_idx + 1]
                else:
                    color_out = blue_cmap[p_idx + 3]
                    color_in = blue_cmap[p_idx + 1]

                colors_out.append(color_out)
                colors_in.append(color_in)
                if col_idx == 0:
                    labels.append(rf"$\mathbf{{{portions_val[0]*100:.0f}\%}}$/{int(portions_val[1]*100)}%")
                else:
                    labels.append(rf"{int(portions_val[0]*100)}%/$\mathbf{{{portions_val[1]*100:.0f}\%}}$")


                medians.append(np.median(adv))

        for d, color_out, color_in, pos in zip(data, colors_out, colors_in, positions):
            vp = ax.violinplot([d], positions=[pos], showmeans=False, showmedians=True, showextrema=True, widths=0.9)
            linewidth = 0.6
            # Body
            vp['bodies'][0].set_facecolor(color_in)
            vp['bodies'][0].set_edgecolor(color_out)
            vp['bodies'][0].set_alpha(0.7)
            vp['bodies'][0].set_linewidth(linewidth)

            # Median
            if 'cmedians' in vp:
                vp['cmedians'].set_color(color_out)
                vp['cmedians'].set_linewidth(linewidth)

            # Bars (whiskers)
            if 'cbars' in vp:
                vp['cbars'].set_color(color_out)
                vp['cbars'].set_linewidth(linewidth)

            # Min/max
            if 'cmins' in vp:
                vp['cmins'].set_color(color_out)
                vp['cmins'].set_linewidth(linewidth)

            if 'cmaxes' in vp:
                vp['cmaxes'].set_color(color_out)
                vp['cmaxes'].set_linewidth(linewidth)



        if row_idx == n_rows - 1:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
        else:
            ax.set_xticks([])  # Remove x-ticks completely
            ax.set_xlabel("")  # Optional: clear label
        if col_idx == 0: 
            ax.set_ylabel(f"MIA advantage", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{base_path}/additional_figures/violin_plot_grid_colored.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}/additional_figures/violin_plot_grid_colored.svg", format="svg", bbox_inches='tight')
    plt.close(fig)



    # --- EPSILON vs DELTA PLOT ---

    plt.figure(figsize=(10, 7))

    def compute_epsilon_delta(noise_multiplier, deltas, iterations, sampling_rate, clipping_norm):
        """
        Compute epsilon for a range of deltas using the RDP accountant.
        """
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
        return epsilons, deltas2

    # Get target_delta and target_epsilons for reference lines
    target_delta = None
    target_epsilons = None
    try:
        target_delta = float(next(iter(extracted_info.values()))["target_delta"])
        target_epsilons = [float(e) for e in next(iter(extracted_info.values()))["budgets"]]
    except Exception:
        pass

    # For legend: keep track of which portion labels have been added
    legend_labels = set()

    # Plot each portion and budget with matching color shade
    for p_idx, portion in enumerate(portion_keys):
        color = portion_colors[portion]
        budgets = extracted_info[portion]["budgets"]
        info = extracted_info[portion]
        for b_idx, budget_key in enumerate(budget_keys):
            # Compute epsilon-delta curve for this portion/budget
            epsilon, delta = compute_epsilon_delta(
                info["pg_noise_multiplier_list"][b_idx],
                np.logspace(-14, -1, 50),
                info["num_steps_list"],
                info["pg_sample_rates_list"][b_idx],
                1.0
            )
            # Only add legend entry once per portion (for the first budget)
            label = "/".join(portion.split("_"))
            show_label = label if (label not in legend_labels) else None
            if show_label:
                legend_labels.add(label)
            plt.plot(
                epsilon, delta,
                color=color,
                alpha=0.9,
                linewidth=2,
                label=show_label
            )

    # Add horizontal line at target_delta and vertical lines at target_epsilons
    if target_delta is not None:
        plt.axhline(target_delta, color='red', linestyle='--', linewidth=2, label='target delta')
    if target_epsilons is not None:
        for i, eps in enumerate(target_epsilons):
            plt.axvline(eps, color='red', linestyle=':', linewidth=2, label='target epsilon' if i == 0 else None)

    plt.yscale("log")
    plt.xlabel("Epsilon", fontsize=14)
    plt.ylabel("Delta (log scale)", fontsize=14)
    plt.title("Epsilon vs Delta for all portions and budgets", fontsize=16)
    plt.legend(title="Portion (split)", fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(f"{base_path}/additional_figures/epsilon_delta.png")

if __name__ == "__main__":
    for d, path, bud, por in zip(datasets, paths, budgets_all, portions_list_all): 
        plot_experiment(d, path, bud, por)