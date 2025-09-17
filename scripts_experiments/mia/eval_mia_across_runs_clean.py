# -*- coding: utf-8 -*-
"""
Refactored script for loading, processing, and visualizing results from
privacy-related machine learning experiments.

This script generates three key plots from the experimental output:
1. A grid of violin plots comparing MIA pAUC and advantage across different
   privacy budget allocations.
2. Epsilon-delta privacy curves for various parameter settings.
3. Privacy trade-off curves (Type I vs. Type II error) and the corresponding
   privacy advantage.
"""

import sys
sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from opacus_new.accountants import RDPAccountant
from tqdm import tqdm
import warnings
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import statsmodels.formula.api as smf


# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

def setup_matplotlib_style() -> None:
    """Sets global Matplotlib styling parameters for consistent plots."""
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

portion_list = [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.19999999999999996], # Keeping original float for perfect path matching
        ]
# Define experiments to run. This structure replaces the separate global lists.

EXPERIMENTS = [
    {
        "dataset": "adult",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/adult",
        "budgets": [1.0, 8.0],
        "portions_list": portion_list
    },
    {
        "dataset": "bloodmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/bloodmnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "breast_cancer",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/breast_cancer",
        "budgets": [5.0, 20.0],
        "portions_list": portion_list
    },
    {
        "dataset": "cifar10",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/cifar10",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "credit_card_default",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/credit_card_default",
        "budgets": [4.0, 20.0],
        "portions_list": portion_list
    },
    {
        "dataset": "dermamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/dermamnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "german_credit",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/german_credit",
        "budgets": [4.0, 16.0],
        "portions_list": portion_list
    },
    {
        "dataset": "mnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/mnist",
        "budgets": [4.0, 16.0],
        "portions_list": portion_list
    },
    {
        "dataset": "mnist_4",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/mnist_4",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "organcmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/organcmnist",
        "budgets": [4.0, 16.0],
        "portions_list": portion_list
    },
    {
        "dataset": "organsmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/organsmnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "pneumoniamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/pneumoniamnist",
        "budgets": [4.0, 16.0],
        "portions_list": portion_list
    },
    {
        "dataset": "uci_isolet",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/uci_isolet",
        "budgets": [1.0, 10.0],
        "portions_list": portion_list, # Keeping original float for perfect path matching
    },
    {
        "dataset": "uci_isolet",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/None/uci_isolet",
        "budgets": [1.0, 5.0],
        "portions_list": portion_list, # Keeping original float for perfect path matching
    },

    ## Clipping experiments
    {
        "dataset": "bloodmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/bloodmnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "bloodmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/bloodmnist",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "breast_cancer",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/breast_cancer",
        "budgets": [2.0, 10.0],
        "portions_list": portion_list
    },
    {
        "dataset": "breast_cancer",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/breast_cancer",
        "budgets": [3.0, 10.0],
        "portions_list": portion_list
    },
    {
        "dataset": "breastmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/breastmnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "breastmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/breastmnist",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
        {
        "dataset": "cifar10",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/cifar10",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "credit_card_default",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/credit_card_default",
        "budgets": [4.0, 20.0],
        "portions_list": portion_list
    },
    {
        "dataset": "dermamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/dermamnist",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "dermamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/dermamnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "german_credit",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/german_credit",
        "budgets": [4.0, 16.0],
        "portions_list": portion_list
    },
    {
        "dataset": "mnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/mnist",
        "budgets": [1.0, 8.0],
        "portions_list": portion_list
    },
    {
        "dataset": "mnist_4",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/mnist_4",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "organcmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/organcmnist",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "organcmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/organcmnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "organsmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/organsmnist",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "organsmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/organsmnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "pneumoniamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/pneumoniamnist",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "pneumoniamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/pneumoniamnist",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },
    {
        "dataset": "uci_isolet",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None_2/uci_isolet",
        "budgets": [1.0, 10.0],
        "portions_list": portion_list, # Keeping original float for perfect path matching
    },
    ###### resnet 9 #####
    {
        "dataset": "breastmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/breastmnist_500",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "dermamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/dermamnist_500",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
        {
        "dataset": "organcmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/organcmnist_500",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
        {
        "dataset": "organsmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/organsmnist_500",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
            {
        "dataset": "pneumoniamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/pneumoniamnist_500",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
        {
        "dataset": "breastmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/breastmnist_2000",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "dermamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/dermamnist_2000",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
        {
        "dataset": "organcmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/organcmnist_2000",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
        {
        "dataset": "organsmnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/organsmnist_2000",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },
    {
        "dataset": "pneumoniamnist",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling/resnet9/pneumoniamnist_2000",
        "budgets": [8.0, 32.0],
        "portions_list": portion_list
    },

    {
        "dataset": "credit_card_default",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None/credit_card_default",
        "budgets": [4.0, 20.0],
        "portions_list": portion_list
    },
    {
        "dataset": "german_credit",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None/german_credit",
        "budgets": [4.0, 16.0],
        "portions_list": portion_list
    },
    {
        "dataset": "mnist_4",
        "path": "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping/None/mnist_4",
        "budgets": [16.0, 50.0],
        "portions_list": portion_list
    },    

]

# Constants for file names and dictionary keys to avoid string literals
ARGS_FILE = "args.json"
NUM_STEPS_FILE = "num_steps_list.npy"
NOISE_MULT_FILE = "pg_noise_multiplier_list.npy"
SAMPLE_RATES_FILE = "pg_sample_rates_list.npy"
INTEGRALS_FILE = "integrals_all.npy"
ADVANTAGE_FILE = "adv.npy"
PRIVACY_SCORES_FILE = "priv_all.npy"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # convert array to list
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        return obj
    
def convert_string(s: str) -> list[float]:
    parts = s.split('_')
    return [int(p)/10 for p in parts]


def format_list_for_path(lst: List[float]) -> str:
    """Converts a list of floats to a file-system-safe string."""
    return "_".join(str(x).replace('.', '').replace('-', 'm') for x in lst)

def generate_latex_label(portions: List[float], bold_index: int) -> str:
    """
    rf"$\mathbf{{{math.ceil(p[0]*100)}\%}}$/{math.ceil(p[1]*100)}%"
    Creates a LaTeX formatted label for plots, highlighting one portion.
    Example: [0.2, 0.8] with bold_index=0 -> "$\\mathbf{20\\%}$/80%"
    """
    if bold_index == 0:
        labels = rf"$\mathbf{{{math.ceil(portions[0]*100)}\%}}$/{math.ceil(portions[1]*100)}%"
    else:
        labels = rf"{math.ceil(portions[0]*100)}%/$\mathbf{{{math.ceil(portions[1]*100)}\%}}$"
    return labels

def lighten_color(color, amount=0.5):
    """Lightens a color by a given amount."""
    try:
        c = mcolors.to_rgb(color)
        return tuple(1 - (1 - x) * amount for x in c)
    except ValueError:
        return color

# =============================================================================
# CORE VISUALIZATION CLASS
# =============================================================================

class ExperimentVisualizer:
    """
    Handles data loading, processing, and plotting for an experiment.
    """
    def __init__(self, dataset: str, base_path: str, budgets: List[float], portions_list: List[List[float]]):
        self.dataset = dataset
        self.base_path = Path(base_path)
        self.budgets = budgets
        self.portions_list = portions_list
        self.output_dir = self.base_path / f"additional_figures_{format_list_for_path(budgets)}"
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.aggregated_integrals: Dict[str, Dict[str, Any]] = {}
        self.aggregated_adv: Dict[str, Dict[str, Any]] = {}
        self.aggregated_priv: Dict[str, Dict[str, Any]] = {}
        self.portion_keys = [format_list_for_path(p) for p in self.portions_list]
        self.budget_keys = [str(b) for b in self.budgets]
        
        cmaps = [plt.get_cmap('Oranges'), plt.get_cmap('Blues')]
        self.colors = [[cmap(x) for x in np.linspace(0, 1, len(self.portion_keys) + 3)] for cmap in cmaps]
        # Color scheme for plots
        # base_color = mcolors.to_rgb("tab:blue")
        # num_portions = len(self.portion_keys)
        # shades = [lighten_color(base_color, 1 - 0.7 * (i / max(num_portions - 1, 1))) for i in range(num_portions)]
        # self.portion_colors = {key: shade for key, shade in zip(self.portion_keys, shades)}

        self._load_and_process_data()

    def _load_and_process_data(self):
        """
        Discovers experiment directories, loads raw data, aggregates results
        across multiple runs, and validates metadata consistency.
        """
        raw_data = {key: [] for key in self.portion_keys}
        for portions in self.portions_list:
            portion_key = format_list_for_path(portions)
            exp_dir_pattern = f"{self.dataset}_[{format_list_for_path(self.budgets)}]_[{portion_key}]"
            path = self.base_path / exp_dir_pattern
            
            if not path.is_dir(): 
                continue
            for run_dir in path.iterdir():
                if not run_dir.is_dir(): 
                    continue
                results_path = run_dir / "results"
                try:
                    with open(results_path / ARGS_FILE, "r") as f: 
                        args = json.load(f)
                    run_info = {
                        k: np.load(results_path / v, allow_pickle=True).tolist()
                        for k, v in {
                            "num_steps_list": NUM_STEPS_FILE,
                            "pg_noise_multiplier_list": NOISE_MULT_FILE,
                            "pg_sample_rates_list": SAMPLE_RATES_FILE,
                        }.items()
                    }
                    run_info.update({
                        k: np.load(results_path / v, allow_pickle=True).item()
                        for k, v in {
                            "integrals_all": INTEGRALS_FILE,
                            "adv": ADVANTAGE_FILE,
                            "priv_all": PRIVACY_SCORES_FILE
                        }.items()
                    })

                    run_info.update(args)
                    run_info["portions"] = portions
                    raw_data[portion_key].append(run_info)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Skipping malformed run at {results_path}: {e}")
                    continue
        self._aggregate_runs(raw_data)
        self._consolidate_metadata(raw_data)
    
    def _aggregate_runs(self, raw_data: Dict[str, List[Dict]]):
        """Averages metrics across all runs for each portion."""
        for key, runs in raw_data.items():
            if not runs: 
                continue
            for metric in ["integrals_all", "adv", "priv_all"]:
                all_metrics = [run[metric] for run in runs]
                if all_metrics:
                    metric_keys = all_metrics[0].keys()
                    aggregated = {
                        k: np.mean([np.array(d[k]) for d in all_metrics], axis=0).tolist()
                        for k in metric_keys
                    }
                    # aggregated = {
                    #     k: [item for d in all_metrics for item in d[k]]  # concatenate all lists for key k
                    #     for k in metric_keys
                    # }    
                    setattr(self, f"aggregated_{metric.replace('_all', '')}", {**getattr(self, f"aggregated_{metric.replace('_all', '')}"), key: aggregated})
    
    def _consolidate_metadata(self, raw_data: Dict[str, List[Dict]]):
        """Extracts and validates metadata from runs."""
        for portion_key, runs in raw_data.items():
            if not runs: 
                continue
            first_run = runs[0]
            self.metadata[portion_key] = {
                k: first_run.get(k) for k in [
                    "budgets", "portions", "target_delta", "num_steps_list",
                    "pg_noise_multiplier_list", "pg_sample_rates_list"
                ]
            }
            # Sanity check for consistency
            for run in runs[1:]:
                for key, ref_value in self.metadata[portion_key].items():
                    run_value = run.get(key)

                    # Case 1: both values are numeric -> compare with tolerance
                    if isinstance(ref_value, (int, float)) and isinstance(run_value, (int, float)):
                        if not np.isclose(run_value, ref_value, rtol=0.02, atol=0):  # 2% relative tolerance
                            raise ValueError(
                                f"Inconsistent '{key}' in '{portion_key}': "
                                f"{run_value} vs {ref_value} (tolerance 2%)"
                            )
                    # Case 2: list of numbers
                    elif isinstance(ref_value, list) and isinstance(run_value, list):
                        if len(ref_value) != len(run_value):
                            raise ValueError(
                                f"Inconsistent '{key}' in '{portion_key}': lists have different lengths "
                                f"({len(run_value)} vs {len(ref_value)})"
                            )
                        # if np.all(np.array(run_value) == None):
                        #     try:
                        #         if not np.allclose(run_value, ref_value, rtol=0.02, atol=0):
                        #             raise ValueError(
                        #                 f"Inconsistent '{key}' in '{portion_key}': "
                        #                 f"{run_value} vs {ref_value} (list, tolerance 2%)"
                        #         )
                        #     except:
                        #         print("something went wrong")
                    # Case 3: everything else -> compare strictly
                    else:
                        if run_value != ref_value:
                            raise ValueError(f"Inconsistent '{key}' in '{portion_key}'.")
    
    def _remove_outliers_as_does_boxplot(self, data):
        """Remove outliers based on the standard boxplot rule (1.5 * IQR)."""
        if len(data) == 0:
            return data
        arr = np.array(data)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        return arr[(arr >= lower_fence) & (arr <= upper_fence)].tolist()

    def statistical_tests(self, data_list, rem_outliers):
        if rem_outliers:
            data_list = [self._remove_outliers_as_does_boxplot(data) for data in data_list]
        results = {}
        results["unpaired_ttests"] = self.unpaired_ttests(data_list)
        # results["repeated_measures_mixed_model"] = self.repeated_measures_mixed_model(data_list)
        results["jonckheere_terpstra"] = self.jonckheere_terpstra(data_list)
        results["jonckheere_terpstra_top_10"] = self.jonckheere_terpstra([sorted(inner, reverse=True)[:int(len(inner)/10)] for inner in data_list])
        return results

    def run_nested_tests(self, nested_data, name, rem_outliers):
        inner_keys = list(next(iter(nested_data.values())).keys())
        list_from_dict = [[outer[k] for outer in nested_data.values()] for k in inner_keys]
        raw_results = {inner: self.statistical_tests(data, rem_outliers) for inner, data in zip(inner_keys, list_from_dict)}
        print(f"Statistical tests for {name} (outliers removed: {rem_outliers}):")
        print(f"{raw_results[inner_keys[0]]['jonckheere_terpstra']}")
        print(f"{raw_results[inner_keys[1]]['jonckheere_terpstra']}")
        print(f"{raw_results[inner_keys[0]]['jonckheere_terpstra_top_10']}")
        print(f"{raw_results[inner_keys[1]]['jonckheere_terpstra_top_10']}")
        with open(self.output_dir / f"statistical_tests_{name}_{rem_outliers}_{self.dataset}.json", "w") as f:
            json.dump(make_serializable(raw_results), f, indent=4)
   

    def unpaired_ttests(self, data_list, alpha=0.05):
        pvals, comparisons = [], []
        for i in range(len(data_list) - 1):
            _, p = stats.ttest_ind(data_list[i], data_list[i+1])
            pvals.append(p)
            comparisons.append(f"Group{i+1} vs Group{i+2}")
        bonf = multipletests(pvals, alpha=alpha, method='bonferroni')
        fdr = multipletests(pvals, alpha=alpha, method='fdr_bh')
        return {
            "comparisons": comparisons,
            "raw_pvals": pvals,
            "bonf_significant": bonf[0],
            "bonf_pvals": bonf[1],
            "fdr_significant": fdr[0],
            "fdr_pvals": fdr[1]
        }

    def repeated_measures_mixed_model(self, data_list):
        # Build DataFrame in long format
        df = pd.DataFrame({
            "value": np.concatenate(data_list),
            "time": np.concatenate([[t+1]*len(data_list[t]) for t in range(len(data_list))]),
            "subject": np.concatenate([np.arange(len(data_list[t])) for t in range(len(data_list))])
        })
        # Fit mixed-effects model
        model = smf.mixedlm("value ~ time", df, groups=df["subject"])
        result = model.fit()
        
        # Compute linear trend correlation
        means = [np.mean(g) for g in data_list]
        r, p = stats.pearsonr(range(1, len(data_list)+1), means)
        
        return {"mixed_model_summary": result.summary(), "trend_correlation": r, "trend_pval": p}

    def jonckheere_terpstra(self, data_list, alternative="increasing"):
        groups = [np.asarray(g) for g in data_list]
        n_tot = sum(len(g) for g in groups)
        JT = 0
        groups_np = [np.array(g) for g in groups]
        for i, j in list(combinations(range(len(groups_np)), 2)):
            x = groups_np[i][:, None]  # shape (len(groups[i]), 1)
            y = groups_np[j][None, :]  # shape (1, len(groups[j]))
            
            # Compare all pairs at once
            JT += np.sum(x < y) + 0.5 * np.sum(x == y)
                
        n_i = np.array([len(g) for g in groups])
        E_JT = 0.25 * (n_tot**2 - np.sum(n_i**2))
        Var_JT = (1/72) * (n_tot**2*(2*n_tot+3) - np.sum(n_i**2*(2*n_i+3)))
        
        z = (JT - E_JT) / np.sqrt(Var_JT)
        
        if alternative == "increasing":
            pval = 1 - stats.norm.cdf(z)
        elif alternative == "decreasing":
            pval = stats.norm.cdf(z)
        else:
            pval = 2*(1 - stats.norm.cdf(abs(z)))
        
        return {"JT": JT, "z": z, "pval": pval}

    def plot_single_boxplots(self, data_dict, name):
        """
        Creates a grid of boxplots for pAUC and MIA advantage.
        """

        base_axes_height = 0.7  # height of the actual plot region (inches)
        extra_label_height = 0.2  # extra space for x-axis labels (inches)

        for c, bud_key in enumerate(self.budget_keys):
            # Collect data + labels
            data = [np.array(data_dict[p_key][bud_key]) for p_key in self.portion_keys]
            labels = [generate_latex_label(self.metadata[k]["portions"], c) for k in self.metadata.keys()]

            for with_labels in [True, False]:
                # Figure height = axes height + possible label space
                fig_height = base_axes_height + (extra_label_height if with_labels else 0)

                fig = plt.figure(figsize=(2.5, fig_height))

                # Fixed axes rectangle: [left, bottom, width, height]
                axes_height_rel = base_axes_height / fig_height
                ax = fig.add_axes([0.2, 1 - axes_height_rel - 0.1, 0.75, axes_height_rel])

                # Draw boxplot
                self._draw_box_subplot(ax, data, labels, self.colors[c], is_bottom=with_labels)

                if not with_labels:
                    # Keep ticks but hide ticklabels
                    ax.set_xticklabels([])

                # Save
                suffix = "with_labels" if with_labels else "no_labels"
                for ext in ["png", "svg"]:
                    fig.savefig(
                        self.output_dir / f"boxplot_single_{name}_{self.dataset}_{c}_{suffix}.{ext}",
                        dpi=300, bbox_inches="tight"
                    )
                plt.close(fig)

        print(f"✅ MIA analysis boxplots saved to {self.output_dir}")


    def _draw_box_subplot(self, ax: plt.Axes, data: List, labels: List[str], palette: List, is_bottom: bool):
        """Helper to draw and style a single subplot of boxplots."""
        positions = np.arange(len(data)) + 1  # boxplot expects positions starting at 1

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,  # allows facecolor
            widths=0.6,
            medianprops=dict(linewidth=0.8, color="black"),
            whiskerprops=dict(linewidth=0.7, color="black"),
            capprops=dict(linewidth=0.7, color="black"),
            showfliers=False,
        )

        for i, (box, color) in enumerate(zip(bp["boxes"], palette)):
            box.set(facecolor=palette[i + 1], edgecolor=palette[i + 3], linewidth=0.7, alpha=0.7)

        for whisker in bp["whiskers"]:
            whisker.set(linewidth=0.7, color=palette[i + 3])

        for cap in bp["caps"]:
            cap.set(linewidth=0.7, color=palette[i + 3])

        for median in bp["medians"]:
            median.set(linewidth=0.8, color=palette[i + 3])

        ax.set_xticks(positions if is_bottom else [])
        if is_bottom:
            ax.set_xticklabels(labels, rotation=45, ha="right")


    def plot_single_violings(self, data_dict, name):
        """
        Creates a grid of violin plots for pAUC and MIA advantage.
        """

        base_axes_height = 0.7  # height of the actual plot region (inches)
        extra_label_height = 0.2  # extra space for x-axis labels (inches)

        for c, bud_key in enumerate(self.budget_keys):
            # Collect data + labels
            data = [np.array(data_dict[p_key][bud_key]) for p_key in self.portion_keys]
            labels = [generate_latex_label(self.metadata[k]["portions"], c) for k in self.metadata.keys()]

            for with_labels in [True, False]:
                # Figure height = axes height + possible label space
                fig_height = base_axes_height + (extra_label_height if with_labels else 0)

                fig = plt.figure(figsize=(2.5, fig_height))

                # Fixed axes rectangle: [left, bottom, width, height]
                # width=0.8, height in relative units (normalized by fig size)
                axes_height_rel = base_axes_height / fig_height
                ax = fig.add_axes([0.2, 1 - axes_height_rel - 0.1, 0.75, axes_height_rel])

                # Draw violin plot
                self._draw_violin_subplot(ax, data, labels, self.colors[c], is_bottom=with_labels)

                # # Y-axis label only for the first subplot
                # if c == 0:
                #     ax.set_ylabel("MIA advantage")

                if not with_labels:
                    # Keep ticks but hide ticklabels
                    ax.set_xticklabels([])

                # Save
                suffix = "with_labels" if with_labels else "no_labels"
                for ext in ["png", "svg"]:
                    fig.savefig(
                        self.output_dir / f"violin_plot_single_{name}_{self.dataset}_{c}_{suffix}.{ext}",
                        dpi=300, bbox_inches="tight"
                    )
                plt.close(fig)

        print(f"✅ MIA analysis singles saved to {self.output_dir}")



    def plot_mia_analysis_grid(self):
        """
        Creates a grid of violin plots for pAUC and MIA advantage.
        """
        integral_slices = [0.01, 0.05, 0.1]
        n_rows, n_cols = len(integral_slices) + 1, len(self.budget_keys)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 1.4 * n_rows))
        if n_rows == 1: 
            axes = axes.reshape(1, -1)

        for r, sl in enumerate(integral_slices):
            for c, bud_key in enumerate(self.budget_keys):
                data = [np.array(self.aggregated_integrals[p_key][bud_key])[:, r] for p_key in self.portion_keys]
                labels = [generate_latex_label(self.metadata[k]["portions"], c) for k in self.metadata.keys()]
                self._draw_violin_subplot(axes[r, c], data, labels, self.colors[c], is_bottom=False)
                if c == 0: 
                    axes[r, c].set_ylabel(f"pAUC of MIA\nat FPR={sl}")
                if r == 0: 
                    axes[r, c].set_title(rf"$\varepsilon_{c}$ = {bud_key}")

        for c, bud_key in enumerate(self.budget_keys):
            ax = axes[-1, c]
            data = [np.array(self.aggregated_adv[p_key][bud_key]) for p_key in self.portion_keys]
            labels = [generate_latex_label(self.metadata[k]["portions"], c) for k in self.metadata.keys()]
            self._draw_violin_subplot(ax, data, labels, self.colors[c], is_bottom=True)
            if c == 0: 
                ax.set_ylabel("MIA advantage")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        for ext in ["png", "svg"]:
            fig.savefig(self.output_dir / f"violin_plot_grid.{ext}", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ MIA analysis grid saved to {self.output_dir}")

    def _draw_violin_subplot(self, ax: plt.Axes, data: List, labels: List[str], palette: List, is_bottom: bool):
        """Helper to draw and style a single subplot of violin plots."""
        positions = np.arange(len(data))
        for i, (d, color) in enumerate(zip(data, palette)):
            vp = ax.violinplot([d], positions=[i], showmeans=False, showmedians=True, showextrema=True, widths=0.9)
            linewidth = 0.7
            face_color = palette[i + 1]
            edge_color = palette[i + 3]
            for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'): 
                vp[part].set_edgecolor(edge_color)
                vp[part].set_linewidth(linewidth)
            vp['bodies'][0].set_facecolor(face_color)
            vp['bodies'][0].set_edgecolor(edge_color)
            vp['bodies'][0].set_linewidth(linewidth)
            vp['bodies'][0].set_alpha(0.7)
        ax.set_xticks(positions if is_bottom else [])
        if is_bottom: 
            ax.set_xticklabels(labels, rotation=45, ha="right")
        # if not ax.get_subplotspec().is_first_col(): 
        #     ax.set_yticklabels([])
        
    def plot_epsilon_delta_curves(self):
        """
        Creates a plot showing the Epsilon vs. Delta privacy curve for each
        experimental portion, colored consistently.
        """
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        
        # Store handles and labels as flat lists
        handles = []
        labels = []
        linewidth = 1.2
        sr = []
        nm = []
        num_steps = []
        for p_idx, (p_key, p_info) in enumerate(self.metadata.items()):
            sr.append(p_info["pg_sample_rates_list"][0])
            nm.append(p_info["pg_noise_multiplier_list"][0])
            num_steps.append(p_info["num_steps_list"][0])
        for i, budget in enumerate(self.budget_keys):
            for p_idx, (p_key, p_info) in enumerate(self.metadata.items()):
                epsilons, deltas = self._compute_epsilon_delta_curve(
                    noise_multiplier=p_info["pg_noise_multiplier_list"][0][i],
                    deltas=np.logspace(-14, -1, 50),
                    iterations=p_info["num_steps_list"][0],
                    sampling_rate=p_info["pg_sample_rates_list"][0][i],
                    clipping_norm=1
                )
                label = generate_latex_label(p_info["portions"], i)
                plot_handle, = ax.plot(epsilons, deltas, color=self.colors[i][p_idx + 3], alpha=0.9, linewidth=linewidth, label=label)
                
                # Append handles and labels in the desired order
                handles.append(plot_handle)
                labels.append(label)


        target_delta = self.metadata[self.portion_keys[0]]["target_delta"]
        target_epsilons = self.metadata[self.portion_keys[0]]["budgets"]
        rest_handles = []
        rest_labels = []

        if target_delta:
            plot_handle = ax.axhline(target_delta, color='red', linestyle='--', linewidth=linewidth, label='Target $\delta$')
            rest_handles.append(plot_handle)
            rest_labels.append('Target $\delta$')
        
        if target_epsilons:
            for i, eps in enumerate(target_epsilons):
                plot_handle = ax.axvline(eps, color='red', linestyle=':', linewidth=linewidth, label='Target $\epsilon$' if i == 0 else None)
                if i == 0:
                    rest_handles.append(plot_handle)
                    rest_labels.append('Target $\epsilon$')
        
        # Correctly order and flatten the lists for the two-column legend
        handles_all = handles[:int(len(handles)/2)] + rest_handles + handles[int(len(handles)/2):]
        labels_all = labels[:int(len(handles)/2)] + rest_labels + labels[int(len(handles)/2):]
        
        # Add dummy entries to the right column to align the last items correctly
        for _ in rest_handles:
            handles_all.append(mpl.lines.Line2D([0], [0], color='none'))
            labels_all.append("")
        
        ax.set_yscale("log")
        ax.set_xlabel("Epsilon ($\epsilon$)")
        ax.set_ylabel("Delta ($\delta$, log scale)")
        
        # Pass the correctly ordered flat lists to the legend
        ax.legend(handles=handles_all, labels=labels_all, ncol=2, loc="upper right")
        plt.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(self.output_dir / f"epsilon_delta_curve.{ext}", dpi=300, bbox_inches='tight')
        print(f"✅ Epsilon-Delta curves saved to {self.output_dir}")

    def _compute_epsilon_delta_curve(self, noise_multiplier, deltas, iterations, sampling_rate, clipping_norm):
        """
        Compute epsilon for a range of deltas using the RDP accountant.
        """
        def compute_epsilon(noise_multiplier, delta, iterations, sampling_rate, clipping_norm):
            accountant = RDPAccountant()
            if iterations[0] is None:
                iterations[0] = 100
            if "exp_mia_final_clipping" in str(self.base_path):
                # Custom logic if needed for clipping experiments
                pass
            accountant.history = [(noise_multiplier, sampling_rate, int(iterations[0]))]
            return accountant.get_epsilon(delta)
        epsilons = []
        deltas2 = []
        for delta in deltas:
            epsilon = compute_epsilon(noise_multiplier, delta, iterations, sampling_rate, clipping_norm)
            epsilons.append(epsilon)
            deltas2.append(delta)
        return epsilons, deltas2

    def plot_privacy_tradeoff(self):
        """
        Creates a grid showing the Type I vs. Type II error trade-off and
        the resulting privacy advantage for each budget.
        """
        n_budgets = len(self.budget_keys)
        fig, axes = plt.subplots(n_budgets, 2, figsize=(6, 4 * n_budgets), squeeze=False)
        alpha_grid = np.linspace(0, 1, 200)

        for r, budget_key in enumerate(self.budget_keys):
            ax_curve, ax_bar = axes[r, 0], axes[r, 1]
            advantages = {}
            
            for p_idx, (p_key, p_info) in enumerate(self.metadata.items()):
                params = {
                    "noise_multiplier": p_info["pg_noise_multiplier_list"][0][r],
                    "steps": p_info["num_steps_list"][0],
                    "sample_rate": p_info["pg_sample_rates_list"][0][r],
                }
                envelope = self._compute_tradeoff_envelope(params, alpha_grid)
                
                # Plotting
                label = generate_latex_label(p_info["portions"], 1 if r == 0 else 0)
                ax_curve.plot(alpha_grid, envelope, color=self.colors[r][p_idx + 3], linewidth=2, label=label)
                
                # Advantage calculation
                trivial = 1 - alpha_grid
                diff = trivial - envelope
                max_idx = np.argmax(np.nan_to_num(diff))
                advantages[p_key] = diff[max_idx]
                ax_curve.plot([alpha_grid[max_idx], alpha_grid[max_idx]], [envelope[max_idx], trivial[max_idx]],
                              color=self.colors[r][p_idx + 3], linestyle=":", linewidth=1)

            # Format curve plot
            ax_curve.plot([0, 1], [1, 0], 'k:', label="_nolegend_")
            ax_curve.set_title(f"Budget ($\epsilon$) = {budget_key}")
            ax_curve.set_xlabel("Type I error $\\alpha$")
            ax_curve.set_ylabel("Type II error $\\beta$")
            ax_curve.grid(True)
            ax_curve.legend()

            # Format bar plot
            x_labels = [generate_latex_label(self.metadata[k]["portions"], 1 if r == 0 else 0) for k in advantages.keys()]
            y_values = list(advantages.values())
            colors = self.colors[r][3:]
            ax_bar.bar(range(len(x_labels)), y_values, color=colors, width=0.5)
            ax_bar.set_ylabel("Advantage")
            ax_bar.set_xticks(range(len(x_labels)))
            ax_bar.set_xticklabels(x_labels, rotation=90)
            margin = 0.1 * (max(y_values) - min(y_values)) if len(y_values) > 1 and max(y_values) != min(y_values) else 0.1
            ax_bar.set_ylim(min(y_values) - margin, max(y_values) + margin)

        plt.tight_layout()
        for ext in ["png", "svg"]: 
            fig.savefig(self.output_dir / f"trade_off_curves.{ext}", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Privacy trade-off plots saved to {self.output_dir}")

    @staticmethod
    def _f_eps_delta(alpha: np.ndarray, epsilon: float, delta: float) -> np.ndarray:
        """Piecewise linear trade-off function f_{ε, δ}."""
        return np.maximum(0, np.maximum(1 - delta - np.exp(epsilon) * alpha, np.exp(-epsilon) * (1 - delta - alpha)))

    def _compute_tradeoff_envelope(self, params: Dict, alpha_grid: np.ndarray) -> np.ndarray:
        """Computes the envelope of all trade-off curves for given params."""
        accountant = RDPAccountant()
        accountant.history = [(params["noise_multiplier"], params["sample_rate"], params["steps"])]
        deltas = np.linspace(0, 1, 2000)
        
        f_values = [self._f_eps_delta(alpha_grid, accountant.get_epsilon(d), d) for d in deltas]
        return np.max(np.array(f_values), axis=0)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # If a custom path is needed for modules like opacus_new
    # sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")
    
    setup_matplotlib_style()
    
    for config in tqdm(EXPERIMENTS):


        print(f"--- Running analysis for dataset: {config['dataset']} ---")
        try:
            visualizer = ExperimentVisualizer(
                dataset=config["dataset"],
                base_path=config["path"],
                budgets=config["budgets"],
                portions_list=config["portions_list"],
            )
            # visualizer.run_nested_tests(visualizer.aggregated_priv, "priv", rem_outliers=True)
            # visualizer.run_nested_tests(visualizer.aggregated_priv, "priv", rem_outliers=False)
            # visualizer.run_nested_tests(visualizer.aggregated_adv, "adv", rem_outliers=True)
            # visualizer.run_nested_tests(visualizer.aggregated_adv, "adv", rem_outliers=False)
            # visualizer.plot_single_violings(visualizer.aggregated_adv, "adv")
            # visualizer.plot_single_violings(visualizer.aggregated_priv, "priv")
            # visualizer.plot_single_boxplots(visualizer.aggregated_adv, "adv")
            # visualizer.plot_single_boxplots(visualizer.aggregated_priv, "priv")
            # visualizer.plot_mia_analysis_grid()
            visualizer.plot_epsilon_delta_curves()
            # visualizer.plot_privacy_tradeoff()
            
        except Exception as e:
            print(f"💥 Error processing experiment for {config['dataset']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Analysis complete. ---")