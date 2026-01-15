"""Membership Inference Attack (MIA) utilities for differential privacy research.

This module provides functions for:
- Computing and scoring membership inference attacks
- Fitting attack models (online and offline variants)
- Computing privacy metrics using ROC curves and AUC
- Privacy accounting via RDP and privacy budgets
- Visualization of attack results and privacy metrics
"""

import functools
import os
import time
from typing import List, Tuple, Dict, Any, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Subset

from opacus_new.accountants import RDPAccountant

from utils.utils_general import load_dataset


# ============================================================================
# Scoring and Basic Utilities
# ============================================================================


def softmax(preds: np.ndarray) -> np.ndarray:
    """Apply softmax transformation to logits.
    
    Numerically stable softmax computation using log-sum-exp trick.
    
    Args:
        preds: Logits array of shape (..., num_classes).
        
    Returns:
        Softmax probabilities with same shape as input.
    """
    predictions = preds - np.max(preds, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
    return predictions


def score_mia_one(input_tuple: Tuple[str, str, Optional[int]]) -> None:
    """Compute MIA scores for a single model's logits.
    
    Converts model logits to membership scores using the standard
    MIA approach: log likelihood of true class minus log likelihood of alternatives.
    
    Args:
        input_tuple: Tuple of (path, dataset_name, num_max_per_class_samples)
                    where path contains a 'logits.npy' file.
                    
    Returns:
        None. Scores are saved to path/scores.npy.
    """
    path, dataset_name, num_max_per_class_samples = input_tuple
    file_path = os.path.join(path, "scores.npy")
    
    if os.path.exists(file_path):
        return  # Already computed
    
    # Load labels
    dataset = load_dataset(dataset_name, train=True, num_max_samples=num_max_per_class_samples)
    if isinstance(dataset, Subset):
        labels = dataset.dataset.tensors[1].numpy()
    else:
        labels = dataset.tensors[1].numpy()
    
    # Load logits and compute probabilities
    logits = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]
    predictions = softmax(logits)
    
    assert predictions.shape[0] == labels.shape[0], "Mismatch between predictions and labels"
    
    # Compute MIA scores
    count = predictions.shape[0]
    y_true = predictions[np.arange(count), :, labels]
    predictions[np.arange(count), :, labels] = 0
    y_wrong = predictions.sum(axis=-1)
    
    # Log odds: ln(P(y|x)) - ln(P(wrong|x))
    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(file_path, logit)


def score_mia(params: Any, savedir: str) -> None:
    """Score all model logits in a directory for MIA.
    
    Args:
        params: Configuration with 'dataset' and 'num_max_per_class_samples' attributes.
        savedir: Directory containing model subdirectories with logits.
    """
    dataset = params.dataset if hasattr(params, "dataset") else params["dataset"]
    num_max_per_class_samples = getattr(params, "num_max_per_class_samples", None)
    
    for x in os.listdir(savedir):
        score_mia_one((os.path.join(savedir, x), dataset, num_max_per_class_samples))



# ============================================================================
# ROC Analysis and Attack Evaluation
# ============================================================================


def sweep(score: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute ROC curve metrics from scores and labels.
    
    Args:
        score: Attack scores for each sample.
        x: Binary ground truth (membership labels).
        
    Returns:
        Tuple of (fpr, tpr, auc_score, accuracy) where accuracy is
        the best balanced accuracy across all score thresholds.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)  # Balanced accuracy
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score, acc


def load_data(savedir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load membership labels and attack scores from saved files.
    
    Loads 'keep.npy' (membership labels) and 'scores.npy' (attack scores)
    from all subdirectories in savedir.
    
    Args:
        savedir: Directory containing model subdirectories.
        
    Returns:
        Tuple of (keep, scores) where keep indicates training membership
        and scores are the attack scores from each model.
    """
    scores = []
    keep = []

    for path in os.listdir(savedir):
        scores.append(np.load(os.path.join(savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(savedir, path, "keep.npy")))
    
    scores = np.array(scores)
    keep = np.array(keep)
    return keep, scores


def load_data_adv(savedir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load membership labels and adversarial attack scores.
    
    Variant of load_data for adversarial evaluation settings.
    
    Args:
        savedir: Directory containing model subdirectories.
        
    Returns:
        Tuple of (keep, scores) as in load_data.
    """
    scores = []
    keep = []

    for path in os.listdir(savedir):
        scores.append(np.load(os.path.join(savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(savedir, path, "keep.npy")))
    
    scores = np.array(scores)
    keep = np.array(keep)
    return keep, scores



def generate_ours(
    keep: np.ndarray,
    scores: np.ndarray,
    check_keep: np.ndarray,
    check_scores: np.ndarray,
    in_size: int = 100000,
    out_size: int = 100000,
    fix_variance: bool = False,
) -> Tuple[List, List]:
    """Online attack fitting two Gaussian distributions for in/out.
    
    Fits independent Gaussian models for in-distribution and out-of-distribution
    scores using median and standard deviation, then scores test samples.
    
    Args:
        keep: Boolean array indicating training membership across models.
        scores: Attack scores of shape (n_models, n_samples, n_score_dims).
        check_keep: Test membership labels.
        check_scores: Test attack scores.
        in_size: Maximum samples for in-distribution model.
        out_size: Maximum samples for out-distribution model.
        fix_variance: If True, use global variance; else per-model variance.
        
    Returns:
        Tuple of (predictions, ground_truth_answers).
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
        pr_in = -stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(
    keep: np.ndarray,
    scores: np.ndarray,
    check_keep: np.ndarray,
    check_scores: np.ndarray,
    in_size: int = 100000,
    out_size: int = 100000,
    fix_variance: bool = False,
) -> Tuple[List, List]:
    """Offline attack fitting single Gaussian for out-of-distribution only.
    
    Simpler variant that only models the out-of-distribution scores,
    computing likelihood under that distribution.
    
    Args:
        keep: Boolean array indicating training membership across models.
        scores: Attack scores of shape (n_models, n_samples, n_score_dims).
        check_keep: Test membership labels.
        check_scores: Test attack scores.
        in_size: Unused (for API compatibility).
        out_size: Maximum samples for out-distribution model.
        fix_variance: If True, use global variance; else per-model variance.
        
    Returns:
        Tuple of (predictions, ground_truth_answers).
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
        score = stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        prediction.extend(score.mean(1))
        answers.extend(ans)
    
    return prediction, answers


def generate_global(
    keep: np.ndarray,
    scores: np.ndarray,
    check_keep: np.ndarray,
    check_scores: np.ndarray,
) -> Tuple[List, List]:
    """Baseline attack using global threshold on raw scores.
    
    Simple attack that uses mean score directly without fitting distributions.
    
    Args:
        keep: Unused (for API compatibility).
        scores: Unused (for API compatibility).
        check_keep: Test membership labels.
        check_scores: Test attack scores.
        
    Returns:
        Tuple of (predictions, ground_truth_answers).
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)
    return prediction, answers




# ============================================================================
# Visualization and Reporting
# ============================================================================


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_val: float,
    acc: float,
    legend: str,
    metric: str,
    **plot_kwargs
) -> None:
    """Plot a single ROC curve with annotation.
    
    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_val: Area under the ROC curve.
        acc: Best accuracy achieved.
        legend: Legend label prefix.
        metric: Metric to display ("auc", "acc", or "").
        **plot_kwargs: Additional plotting arguments.
    """
    if metric == "auc":
        metric_text = f"auc={auc_val:.3f}"
    elif metric == "acc":
        metric_text = f"acc={acc:.3f}"
    else:
        metric_text = ""
    plt.plot(fpr, tpr, label=f"{legend}{metric_text}", **plot_kwargs)


def run_attack_and_get_roc(
    fn: Callable,
    keep: np.ndarray,
    scores: np.ndarray,
    ntest: int,
    sweep_fn: Callable,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Execute attack and compute ROC metrics.
    
    Args:
        fn: Attack function taking (keep_train, scores_train, keep_test, scores_test).
        keep: Membership labels for all models/samples.
        scores: Attack scores for all models/samples.
        ntest: Number of test samples.
        sweep_fn: Function to compute ROC curve from predictions and labels.
        
    Returns:
        Tuple of (fpr, tpr, auc, accuracy, tpr_at_low_fpr).
    """
    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep[-ntest:], scores[-ntest:])
    fpr, tpr, auc_val, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    
    # Find TPR at FPR < 0.001
    try:
        low = tpr[np.where(fpr < 0.001)[0][-1]]
    except IndexError:
        low = 0.0
    
    return fpr, tpr, auc_val, acc, low


def fig_fpr_tpr(args: Any, keep: np.ndarray, scores: np.ndarray, savedir: str) -> None:
    """Plot and save ROC curves for multiple attack variants.
    
    Args:
        args: Configuration namespace (unused).
        keep: Membership labels.
        scores: Attack scores.
        savedir: Directory to save figure.
    """
    os.makedirs(savedir, exist_ok=True)
    plt.figure(figsize=(4, 3))

    attacks = [
        (generate_ours, "Ours (online)\n", {}),
        (functools.partial(generate_ours, fix_variance=True), "Ours (online, fixed variance)\n", {}),
        (functools.partial(generate_ours_offline), "Ours (offline)\n", {}),
        (functools.partial(generate_ours_offline, fix_variance=True), "Ours (offline, fixed variance)\n", {}),
        (generate_global, "Global threshold\n", {}),
    ]

    for fn, legend, kwargs in attacks:
        fpr, tpr, auc_val, acc, low = run_attack_and_get_roc(fn, keep, scores, 1, sweep)
        print(f"Attack {legend}   AUC {auc_val:.4f}, Accuracy {acc:.4f}, TPR@0.1%FPR {low:.4f}")
        plot_roc_curve(fpr, tpr, auc_val, acc, legend, "auc", **kwargs)

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
    plt.close()


def run_attack_and_get_roc_target(
    fn: Callable,
    keep: np.ndarray,
    scores: np.ndarray,
    keep_target: np.ndarray,
    scores_target: np.ndarray,
    ntest: int,
    sweep_fn: Callable,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Execute target attack and compute ROC metrics.
    
    Like run_attack_and_get_roc but tests on a different target distribution.
    
    Args:
        fn: Attack function.
        keep: Source membership labels.
        scores: Source attack scores.
        keep_target: Target membership labels.
        scores_target: Target attack scores.
        ntest: Number of test samples.
        sweep_fn: ROC computation function.
        
    Returns:
        Tuple of (fpr, tpr, auc, accuracy, tpr_at_low_fpr).
    """
    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep_target[-ntest:], scores_target[-ntest:])
    fpr, tpr, auc_val, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    try:
        low = tpr[np.where(fpr < 0.001)[0][-1]]
    except IndexError:
        low = 0.0
    return fpr, tpr, auc_val, acc, low


def fig_fpr_tpr_target(
    args: Any,
    keep: np.ndarray,
    scores: np.ndarray,
    keep_target: np.ndarray,
    scores_target: np.ndarray,
    savedir: str,
    name: str = "fprtpr_target",
) -> None:
    """Plot and save ROC curves for attacks evaluated on target distribution.
    
    Args:
        args: Configuration namespace (unused).
        keep: Source membership labels.
        scores: Source attack scores.
        keep_target: Target membership labels.
        scores_target: Target attack scores.
        savedir: Directory to save figure.
        name: Base filename for the figure.
    """
    os.makedirs(savedir, exist_ok=True)
    plt.figure(figsize=(4, 3))

    attacks = [
        (generate_ours, "Ours (online)\n", {}),
        (functools.partial(generate_ours, fix_variance=True), "Ours (online, fixed variance)\n", {}),
        (functools.partial(generate_ours_offline), "Ours (offline)\n", {}),
        (functools.partial(generate_ours_offline, fix_variance=True), "Ours (offline, fixed variance)\n", {}),
        (generate_global, "Global threshold\n", {}),
    ]

    for fn, legend, kwargs in attacks:
        fpr, tpr, auc_val, acc, low = run_attack_and_get_roc_target(
            fn, keep, scores, keep_target, scores_target, 1, sweep
        )
        print(f"Attack {legend}   AUC {auc_val:.4f}, Accuracy {acc:.4f}, TPR@0.1%FPR {low:.4f}")
        plot_roc_curve(fpr, tpr, auc_val, acc, legend, "auc", **kwargs)

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
    plt.close()



# ============================================================================
# Privacy Analysis - Sample-wise Metrics
# ============================================================================


def fit_mia_in_out_gaussians(keep: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Gaussian distributions for in-distribution and out-distribution samples.
    
    Args:
        keep: Boolean membership labels of shape (n_models, n_samples).
        scores: Attack scores of shape (n_models, n_samples, n_score_dims).
        
    Returns:
        Tuple of (mean_in, mean_out, std_in, std_out) for each model.
    """
    dat_in = []
    dat_out = []
    in_size = 100000
    out_size = 100000
    
    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, 0])
        dat_out.append(scores[~keep[:, j], j, 0])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    std_in = np.std(dat_in, 1)
    std_out = np.std(dat_out, 1)
    
    return mean_in, mean_out, std_in, std_out


def compute_individual_scores(
    mean_in: np.ndarray,
    mean_out: np.ndarray,
    std_in: np.ndarray,
    std_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List, List]:
    """Compute per-model privacy metrics from fitted Gaussians.
    
    Computes AUC, TPR curves, and area-under-curve integrals at various FPR thresholds.
    
    Args:
        mean_in: Mean of in-distribution scores per model.
        mean_out: Mean of out-distribution scores per model.
        std_in: Standard deviation of in-distribution scores per model.
        std_out: Standard deviation of out-distribution scores per model.
        
    Returns:
        Tuple of (auc_scores, fpr, tpr_curves, integrals, advantages, priv_scores).
    """
    priv_scores = []
    auc_scores = []
    advs = []
    tprs = []
    fpr = np.linspace(1e-6, 1 - 1e-6, 10000)

    for j in range(mean_in.shape[0]):
        # Compute AUC using Gaussian parameters (Mos method)
        mu_x = mean_in[j]
        mu_xp = mean_out[j]
        sigma2_x = std_in[j] ** 2
        sigma2_xp = std_out[j] ** 2
        denom = np.sqrt(sigma2_x + sigma2_xp)
        auc_j = norm.cdf((mu_x - mu_xp) / (denom + 1e-30))

        # Compute TPR curve as function of FPR
        mu1 = mean_in[j]
        mu0 = mean_out[j]
        sigma1 = std_in[j]
        sigma0 = std_out[j]
        a = np.abs(mu1 - mu0) / (sigma1 + 1e-30)
        b = sigma0 / (sigma1 + 1e-30)
        R_x = norm.cdf(a + b * norm.ppf(fpr))

        priv_score = abs(mu_x - mu_xp) / (sigma2_x + sigma2_xp + 1e-30)

        priv_scores.append(priv_score)
        auc_scores.append(auc_j)
        tprs.append(np.array(R_x))
    
    # Compute integrals at various FPR thresholds
    high_alpha_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    integrals = []
    for R in tprs:
        integrals_row = []
        for high_alpha in high_alpha_list:
            mask = fpr <= high_alpha
            if np.any(mask):
                integral = np.trapz(R[mask], fpr[mask])
            else:
                integral = 0.0
            integrals_row.append(integral)
        integrals.append(integrals_row)
        adv = np.max(np.array(R) - np.array(fpr))
        advs.append(adv)

    auc_scores = np.array(auc_scores)
    return auc_scores, fpr, np.array(tprs), integrals, advs, priv_scores


def plot_and_save_samplewise_auc(samplewise_auc: Dict[str, np.ndarray], savedir_result: str) -> None:
    """Plot and save sample-wise AUC distributions as boxplot and histogram.
    
    Args:
        samplewise_auc: Dictionary mapping label -> array of AUC values.
        savedir_result: Directory to save figures.
    """
    os.makedirs(savedir_result, exist_ok=True)
    keys = list(samplewise_auc.keys())
    data = [np.array(samplewise_auc[key]) for key in keys]

    # Boxplot with max values highlighted
    plt.figure(figsize=(7, 4))
    box = plt.boxplot(
        data,
        labels=keys,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="darkblue"),
    )
    plt.xlabel("Label")
    plt.ylabel("AUC")
    plt.title("Sample-wise AUC Boxplot")
    
    # Annotate max values
    blues = sns.color_palette("Blues", len(keys) + 2)[2:]
    for i, values in enumerate(data):
        max_val = np.max(values)
        plt.scatter([i + 1], [max_val], color=blues[i], marker="o", s=120, edgecolor="black", zorder=5)
        plt.text(i + 1 + 0.05, max_val, f"{max_val:.4f}", va="center", ha="left", fontsize=10, color=blues[i], fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir_result, "samplewise_auc_boxplot.png"))
    plt.close()

    # Histogram
    plt.figure(figsize=(7, 4))
    n = len(keys)
    blues = sns.color_palette("Blues", n + 2)[2:]
    caption = ""
    
    for i, key in enumerate(keys):
        values = np.array(samplewise_auc[key])
        sns.histplot(values, label=str(key), color=blues[i], kde=False, alpha=0.7, bins=20)
        max_val = np.max(values)
        plt.scatter([max_val], [0], color=blues[i], marker="o", s=80, edgecolor="black", zorder=5)
        caption += f"{key}: {max_val:.4f}\n"
    
    plt.xlabel("AUC")
    plt.ylabel("Count")
    plt.title("Sample-wise AUC Distribution")
    plt.legend(title="Label")
    plt.figtext(0.99, 0.01, caption, ha="right", va="bottom", fontsize=9, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(savedir_result, "samplewise_auc_hist.png"))
    plt.close()


def plot_and_save_integrals(integrals: Dict[str, np.ndarray], savedir_result: str) -> None:
    """Plot violin plots of integral values across different FPR thresholds.
    
    Args:
        integrals: Dictionary mapping label -> array of integral values
                  with shape (n_samples, n_thresholds).
        savedir_result: Directory to save figures.
    """
    os.makedirs(savedir_result, exist_ok=True)
    keys = list(integrals.keys())
    
    # FPR thresholds used in compute_individual_scores
    high_alpha_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    x_labels = [f"{v:.2e}" for v in high_alpha_list]
    m = len(high_alpha_list)

    for setting_idx in range(m):
        data = [np.array(integrals[key])[:, setting_idx] for key in keys]
        plt.figure(figsize=(7, 4))
        sns.violinplot(data=data, inner="box", density_norm="width")
        plt.xticks(ticks=range(len(keys)), labels=keys)
        plt.xlabel("Label")
        plt.ylabel("Integral Value")
        plt.title(f"Distribution of Integrals by Label\nIntegral up to {x_labels[setting_idx]}")
        plt.tight_layout()
        
        fname = f"adv_violin_{setting_idx}_alpha_{x_labels[setting_idx]}.png"
        plt.savefig(os.path.join(savedir_result, fname))
        plt.close()
        print(f"Saved violin plot of integrals to {os.path.join(savedir_result, fname)}")




# ============================================================================
# Matrix and Privacy Budget Utilities
# ============================================================================


def generate_biregular_binary_matrix_random(
    n_rows: int,
    n_cols: int,
    portion: float,
    tol: float = 0.01,
    max_iter: int = 10,
) -> np.ndarray:
    """Generate a biregular binary matrix with specified sparsity.
    
    Creates a matrix where:
    - Approximately 'portion' fraction of entries are 1
    - Row sums are approximately balanced
    - Column sums are approximately balanced
    - Entries are assigned randomly to encourage independence
    
    Args:
        n_rows: Number of rows.
        n_cols: Number of columns.
        portion: Target fraction of ones (0 < portion < 1).
        tol: Tolerance for row/col sum deviations (unused).
        max_iter: Maximum refinement iterations (unused).
    
    Returns:
        Binary matrix of shape (n_rows, n_cols).
    """
    seed = int(time.time() * 1e6) % (2**32)
    np.random.seed(seed)
    total_ones = int(np.round(portion * n_rows * n_cols))
    
    # Calculate target ones per row and column
    row_sum_target = total_ones // n_rows
    col_sum_target = total_ones // n_cols
    
    # Distribute remainder
    row_sums = np.full(n_rows, row_sum_target)
    row_sums[: total_ones - row_sum_target * n_rows] += 1
    np.random.shuffle(row_sums)

    col_sums = np.full(n_cols, col_sum_target)
    col_sums[: total_ones - col_sum_target * n_cols] += 1
    np.random.shuffle(col_sums)
    
    # Initialize empty matrix
    M = np.zeros((n_rows, n_cols), dtype=int)
    col_counts = np.zeros(n_cols, dtype=int)
    
    for i, rsum in enumerate(row_sums):
        # Compute weights inversely proportional to current column counts
        weights = 1 / (col_counts + 1e-6)
        weights = weights ** 2
        probs = weights / weights.sum()
        
        # Sample columns without replacement according to probabilities
        chosen_cols = np.random.choice(n_cols, size=rsum, replace=False, p=probs)
        
        # Assign ones
        M[i, chosen_cols] = 1
        col_counts[chosen_cols] += 1
    
    return M


def compute_epsilon_delta(
    noise_multiplier: float,
    deltas: List[float],
    iterations: int,
    sampling_rate: float,
    clipping_norm: float,
) -> Tuple[List[float], List[float]]:
    """Compute epsilon for given delta values using RDP accounting.
    
    Args:
        noise_multiplier: Noise multiplier for DP.
        deltas: List of delta values to compute epsilon for.
        iterations: Number of training iterations.
        sampling_rate: Sampling rate (fraction of data per batch).
        clipping_norm: Gradient clipping norm.
        
    Returns:
        Tuple of (epsilon_list, delta_list).
    """
    def compute_epsilon_single(noise_multiplier, delta, iterations, sampling_rate, clipping_norm):
        accountant = RDPAccountant()
        for _ in range(int(iterations)):
            accountant.step(noise_multiplier=noise_multiplier / clipping_norm, sample_rate=sampling_rate)
        return accountant.get_epsilon(delta)
    
    epsilons = []
    for delta in deltas:
        epsilon = compute_epsilon_single(noise_multiplier, delta, iterations, sampling_rate, clipping_norm)
        epsilons.append(epsilon)
    
    return epsilons, deltas


def load_pg_lists(args: Any, savedir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load privacy budget lists and compute epsilon-delta curves.
    
    Loads pre-computed noise multiplier and sampling rate lists, then
    computes epsilon values for various delta thresholds using RDP accounting.
    
    Args:
        args: Configuration with 'budgets' attribute.
        savedir: Directory containing:
                - pg_noise_multiplier_list.npy
                - pg_sample_rates_list.npy
                - num_steps_list.npy
    
    Returns:
        Tuple of (noise_multiplier_list, sample_rates_list).
        Side effect: Saves delta_epsilon_plot.png to savedir.
    """
    noise_multiplier_list = np.load(os.path.join(savedir, "pg_noise_multiplier_list.npy"), allow_pickle=True)
    sample_rates_list = np.load(os.path.join(savedir, "pg_sample_rates_list.npy"), allow_pickle=True)
    num_steps_list = np.load(os.path.join(savedir, "num_steps_list.npy"), allow_pickle=True)
    
    # Collapse 2D arrays if all rows are identical
    if sample_rates_list.ndim == 2:
        if np.allclose(sample_rates_list, sample_rates_list[0, :], atol=1e-6):
            sample_rates_list = sample_rates_list[0, :]
        else:
            raise ValueError("sample_rates_list rows are not all (almost) equal; cannot collapse dimension.")

    if noise_multiplier_list.ndim == 2:
        if np.allclose(noise_multiplier_list, noise_multiplier_list[0, :], atol=1e-6):
            noise_multiplier_list = noise_multiplier_list[0, :]
        else:
            raise ValueError("noise_multiplier_list rows are not all (almost) equal; cannot collapse dimension.")

    dict_return = {
        "noise_multiplier_list": noise_multiplier_list,
        "sample_rates_list": sample_rates_list,
        "num_steps_list": num_steps_list,
        "epsilons": {},
        "deltas": {},
    }

    budgets = args.budgets

    # Compute epsilon-delta curves for each budget
    for i, (nm, sr) in enumerate(zip(noise_multiplier_list, sample_rates_list)):
        epsilon, delta = compute_epsilon_delta(nm, np.logspace(-12, -1, 300), int(num_steps_list[0]), sr, 1.0)
        dict_return["epsilons"][budgets[i]] = epsilon
        dict_return["deltas"][budgets[i]] = delta

    # Plot epsilon-delta tradeoff
    for i, (nm, sr) in enumerate(zip(noise_multiplier_list, sample_rates_list)):
        epsilon = dict_return["epsilons"][budgets[i]]
        delta = dict_return["deltas"][budgets[i]]
        plt.semilogy(epsilon, delta, label=(budgets[i]))

    plt.ylabel("Delta")
    plt.xlabel("Epsilon")
    plt.title("Delta vs Epsilon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "delta_epsilon_plot.png"))
    plt.close()
    print(f"Saved delta vs epsilon plot to {os.path.join(savedir, 'delta_epsilon_plot.png')}")
    
    return noise_multiplier_list, sample_rates_list

