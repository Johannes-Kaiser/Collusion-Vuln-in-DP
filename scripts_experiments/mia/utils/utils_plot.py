
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import functools
from sklearn.metrics import auc, roc_curve
import scipy.stats

def plot_roc(fpr_list, tpr_list, roc_auc, path, log=False):
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(range01, range01, "--", label="Random guess")
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    if log:
        plt.xlim([10e-6, 1])
        plt.ylim([10e-6, 1])
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()


def get_fpr_tpr(log_dir, report_folder, model_idx):
                
    if model_idx == "ten" or model_idx == "fifty" :
        if model_idx == "ten":
            nbmodels = 10
        elif model_idx == "fifty":
            nbmodels = 50
        all_fpr, all_tpr = [], []
        for k in range(nbmodels):
            filepath = f"{log_dir}/{report_folder}/attack_stats_{k}.npz"
            
            if os.path.isfile(filepath):
                stats = np.load(filepath, allow_pickle=True)
                fpr_list, tpr_list = stats["fpr_list"], stats["tpr_list"]
                all_fpr.append(fpr_list)
                all_tpr.append(tpr_list)
 
            else:
                print(f"{report_folder} NOT FOUND")
        
        if len(all_fpr) > 0:
            return np.mean(all_fpr, axis=0), np.mean(all_tpr, axis=0)
    else:
        filepath = f"{log_dir}/{report_folder}/attack_stats_{model_idx}.npz"

        if os.path.isfile(filepath):
            stats = np.load(filepath, allow_pickle=True)
            try :
                fpr_list, tpr_list = stats["fpr_list"], stats["tpr_list"]

                return fpr_list, tpr_list

            except Exception as e:
                print(e)
        else:
            print(f"{report_folder} NOT FOUND")

def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc

def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
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
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
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
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers



def do_plot(fn, keep, scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep[-ntest:], scores[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < 0.001)[0][-1]]

    print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)

def do_plot2(fn, keep, scores, keep_target, scores_target, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep_target[-ntest:], scores_target[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < 0.001)[0][-1]]

    print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr(args, keep, scores, savedir):
    os.makedirs(savedir, exist_ok=True)
    plt.figure(figsize=(4, 3))

    do_plot(generate_ours, keep, scores, 1, "Ours (online)\n", metric="auc")

    do_plot(functools.partial(generate_ours, fix_variance=True), keep, scores, 1, "Ours (online, fixed variance)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline), keep, scores, 1, "Ours (offline)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, 1, "Ours (offline, fixed variance)\n", metric="auc")

    do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")

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


def fig_fpr_tpr_target(args, keep, scores, keep_target, scores_target, savedir, name="fprtpr_target"):
    os.makedirs(savedir, exist_ok=True)
    plt.figure(figsize=(4, 3))

    do_plot2(generate_ours, keep, scores, keep_target, scores_target, 1, "Ours (online)\n", metric="auc")

    do_plot2(functools.partial(generate_ours, fix_variance=True), keep, scores, keep_target, scores_target, 1, "Ours (online, fixed variance)\n", metric="auc")

    do_plot2(functools.partial(generate_ours_offline), keep, scores, keep_target, scores_target, 1, "Ours (offline)\n", metric="auc")

    do_plot2(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, keep_target, scores_target, 1, "Ours (offline, fixed variance)\n", metric="auc")

    do_plot2(generate_global, keep, scores, keep_target, scores_target, 1, "Global threshold\n", metric="auc")

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


def plot_and_save_samplewise_auc(samplewise_auc, savedir_result):
    """
    Plots a boxplot and a histogram for each key in samplewise_auc (a dict of key: list/array of values).
    Each key's values are plotted as a separate box in the boxplot, and as a separate histogram in the histogram plot.
    The plots are saved to savedir_result/samplewise_auc_boxplot.png and samplewise_auc_hist.png.
    """

    os.makedirs(savedir_result, exist_ok=True)
    keys = list(samplewise_auc.keys())
    data = [np.array(samplewise_auc[key]) for key in keys]

    # Boxplot
    plt.figure(figsize=(7, 4))
    box = plt.boxplot(data, labels=keys, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='blue'),
                      medianprops=dict(color='darkblue'))
    plt.xlabel("Label")
    plt.ylabel("AUC")
    plt.title("Samplewise AUC Boxplot")
    # Plot the largest value for each box as a larger colored circle and annotate its value
    blues = sns.color_palette("Blues", len(keys) + 2)[2:]  # skip lightest
    for i, values in enumerate(data):
        max_val = np.max(values)
        plt.scatter([i + 1], [max_val], color=blues[i], marker='o', s=120, edgecolor='black', zorder=5)
        plt.text(i + 1 + 0.05, max_val, f"{max_val:.4f}", va='center', ha='left', fontsize=10, color=blues[i], fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir_result, "samplewise_auc_boxplot.png"))
    plt.close()

    # Histogram
    plt.figure(figsize=(7, 4))
    n = len(keys)
    blues = sns.color_palette("Blues", n + 2)[2:]  # skip lightest
    caption = ""
    for i, key in enumerate(keys):
        values = np.array(samplewise_auc[key])
        sns.histplot(values, label=str(key), color=blues[i], kde=False, alpha=0.7, bins=20)
        max_val = np.max(values)
        plt.scatter([max_val], [0], color=blues[i], marker='o', s=80, edgecolor='black', zorder=5)
        caption += f"{key}: {max_val:.4f}\n"
    plt.xlabel("AUC")
    plt.ylabel("Count")
    plt.title("Samplewise AUC Distribution")
    plt.legend(title="Label")
    plt.figtext(0.99, 0.01, caption, ha="right", va="bottom", fontsize=9, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(savedir_result, "samplewise_auc_hist.png"))
    plt.close()


def plot_and_save_integrals(integrals, savedir_result):
    import matplotlib.pyplot as plt

    os.makedirs(savedir_result, exist_ok=True)
    keys = list(integrals.keys())
    m = len([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])  # number of integral settings
    x_labels = [f"{v:.2e}" for v in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]]

    for setting_idx in range(m):
        data = [np.array(integrals[key])[:, setting_idx] for key in keys]
        plt.figure(figsize=(7, 4))
        sns.violinplot(data=data, inner="box", density_norm='width')
        plt.xticks(ticks=range(len(keys)), labels=keys)
        plt.xlabel("Label")
        plt.ylabel("Integral Value")
        plt.title(f"Distribution of Integrals by Label\nIntegral up to {x_labels[setting_idx]}")
        plt.tight_layout()
        fname = f"adv_violin_{setting_idx}_alpha_{x_labels[setting_idx]}.png"
        plt.savefig(os.path.join(savedir_result, fname))
        plt.close()
        print("Saved violin plot of integrals to", os.path.join(savedir_result, fname))
    return
