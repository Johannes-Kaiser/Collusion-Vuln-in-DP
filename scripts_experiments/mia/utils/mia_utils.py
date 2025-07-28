
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))  # add current script dir
from opacus_new.accountants import RDPAccountant

import multiprocessing as mp
import os
from scipy.stats import norm
import numpy as np
import time
from utils_general import load_dataset


def load_one(input_tuple):
    """
    This loads a logits and converts it to a scored prediction.
    """
    path, dataset_name = input_tuple
    opredictions = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]

    # Be exceptionally careful.
    # Numerically stable everything, as described in the paper.
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    labels = load_dataset(dataset_name, train=True).targets.numpy()

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(os.path.join(path, "scores.npy"), logit)    


def load_stats(args, savedir):

    with mp.get_context("spawn").Pool(8) as p:
        p.map(load_one, [(os.path.join(savedir, x), args.dataset) for x in os.listdir(savedir)])


def load_data(args, savedir):
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for path in os.listdir(savedir):
        scores.append(np.load(os.path.join(savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(savedir, path, "keep.npy")))
    scores = np.array(scores)
    keep = np.array(keep)

    return keep, scores


def indiv_scores(keep, scores):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
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

    auc_scores = []
    for j in range(scores.shape[1]):
        # For each record in this model
        mu_x = mean_in[j]
        mu_xp = mean_out[j]
        sigma2_x = std_in[j] ** 2
        sigma2_xp = std_out[j] ** 2
        # Closed-form AUC for normal distributions
        denom = np.sqrt(sigma2_x + sigma2_xp)
        auc_j = norm.cdf((mu_x - mu_xp) / (denom + 1e-30))
        auc_scores.append(auc_j)
        # auc_scores is a list of per-record AUCs (length = scores.shape[1])
        # Compute R(x) = Φ(a + b * Φ^{-1}(x)) for 100 steps between 0 and 1, samplewise
        # a = ||μ1 - μ0|| / σ1, b = σ0 / σ1
    advs = []
    TPRs = []
    FPR = np.linspace(1e-6, 1 - 1e-6, 100000)
    for j in range(scores.shape[1]):
        mu1 = mean_in[j]
        mu0 = mean_out[j]
        sigma1 = std_in[j]
        sigma0 = std_out[j]
        a = np.abs(mu1 - mu0) / (sigma1 + 1e-30)
        b = sigma0 / (sigma1 + 1e-30)
        R_x = norm.cdf(a + b * norm.ppf(FPR))
        TPRs.append(R_x)
        # Integrate R_x from 0 to high_alpha for each sample
    # Compute integrals for a log-spaced list of high_alpha values between 0 and 1 (10 steps)
    high_alpha_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    integrals = []
    for R in TPRs:
        integrals_row = []
        for high_alpha in high_alpha_list:
            mask = FPR <= high_alpha
            if np.any(mask):
                integral = np.trapz(R[mask], FPR[mask])
            else:
                integral = 0.0
            integrals_row.append(integral)
        integrals.append(integrals_row)
        adv = np.max(np.array(R) - np.array(FPR))
        advs.append(adv)
    # Example: Compute the integral of R(x) from 0 to high_alpha using sympy for each sample
    # high_alpha = 0.01  # or any value you want to integrate up to
    # integrals_sympy = []
    # for j in range(scores.shape[1]):
    #     mu1 = mean_in[j]
    #     mu0 = mean_out[j]
    #     sigma1 = std_in[j]
    #     sigma0 = std_out[j]
    #     a = np.abs(mu1 - mu0) / (sigma1 + 1e-30)
    #     b = sigma0 / (sigma1 + 1e-30)
    #     x = sp.symbols('x')
    #     Z = Normal('Z', 0, 1)
    #     phi_inv = quantile(Z)(x)
    #     R_x = cdf(Z)(a + b * phi_inv)
    #     integral = sp.integrate(R_x, (x, 0, high_alpha))
    #     integrals_sympy.append(float(integral.evalf()))
    # integrals_sympy is a list of the definite integrals for each sample up to high_alpha (using sympy)
    # integrals is a list of the definite integrals for each sample up to high_alpha


    auc_scores = np.array(auc_scores)
    max_auc = np.max(auc_scores)
    min_auc = np.min(auc_scores)
    avg_auc = np.mean(auc_scores)    
    return auc_scores, FPR, TPRs, integrals, advs


import numpy as np
from scipy.sparse import coo_matrix
import random
from scipy.stats import norm
import sympy as sp
import seaborn as sns
import os


def generate_biregular_binary_matrix_random(n_rows, n_cols, portion, tol=0.01, max_iter=10):
    """
    Generate a biregular binary matrix with given portion of ones,
    using probabilistic assignment to encourage randomness.
    Ensures that the column sums are the same for all
    Ensures that the row sums are approximatelly the same for all
    
    Args:
        n_rows (int): number of rows
        n_cols (int): number of columns
        portion (float): fraction of ones in the matrix (0 < portion < 1)
        tol (float): tolerance for row/col sum deviations
        max_iter (int): number of refinement iterations
    
    Returns:
        np.ndarray: binary matrix (n_rows x n_cols)
    """
    seed = int(time.time() * 1e6) % (2**32)
    np.random.seed(seed)
    total_ones = int(np.round(portion * n_rows * n_cols))
    
    # Calculate approximate ones per row and column
    row_sum_target = total_ones // n_rows
    col_sum_target = total_ones // n_cols
    
    # Distribute remainder
    row_sums = np.full(n_rows, row_sum_target)
    row_sums[:total_ones - row_sum_target * n_rows] += 1
    np.random.shuffle(row_sums)

    col_sums = np.full(n_cols, col_sum_target)
    col_sums[:total_ones - col_sum_target * n_cols] += 1
    np.random.shuffle(col_sums)
    
    # Initialize empty matrix
    M = np.zeros((n_rows, n_cols), dtype=int)
    col_counts = np.zeros(n_cols, dtype=int)
    
    for i, rsum in enumerate(row_sums):
        # Compute weights inversely proportional to current col_counts (plus a small epsilon to avoid zero div)
        weights = 1 / (col_counts + 1e-6)
        weights = weights**2
        # Normalize weights to get probabilities
        probs = weights / weights.sum()
        
        # Sample columns without replacement according to probabilities
        chosen_cols = np.random.choice(n_cols, size=rsum, replace=False, p=probs)
        
        # Assign ones
        M[i, chosen_cols] = 1
        
        # Update col counts
        col_counts[chosen_cols] += 1
    return M

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
    return epsilons, deltas2

def load_pg_lists(args, savedir):
    """
    Loads 'pg_noise_multiplier_list.npy' and 'pg_sample_rates_list.npy' from the given directory.

    Args:
        savedir (str): Directory containing the .npy files.

    Returns:
        tuple: (noise_multiplier_list, sample_rates_list) as numpy arrays.
    """
    noise_multiplier_list = np.load(os.path.join(savedir, "pg_noise_multiplier_list.npy"), allow_pickle=True)
    sample_rates_list = np.load(os.path.join(savedir, "pg_sample_rates_list.npy"), allow_pickle=True)
    num_steps_list = np.load(os.path.join(savedir, "num_steps_list.npy"), allow_pickle=True)
    # Check if sample_rates_list is 2D and if all values in each row are (almost) the same
    if sample_rates_list.ndim == 2:
        if np.allclose(sample_rates_list, sample_rates_list[0, :], atol=1e-6):
            sample_rates_list = sample_rates_list[0, :]
        else:
            raise ValueError("sample_rates_list rows are not all (almost) equal; cannot collapse dimension.")

    # Check if noise_multiplier_list is 2D and if all values in each row are (almost) the same
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
        "deltas": {}
    }

    budgets = args.budgets

    for i, (nm, sr) in enumerate(zip(noise_multiplier_list, sample_rates_list)):
        epsilon, delta = compute_epsilon_delta(nm, np.logspace(-12, -1, 300), num_steps_list[0], sr, 1.0)
        dict_return["epsilons"][budgets[i]] = epsilon
        dict_return["deltas"][budgets[i]] = delta
    # Plot delta vs epsilon (semilogx: log-scale on delta axis)
    import matplotlib.pyplot as plt

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

