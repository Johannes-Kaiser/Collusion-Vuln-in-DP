# Adaptation of PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)
import sys
sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.utils_general import load_dataset, load_model, save_logits, train_loop, make_private
from utils.utils_mia import (
    indiv_scores,     
    generate_biregular_binary_matrix_random,
    load_stats,
    load_data, 
    plot_and_save_samplewise_auc,
    plot_and_save_integrals,
    load_pg_lists    
)
from tqdm import tqdm
import threading
from queue import Queue
import json
# --- YAML support ---
import yaml


# --- YAML experiment config support ---
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
    parser.add_argument('--exp_yaml', type=str, help='YAML file with experiment parameters', default='./scripts_experiments/mia/exp_yaml/mnist_4.yaml')
    # Do not set defaults for any other arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n_shadows", type=int)
    parser.add_argument("--n_targets", type=int)
    parser.add_argument("--pkeep", type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_queries", type=int)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adapt_weights_to_budgets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--accountant", type=str)
    parser.add_argument("--individualize", type=str)
    parser.add_argument("--weights")
    parser.add_argument("--target_delta", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--portions", type=float, nargs="+")
    parser.add_argument("--budgets", type=float, nargs="+")
    parser.add_argument("--n_parallel", type=int)
    parser.add_argument("--disable_inner", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Load config from YAML
    config = load_yaml_args(args.exp_yaml)
    # Overwrite config with any command-line arguments that are not None
    for k, v in vars(args).items():
        if k == "exp_yaml":
            continue
        if v is not None:
            if k in config and config[k] != v:
                print(f"Overwriting config value for '{k}': {config[k]} -> {v}")
            config[k] = v
    # Convert config dict to Namespace
    config["target_delta"] = float(config["target_delta"])
    return argparse.Namespace(**config)

args = parse_args_with_yaml()

# Construct default_path from dataset, budgets, and portions
def format_list(lst):
    return "_".join(str(x).replace('.', '').replace('-', 'm') for x in lst)

default_path = f"{args.dataset}_[{format_list(args.budgets)}]_[{format_list(args.portions)}]/{str(args.seed)}"
# Set savedir, savedir_target, and savedir_result as hardcoded attributes on args
args.savedir = f"./exp_mia_2/{default_path}/shadow"
args.savedir_target = f"./exp_mia_2/{default_path}/target"
args.savedir_result = f"./exp_mia_2/{default_path}/results"

if args.n_parallel > 1:
    args.disable_inner = True
# args.disable_inner = False
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
DEVICE = torch.device("cpu")  # Force CPU for now

# Save the args dict as a JSON file in savedir_result
os.makedirs(args.savedir_result, exist_ok=True)
args_dict = vars(args)
with open(os.path.join(args.savedir_result, "args.json"), "w") as f:
    json.dump(args_dict, f, indent=2)


        

def train_target_models():

    seeds = []
    args.debug = True
    if args.private == True:
        dummy_train_ds = load_dataset(args.dataset, train=True)
        size = len(dummy_train_ds)

        # Define portions and budgets for privacy
        portions = args.portions 
        budgets = args.budgets  
        assert abs(sum(portions) - 1.0) < 1e-6, "Portions must sum to 1."

        # Assign privacy budgets to each sample in the dataset
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
        np.random.seed(args.seed)  # Set seed
        shuffled_indices = np.random.permutation(size)
        pp_budgets = pp_budgets[shuffled_indices]
        # Log indices for each unique budget
        budget_to_index = {}
        unique_budgets = np.unique(pp_budgets)
        for budget in unique_budgets:
            indices = np.where(pp_budgets == budget)[0]
            budget_to_index[str(budget)] = indices

    if args.private:
        index_order = []
        keep_tensors = []
        for budget, indices in budget_to_index.items():
            keep = generate_biregular_binary_matrix_random(
                args.n_targets, len(indices), args.pkeep
            )
            keep_tensors.append(keep)
            index_order.append(indices)
        keep = np.concatenate([k for k in keep_tensors], axis=1)
        index_order = np.concatenate(index_order)
        # Reorder keep columns according to index_order so that keep[:, index_order[i]] is in column i
        keep_reordered = np.zeros_like(keep)
        for i, idx in enumerate(index_order):
            keep_reordered[:, idx] = keep[:, i]

        # Compute and print column and row sums
        colsum = keep_reordered.sum(axis=0)
        rowsum = keep_reordered.sum(axis=1)
        print("Column sum:", colsum)
        print("Row sum:", rowsum)

        keep = keep_reordered
    else:
        if args.n_targets is not None:
            keep = np.random.uniform(0, 1, size=(args.n_targets, size))
            order = keep.argsort(0)
            keep = order < int(args.pkeep * args.n_targets)
        else:
            keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
            keep.sort()

    threads = []
    seeds_out = [None] * args.n_targets
    pg_sample_rates_list = [None] * args.n_targets
    pg_noise_multiplier_list = [None] * args.n_targets
    num_steps_list = [None] * args.n_targets
    active_threads = []
    n_parallel = getattr(args, "n_parallel", 1)

    def worker(target_id, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, num_steps_list):
        seed = int(time.time() * 1e6) % (2**32)
        np.random.seed(seed)
        seeds_out[target_id] = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if args.private:
            keep_indiv = np.array(keep[target_id], dtype=bool)
            keep_indiv = keep_indiv.nonzero()[0]
        else:
            keep_indiv = np.array(keep[target_id], dtype=bool)
            keep_indiv = keep_indiv.nonzero()[0]

        train_ds = load_dataset(args.dataset, train=True)
        test_ds = load_dataset(args.dataset, train=False)

        if args.private:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
            pp_budgets_subset = pp_budgets[keep_indiv]
        else:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
            pp_budgets_subset = None

        m = model = load_model(args.dataset, model_name=args.model, num_classes=None)
        m = m.to(DEVICE)

        keep_bool = np.full((size), False)
        keep_bool[keep_indiv] = True

        train_dl = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True
        )
        test_dl = DataLoader(
            test_ds, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True
        )

        optim = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()

        if args.private:
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args
            )
            pg_sample_rates_list[target_id] = privacy_engine.weights
            pg_noise_multiplier_list[target_id] = [optim.noise_multiplier] * len(privacy_engine.weights)
        else:
            pg_sample_rates_list[target_id] = None
            pg_noise_multiplier_list[target_id] = None

        # Save number of steps
        num_steps = args.epochs * len(train_dl)
        num_steps_list[target_id] = num_steps

        model = train_loop(
            args,
            model,
            train_dl,
            criterion,
            optim,
            sched,
            DEVICE,
            args.epochs
        )

        savedir = os.path.join(args.savedir_target, str(seed))
        os.makedirs(savedir, exist_ok=True)
        np.save(savedir + "/keep.npy", keep_bool)
        torch.save(m.state_dict(), savedir + "/model.pt")
        with open(os.path.join(savedir, "bti.npy"), "wb") as f:
            np.save(f, budget_to_index, allow_pickle=True)

    for target_id in range(args.n_targets):
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            def worker_with_stream(target_id=target_id, stream=stream):
                with torch.cuda.stream(stream):
                    worker(target_id, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, num_steps_list)
            t = threading.Thread(target=worker_with_stream)
        else:
            t = threading.Thread(target=worker, args=(target_id, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, num_steps_list))
        threads.append(t)

    total_jobs = len(threads)
    active_threads = []
    idx = 0
    with tqdm(total=total_jobs) as pbar:
        while idx < total_jobs or active_threads:
            # Start new threads if we have capacity
            while len(active_threads) < n_parallel and idx < total_jobs:
                t = threads[idx]
                t.start()
                active_threads.append(t)
                idx += 1
            # Remove finished threads
            still_active = []
            for t in active_threads:
                if t.is_alive():
                    still_active.append(t)
                else:
                    pbar.update(1)
            active_threads = still_active
            if active_threads:
                # Avoid busy waiting
                active_threads[0].join(timeout=0.5)
    for t in active_threads:
        t.join()

    # Save the lists as .npy files in the results directory
    np.save(os.path.join(args.savedir_result, "pg_sample_rates_list.npy"), pg_sample_rates_list, allow_pickle=True)
    np.save(os.path.join(args.savedir_result, "pg_noise_multiplier_list.npy"), pg_noise_multiplier_list, allow_pickle=True)
    np.save(os.path.join(args.savedir_result, "num_steps_list.npy"), num_steps_list, allow_pickle=True)

    return seeds_out, num_steps_list


def inference(savedir):
    train_ds = load_dataset(args.dataset, train=True)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    # Infer the logits with multiple queries
    for path in tqdm(os.listdir(savedir)):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(DEVICE)
        save_logits(args, m, train_dl, DEVICE, savedir, path)
    return


if __name__ == "__main__":

    samplewise_auc = {}
    samplewise_auc_R = {}
    integrals_all = {}

    _ = train_target_models()
    seeds = [folder for folder in os.listdir(args.savedir_target) if os.path.isdir(os.path.join(args.savedir_target, folder))]
    inference(args.savedir_target)
    load_stats(args, args.savedir_target)
    keep_target, scores_target = load_data(args, args.savedir_target)
    # Load the saved budget_to_index dictionary for each target model
    budget_to_index_list = []
    # fig_fpr_tpr_target(args, keep, scores, keep_target, scores_target, args.savedir_result)
    for seed in seeds:
        savedir = os.path.join(args.savedir_target, str(seed))
        bti_path = os.path.join(savedir, "bti.npy")
        with open(bti_path, "rb") as f:
            budget_to_index = np.load(f, allow_pickle=True).item()
        budget_to_index_list.append(budget_to_index)
        # Check if all dicts in budget_to_index_list contain the same values
        first_bti = budget_to_index_list[0]
        for bti in budget_to_index_list[1:]:
            assert first_bti.keys() == bti.keys(), "Target models need to have the same pp_budgets (keys mismatch)"
            for key in first_bti.keys():
                assert np.array_equal(first_bti[key], bti[key]), f"Target models need to have the same pp_budgets for key {key}"
                
    for key in first_bti.keys():
        indices = first_bti[key]
        # keep_budget = keep[:, indices]
        # scores_budget = scores[:, indices, :]
        keep_target_budget = keep_target[:, indices]
        scores_target_budget = scores_target[:, indices, :]
        # fig_fpr_tpr_target(args, keep_budget, scores_budget, keep_target_budget, scores_target_budget, args.savedir_result, name=f"fprtpr_target_{key}")
        indiv_scores_val, x_vals, samplewise_R, integrals, adv = indiv_scores(keep_target_budget, scores_target_budget)

        samplewise_auc[key] = indiv_scores_val
        samplewise_auc_R[key] = samplewise_R
        integrals_all[key] = integrals

    # Save the dictionaries as .npy files in the results directory
    np.save(os.path.join(args.savedir_result, "samplewise_auc.npy"), samplewise_auc, allow_pickle=True)
    np.save(os.path.join(args.savedir_result, "samplewise_auc_R.npy"), samplewise_auc_R, allow_pickle=True)
    np.save(os.path.join(args.savedir_result, "integrals_all.npy"), integrals_all, allow_pickle=True)
    np.save(os.path.join(args.savedir_result, "adv.npy"), adv, allow_pickle=True)

    plot_and_save_samplewise_auc(
        samplewise_auc,
        args.savedir_result,
    )
    plot_and_save_integrals(
        integrals_all,
        args.savedir_result,
    )

    load_pg_lists(
        args,
        savedir=args.savedir_result
    )