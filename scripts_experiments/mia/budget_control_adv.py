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
from opacus_new import PrivacyEngine
import threading
from queue import Queue
import json
# --- YAML support ---
import yaml
import pickle


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
    parser.add_argument('--exp_yaml', type=str, help='YAML file with experiment parameters', default='./scripts_experiments/mia/exp_yaml/mnist_4_budget_adv.yaml')
    parser.add_argument('--idx_start', type=int, help='Start index for attackee', default=0)
    parser.add_argument('--idx_end', type=int, help='End index for attackee', default=100)
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

if args.n_parallel > 1:
    args.disable_inner = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
# DEVICE = torch.device("cpu")  # Force CPU for now

def train_target_models(attackee_idx, attacker_budgets):
    if args.private == True:
        dummy_train_ds = load_dataset(args.dataset, train=True)
        size = len(dummy_train_ds)
        # Define portions and budgets for privacy
        budgets_attacker = attacker_budgets
        budgets_attackee = args.budgets_attackee  

        # Assign privacy budgets to each sample in the dataset
        pp_budgets = np.zeros(size) + budgets_attacker
        pp_budgets[attackee_idx] = budgets_attackee

        budget_to_index = {}
        unique_budgets = np.array([budgets_attacker, budgets_attackee])
        for budget in unique_budgets:
            indices = np.where(pp_budgets == budget)[0]
            budget_to_index[str(budget)] = indices

    keep_tensors = np.ones((args.n_targets, size), dtype=bool)
    keep_tensors[0:int(args.n_targets/2), attackee_idx] = False
    keep = keep_tensors

    threads = []
    seeds_out = [None] * args.n_targets
    pg_sample_rates_list = [None] * args.n_targets
    pg_noise_multiplier_list = [None] * args.n_targets
    num_steps_list = [None] * args.n_targets
    active_threads = []
    n_parallel = getattr(args, "n_parallel", 1)

    def worker(target_id, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, num_steps_list):
        seed = target_id # int(time.time() * 1e6) % (2**32)
        np.random.seed(seed)
        seeds_out[target_id] = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        keep_indiv = np.array(keep[target_id], dtype=bool)
        keep_indiv = keep_indiv.nonzero()[0]

        train_ds = load_dataset(args.dataset, train=True)
        train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
        train_dl = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True
        )

        m = model = load_model(args.dataset, model_name=args.model, num_classes=None)
        model = model.to(DEVICE)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()

        pp_budgets_subset = pp_budgets[keep_indiv]
        model, optim, train_dl, privacy_engine = make_private(
            model, train_dl, optim, pp_budgets_subset, args
        )
        pg_sample_rates_list[target_id] = privacy_engine.weights
        pg_noise_multiplier_list[target_id] = [optim.noise_multiplier] * len(privacy_engine.weights)


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
        np.save(savedir + "/keep.npy", keep[target_id])
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
    with open(os.path.join(args.savedir_result, "pg_sample_rates_list.pkl"), "wb") as f:
        pickle.dump(pg_sample_rates_list, f)
    with open(os.path.join(args.savedir_result, "pg_noise_multiplier_list.pkl"), "wb") as f:
        pickle.dump(pg_noise_multiplier_list, f)
    with open(os.path.join(args.savedir_result, "num_steps_list.pkl"), "wb") as f:
        pickle.dump(num_steps_list, f)

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

def test_models(savedir):
    test_ds = load_dataset(args.dataset, train=False)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    # Infer the logits with multiple queries
    for path in tqdm(os.listdir(savedir)):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(DEVICE)
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_dl:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = m(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        np.save(os.path.join(savedir, path, "accuracy.npy"), np.array([accuracy]))
    return




if __name__ == "__main__":

    for attackee_idx in range(args.idx_start, args.idx_end):
        for attacker_budgets in args.budgets:
            default_path = f"{args.dataset}_[{attacker_budgets}_{args.budgets_attackee}]"
            # Set savedir, savedir_target, and savedir_result as hardcoded attributes on args
            args.savedir_target = f"./budget_adv/{attackee_idx}/{default_path}/target"
            args.savedir_result = f"./budget_adv/{attackee_idx}/{default_path}/results"
            os.makedirs(args.savedir_result, exist_ok=True)
            os.makedirs(args.savedir_target, exist_ok=True)

            with open(os.path.join(args.savedir_result, "args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            _ = train_target_models(attackee_idx, attacker_budgets)
            seeds = [folder for folder in os.listdir(args.savedir_target) if os.path.isdir(os.path.join(args.savedir_target, folder))]
            inference(args.savedir_target)
            test_models(args.savedir_target)
            load_stats(args, args.savedir_target)
            keep_target, scores_target = load_data(args, args.savedir_target)

            indices_all = np.arange(keep_target.shape[1])
            indices_all_except_idx = np.delete(indices_all, attackee_idx)
            
            # keep_target_budget = keep_target[:, indices_all_except_idx]
            # scores_target_budget = scores_target[:, indices_all_except_idx, :]
            # indiv_scores_val_all_except_idx, x_vals_all_except_idx, samplewise_R_all_except_idx, integrals_all_except_idx, adv_all_except_idx = indiv_scores(keep_target_budget, scores_target_budget)

            keep_target_budget = np.expand_dims(keep_target[:, attackee_idx], axis=1)
            scores_target_budget = np.expand_dims(scores_target[:, attackee_idx, :], axis=1)
            indiv_scores_val_attackee, x_vals_attackee, samplewise_R_attackee, integrals_attackee, adv_attackee = indiv_scores(keep_target_budget, scores_target_budget)
            # Save the computed arrays/lists as .npy files in args.savedir_result
            np.save(os.path.join(args.savedir_result, "indiv_scores_val_attackee.npy"), np.array(indiv_scores_val_attackee))
            np.save(os.path.join(args.savedir_result, "x_vals_attackee.npy"), np.array(x_vals_attackee))
            np.save(os.path.join(args.savedir_result, "samplewise_R_attackee.npy"), np.array(samplewise_R_attackee))
            np.save(os.path.join(args.savedir_result, "integrals_attackee.npy"), np.array(integrals_attackee))
            np.save(os.path.join(args.savedir_result, "adv_attackee.npy"), np.array(adv_attackee))
            # Save the computed arrays/lists as YAML for visual inspection
            yaml_data = {
                "indiv_scores_val_attackee": np.array(indiv_scores_val_attackee).tolist(),
                "x_vals_attackee": np.array(x_vals_attackee).tolist(),
                "samplewise_R_attackee": np.array(samplewise_R_attackee).tolist(),
                "integrals_attackee": np.array(integrals_attackee).tolist(),
                "adv_attackee": np.array(adv_attackee).tolist(),
            }
            with open(os.path.join(args.savedir_result, "attackee_results.yaml"), "w") as f:
                yaml.dump(yaml_data, f)
            print("done")

# loop for different attacker budgets
# loop for different attackee index