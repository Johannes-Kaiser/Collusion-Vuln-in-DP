import sys
sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")

import gc

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.utils_general import load_dataset, load_model, save_logits, train_loop, make_private, generate_string, get_dataset_size
from utils.utils_mia import (
    score_mia,
    load_data_adv,  
    fit_mia_in_out_gaussians,
    compute_individual_scores   
)
from tqdm import tqdm
import json
# --- YAML support ---
import yaml
import pickle
import torch.multiprocessing as mp
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore", category=UserWarning)


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
    parser.add_argument('--exp_yaml', type=str, help='YAML file with experiment parameters', default='./scripts_experiments/mia/exp_yaml_adv/credit_card_default.yaml')
    parser.add_argument('--idx_start', type=int, help='Start index for attackee', default=0)
    parser.add_argument('--idx_end', type=int, help='End index for attackee', default=400)
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


# Construct default_path from dataset, budgets, and portions
def format_list(lst):
    return "_".join(str(x).replace('.', '').replace('-', 'm') for x in lst)    


def train_target_models(args, attackee_idx, attacker_budgets, SR_mp, NM_mp, device, lock):
    times_start = time.time()
    if args.private:
        size = get_dataset_size(args.dataset, train=True, num_max_per_class_samples=args.num_max_per_class_samples)
        budgets_attacker = attacker_budgets
        budgets_attackee = args.budgets_attackee  
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

    seeds_out = list([None] * args.n_targets)
    pg_sample_rates_list =list([None] * args.n_targets)
    pg_noise_multiplier_list = list([None] * args.n_targets)
    num_steps_list = list([None] * args.n_targets)
    
    train_ds = load_dataset(args.dataset, train=True, num_max_samples=args.num_max_per_class_samples)
    previous_keep_indiv = []
    previous_train_dl = None

    for target_id in range(args.n_targets):
        seed = target_id  # or use int(time.time() * 1e6) % (2**32) for more randomness
        np.random.seed(seed)
        seeds_out[target_id] = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        keep_indiv = np.array(keep[target_id], dtype=bool)
        keep_indiv = keep_indiv.nonzero()[0]

        if np.array_equal(keep_indiv, previous_keep_indiv):
            train_dl = previous_train_dl
        else:
            train_ds_sub = torch.utils.data.Subset(train_ds, keep_indiv)
            train_dl = DataLoader(
                train_ds_sub, batch_size=args.batchsize, shuffle=True, num_workers=0  # set num_workers=0 to reduce memory
            )
        previous_keep_indiv = keep_indiv
        previous_train_dl = train_dl

        m = model = load_model(args.dataset, model_name=args.model, num_classes=None)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        # sched = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.4, total_iters=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()

        pp_budgets_subset = pp_budgets[keep_indiv]
        unique, counts = np.unique(pp_budgets_subset, return_counts=True)
        str_budget = generate_string(unique, counts)
        # print(NM_mp)
        # print(SR_mp)
        with lock:
            use_cached = str_budget in SR_mp and str_budget in NM_mp
            if use_cached:
                nm_local = NM_mp[str_budget]
                sr_local = SR_mp[str_budget]
        if use_cached:
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args,
                adapt_weights_to_budgets=False, nm=nm_local, sr=sr_local
            )
        else:
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args
            )

        pg_sample_rates_list[target_id] = privacy_engine.weights
        pg_noise_multiplier_list[target_id] = [optim.noise_multiplier] * len(privacy_engine.weights)
        with lock:
            if str_budget not in SR_mp:
                SR_mp[str_budget] = pg_sample_rates_list[target_id]
            if str_budget not in NM_mp:
                NM_mp[str_budget] = pg_noise_multiplier_list[target_id]

        num_steps = args.epochs * len(train_dl)
        num_steps_list[target_id] = num_steps

        model = train_loop(
            args,
            model,
            train_dl,
            criterion,
            optim,
            sched,
            device,
            args.epochs
        )

        def save_artifacts(savedir, keep_arr, model, budget_to_index, seed):
            os.makedirs(savedir, exist_ok=True)
            np.save(os.path.join(savedir, "keep.npy"), keep_arr)
            torch.save(model.state_dict(), os.path.join(savedir, "model.pt"))
            with open(os.path.join(savedir, "bti.npy"), "wb") as f:
                np.save(f, budget_to_index, allow_pickle=True)

        savedir = os.path.join(args.savedir_target, str(seed))
        save_artifacts(savedir, keep[target_id], m, budget_to_index, seed)

        # --- Explicit cleanup to avoid memory/data leaks ---
        del model, m, optim, sched, criterion, privacy_engine
        torch.cuda.empty_cache()
        gc.collect()

        # print(f"One shadow model took {(time.time() - time_start_single):.2f} seconds to train.")

    seeds_out = list(seeds_out)
    pg_sample_rates_list = list(pg_sample_rates_list)
    pg_noise_multiplier_list = list(pg_noise_multiplier_list)
    num_steps_list = list(num_steps_list)

    with open(os.path.join(args.savedir_result, "pg_sample_rates_list.pkl"), "wb") as f:
        pickle.dump(pg_sample_rates_list, f)
    with open(os.path.join(args.savedir_result, "pg_noise_multiplier_list.pkl"), "wb") as f:
        pickle.dump(pg_noise_multiplier_list, f)
    with open(os.path.join(args.savedir_result, "num_steps_list.pkl"), "wb") as f:
        pickle.dump(num_steps_list, f)

    print(f"All shadow models took {(time.time() - times_start):.2f} seconds to train.")

    return seeds_out, num_steps_list

def inference(args, savedir, device):
    train_ds = load_dataset(args.dataset, train=True, num_max_samples=args.num_max_per_class_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batchsize, shuffle=False, num_workers=0)  # num_workers=0
    # Infer the logits with multiple queries
    for path in os.listdir(savedir):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(device)
        with torch.no_grad():
            save_logits(args, m, train_dl, device, savedir, path)
        del m
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    del train_dl, train_ds
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return

def test_models(args, savedir, device):
    test_ds = load_dataset(args.dataset, train=False)
    test_dl = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=0)  # num_workers=0
    # Infer the logits with multiple queries
    for path in os.listdir(savedir):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(device)
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_dl:
                data, target = data.to(device), target.to(device)
                outputs = m(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        print(f"Accuracy for model in {path}: {accuracy:.4f}")
        np.save(os.path.join(savedir, path, "accuracy.npy"), np.array([accuracy]))
        del m
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    del test_dl, test_ds
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return


def worker_mp(args_dict, SR_mp, NM_mp, attackee_idx, lock):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = SimpleNamespace(**args_dict)
    if args.num_max_per_class_samples is None:
        num_max_per_class_samples_name_ext = ""
    else:
        num_max_per_class_samples_name_ext = f"_{args.num_max_per_class_samples}"
    for attacker_budgets in args.budgets:
        default_path = f"{args.dataset}_[{attacker_budgets}_{args.budgets_attackee}]"
        # Set savedir, savedir_target, and savedir_result as hardcoded attributes on args
        args.savedir_target = f"./budget_adv_final_by_dataset/{args.dataset}{num_max_per_class_samples_name_ext}/{attackee_idx}/{default_path}/target"
        args.savedir_result = f"./budget_adv_final_by_dataset/{args.dataset}{num_max_per_class_samples_name_ext}/{attackee_idx}/{default_path}/results"
        os.makedirs(args.savedir_result, exist_ok=True)
        os.makedirs(args.savedir_target, exist_ok=True)

        with open(os.path.join(args.savedir_result, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        if args.train_and_save_models:
             _ = train_target_models(args=args, attackee_idx=attackee_idx, attacker_budgets=attacker_budgets, SR_mp=SR_mp, NM_mp=NM_mp, device=device, lock=lock)
        if args.test_model:
            test_models(args, args.savedir_target, device)
        if args.compute_and_save_logits:
            inference(args, args.savedir_target, device=device)
        if args.compute_and_save_scores:
            score_mia(args, args.savedir_target)
        if args.compute_and_save_stats:
            keep, scores = load_data_adv(args.savedir_target)

        keep_target_budget = np.expand_dims(keep[:, attackee_idx], axis=1)
        scores_target_budget = np.expand_dims(scores[:, attackee_idx, :], axis=1)
        mean_in, mean_out, std_in, std_out = fit_mia_in_out_gaussians(keep_target_budget, scores_target_budget)
        indiv_scores_val_attackee, x_vals_attackee, samplewise_R_attackee, integrals_attackee, adv_attackee, priv_scores = compute_individual_scores(mean_in, mean_out, std_in, std_out)

        # Save the computed arrays/lists as .npy files in args.savedir_result
        np.save(os.path.join(args.savedir_result, "indiv_scores_val_attackee.npy"), np.array(indiv_scores_val_attackee))
        np.save(os.path.join(args.savedir_result, "x_vals_attackee.npy"), np.array(x_vals_attackee))
        np.save(os.path.join(args.savedir_result, "samplewise_R_attackee.npy"), np.array(samplewise_R_attackee))
        np.save(os.path.join(args.savedir_result, "integrals_attackee.npy"), np.array(integrals_attackee))
        np.save(os.path.join(args.savedir_result, "adv_attackee.npy"), np.array(adv_attackee))
        np.save(os.path.join(args.savedir_result, "priv_scores.npy"), np.array(priv_scores))
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

if __name__ == "__main__":
    args = parse_args_with_yaml()

    for k, v in vars(args).items():
        try:
            pickle.dumps(v)
        except Exception as e:
            print(f"[UNPICKLEABLE] {k}: {type(v)} -> {e}")
    args.batchsize = getattr(args, 'batchsize', 128)
    args.num_max_per_class_samples = getattr(args, 'num_max_per_class_samples', None)
    if args.n_parallel > 1:
        args.disable_inner = True
    DEVICE = torch.device("cpu") #  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {DEVICE}")
    # device = torch.device("cpu")  # Force CPU for now

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        SR_mp = manager.dict()  # <-- managed dict for SR
        NM_mp = manager.dict()  # <-- managed dict for NM
        n_parallel = getattr(args, "n_parallel", 1)
        lock = ctx.Lock()

        processes = []
        for attackee_idx in range(args.idx_start, args.idx_end):
            p = ctx.Process(
                        target=worker_mp,
                        args=(
                            vars(args),
                            SR_mp,  # pass managed dict
                            NM_mp,  # pass managed dict
                            attackee_idx,
                            lock
                        ),
                    )
            processes.append(p)
        total_jobs = len(processes)
        active_processes = []
        idx = 0
        with tqdm(total=total_jobs) as pbar:
            while idx < total_jobs or active_processes:
                # Start new processes if we have capacity
                while len(active_processes) < n_parallel and idx < total_jobs:
                    p = processes[idx]
                    p.start()
                    print("Starting process for attackee index:", idx)
                    active_processes.append(p)
                    if idx == 0:
                        time.sleep(3 * 60)
                    idx += 1
                # Remove finished processes
                still_active = []
                for p in active_processes:
                    if p.is_alive():
                        still_active.append(p)
                    else:
                        pbar.update(1)
                active_processes = still_active
                if active_processes:
                    # Avoid busy waiting
                    active_processes[0].join(timeout=0.5)
        for p in active_processes:
            p.join()

            




