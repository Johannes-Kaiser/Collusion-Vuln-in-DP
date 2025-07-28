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
from mia_utils import (
    load_stats,
    load_data,
    save_logits,
    indiv_scores,
    generate_biregular_binary_matrix_random,
    load_pg_lists
)
from utils_general import  (
    train_loop,
    load_dataset,
    load_model,
)

from utils_plot import (
    plot_and_save_integrals,
    plot_and_save_samplewise_auc,
    fig_fpr_tpr,
    fig_fpr_tpr_target,
)

from tqdm import tqdm
from opacus_new import PrivacyEngine
import threading
from queue import Queue
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist_4", type=str)
parser.add_argument("--model", default=None, type=str)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--n_shadows", default=5, type=int)
parser.add_argument("--n_targets", default=32, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable private mode (default: True)")
parser.add_argument("--adapt_weights_to_budgets", action=argparse.BooleanOptionalAction, default=True, help="Adapt weights from budgets (default: True)")
parser.add_argument("--accountant", default="rdp", type=str, help="Type of privacy accountant (default: RDP)")
parser.add_argument("--individualize", default="sampling", type=str, help="Individualization method (default: sampling)")
parser.add_argument("--weights", default=None)
parser.add_argument("--target_delta", default=1e-10, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--budgets", type=float, nargs="+", default=[16.0, 50.0], help="List of epsilon values for budgets")
parser.add_argument("--n_parallel", default=16, type=int)
parser.add_argument("--disable_inner", action=argparse.BooleanOptionalAction, default=False, help="Disable inner parallelism (default: False)")
parser.add_argument("--seed", default=42, type=int, help="Random seed (default: 42)")
parser.add_argument("--begin", default=0, type=int)
parser.add_argument("--end", default=1000, type=int)
args = parser.parse_args()

# Construct default_path from dataset, budgets, and portions
def format_list(lst):
    return "_".join(str(x).replace('.', '').replace('-', 'm') for x in lst)

default_path = f"{args.dataset}_[{format_list(args.budgets)}]"
# Set savedir, savedir_target, and savedir_result as hardcoded attributes on args
args.savedir = f"./exp_mia_budg_adv_1/{default_path}"


if args.n_parallel > 1:
    args.disable_inner = True
args.disable_inner = True
# args.disable_inner = False
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")  # Force CPU for now

# Save the args dict as a JSON file in savedir_result
# os.makedirs(args.savedir_result, exist_ok=True)
# args_dict = vars(args)
# with open(os.path.join(args.savedir_result, "args.json"), "w") as f:
#     json.dump(args_dict, f, indent=2)




def make_private(args, model, train_loader, optimizer, pp_budgets=None, sr=None, nm=None):
    # modulevalidator = ModuleValidator()
    # model = modulevalidator.fix_and_validate(model)
    pp_budgets = np.array(pp_budgets)
    sr = np.array(sr).squeeze()
    privacy_engine = PrivacyEngine(accountant=args.accountant,
                                   individualize=args.individualize,
                                   weights=args.weights,
                                   pp_budgets=pp_budgets)
    if args.adapt_weights_to_budgets:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private_with_epsilon(module=model,
                                       optimizer=optimizer,
                                       data_loader=train_loader,
                                       target_epsilon=min(pp_budgets),
                                       target_delta=args.target_delta,
                                       epochs=args.epochs,
                                       max_grad_norm=args.max_grad_norm,
                                       optimal=True,
                                       max_alpha=10_000,
                                       numeric=True)
    else:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private(module=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          noise_multiplier=nm,
                          max_grad_norm=args.max_grad_norm,
                          pp_weights=sr)
    # print(np.unique(pp_budgets))
    # print(privacy_engine.weights)

    return private_model, private_optimizer, private_loader, privacy_engine


def get_sr_and_nm():
    budget_to_index = {}
    if args.private == True:
        dummy_train_ds = load_dataset(args.dataset, train=True)
        size = len(dummy_train_ds)

        # Assign privacy budgets to each sample in the dataset
        pp_budgets = np.zeros(size) + args.budgets[0]
        pp_budgets[0] = args.budgets[1]
        unique_budgets = np.unique(pp_budgets)
        for budget in unique_budgets:
            indices = np.where(pp_budgets == budget)[0]
            budget_to_index[str(budget)] = indices

    keep = np.ones((10, size))
    keep[0:int(10/2), 0] = 0


    sample_rates = {}
    noise_multiplier = {}
    num_steps_dict = None

    #### Get noise multiplier and sample rates for in and out setting ####
    for setting_name in ["exclude", "include"]:
        setting = 0 if setting_name == "exclude" else -1
        keep_indiv = np.array(keep[setting], dtype=bool)
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

        optim = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=5e-4)

        if args.private:
            model, optim, train_dl, privacy_engine = make_private(
               args, model, train_dl, optim, pp_budgets=pp_budgets_subset 
            )

        # Save number of steps
        num_steps = args.epochs * len(train_dl)
        num_steps_dict = num_steps

        sample_rates[setting_name] = privacy_engine.weights
        noise_multiplier[setting_name] = optim.noise_multiplier
    return sample_rates, noise_multiplier, num_steps_dict


def train_target_models_budg_adv(idx, sample_rates, noise_multiplier, num_steps_dict):
    args.adapt_weights_to_budgets = False  # Disable adapting weights to budgets for budg_adv
    budget_to_index = {}

    if args.private == True:
        dummy_train_ds = load_dataset(args.dataset, train=True)
        size = len(dummy_train_ds)

        # Assign privacy budgets to each sample in the dataset
        pp_budgets = np.zeros(size) + args.budgets[0]
        pp_budgets[idx] = args.budgets[1]
        unique_budgets = np.unique(pp_budgets)
        for budget in unique_budgets:
            indices = np.where(pp_budgets == budget)[0]
            budget_to_index[str(budget)] = indices

    keep = np.ones((args.n_targets, size))
    keep[0:int(args.n_targets/2), idx] = 0

    threads = []
    active_threads = []
    n_parallel = getattr(args, "n_parallel", 1)

        

    def worker(target_id, idx, keep, size, args, DEVICE, budget_to_index, sample_rates, noise_multiplier_dict):
        seed = int(time.time() * 1e6) % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        keep_indiv = np.array(keep[target_id], dtype=bool)

        
        
        if np.any(keep_indiv == 0):
            pp_sampling_rates = [sample_rates["exclude"]] * len(keep_indiv.nonzero()[0])
            noise_multiplier = noise_multiplier_dict["exclude"]
            pp_budgets = np.zeros(size) + args.budgets[0]
        else:
            pp_sampling_rates = [sample_rates["include"][0]] * len(keep_indiv.nonzero()[0])
            pp_sampling_rates[idx] = sample_rates["include"][1]
            noise_multiplier = noise_multiplier_dict["include"]
            pp_budgets = np.zeros(size) + args.budgets[0]
            pp_budgets[idx] = args.budgets[1]
        keep_indiv = keep_indiv.nonzero()[0]


        train_ds = load_dataset(args.dataset, train=True)
        test_ds = load_dataset(args.dataset, train=False)

        if args.private:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
        else:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)

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

        model, optim, train_dl, privacy_engine = make_private(
            args, model, train_dl, optim, pp_budgets=pp_budgets, sr=pp_sampling_rates, nm=noise_multiplier
        )

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

        savedir_target = os.path.join(args.savedir, str(idx), "target", str(seed))
        os.makedirs(savedir_target, exist_ok=True)
        np.save(savedir_target + "/keep.npy", keep_bool)
        torch.save(m.state_dict(), savedir_target + "/model.pt")
        with open(os.path.join(savedir_target, "bti.npy"), "wb") as f:
            np.save(f, budget_to_index, allow_pickle=True)


    for target_id in range(args.n_targets):
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            def worker_with_stream(target_id=target_id, stream=stream):
                with torch.cuda.stream(stream):
                    try: 
                        worker(target_id, idx, keep, size, args, DEVICE, budget_to_index, sample_rates, noise_multiplier)
                    except Exception as e:
                        print(f"Error occurred in worker thread for target {target_id}: {e}")
            t = threading.Thread(target=worker_with_stream)
        else:
            t = threading.Thread(target=worker, args=(target_id, idx, keep, size, args, DEVICE, budget_to_index, sample_rates, noise_multiplier))
        threads.append(t)

    total_jobs = len(threads)
    active_threads = []
    idx_threads = 0
    with tqdm(total=total_jobs) as pbar:
        while idx_threads < total_jobs or active_threads:
            # Start new threads if we have capacity
            while len(active_threads) < n_parallel and idx_threads < total_jobs:
                t = threads[idx_threads]
                t.start()
                active_threads.append(t)
                idx_threads += 1
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
    savedir_results = os.path.join(args.savedir, str(idx), "results")
    os.makedirs(savedir_results, exist_ok=True)
    # Save the dicts as JSON files instead of .npy, since they are now dictionaries
    with open(os.path.join(savedir_results, "sample_rates.json"), "w") as f:
        json.dump(sample_rates, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    with open(os.path.join(savedir_results, "noise_multiplier.json"), "w") as f:
        json.dump(noise_multiplier, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    with open(os.path.join(savedir_results, "num_steps_dict.json"), "w") as f:
        json.dump(num_steps_dict, f, indent=2)

    return num_steps_dict


def inference(savedir):
    train_ds = load_dataset(args.dataset, train=True)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    # Infer the logits with multiple queries
    for path in tqdm(os.listdir(savedir), desc="inference"):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(DEVICE)
        save_logits(args, m, train_dl, DEVICE, savedir, path)
    return

if __name__ == "__main__":

    train_ds_init = load_dataset(args.dataset, train=True)
    sample_rates, noise_multiplier, num_steps_dict = get_sr_and_nm()
    for idx, _ in tqdm(enumerate(range(args.begin, args.end)), desc="Individual idx"): #train_ds_init):
        samplewise_auc = {}
        samplewise_auc_R = {}
        integrals_all = {}
        savedir_result = os.path.join(args.savedir, str(idx), "results")
        savedir_target = os.path.join(args.savedir, str(idx), "target")
        _ = train_target_models_budg_adv(idx, sample_rates, noise_multiplier, num_steps_dict)
        seeds = [folder for folder in os.listdir(savedir_target) if os.path.isdir(os.path.join(savedir_target, folder))]
        inference(savedir_target)
        load_stats(args, savedir_target)
        keep_target, scores_target = load_data(args, savedir_target)
        # Load the saved budget_to_index dictionary for each target model
        budget_to_index_list = []
        # fig_fpr_tpr_target(args, keep, scores, keep_target, scores_target, args.savedir_result)
        for seed in seeds:
            savedir = os.path.join(savedir_target, str(seed))
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
        np.save(os.path.join(savedir_result, "samplewise_auc.npy"), samplewise_auc, allow_pickle=True)
        np.save(os.path.join(savedir_result, "samplewise_auc_R.npy"), samplewise_auc_R, allow_pickle=True)
        np.save(os.path.join(savedir_result, "integrals_all.npy"), integrals_all, allow_pickle=True)
        np.save(os.path.join(savedir_result, "adv.npy"), adv, allow_pickle=True)

        plot_and_save_samplewise_auc(
            samplewise_auc,
            savedir_result,
        )
        plot_and_save_integrals(
            integrals_all,
            savedir_result,
        )

        load_pg_lists(
            args,
            savedir=savedir_result
        )
