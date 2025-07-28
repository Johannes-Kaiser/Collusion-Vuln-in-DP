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
from utils.utils_general import (
    train_loop,
    load_dataset,
    load_model,
    parse_extra
)
from utils.mia_utils import generate_biregular_binary_matrix_random
from tqdm import tqdm
from opacus_new import PrivacyEngine
import threading
import yaml
DEVICE = "cpu"

def make_private(model, train_loader, optimizer, pp_budgets, args):
    privacy_engine = PrivacyEngine(accountant=args["train"]["accountant"],
                                   individualize=args["train"]["individualize"],
                                   weights=args["train"]["weights"],
                                   pp_budgets=pp_budgets)
    if args["train"]["adapt_weights_to_budget"]:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private_with_epsilon(module=model,
                                       optimizer=optimizer,
                                       data_loader=train_loader,
                                       target_epsilon=min(pp_budgets),
                                       target_delta=args["train"]["target_delta"],
                                       epochs=args["train"]["epochs"],
                                       max_grad_norm=args["train"]["max_grad_norm"],
                                       optimal=True,
                                       max_alpha=10_000,
                                       numeric=True)
    else:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private(module=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          noise_multiplier=args["train"]["noise_multiplier"],
                          max_grad_norm=args["train"]["max_grad_norm"])
    # print(np.unique(pp_budgets))
    # print(privacy_engine.weights)
    return private_model, private_optimizer, private_loader, privacy_engine
        

def train_target_models(seed):

    if args["train"]["private"] == True:
        dummy_train_ds = load_dataset(args["train"]["dataset"], train=True)
        size = len(dummy_train_ds)

        # Define portions and budgets for privacy
        portions = args["train"]["portions"]
        budgets = args["train"]["budgets"]
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
        np.random.seed(seed)  # Set seed
        shuffled_indices = np.random.permutation(size)
        pp_budgets = pp_budgets[shuffled_indices]
        # Log indices for each unique budget
        budget_to_index = {}
        unique_budgets = np.unique(pp_budgets)
        for budget in unique_budgets:
            indices = np.where(pp_budgets == budget)[0]
            budget_to_index[str(budget)] = indices

    if args["train"]["private"]:
        index_order = []
        keep_tensors = []
        for budget, indices in budget_to_index.items():
            keep = generate_biregular_binary_matrix_random(
                args["train"]["n_targets"], len(indices), args["train"]["pkeep"]
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
        if args["train"]["n_targets"] is not None:
            keep = np.random.uniform(0, 1, size=(args["train"]["n_targets"], size))
            order = keep.argsort(0)
            keep = order < int(args["train"]["pkeep"] * args["train"]["n_targets"])
        else:
            keep = np.random.choice(size, size=int(args["train"]["pkeep"] * size), replace=False)
            keep.sort()

    threads = []
    seeds_out = [None] * args["train"]["n_targets"]
    pg_sample_rates_list = [None] * args["train"]["n_targets"]
    pg_noise_multiplier_list = [None] * args["train"]["n_targets"]
    num_steps_list = [None] * args["train"]["n_targets"]
    active_threads = []
    n_parallel = args["train"]["n_parallel"]

    def worker(target_id, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, num_steps_list):
        seed = int(time.time() * 1e6) % (2**32)
        np.random.seed(seed)
        seeds_out[target_id] = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if args["train"]["private"]:
            keep_indiv = np.array(keep[target_id], dtype=bool)
            keep_indiv = keep_indiv.nonzero()[0]
        else:
            keep_indiv = np.array(keep[target_id], dtype=bool)
            keep_indiv = keep_indiv.nonzero()[0]

        train_ds = load_dataset(args["train"]["dataset"], train=True)
        test_ds = load_dataset(args["train"]["dataset"], train=False)

        if args["train"]["private"]:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
            pp_budgets_subset = pp_budgets[keep_indiv]
        else:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
            pp_budgets_subset = None

        m = model = load_model(args["train"]["dataset"], model_name=args["train"]["model"], num_classes=None)
        m = m.to(DEVICE)

        keep_bool = np.full((size), False)
        keep_bool[keep_indiv] = True

        train_dl = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True
        )
        test_dl = DataLoader(
            test_ds, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True
        )

        optim = torch.optim.AdamW(m.parameters(), lr=args["train"]["lr"], weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args["train"]["epochs"])
        criterion = torch.nn.CrossEntropyLoss()

        if args["train"]["private"]:
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args
            )
            pg_sample_rates_list[target_id] = privacy_engine.weights
            pg_noise_multiplier_list[target_id] = [optim.noise_multiplier] * len(privacy_engine.weights)
        else:
            pg_sample_rates_list[target_id] = None
            pg_noise_multiplier_list[target_id] = None

        # Save number of steps
        num_steps = args["train"]["epochs"] * len(train_dl)
        num_steps_list[target_id] = num_steps

        model = train_loop(
            args,
            model,
            train_dl,
            criterion,
            optim,
            sched,
            DEVICE,
            args["train"]["epochs"]
        )

        savedir = os.path.join(args["data"]["savedir_target"], str(seed))
        os.makedirs(savedir, exist_ok=True)
        np.save(savedir + "/keep.npy", keep_bool)
        torch.save(m.state_dict(), savedir + "/model.pt")
        with open(os.path.join(savedir, "bti.npy"), "wb") as f:
            np.save(f, budget_to_index, allow_pickle=True)

    for target_id in range(args["train"]["n_targets"]):
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            def worker_with_stream(target_id=target_id, stream=stream):
                with torch.cuda.stream(stream):
                    # try: 
                    worker(target_id, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, num_steps_list)
                    # except Exception as e:
                        # print(f"Error occurred in worker thread for target {target_id}: {e}")
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
    os.makedirs(args["data"]["savedir_result"], exist_ok=True)
    np.save(os.path.join(args["data"]["savedir_result"], "pg_sample_rates_list.npy"), pg_sample_rates_list, allow_pickle=True)
    np.save(os.path.join(args["data"]["savedir_result"], "pg_noise_multiplier_list.npy"), pg_noise_multiplier_list, allow_pickle=True)
    np.save(os.path.join(args["data"]["savedir_result"], "num_steps_list.npy"), num_steps_list, allow_pickle=True)

    return seeds_out, num_steps_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cf",
        type=str,
        default="/vol/miltank/users/kaiserj/Clipping_vs_Sampling/scripts_experiments/mia/config_files/rmia_mnist_4k.yaml",
        help="Yaml file which contains the configurations",
    )

    # Load the parameters
    args, unknown = parser.parse_known_args()
    with open(args.cf, "rb") as f:
        args = yaml.safe_load(f)

    args = parse_extra(parser, args) # parsing more stuff
    args["train"]["target_delta"] = float(args["train"]["target_delta"])
    for seed in range(args["train"]["seeds"]):
        args["data"]["savedir_target"] = args["data"]["savedir"] + f"/{str(seed)}/target"
        args["data"]["savedir_result"] = args["data"]["savedir"] + f"/{str(seed)}/result"
        _ = train_target_models(seed)
