# Adaptation of PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)
import sys
sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")

import os
import time
import json
import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from utils.utils_general import (
    parse_args_with_yaml,
    load_dataset, 
    load_model, 
    save_logits, 
    train_loop, 
    make_private, 
    get_dataset_size,
    get_budget_to_index_from_seeds,
    format_list,
    generate_pp_budgets, 
    generate_string,
    to_padded_array,
    assign_pp_values,
    PrefetchedLoader
)
from utils.utils_mia import (
    generate_biregular_binary_matrix_random,
    score_mia,
    load_data, 
    plot_and_save_samplewise_auc,
    plot_and_save_integrals,
    load_pg_lists,
    fit_mia_in_out_gaussians,
    compute_individual_scores    
)

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda")  # if torch.cuda.is_available() else torch.device("cpu")




def mp_worker(target_id, train_ds, NM_mp, SR_mp, CW_mp, keep, size, args, DEVICE, pp_budgets, budget_to_index, seeds_out, pg_sample_rates_list, pg_noise_multiplier_list, pg_cw_list, num_steps_list):
    # Each process needs its own seed and device context
    # print(f"starting {target_id}")
    try:
        num_existing_models = sum(os.path.isdir(os.path.join(args.savedir_target, d)) for d in os.listdir(args.savedir_target))
        targets_to_compute = list(set(range(args.n_targets)) - set(range(num_existing_models)))
    except FileNotFoundError:
        targets_to_compute = list(range(args.n_targets))
    if targets_to_compute == 0:
        return
    created_folder = False
    while not created_folder:
        seed = args.seed*10000 + int(time.time() * 1e6) % (2**32)
        savedir = os.path.join(args.savedir_target, str(seed))
        if os.path.isdir(savedir):
            continue
        else:
            created_folder = True
    os.makedirs(savedir, exist_ok=True)
    target_id = len(targets_to_compute) - 1
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

    if args.private:
        train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
        pp_budgets_subset = pp_budgets[keep_indiv]
        unique, counts = np.unique(pp_budgets_subset, return_counts=True)
        str_budget = generate_string(unique, counts)
    else:
        train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
        pp_budgets_subset = None
        str_budget = "non_private"


    m = model = load_model(args.dataset, model_name=args.model, num_classes=None)
    m = m.to(DEVICE)

    keep_bool = np.full((size), False)
    keep_bool[keep_indiv] = True

    train_dl = DataLoader(
        train_ds, batch_size=args.batchsize, shuffle=True, num_workers=0  # , persistent_workers=True
    )
    train_dl = PrefetchedLoader(train_dl, device=DEVICE)

    optim = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.private:
        use_cached = str_budget in SR_mp and str_budget in NM_mp and str_budget in CW_mp
        if use_cached:
            nm_local = NM_mp[str_budget]
            sr_local = SR_mp[str_budget]
            cw_local = CW_mp[str_budget]
        if use_cached:
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args,
                adapt_weights_to_budgets=False, nm=nm_local, sr=sr_local, cw=cw_local
            )
        else:
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args
            )
        if args.individualize == "sampling":
            sr = privacy_engine.weights
            nm = [optim.noise_multiplier] * len(privacy_engine.weights)
            cw = [args.max_grad_norm] * len(privacy_engine.weights)
            pg_sample_rates_list[target_id] = sr
            pg_noise_multiplier_list[target_id] = nm
            pg_cw_list[target_id] = cw
            if str_budget not in SR_mp:
                SR_mp[str_budget] = pg_sample_rates_list[target_id]
            if str_budget not in NM_mp:
                NM_mp[str_budget] = pg_noise_multiplier_list[target_id]
            if str_budget not in CW_mp:
                CW_mp[str_budget] = pg_cw_list[target_id]
        else:
            sr = [1 / len(train_dl)] * len(privacy_engine.weights)
            nm = [optim.noise_multiplier] * len(privacy_engine.weights)
            cw = privacy_engine.weights
            pg_sample_rates_list[target_id] = sr
            pg_noise_multiplier_list[target_id] = nm
            pg_cw_list[target_id] = cw
            if str_budget not in SR_mp:
                SR_mp[str_budget] = pg_sample_rates_list[target_id]
            if str_budget not in NM_mp:
                NM_mp[str_budget] = pg_noise_multiplier_list[target_id]
            if str_budget not in CW_mp:
                CW_mp[str_budget] = pg_cw_list[target_id]
            
    else:
        pg_sample_rates_list[target_id] = None
        pg_noise_multiplier_list[target_id] = None

    num_steps = args.epochs * len(train_dl)
    num_steps_list[target_id] = num_steps
    if args.individualize == "clipping":
        pp_max_grad_norms = assign_pp_values(pp_budgets_subset, privacy_engine.weights) # sample rate is clipping range in that setting
    else:
        pp_max_grad_norms = None
    model = train_loop(
        args,
        model,
        train_dl,
        criterion,
        optim,
        sched,
        DEVICE,
        args.epochs,
        pp_max_grad_norms
    )
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")
    np.save(os.path.join(savedir, "pg_sample_rates.npy"), sr, allow_pickle=True)
    np.save(os.path.join(savedir, "pg_noise_multiplier.npy"), nm, allow_pickle=True)
    np.save(os.path.join(savedir, "pg_cw.npy"), cw, allow_pickle=True)
    np.save(os.path.join(savedir, "num_steps.npy"), num_steps, allow_pickle=True)
    with open(os.path.join(savedir, "bti.npy"), "wb") as f:
        np.save(f, budget_to_index, allow_pickle=True)        

def train_target_models(portions, ctx, manager, SR_mp, NM_mp, CW_mp):

    if args.private:
        size = get_dataset_size(args.dataset, train=True, num_max_per_class_samples=args.num_max_per_class_samples)
        pp_budgets, budget_to_index = generate_pp_budgets(1, size, portions, args.budgets)

    if args.private:
        index_order = []
        keep_tensors = []
        for _, indices in budget_to_index.items():
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
        keep = np.random.uniform(0, 1, size=(args.n_targets, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_targets)


    seeds_out = [0] * args.n_targets
    pg_sample_rates_list = [None] * args.n_targets
    pg_noise_multiplier_list = [None] * args.n_targets
    pg_cw_list = [None] * args.n_targets
    num_steps_list = [None] * args.n_targets
    n_parallel = getattr(args, "n_parallel", 1)
  


    # Use managed lists for shared memory between processes
    seeds_out_mp = manager.list([None] * args.n_targets)
    pg_sample_rates_list_mp = manager.list([None] * args.n_targets)
    pg_noise_multiplier_list_mp = manager.list([None] * args.n_targets)
    pg_cw_list_mp = manager.list([None] * args.n_targets)
    num_steps_list_mp = manager.list([None] * args.n_targets)

    train_ds = load_dataset(args.dataset, train=True, num_max_samples=args.num_max_per_class_samples)#
    try:
        num_existing_models = sum(os.path.isdir(os.path.join(args.savedir_target, d)) for d in os.listdir(args.savedir_target))
        targets_to_compute = list(set(range(args.n_targets)) - set(range(num_existing_models)))
    except FileNotFoundError:#
        targets_to_compute = list(range(args.n_targets))
    if n_parallel > 1:
        processes = []
        for target_id in targets_to_compute:
            p = ctx.Process(
                target=mp_worker,
                args=(
                    target_id,
                    train_ds,
                    NM_mp,
                    SR_mp,
                    CW_mp,
                    keep,
                    size,
                    args,
                    DEVICE,
                    pp_budgets,
                    budget_to_index,
                    seeds_out_mp,
                    pg_sample_rates_list_mp,
                    pg_noise_multiplier_list_mp,
                    pg_cw_list,
                    num_steps_list_mp,
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
                    active_processes.append(p)
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
                    active_processes[0].join(timeout=0.5)
        for p in active_processes:
            p.join()
    else:
        for target_id in targets_to_compute:
            mp_worker(target_id,
                    train_ds,
                    NM_mp,
                    SR_mp,
                    CW_mp,
                    keep,
                    size,
                    args,
                    DEVICE,
                    pp_budgets,
                    budget_to_index,
                    seeds_out_mp,
                    pg_sample_rates_list_mp,
                    pg_noise_multiplier_list_mp,
                    pg_cw_list_mp,
                    num_steps_list_mp)

    # Copy managed lists back to numpy arrays/lists
    for i in range(args.n_targets):
        pg_sample_rates_list[i] = pg_sample_rates_list_mp[i]
        pg_noise_multiplier_list[i] = pg_noise_multiplier_list_mp[i]
        num_steps_list[i] = num_steps_list_mp[i]
        pg_cw_list[i] = pg_cw_list_mp[i]
    pg_sample_rates_list = to_padded_array(pg_sample_rates_list)
    pg_noise_multiplier_list = to_padded_array(pg_noise_multiplier_list)
    num_steps_list = to_padded_array(num_steps_list)
    pg_cw_list = to_padded_array(pg_cw_list)
    # Save the lists as .npy files in the results directory
    if np.max(num_steps_list) != 0:
        np.save(os.path.join(args.savedir_result, "pg_sample_rates_list.npy"), pg_sample_rates_list, allow_pickle=True)
        np.save(os.path.join(args.savedir_result, "pg_noise_multiplier_list.npy"), pg_noise_multiplier_list, allow_pickle=True)
        np.save(os.path.join(args.savedir_result, "pg_cw_list.npy"), pg_cw_list, allow_pickle=True)
        np.save(os.path.join(args.savedir_result, "num_steps_list.npy"), num_steps_list, allow_pickle=True)

    return seeds_out, num_steps_list

def test_models(args, savedir, device):
    test_ds = load_dataset(args.dataset, train=False)
    test_dl = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=0)  # num_workers=0
    # Infer the logits with multiple queries
    accuracies = []
    for path in os.listdir(savedir)[:10]:
        try:
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
            accuracies.append(accuracy)
            # print(f"Accuracy for model in {path}: {accuracy:.4f}")
            np.save(os.path.join(savedir, path, "accuracy.npy"), np.array([accuracy]))
            del m
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            continue
    print(f"{savedir}")
    print(f"Mean accuracy over {len(accuracies)} models: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    del test_dl, test_ds
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return

def inference(savedir):
    train_ds = load_dataset(args.dataset, train=True, num_max_samples=args.num_max_per_class_samples)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    # Infer the logits with multiple queries
    for path in tqdm(os.listdir(savedir)):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(DEVICE)
        save_logits(args, m, train_dl, DEVICE, savedir, path)
    return


if __name__ == "__main__":

    args = parse_args_with_yaml()
    if args.n_parallel > 1:
        args.disable_inner = True
    args.batchsize = getattr(args, 'batchsize', 128)
    args.num_max_per_class_samples = getattr(args, 'num_max_per_class_samples', None)
    if args.num_max_per_class_samples is None:
        num_max_per_class_samples_name_ext = ""
    else:
        num_max_per_class_samples_name_ext = f"_{args.num_max_per_class_samples}"
    # Use multiprocessing to spawn processes
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    SR_mp = manager.dict()  # <-- managed dict for SR
    NM_mp = manager.dict()  # <-- managed dict for NM
    CW_mp = manager.dict()  # <-- managed dict for CW
    print(f"Looking at dataset {args.dataset}")
    for portion in args.portions:
        portions = [portion, 1-portion]
        default_path = f"{args.model}/{args.dataset}{num_max_per_class_samples_name_ext}/{args.dataset}_[{format_list(args.budgets)}]_[{format_list(portions)}]/{str(args.seed)}"
        args.savedir = f"./exp_mia_final_{args.individualize}_{args.name_ext}/{default_path}/shadow"
        args.savedir_target = f"./exp_mia_final_{args.individualize}_{args.name_ext}/{default_path}/target"
        args.savedir_result = f"./exp_mia_final_{args.individualize}_{args.name_ext}/{default_path}/results"

        # Save the args dict as a JSON file in savedir_result
        os.makedirs(args.savedir_result, exist_ok=True)
        args_dict = vars(args)
        with open(os.path.join(args.savedir_result, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=2)

        samplewise_auc = {}
        samplewise_auc_R = {}
        integrals_all = {}
        adv_all = {}
        priv_all = {}

        if args.train_and_save_models:
            _ = train_target_models(portions, ctx, manager, SR_mp, NM_mp, CW_mp)
        if args.test_model:
            test_models(args, args.savedir_target, DEVICE)
        if args.compute_and_save_logits:
            inference(args.savedir_target)
        if args.compute_and_save_scores:
            score_mia(args, args.savedir_target)
        if args.compute_and_save_stats:
            keep, scores = load_data(args.savedir_target)
            # Load the saved budget_to_index dictionary for each target model
            seeds = [folder for folder in os.listdir(args.savedir_target) if os.path.isdir(os.path.join(args.savedir_target, folder))]
            first_bti = get_budget_to_index_from_seeds(args.savedir_target, seeds)
            
                        
            for key in first_bti.keys():
                indices = first_bti[key]

                keep_budget = keep[:, indices]
                scores_budget = scores[:, indices, :]
                mean_in, mean_out, std_in, std_out = fit_mia_in_out_gaussians(keep_budget, scores_budget)
                indiv_scores_val, x_vals, samplewise_R, integrals, adv, priv_scores = compute_individual_scores(mean_in, mean_out, std_in, std_out)

                samplewise_auc[key] = indiv_scores_val
                samplewise_auc_R[key] = samplewise_R
                integrals_all[key] = integrals
                adv_all[key] = adv
                priv_all[key] = priv_scores

            # Save the dictionaries as .npy files in the results directory
            os.makedirs(args.savedir_result, exist_ok=True)
            np.save(os.path.join(args.savedir_result, "samplewise_auc.npy"), samplewise_auc, allow_pickle=True)
            np.save(os.path.join(args.savedir_result, "samplewise_auc_R.npy"), samplewise_auc_R, allow_pickle=True)
            np.save(os.path.join(args.savedir_result, "integrals_all.npy"), integrals_all, allow_pickle=True)
            np.save(os.path.join(args.savedir_result, "adv_all.npy"), adv_all, allow_pickle=True)
            np.save(os.path.join(args.savedir_result, "priv_all.npy"), priv_all, allow_pickle=True)
            print(f"Saved samplewise AUCs and integrals to {args.savedir_result}")

        if args.plot_results:
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