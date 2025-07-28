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
    train_loop,
    load_dataset,
    load_model,
    save_logits,
    load_stats,
    load_data,
    fig_fpr_tpr,
    fig_fpr_tpr_target,
    indiv_scores,
    plot_and_save_samplewise_auc,
    generate_biregular_binary_matrix_random,
    plot_and_save_integrals,
    load_pg_lists
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
parser.add_argument("--n_targets", default=3, type=int)
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
parser.add_argument("--portions", type=float, nargs="+", default=[0.1, 0.9], help="List of portions (must sum to 1.0)")
parser.add_argument("--budgets", type=float, nargs="+", default=[16.0, 50.0], help="List of epsilon values for budgets")
parser.add_argument("--n_parallel", default=8, type=int)
parser.add_argument("--disable_inner", action=argparse.BooleanOptionalAction, default=False, help="Disable inner parallelism (default: False)")
parser.add_argument("--seed", default=42, type=int, help="Random seed (default: 42)")
args = parser.parse_args()

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