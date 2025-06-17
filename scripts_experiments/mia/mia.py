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
from sklearn.metrics import roc_curve, auc
import scipy
from mia_utils import (
    train_loop,
    load_dataset,
    load_model,
    save_logits,
    load_stats,
    load_data,
    fig_fpr_tpr,
    fig_fpr_tpr_target,
)
from tqdm import tqdm
from opacus_new import PrivacyEngine

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnistbinary", type=str)
parser.add_argument("--model", default=None, type=str)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--n_shadows", default=128, type=int)
parser.add_argument("--n_targets", default=2, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="./exp_mia/mnistbinary_shadow", type=str)
parser.add_argument("--savedir_target", default="./exp_mia/mnistbinary_target", type=str)
parser.add_argument("--savedir_result", default="./exp_mia/mnistbinary_results", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable private mode (default: True)")
parser.add_argument("--adapt_weights_to_budgets", action=argparse.BooleanOptionalAction, default=True, help="Adapt weights from budgets (default: True)")
parser.add_argument("--accountant", default="rdp", type=str, help="Type of privacy accountant (default: RDP)")
parser.add_argument("--individualize", default="sampling", type=str, help="Individualization method (default: sampling)")
parser.add_argument("--weights", default=None)
parser.add_argument("--target_delta", default=1e-5, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
args = parser.parse_args()

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
DEVICE = torch.device("cpu")  # Force CPU for now

def make_private(model, train_loader, optimizer, pp_budgets, args):
    # modulevalidator = ModuleValidator()
    # model = modulevalidator.fix_and_validate(model)

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
                                       max_alpha=10_000)
    else:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private(module=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          noise_multiplier=args.noise_multiplier,
                          max_grad_norm=args.max_grad_norm)
    return private_model, private_optimizer, private_loader, privacy_engine
        

def train_shadow_models():
    seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    args.debug = True

    dummy_train_ds = load_dataset(args.dataset, train=True)
    size = len(dummy_train_ds)

    if args.n_shadows is not None:
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()

    for shadow_id in tqdm(range(args.n_shadows), desc="Shadow Models"):
        seed = np.random.randint(0, 1000000000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        keep_indiv = np.array(keep[shadow_id], dtype=bool)
        keep_indiv = keep_indiv.nonzero()[0]
        train_ds = load_dataset(args.dataset, train=True)
        test_ds = load_dataset(args.dataset, train=False)

        m = model = load_model(args.dataset, model_name=args.model, num_classes=None)
        m = m.to(DEVICE)

        keep_bool = np.full((size), False)
        keep_bool[keep_indiv] = True

        train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
        print(len(train_ds), "samples in shadow dataset", shadow_id)
        train_dl = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True
        )
        test_dl = DataLoader(
            test_ds, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True
        )

        optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()


        model = train_loop(
            model,
            train_dl,
            criterion,
            optim,
            sched,
            DEVICE,
            args.epochs
        )

        savedir = os.path.join(args.savedir, str(shadow_id))
        os.makedirs(savedir, exist_ok=True)
        np.save(savedir + "/keep.npy", keep_bool)
        torch.save(m.state_dict(), savedir + "/model.pt")

def train_target_models():
    args.debug = True

    dummy_train_ds = load_dataset(args.dataset, train=True)
    size = len(dummy_train_ds)

    # Define portions and budgets for privacy
    portions = [0.5, 0.5]  # Example: 50%, 30%, 20%
    budgets = [50.0, 50.0]   # Example: epsilon values
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

    if args.private:
        # For each target, sample a subset of the dataset such that the privacy portions are preserved
        n_keep = int(args.pkeep * size)
        keep = np.zeros((args.n_targets, size), dtype=bool)
        for target_id in range(args.n_targets):
            # For each portion, sample the correct number of indices
            indices = []
            start = 0
            for portion in portions:
                end = start + int(round(portion * size))
                if end > size:
                    end = size
                portion_indices = np.arange(start, end)
                n_portion_keep = int(round(portion * n_keep))
                if len(portion_indices) > 0 and n_portion_keep > 0:
                    chosen = np.random.choice(portion_indices, size=min(n_portion_keep, len(portion_indices)), replace=False)
                    indices.extend(chosen)
                start = end
            # If rounding caused a mismatch, fill up to n_keep
            if len(indices) < n_keep:
                remaining = np.setdiff1d(np.arange(size), indices)
                extra = np.random.choice(remaining, size=n_keep - len(indices), replace=False)
                indices.extend(extra)
            indices = np.array(indices)
            np.random.shuffle(indices)
            keep[target_id, indices] = True
    else:
        # Use the same logic as in train_shadow_models for keep
        if args.n_targets is not None:
            keep = np.random.uniform(0, 1, size=(args.n_targets, size))
            order = keep.argsort(0)
            keep = order < int(args.pkeep * args.n_targets)
        else:
            keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
            keep.sort()

    for target_id in tqdm(range(args.n_targets), desc="Target Models"):
        seed = np.random.randint(0, 1000000000)
        torch.manual_seed(seed)
        np.random.seed(seed)
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

        # Subset the dataset and budgets accordingly
        if args.private:
            # Subset both dataset and budgets
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
            pp_budgets_subset = pp_budgets[keep_indiv]
        else:
            train_ds = torch.utils.data.Subset(train_ds, keep_indiv)
            pp_budgets_subset = None  # Not used

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

        optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()

        if args.private:
            # Make model private using the subsetted budgets
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_subset, args
            )

        model = train_loop(
            model,
            train_dl,
            criterion,
            optim,
            sched,
            DEVICE,
            args.epochs
        )

        savedir = os.path.join(args.savedir_target, str(target_id))
        os.makedirs(savedir, exist_ok=True)
        np.save(savedir + "/keep.npy", keep_bool)
        torch.save(m.state_dict(), savedir + "/model.pt")
    


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

    # train_shadow_models()
    # inference(args.savedir)
    # load_stats(args, args.savedir)
    keep, scores = load_data(args, args.savedir)
    # fig_fpr_tpr(args, keep, scores, args.savedir_result)

    train_target_models()
    inference(args.savedir_target)
    load_stats(args, args.savedir_target)
    keep_target, scores_target = load_data(args, args.savedir_target)
    fig_fpr_tpr_target(args, keep, scores, keep_target, scores_target, args.savedir_result)
