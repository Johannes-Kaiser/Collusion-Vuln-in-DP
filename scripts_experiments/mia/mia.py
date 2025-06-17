# Adaptation of PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import scipy
from mia_utils import train_loop, load_dataset, load_model, save_logits, load_stats, load_data, fig_fpr_tpr
from tqdm import tqdm
from opacus_new import PrivacyEngine

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnistbinary", type=str)
parser.add_argument("--model", default=None, type=str)
parser.add_argument("--lr", default=0.02, type=float)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--n_shadows", default=32, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="./exp_mia/mnistbinary", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable private mode (default: True)")
parser.add_argument("--adapt_weights_to_budgets", action=argparse.BooleanOptionalAction, default=True, help="Adapt weights from budgets (default: True)")
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


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
        

def train_models():
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

    if args.private:
        if hasattr(args, "indiv_budgets") and hasattr(args, "indiv_portions") and args.indiv_budgets is not None and args.indiv_portions is not None:
            # Compute pp_budgets according to indiv_budgets and indiv_portions
            total = len(dummy_train_ds)
            pp_budgets = []
            for budget, portion in zip(args.indiv_budgets, args.indiv_portions):
                count = int(portion * total)
                pp_budgets.extend([budget] * count)
            # If rounding caused a mismatch, pad or trim
            if len(pp_budgets) < total:
                pp_budgets.extend([args.indiv_budgets[-1]] * (total - len(pp_budgets)))
            elif len(pp_budgets) > total:
                pp_budgets = pp_budgets[:total]
        else:
            pp_budgets = [args.target_epsilon] * len(dummy_train_ds)

    for shadow_id in tqdm(range(args.n_shadows), desc="Shadow Models"):
        seed = np.random.randint(0, 1000000000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        keep_indiv = np.array(keep[shadow_id], dtype=bool)
        keep_indiv = keep_indiv.nonzero()[0]
        pp_budgets_indiv = np.array(pp_budgets)[keep_indiv]
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

        privacy_engine = None
        if args.private:               
            model, optim, train_dl, privacy_engine = make_private(
                model, train_dl, optim, pp_budgets_indiv, args
            )


        model = train_loop(
            model,
            train_dl,
            criterion,
            optim,
            sched,
            DEVICE,
            args.epochs,
            privacy_engine=privacy_engine,
        )

        savedir = os.path.join(args.savedir, str(shadow_id))
        os.makedirs(savedir, exist_ok=True)
        np.save(savedir + "/keep.npy", keep_bool)
        torch.save(m.state_dict(), savedir + "/model.pt")
        


def inference():
    train_ds = load_dataset(args.dataset, train=True)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    # Infer the logits with multiple queries
    for path in os.listdir(args.savedir):
        m = load_model(args.dataset, model_name=args.model, num_classes=None)
        m.load_state_dict(torch.load(os.path.join(args.savedir, path, "model.pt")))
        m.to(DEVICE)
        save_logits(args, m, train_dl, DEVICE, path)
    return


if __name__ == "__main__":

    train_models()
    inference()
    load_stats(args)
    keep, scores = load_data(args)
    fig_fpr_tpr(args, keep, scores)
