# Adaptation of PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)
import sys
sys.path.append("/vol/miltank/users/kaiserj/Clipping_vs_Sampling/")
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.utils_general import (
    load_dataset,
    load_model,
    parse_extra
)
from tqdm import tqdm
import yaml

DEVICE = "cpu"


def save_logits(args, model, train_dl, DEVICE, savedir, path):
    """
    Computes and saves logits for all data in the dataloader.

    Args:s
        model: PyTorch model.
        dataloader: DataLoader providing the data.
        device: Device to run the model on.
        save_path: Path to save the logits (as a .pt file).
    """
    model.eval()

    logits_n = []
    for i in range(args["train"]["n_queries"]):
        logits = []
        for x, _ in train_dl:
            x = x.to(DEVICE)
            outputs = model(x)
            logits.append(outputs.detach().cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)

    np.save(os.path.join(savedir, path, "logits.npy"), logits_n)

def inference(savedir):
    train_ds = load_dataset(args["train"]["dataset"], train=True)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    # Infer the logits with multiple queries
    for path in tqdm(os.listdir(savedir)):
        m = load_model(args["train"]["dataset"], model_name=args["train"]["model"], num_classes=None)
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(DEVICE)
        save_logits(args, m, train_dl, DEVICE, savedir, path)
    return

def save_targets(args):
    train_ds = load_dataset(args["train"]["dataset"], train=True)
    targets = train_ds.targets
    targets = targets.cpu().numpy()  # use .cpu() in case the tensor is on GPU
    path = args["data"]["savedir_result"] + '/y_train.npy'
    np.save(path, targets)
    return


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
        args = yaml.load(f, Loader=yaml.Loader)

    args = parse_extra(parser, args) # parsing more stuff

    # seeds = [folder for folder in os.listdir(args["data"]["savedir_target"]) if os.path.isdir(os.path.join(args["data"]["savedir_target"], folder))]
    for seed in range(args["train"]["seeds"]):
        args["data"]["savedir_target"] = args["data"]["savedir"] + f"/{str(seed)}/target"
        args["data"]["savedir_result"] = args["data"]["savedir"] + f"/{str(seed)}/result"
        save_targets(args) 
        inference(args["data"]["savedir_target"])

