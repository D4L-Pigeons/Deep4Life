import argparse
import numpy as np
import os
import torch
from src.utils import prepare_save_dir
from src.models.STELLAR import STELLAR
from src.datasets.datasets import GraphDataset
from src.datasets.load_d4ls import load_d4ls_data, load_d4ls_graph_data
from src.models.linear import LinearModel
from src.models.linear import XGBoostModel


def run_linear(args, dataset):
    linear_model = LinearModel(args, dataset)
    linear_model.train()
    _, results = linear_model.pred()
    np.save(os.path.join(args.savedir, args.dataset + "_results.npy"), results)


def run_xgboost(args, dataset):
    linear_model = XGBoostModel(args, dataset)
    linear_model.train()
    _, results = linear_model.pred()
    np.save(os.path.join(args.savedir, args.dataset + "_results.npy"), results)


def run_stellar(args, dataset):
    stellar = STELLAR(args, dataset)
    stellar.train()
    _, results = stellar.pred()
    np.save(os.path.join(args.savedir, args.dataset + "_results.npy"), results)


def main():
    parser = argparse.ArgumentParser(description="STELLAR")
    parser.add_argument("--dataset", default="d4ls", help="dataset setting")
    parser.add_argument("--dataset-path", default="data/train", help="dataset path")
    parser.add_argument(
        "--method",
        default="stellar",
        help="method chosen (stellar, linear, xgboost, ...)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument("--num-heads", type=int, default=22)
    parser.add_argument("--num-seed-class", type=int, default=0)
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--distance_thres", default=50, type=int)
    parser.add_argument("--savedir", type=str, default="./")
    args = parser.parse_args()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    args.method = "_".join([args.dataset, args.method])
    args = prepare_save_dir(args, __file__)

    if args.dataset == "d4ls":
        data = load_d4ls_data(args.dataset_path)
        (
            labeled_X,
            labeled_y,
            unlabeled_X,
            labeled_edges,
            unlabeled_edges,
            inverse_dict,
        ) = load_d4ls_graph_data()
        dataset = GraphDataset(
            labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges
        )

    if args.method == "stellar":
        run_stellar(dataset, args)
    elif args.method == "linear":
        run_linear(data, args)
    elif args.method == "xgboost":
        run_xgboost(data, args)
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")


if __name__ == "__main__":
    main()
