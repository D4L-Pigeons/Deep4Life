import argparse
import datetime
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import torch
import yaml
from sklearn.model_selection import KFold

from datasets.data_utils import load_full_anndata
from models.custom_stellar import CustomStellarReduced
from models.ModelBase import ModelBase
from models.sklearn_mlp import SklearnMLP
from models.torch_mlp import TorchMLP
from models.sklearn_svm import SVMSklearnSVC
from models.vanilla_stellar import VanillaStellarReduced
from models.xgboost import XGBoostModel
from utils import cross_validation

CONFIG_PATH: Path = Path(__file__).parent / "config"
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument("--dataset-path", default="data/train", help="dataset path")
    parser.add_argument(
        "--method",
        default="stellar",
        choices=["stellar", "torch_mlp", "sklearn_mlp", "xgboost", "sklearn_svm/svc"],
    )
    parser.add_argument(
        "--config",
        default="standard",
        help="Name of a configuration in src/config/{method} directory.",
    )
    parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    parser.add_argument(
        "--n-folds", default=5, help="Number of folds in cross validation."
    )
    parser.add_argument("--test", action="store_true", help="Test mode.")
    parser.add_argument(
        "--retrain", default=True, help="Retrain a model using the whole dataset."
    )

    args = parser.parse_args()

    config = load_config(args)
    model = create_model(args, config)

    data = load_full_anndata(test=args.test)

    cross_validation_metrics = cross_validation(
        data, model, random_state=args.cv_seed, n_folds=args.n_folds
    )

    # Save results
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create directories if they don't exist
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    results_path = RESULTS_PATH / args.method
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_path = (
        RESULTS_PATH
        / args.method
        / f"{args.config}_{formatted_time}_seed_{args.cv_seed}_folds_{args.n_folds}"
    )
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Save config
    with open(results_path / "config.yaml", "w") as file:
        yaml.dump(config.__dict__, file)
        print(f"Config saved to: {results_path / 'config.yaml'}")

    # Save metrics
    cross_validation_metrics.to_json(results_path / "metrics.json", indent=4)
    print(f"Metrics saved to: {results_path / 'metrics.json'}")

    # Retrain and save model
    if args.retrain:
        print("Retraining model...")
        model.train(data)
        saved_model_path = model.save(str(results_path / "saved_model"))
        print(f"Model saved to: {saved_model_path}")


def load_config(args) -> argparse.Namespace:
    with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
        config = yaml.safe_load(file)

    return argparse.Namespace(**config)


def create_model(args, config) -> ModelBase:
    if args.method == "stellar":
        if args.config == "standard":
            return VanillaStellarReduced(config)
        else:
            return CustomStellarReduced(config)
    elif args.method == "torch_mlp":
        return TorchMLP(config)
    elif args.method == "sklearn_mlp":
        return SklearnMLP(config)
    elif args.method == "xgboost":
        return XGBoostModel(config)
    elif args.method == "sklearn_svm/svc":
        return SVMSklearnSVC(vars(config))
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")


if __name__ == "__main__":
    main()
