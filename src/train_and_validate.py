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
from eval.utils import cross_validation

CONFIG_PATH: Path = Path(__file__).parent / "config"
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Script to train or test a model.")
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

    subparsers = parser.add_subparsers(help="Train or test a model.", dest="mode")
    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    train_parser.add_argument(
        "--n-folds", default=5, help="Number of folds in cross validation."
    )
    train_parser.add_argument(
        "--retrain", default=True, help="Retrain a model using the whole dataset."
    )

    test_parser = subparsers.add_parser("test", help="Test a model.")
    test_parser.add_argument(
        "model_name",
        help="Name of the model to test (subdirectory of src/results containing the model and its config).",
    )

    args = parser.parse_args()
    config = load_config(args)
    model = create_model(args, config)
    test_mode = args.mode == "test"
    data = load_full_anndata(test=test_mode)

    # Create directories if they don't exist
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    results_path = RESULTS_PATH / args.method
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save results
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    if not test_mode:
        results_path = (
            RESULTS_PATH
            / args.method
            / f"{args.config}_{formatted_time}_seed_{args.cv_seed}_folds_{args.n_folds}"
        )
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        cross_validation_metrics = cross_validation(
            data, model, random_state=args.cv_seed, n_folds=args.n_folds
        )
        # Save results
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

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

    else:
        model_path = RESULTS_PATH / args.method / args.model_name / "saved_model"
        model.load(model_path)
        prediction = model.predict(data)
        prediction_probability = model.predict_proba(data)

        # Save prediction and prediction probability
        np.save(results_path / "prediction.npy", prediction)
        np.save(results_path / "prediction_probability.npy", prediction_probability)


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

# cp first 10 from data/train/images_masks/img to data/test/images_masks/img
# ! ls data/train/images_masks/img | head -10 | xargs -I {} cp data/train/images_masks/img/{} data/test/images_masks/img
