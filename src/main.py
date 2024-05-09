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
from eval.metrics import calculate_metrics

CONFIG_PATH: Path = Path(__file__).parent / "config"
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"
FINAL_RESULTS_PATH: Path = Path(__file__).parent.parent / "best_results"


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Script to train or test a model.")
    subparsers = parser.add_subparsers(help="Train, test or validate a model.", dest="mode", required=True)
    
    parser.add_argument(
        "--method",
        default="stellar",
        choices=["stellar", "torch_mlp", "sklearn_mlp", "xgboost", "sklearn_svm/svc"],
    )
    parser.add_argument(
        "--data",
        default="train",
        choices=["train", "test"],
        help="Indicates the directory from which to read data: data/train or data/test.",
    )
    parser.add_argument(
        "--config",
        default="standard",
        help="Name of a configuration in src/config/{method} directory.",
    )
    
    # Train model
    train_parser = subparsers.add_parser("train", help="Train a model.")
    
    # Validate model
    validate_parser = subparsers.add_parser("validate", help="Validate a model using cross validation.")
    validate_parser.add_argument(
        "--cv-seed", default=42, help="Seed used to make k folds for cross validation."
    )
    validate_parser.add_argument(
        "--n-folds", default=5, help="Number of folds in cross validation."
    )
    validate_parser.add_argument(
        "--retrain", default=True, help="Retrain a model using the whole dataset."
    )
    
    # Test
    test_parser = subparsers.add_parser("test", help="Test a model.")
    test_parser.add_argument(
        "--model_name",
        default="",
        help="Name of the model to test (subdirectory of results containing the model and its config).",
    )
    test_parser.add_argument(
        "--evaluate",
        default=True,
        help="Indicates if the evaluation will be made based on true values.",
    )
    test_parser.add_argument(
        "--final",
        default=True,
        # help="Indicates if the evaluation will be made based on true values.",
    )


    args = parser.parse_args()
    config = load_config(args)
    model = create_model(args, config)
    test_data = args.data == "test"
    data = load_full_anndata(test=test_data)

    # Create directories if they don't exist
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    if args.mode == 'train':
        train_model(args, model, data, config)
    elif args.mode == 'test':
        test_model(args, model, data)
    elif args.mode == 'validate':
        validate_model(args, model, data, config)
        

def train_model(args, model, data, config):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    results_path = (
        RESULTS_PATH
        / args.method
        / f"{args.config}_{formatted_time}_seed_{args.cv_seed}_folds_{args.n_folds}"
    )
    os.makedirs(results_path, exist_ok=True)

    model.train(data)
    saved_model_path = model.save(str(results_path / "saved_model"))
    print(f"Model saved to: {saved_model_path}")
    
    # Save config
    with open(results_path / "config.yaml", "w") as file:
        yaml.dump(config.__dict__, file)
        print(f"Config saved to: {results_path / 'config.yaml'}")
        
    # Save predictions
    prediction = model.predict(data)
    prediction_probability = model.predict_proba(data)

    # Save prediction and prediction probability
    np.save(results_path / "train_prediction.npy", prediction)
    np.save(results_path / "train_prediction_probability.npy", prediction_probability)

def test_model(args, model, data):
    if args.final:
        results_path = FINAL_RESULTS_PATH / args.method / args.model_name
    else:
        results_path = RESULTS_PATH / args.method / args.model_name
    
    model_path = results_path / "saved_model"
    model.load(str(model_path))
    prediction = model.predict(data)
    prediction_probability = model.predict_proba(data)

    # Save prediction and prediction probability
    np.save(results_path / "test_prediction.npy", prediction)
    np.save(results_path / "test_prediction_probability.npy", prediction_probability)
    
    if args.evaluate:
        metrics_names = [
            "f1_score_per_cell_type",
            "f1_score",
            "accuracy",
            "average_precision_per_cell_type",
            "roc_auc_per_cell_type",
            "confusion_matrix",
        ]

        validation_metrics = pd.DataFrame(columns=metrics_names)
        ground_truth = data.obs["cell_labels"]

        calculate_metrics(
            ground_truth,
            prediction,
            prediction_probability,
            data.obs["cell_labels"].cat.categories,
            validation_metrics,
        )
        
        validation_metrics.to_json(results_path / "test_metrics.json", indent=4)
        print(f"Test metrics saved to: {results_path / 'test_metrics.json'}")
        
def validate_model(args, model, data, config):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    results_path = (
        RESULTS_PATH
        / args.method
        / f"{args.config}_{formatted_time}_seed_{args.cv_seed}_folds_{args.n_folds}"
    )
    os.makedirs(results_path)

    cross_validation_metrics = cross_validation(
        data, model, random_state=args.cv_seed, n_folds=args.n_folds
    )

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
    if args.mode != "test":
        with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
            config = yaml.safe_load(file)
    else:
        if args.final:
            config_path = FINAL_RESULTS_PATH / args.method / args.model_name / "config.yaml"
        else:
            config_path = RESULTS_PATH / args.method / args.model_name / "config.yaml"
        with open(config_path) as file:
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
