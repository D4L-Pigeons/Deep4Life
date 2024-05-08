import torch
import anndata
from sklearn.model_selection import KFold
from datasets.data_utils import load_full_anndata
from models.ModelBase import ModelBase
from models.xgboost import XGBoostModel
import argparse
import datetime
import os
from pathlib import Path
import anndata
import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import torch
from models.vanilla_stellar import VanillaStellarReduced
from models.custom_stellar import CustomStellarReduced
import yaml
from sklearn.model_selection import KFold
from models.sklearn_mlp import SklearnMLP
from models.torch_mlp import TorchMLP

CONFIG_PATH: Path = Path(__file__).parent / "config"
RESULTS_PATH: Path = Path(__file__).parent.parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument("--dataset-path", default="data/train", help="dataset path")
    parser.add_argument(
        "--method",
        default="stellar",
        choices=["stellar", "torch_mlp", "sklearn_mlp", "xgboost"],
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

    # Create directories if it doesn't exist
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    results_path = RESULTS_PATH / args.method
    if not os.path.exists(results_path):
        os.mkdir(results_path)

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
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")


def cross_validation(
    data: anndata.AnnData, model: ModelBase, random_state: int = 42, n_folds: int = 5
) -> pd.DataFrame:
    torch.manual_seed(42)

    metrics_names = [
        "f1_score_per_cell_type",
        "f1_score",
        "accuracy",
        "average_precision_per_cell_type",
        "roc_auc_per_cell_type",
        "confusion_matrix",
    ]

    cross_validation_metrics = pd.DataFrame(columns=metrics_names)

    for i, (train_data, test_data) in enumerate(k_folds(data, n_folds, random_state)):
        model.train(train_data)
        prediction = model.predict(test_data)
        prediction_probability = model.predict_proba(test_data)
        ground_truth = test_data.obs["cell_labels"]

        calculate_metrics(
            ground_truth,
            prediction,
            prediction_probability,
            test_data.obs["cell_labels"].cat.categories,
            cross_validation_metrics,
        )

        print(
            f"Validation accuracy of {i} fold:",
            cross_validation_metrics.loc[i]["accuracy"],
        )

    average_metrics = {
        metric_name: cross_validation_metrics[metric_name].mean()
        for metric_name in metrics_names
    }
    cross_validation_metrics.loc[len(cross_validation_metrics.index)] = average_metrics

    return cross_validation_metrics


def k_folds(data: anndata.AnnData, n_folds: int, random_state: int):
    sample_ids = data.obs["sample_id"].cat.remove_unused_categories()
    sample_ids_unique = sample_ids.cat.categories

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    split = kfold.split(sample_ids_unique.tolist())

    for train, test in split:
        train_mask = data.obs["sample_id"].isin(sample_ids_unique[train])
        test_mask = data.obs["sample_id"].isin(sample_ids_unique[test])

        yield data[train_mask], data[test_mask]


# def macro_average_precision(ground_truth, prediction_probability):
#   """
#   Calculates macro-averaged precision for multi-class classification.

#   Args:
#       ground_truth (array-like): Array of true labels.
#       prediction_probability (array-like): Array of predicted class probabilities.

#   Returns:
#       float: Macro-averaged precision score.
#   """
#   num_classes = len(ground_truth.unique())

#   precision_per_class = []

#   # Calculate precision score for each class
#   for class_label in range(num_classes):
#     class_mask = ground_truth == class_label
#     ground_truth_filtered = ground_truth[class_mask]
#     prediction_probability_filtered = prediction_probability[class_mask]
#     # Calculate precision for this class
#     precision = sklearn.metrics.precision_score(ground_truth_filtered, prediction_probability_filtered[:, class_label], average='binary', zero_division=0)
#     precision_per_class.append(precision)

#   # Macro-average the precision scores
#   macro_average_precision = np.mean(precision_per_class)
#   return macro_average_precision



def calculate_metrics(
    ground_truth, prediction, prediction_probability, classes, cross_validation_metrics
):
    f1_score_per_cell_type = sklearn.metrics.f1_score(
        ground_truth, prediction, labels=classes, average=None
    )
    f1_score = sklearn.metrics.f1_score(
        ground_truth, prediction, labels=classes, average="macro"
    )
    accuracy = sklearn.metrics.accuracy_score(ground_truth, prediction)
    average_precision_per_cell_type = sklearn.metrics.average_precision_score(
        ground_truth, prediction_probability, average=None
    )
    roc_auc_per_cell_type = sklearn.metrics.roc_auc_score(
        ground_truth,
        prediction_probability,
        multi_class="ovr",
        average=None,
        labels=classes,
    )
    confusion_matrix = sklearn.metrics.confusion_matrix(
        ground_truth, prediction, labels=classes
    )

    metrics = {
        "f1_score_per_cell_type": f1_score_per_cell_type,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "average_precision_per_cell_type": average_precision_per_cell_type,
        "roc_auc_per_cell_type": roc_auc_per_cell_type,
        "confusion_matrix": confusion_matrix,
    }

    cross_validation_metrics.loc[len(cross_validation_metrics.index)] = metrics


if __name__ == "__main__":
    main()
