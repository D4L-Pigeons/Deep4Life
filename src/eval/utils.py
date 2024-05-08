import pandas as pd
import anndata
import torch
from sklearn.model_selection import KFold
from eval.metrics import calculate_metrics
from models.ModelBase import ModelBase


def cross_validation(
    data: anndata.AnnData, model: ModelBase, random_state: int = 42, n_folds: int = 5
) -> pd.DataFrame:

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
