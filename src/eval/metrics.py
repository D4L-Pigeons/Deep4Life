import sklearn


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
    if prediction_probability is not None:
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
    else:
        average_precision_per_cell_type = None
        roc_auc_per_cell_type = None
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
