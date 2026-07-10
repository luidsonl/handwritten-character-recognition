import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from .data_loader import EMNIST_BALANCED_LABELS


EvaluationResults = dict[str, float | NDArray[np.int64]]


def evaluate_model(model, x_test: NDArray, y_test: NDArray) -> EvaluationResults:
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
    }


ConfusedPair = tuple[str, str, int]


def get_most_confused_pairs(
    confusion_matrix: NDArray[np.int64],
    labels: list[str] | None = None,
    top_n: int = 10,
) -> list[ConfusedPair]:
    if labels is None:
        labels = EMNIST_BALANCED_LABELS

    cm = confusion_matrix.copy()
    np.fill_diagonal(cm, 0)

    pairs: list[ConfusedPair] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i][j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i][j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def print_evaluation_report(results: EvaluationResults) -> None:
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print()
    print("Most confused pairs (true -> predicted, count):")
    for true_label, pred_label, count in get_most_confused_pairs(results['confusion_matrix']):
        print(f"  '{true_label}' -> '{pred_label}': {count}")
