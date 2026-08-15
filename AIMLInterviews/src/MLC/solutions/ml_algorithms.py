"""Interview-sized NumPy reference implementations for common ML coding problems."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
import random

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax over the last dimension."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Return mean multiclass cross-entropy for integer class targets."""
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets)
    if scores.ndim != 2 or labels.shape != (scores.shape[0],):
        raise ValueError("expected logits [N, C] and targets [N]")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("targets must contain integer class indices")
    if np.any((labels < 0) | (labels >= scores.shape[1])):
        raise ValueError("target class index is out of range")

    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probabilities[np.arange(scores.shape[0]), labels].mean())


def linear_regression_gradient_descent(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float = 0.05,
    steps: int = 1_000,
) -> tuple[np.ndarray, float]:
    """Fit linear regression with batch gradient descent and MSE loss."""
    x, y = _validate_supervised_inputs(features, targets)
    if learning_rate <= 0 or steps < 1:
        raise ValueError("learning_rate and steps must be positive")

    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        errors = x @ weights + bias - y
        weights -= learning_rate * (2.0 / x.shape[0]) * (x.T @ errors)
        bias -= learning_rate * 2.0 * float(errors.mean())
    return weights, bias


def logistic_regression_gradient_descent(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float = 0.1,
    steps: int = 1_000,
) -> tuple[np.ndarray, float]:
    """Fit binary logistic regression with batch gradient descent."""
    x, y = _validate_supervised_inputs(features, targets)
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("targets must be binary")
    if learning_rate <= 0 or steps < 1:
        raise ValueError("learning_rate and steps must be positive")

    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        probabilities = _sigmoid(x @ weights + bias)
        errors = probabilities - y
        weights -= learning_rate * (x.T @ errors) / x.shape[0]
        bias -= learning_rate * float(errors.mean())
    return weights, bias


def knn_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    k: int = 3,
) -> np.ndarray:
    """Predict labels using vectorized squared Euclidean distances."""
    train_x = _as_feature_matrix(train_features, "train_features")
    query_x = _as_feature_matrix(query_features, "query_features")
    labels = np.asarray(train_labels)
    if labels.shape != (train_x.shape[0],):
        raise ValueError("train_labels must have one value per training row")
    if query_x.shape[1] != train_x.shape[1]:
        raise ValueError("training and query feature widths must match")
    if not 1 <= k <= train_x.shape[0]:
        raise ValueError("k must be between 1 and the number of training rows")

    distances = ((query_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    neighbor_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    predictions = []
    for indices in neighbor_indices:
        values, counts = np.unique(labels[indices], return_counts=True)
        predictions.append(values[np.argmax(counts)])
    return np.asarray(predictions, dtype=labels.dtype)


def kmeans(
    features: np.ndarray,
    k: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rows with Lloyd's algorithm and deterministic initialization."""
    x = _as_feature_matrix(features, "features")
    if not 1 <= k <= x.shape[0]:
        raise ValueError("k must be between 1 and the number of rows")
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("max_iterations must be positive and tolerance non-negative")

    generator = np.random.default_rng(seed)
    centers = x[generator.choice(x.shape[0], size=k, replace=False)].copy()

    for _ in range(max_iterations):
        squared_distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared_distances.argmin(axis=1)
        new_centers = centers.copy()
        nearest_distance = squared_distances[np.arange(x.shape[0]), labels].copy()

        for cluster_index in range(k):
            members = x[labels == cluster_index]
            if len(members):
                new_centers[cluster_index] = members.mean(axis=0)
            else:
                # Re-seed an empty cluster with a currently poorly represented point.
                farthest_index = int(np.argmax(nearest_distance))
                new_centers[cluster_index] = x[farthest_index]
                nearest_distance[farthest_index] = -1.0

        if np.linalg.norm(new_centers - centers) <= tolerance:
            centers = new_centers
            break
        centers = new_centers

    final_distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return centers, final_distances.argmin(axis=1)


def gini_impurity(labels: np.ndarray) -> float:
    """Compute Gini impurity for a one-dimensional label array."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    return float(1.0 - np.sum(probabilities**2))


def best_gini_split(
    features: np.ndarray, labels: np.ndarray
) -> tuple[int, float, float] | None:
    """Find the feature and threshold with minimum weighted Gini impurity."""
    x = _as_feature_matrix(features, "features")
    y = np.asarray(labels)
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have one value per row")

    best: tuple[int, float, float] | None = None
    for feature_index in range(x.shape[1]):
        unique_values = np.unique(x[:, feature_index])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
        for threshold in thresholds:
            left = x[:, feature_index] <= threshold
            if left.all() or (~left).all():
                continue
            weighted_impurity = (
                left.mean() * gini_impurity(y[left])
                + (~left).mean() * gini_impurity(y[~left])
            )
            candidate = (feature_index, float(threshold), float(weighted_impurity))
            if best is None or candidate[2] < best[2]:
                best = candidate
    return best


def principal_component_analysis(
    features: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projected data, principal axes, and explained variances."""
    x = _as_feature_matrix(features, "features")
    if not 1 <= n_components <= min(x.shape):
        raise ValueError("n_components must not exceed min(num_rows, num_features)")

    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:n_components]
    transformed = centered @ components.T
    denominator = max(x.shape[0] - 1, 1)
    explained_variance = singular_values[:n_components] ** 2 / denominator
    return transformed, components, explained_variance


def conv2d_valid(
    image: np.ndarray, kernel: np.ndarray, *, stride: int = 1
) -> np.ndarray:
    """Compute a single-channel valid 2D cross-correlation."""
    x = np.asarray(image, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("image and kernel must both be two-dimensional")
    if stride < 1 or np.any(np.asarray(weights.shape) > np.asarray(x.shape)):
        raise ValueError("stride must be positive and kernel must fit inside image")

    output_height = 1 + (x.shape[0] - weights.shape[0]) // stride
    output_width = 1 + (x.shape[1] - weights.shape[1]) // stride
    output = np.empty((output_height, output_width), dtype=np.float64)
    for row in range(output_height):
        for column in range(output_width):
            row_start = row * stride
            column_start = column * stride
            window = x[
                row_start : row_start + weights.shape[0],
                column_start : column_start + weights.shape[1],
            ]
            output[row, column] = np.sum(window * weights)
    return output


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-head scaled dot-product attention and attention weights."""
    q = _as_feature_matrix(queries, "queries")
    k = _as_feature_matrix(keys, "keys")
    v = _as_feature_matrix(values, "values")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q/K widths and K/V sequence lengths must match")

    scores = q @ k.T / math.sqrt(q.shape[1])
    if mask is not None:
        allowed = np.asarray(mask, dtype=bool)
        if allowed.shape != scores.shape:
            raise ValueError("mask must have shape [num_queries, num_keys]")
        if np.any(~allowed.any(axis=1)):
            raise ValueError("every query must be allowed to attend to at least one key")
        scores = np.where(allowed, scores, -np.inf)
    weights = softmax(scores)
    return weights @ v, weights


def binary_classification_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 from binary labels."""
    target_values = np.asarray(targets)
    prediction_values = np.asarray(predictions)
    if target_values.ndim != 1 or target_values.shape != prediction_values.shape:
        raise ValueError("targets and predictions must be equal-length vectors")
    if not np.all(np.isin(target_values, (0, 1))) or not np.all(
        np.isin(prediction_values, (0, 1))
    ):
        raise ValueError("targets and predictions must be binary")
    y_true = target_values.astype(bool)
    y_pred = prediction_values.astype(bool)

    true_positive = int(np.sum(y_true & y_pred))
    true_negative = int(np.sum(~y_true & ~y_pred))
    false_positive = int(np.sum(~y_true & y_pred))
    false_negative = int(np.sum(y_true & ~y_pred))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "accuracy": _safe_divide(true_positive + true_negative, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC-AUC using average ranks, including tied scores."""
    y = np.asarray(targets)
    predictions = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != predictions.shape:
        raise ValueError("targets and scores must be equal-length vectors")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("targets must be binary")

    positive_count = int(y.sum())
    negative_count = len(y) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative examples")

    order = np.argsort(predictions, kind="mergesort")
    sorted_scores = predictions[order]
    ranks = np.empty(len(predictions), dtype=np.float64)
    start = 0
    while start < len(predictions):
        end = start + 1
        while end < len(predictions) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def reservoir_sample(items: Iterable[object], k: int, *, seed: int = 0) -> list[object]:
    """Select k uniformly distributed items from a stream of unknown length."""
    if k < 0:
        raise ValueError("k must be non-negative")
    generator = random.Random(seed)
    reservoir: list[object] = []
    for index, item in enumerate(items):
        if index < k:
            reservoir.append(item)
            continue
        replacement_index = generator.randint(0, index)
        if replacement_index < k:
            reservoir[replacement_index] = item
    return reservoir


def tfidf(documents: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Build a basic lowercase whitespace-tokenized TF-IDF matrix."""
    tokenized = [document.lower().split() for document in documents]
    vocabulary = sorted({token for document in tokenized for token in document})
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    if not vocabulary:
        return matrix, vocabulary

    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    document_frequency = np.zeros(len(vocabulary), dtype=np.float64)
    for row, document in enumerate(tokenized):
        if not document:
            continue
        counts: dict[str, int] = {}
        for token in document:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            column = token_to_index[token]
            matrix[row, column] = count / len(document)
            document_frequency[column] += 1

    inverse_document_frequency = np.log(
        (1.0 + len(documents)) / (1.0 + document_frequency)
    ) + 1.0
    return matrix * inverse_document_frequency, vocabulary


def _as_feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_supervised_inputs(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_feature_matrix(features, "features")
    y = np.asarray(targets, dtype=np.float64)
    if y.shape != (x.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("targets must be a finite vector with one value per row")
    return x, y


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (1.0 + exponentials)
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
