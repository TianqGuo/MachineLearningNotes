"""
Waymo ML Coding prep — drill stubs.

Fill in each function. Run `python check.py` to test everything,
or `python check.py kmeans` to test one drill.

RULES:
  - No sklearn / scipy imports in THIS file. NumPy only.
  - No Python for-loops over data points. Loops over iterations
    (e.g. GD steps, k-means rounds, tree depth) are fine.
  - Comment your shapes as you go: # [B, N, D]
"""

import numpy as np


# ===========================================================================
# TIER 1
# ===========================================================================

def masked_metrics(preds, labels, valid_mask, group_ids, n_groups):
    """Per-group classification metrics over a batch with invalid entries.

    This is the 'simulation agent metrics' shape. Highest-probability problem type.

    Args:
        preds:      [N] int array of predicted binary labels (0/1)
        labels:     [N] int array of ground-truth binary labels (0/1)
        valid_mask: [N] bool array; False entries must be excluded entirely
        group_ids:  [N] int array in [0, n_groups), e.g. agent id
        n_groups:   int

    Returns:
        dict with keys 'precision', 'recall', 'accuracy', each a [n_groups]
        float array. Groups with no valid entries -> 0.0. Zero-denominator
        precision/recall -> 0.0.

    Hint: np.bincount with `weights=` and `minlength=` does the aggregation
    without a loop.
    """
    raise NotImplementedError


def kmeans(X, k, init_centroids, max_iters=100, tol=1e-6):
    """Lloyd's algorithm.

    Args:
        X:              [N, D] float
        k:              int
        init_centroids: [k, D] float — use these, do not random-init
        max_iters:      int
        tol:            stop when max centroid shift < tol

    Returns:
        (centroids [k, D], labels [N] int)

    Requirements:
        - Assignment step fully vectorized (broadcasting, no loop over points)
        - Empty cluster: keep its previous centroid, do not produce NaN
    """
    raise NotImplementedError


def knn_predict(X_train, y_train, X_test, k):
    """K-nearest neighbours, Euclidean distance, majority vote.

    Args:
        X_train: [N, D], y_train: [N] int labels in [0, C)
        X_test:  [M, D]
        k:       int

    Returns:
        [M] int predicted labels. Ties broken toward the smallest label.

    Requirements:
        - Pairwise distances vectorized (no loop over test points)
        - Use np.argpartition, not a full argsort
    """
    raise NotImplementedError


def softmax(logits, axis=-1):
    """Numerically stable softmax. logits: any shape. Returns same shape.

    Must not overflow on inputs like 1e5.
    """
    # raise NotImplementedError
    e_score =  np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e_score / np.sum(e_score, axis=axis, keepdims=True)


def cross_entropy_loss(logits, labels):
    """Mean cross-entropy.

    logits: [N, C] float, labels: [N] int class indices.
    Returns: scalar float. Must be stable (log-sum-exp, not log(softmax(x))).
    """
    # raise NotImplementedError
    n = logits.shape[0]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))  # [N], no keepdims
    return -np.mean(shifted[np.arange(n), labels] - log_sum_exp)


def cross_entropy_grad(logits, labels):
    """Gradient of `cross_entropy_loss` w.r.t. logits. Returns [N, C]."""
    n = logits.shape[0]
    probs = softmax(logits, axis=-1)                # [N, C]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return (probs - onehot) / n



def pca(X, n_components):
    """Principal component analysis.

    Args:
        X: [N, D] float (not pre-centered)
        n_components: int

    Returns:
        (components [n_components, D], projected [N, n_components],
         explained_variance [n_components])

    Requirements:
        - Center the data first
        - Components sorted by descending explained variance
        - Sign convention: make the largest-|value| entry of each component positive
    """
    raise NotImplementedError


def logistic_regression_fit(X, y, lr=0.1, n_iters=1000):
    """Binary logistic regression via full-batch gradient descent.

    Args:
        X: [N, D] float (no bias column — add one internally)
        y: [N] int in {0, 1}

    Returns:
        (w [D], b scalar)

    Requirements:
        - Vectorized gradient, no loop over samples
        - Sigmoid must not overflow for large |z|
    """
    raise NotImplementedError


def conv2d(x, kernel, stride=1, padding=0):
    """2D cross-correlation (the 'convolution' every framework actually does).

    Args:
        x:      [H, W] float
        kernel: [KH, KW] float
        stride: int
        padding: int, zero-padding on all four sides

    Returns:
        [H_out, W_out] where H_out = (H + 2p - KH)//stride + 1

    Requirements:
        - No nested loop over output pixels. Use np.lib.stride_tricks.
          sliding_window_view (then slice by stride) or an im2col construction.
    """
    raise NotImplementedError


# ===========================================================================
# TIER 2
# ===========================================================================

def linear_regression_normal_eq(X, y):
    """Closed-form OLS with intercept. X: [N, D], y: [N]. Returns (w [D], b)."""
    raise NotImplementedError


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Args:
        Q: [..., Lq, D], K: [..., Lk, D], V: [..., Lk, Dv]
        mask: [..., Lq, Lk] bool or None. True = KEEP, False = mask out (-inf).

    Returns:
        ([..., Lq, Dv] output, [..., Lq, Lk] attention weights)
    """
    raise NotImplementedError


def multihead_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """Multi-head self-attention.

    Args:
        X:  [B, L, D]
        Wq, Wk, Wv, Wo: [D, D]
        n_heads: int, divides D
        causal: if True, position i may only attend to j <= i

    Returns:
        [B, L, D]
    """
    raise NotImplementedError


def gini_impurity(y):
    """Gini impurity of a label array. y: [N] int. Returns scalar float.
    Empty input -> 0.0."""
    raise NotImplementedError


def best_split(X, y):
    """Find the split minimizing weighted child Gini.

    Args:
        X: [N, D] float, y: [N] int

    Returns:
        (feature_index int, threshold float, weighted_gini float)
        Split rule is X[:, f] <= threshold goes left.
        Candidate thresholds: midpoints between consecutive unique values.
        No valid split -> (-1, 0.0, gini_impurity(y))
    """
    raise NotImplementedError


def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """Two-layer MLP: X -> Linear -> ReLU -> Linear -> softmax-CE. Manual backprop.

    Args:
        X: [N, D], y: [N] int class indices
        W1: [D, H], b1: [H], W2: [H, C], b2: [C]

    Returns:
        (loss scalar, grads dict with keys 'W1','b1','W2','b2' matching shapes)
    """
    raise NotImplementedError


def vae_forward(x, W_enc, b_enc, W_mu, b_mu, W_logvar, b_logvar,
                W_dec, b_dec, eps):
    """VAE forward pass with the reparameterization trick.

    Encoder:  h = relu(x @ W_enc + b_enc)
              mu = h @ W_mu + b_mu ; logvar = h @ W_logvar + b_logvar
    Sample:   z = mu + exp(0.5 * logvar) * eps      (eps passed in for determinism)
    Decoder:  x_hat = sigmoid(z @ W_dec + b_dec)

    Args:
        x:   [N, D]
        eps: [N, Z] standard-normal draw, supplied so the test is deterministic

    Returns:
        (x_hat [N, D], mu [N, Z], logvar [N, Z], loss scalar)

    Loss = mean over batch of (binary cross-entropy recon summed over D)
           + KL(q(z|x) || N(0,I)) summed over Z.
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    raise NotImplementedError


# ===========================================================================
# TIER 3 — write the update rule, not a full optimizer
# ===========================================================================

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam update with bias correction.

    Returns: (new_param, new_m, new_v) — all same shape as param.
    """
    raise NotImplementedError


def reservoir_sample(stream, k, rng):
    """Algorithm R. `stream` is an iterable of unknown length, `rng` a
    np.random.Generator. Returns a list of k items (or all items if fewer).

    Use rng.integers(0, i+1) for the i-th item (0-indexed) after the first k.
    """
    raise NotImplementedError


def stratified_sample(y, n_per_class, rng):
    """Return indices sampling exactly n_per_class from each class in y.

    y: [N] int. Returns [n_classes * n_per_class] int index array, sorted
    ascending. If a class has fewer than n_per_class members, take all of them.
    Use rng.permutation on each class's index array.
    """
    raise NotImplementedError
