"""
Waymo prep — drill checker.

Usage:
    python check.py               # run everything
    python check.py kmeans        # run one drill
    python check.py kmeans knn    # run several
    python check.py --list        # list drill names

Every check compares your implementation in drills.py against a reference
(sklearn / scipy / a hand-verified expected value) plus the edge cases the
interviewer is most likely to probe.
"""

import sys
import time
import traceback

import numpy as np

import drills

RNG = np.random.default_rng(0)
CHECKS = {}


def check(name):
    def deco(fn):
        CHECKS[name] = fn
        return fn
    return deco


def close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float),
                       rtol=tol, atol=tol)


# ---------------------------------------------------------------- tier 1

@check("masked_metrics")
def _():
    N, G = 200, 4
    preds = RNG.integers(0, 2, N)
    labels = RNG.integers(0, 2, N)
    valid = RNG.random(N) > 0.3
    groups = RNG.integers(0, G, N)

    out = drills.masked_metrics(preds, labels, valid, groups, G)

    # reference: explicit loop (fine here, not in your solution)
    for g in range(G):
        m = valid & (groups == g)
        p, l = preds[m], labels[m]
        tp = np.sum((p == 1) & (l == 1))
        fp = np.sum((p == 1) & (l == 0))
        fn = np.sum((p == 0) & (l == 1))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        acc = np.mean(p == l) if m.sum() else 0.0
        assert close(out["precision"][g], prec), f"precision group {g}"
        assert close(out["recall"][g], rec), f"recall group {g}"
        assert close(out["accuracy"][g], acc), f"accuracy group {g}"

    # edge: a group with zero valid entries
    valid2 = valid.copy()
    valid2[groups == 2] = False
    out2 = drills.masked_metrics(preds, labels, valid2, groups, G)
    for key in ("precision", "recall", "accuracy"):
        v = out2[key][2]
        assert not np.isnan(v) and close(v, 0.0), f"empty group -> {key} should be 0.0"

    # edge: everything invalid
    out3 = drills.masked_metrics(preds, labels, np.zeros(N, bool), groups, G)
    assert not np.isnan(np.asarray(out3["precision"], float)).any(), "all-invalid produced NaN"

    # edge: N = 0
    z = np.array([], int)
    out4 = drills.masked_metrics(z, z, np.array([], bool), z, G)
    assert len(out4["recall"]) == G, "n_groups length not preserved on empty input"


@check("kmeans")
def _():
    from sklearn.cluster import KMeans

    X = np.vstack([RNG.normal(0, 1, (60, 3)),
                   RNG.normal(6, 1, (60, 3)),
                   RNG.normal(-6, 1, (60, 3))])
    init = np.array([X[0], X[70], X[130]], float)

    C, lab = drills.kmeans(X, 3, init.copy())

    ref = KMeans(n_clusters=3, init=init.copy(), n_init=1,
                 max_iter=100, tol=1e-10).fit(X)
    assert close(np.sort(C, axis=0), np.sort(ref.cluster_centers_, axis=0), 1e-3), \
        "centroids differ from sklearn"
    assert len(lab) == len(X) and lab.dtype.kind in "iu", "labels must be [N] int"

    # edge: an init centroid far from all data -> that cluster goes empty
    init_bad = np.array([X[0], X[70], [1e6, 1e6, 1e6]], float)
    C2, _ = drills.kmeans(X, 3, init_bad)
    assert not np.isnan(C2).any(), "empty cluster produced NaN centroid"

    # no data-point loop
    _assert_no_point_loop("kmeans")


@check("knn")
def _():
    from sklearn.neighbors import KNeighborsClassifier

    Xtr = RNG.normal(0, 1, (150, 4))
    ytr = RNG.integers(0, 3, 150)
    Xte = RNG.normal(0, 1, (40, 4))

    got = drills.knn_predict(Xtr, ytr, Xte, k=5)
    ref = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).predict(Xte)
    agree = np.mean(got == ref)
    assert agree > 0.95, f"only {agree:.0%} agreement with sklearn (ties aside, expect ~100%)"

    # edge: k = 1 must reproduce the nearest training label exactly
    got1 = drills.knn_predict(Xtr, ytr, Xtr[:10], k=1)
    assert np.array_equal(got1, ytr[:10]), "k=1 on training points must return their own labels"

    _assert_no_point_loop("knn_predict")


@check("softmax")
def _():
    from scipy.special import softmax as ref_softmax

    x = RNG.normal(0, 2, (7, 5))
    assert close(drills.softmax(x), ref_softmax(x, axis=-1)), "softmax mismatch"
    assert close(drills.softmax(x, axis=0), ref_softmax(x, axis=0)), "axis=0 mismatch"

    big = np.array([[1e5, 1e5 + 1, 1e5 - 1]])
    out = drills.softmax(big)
    assert np.isfinite(out).all(), "overflow on large logits — subtract the max"
    assert close(out.sum(), 1.0), "rows must sum to 1"

    neg = np.array([[-1e5, -1e5 - 1]])
    assert np.isfinite(drills.softmax(neg)).all(), "underflow on large negative logits"


@check("cross_entropy")
def _():
    from scipy.special import log_softmax

    logits = RNG.normal(0, 2, (32, 6))
    labels = RNG.integers(0, 6, 32)

    ref_loss = -log_softmax(logits, axis=-1)[np.arange(32), labels].mean()
    assert close(drills.cross_entropy_loss(logits, labels), ref_loss), "loss mismatch"

    # analytic gradient vs finite differences
    g = drills.cross_entropy_grad(logits, labels)
    assert g.shape == logits.shape, "gradient shape mismatch"
    num = np.zeros_like(logits)
    h = 1e-5
    for i in range(4):
        for j in range(6):
            p = logits.copy(); p[i, j] += h
            m = logits.copy(); m[i, j] -= h
            num[i, j] = (drills.cross_entropy_loss(p, labels)
                         - drills.cross_entropy_loss(m, labels)) / (2 * h)
    assert close(g[:4], num[:4], 1e-4), "gradient disagrees with finite differences"

    huge = np.array([[1e5, 0.0, -1e5]])
    assert np.isfinite(drills.cross_entropy_loss(huge, np.array([0]))), \
        "unstable on large logits — use log-sum-exp"


@check("pca")
def _():
    from sklearn.decomposition import PCA

    X = RNG.normal(0, 1, (100, 6)) @ RNG.normal(0, 1, (6, 6))
    comps, proj, var = drills.pca(X, 3)

    ref = PCA(n_components=3).fit(X)
    assert close(np.abs(comps), np.abs(ref.components_), 1e-4), "components mismatch"
    assert close(var, ref.explained_variance_, 1e-4), "explained variance mismatch"
    assert close(np.abs(proj), np.abs(ref.transform(X)), 1e-4), "projection mismatch"
    assert np.all(np.diff(var) <= 1e-9), "variances must be descending"

    # edge: D > N
    Xw = RNG.normal(0, 1, (5, 20))
    c2, p2, v2 = drills.pca(Xw, 2)
    assert c2.shape == (2, 20) and p2.shape == (5, 2), "wide-matrix shapes wrong"
    assert np.isfinite(c2).all(), "NaN with D > N"


@check("logreg")
def _():
    X = RNG.normal(0, 1, (400, 3))
    w_true = np.array([1.5, -2.0, 0.5])
    p = 1 / (1 + np.exp(-(X @ w_true + 0.3)))
    y = (RNG.random(400) < p).astype(int)

    w, b = drills.logistic_regression_fit(X, y, lr=0.5, n_iters=3000)
    assert np.asarray(w).shape == (3,), "w must be shape [D]"
    assert np.corrcoef(w, w_true)[0, 1] > 0.9, "learned weights don't track the truth"

    acc = np.mean(((X @ w + b) > 0).astype(int) == y)
    assert acc > 0.8, f"train accuracy only {acc:.2f}"

    # sigmoid stability
    Xb = np.full((4, 3), 500.0)
    yb = np.array([1, 1, 0, 0])
    wb, bb = drills.logistic_regression_fit(Xb, yb, lr=0.1, n_iters=50)
    assert np.isfinite(wb).all() and np.isfinite(bb), "overflow on large inputs"

    _assert_no_point_loop("logistic_regression_fit")


@check("conv2d")
def _():
    from scipy.signal import correlate2d

    x = RNG.normal(0, 1, (9, 11))
    k = RNG.normal(0, 1, (3, 3))

    assert close(drills.conv2d(x, k, 1, 0), correlate2d(x, k, mode="valid")), \
        "stride 1, no padding"

    xp = np.pad(x, 1)
    assert close(drills.conv2d(x, k, 1, 1), correlate2d(xp, k, mode="valid")), \
        "padding=1"

    full = correlate2d(xp, k, mode="valid")
    assert close(drills.conv2d(x, k, 2, 1), full[::2, ::2]), "stride 2 + padding 1"

    out = drills.conv2d(x, np.ones((2, 4)), 3, 2)
    H = (9 + 4 - 2) // 3 + 1
    W = (11 + 4 - 4) // 3 + 1
    assert out.shape == (H, W), f"expected shape {(H, W)}, got {out.shape}"

    _assert_no_output_pixel_loop("conv2d")


# ---------------------------------------------------------------- tier 2

@check("linreg")
def _():
    from sklearn.linear_model import LinearRegression
    X = RNG.normal(0, 1, (80, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.7 + RNG.normal(0, 0.1, 80)
    w, b = drills.linear_regression_normal_eq(X, y)
    ref = LinearRegression().fit(X, y)
    assert close(w, ref.coef_, 1e-5) and close(b, ref.intercept_, 1e-5), "OLS mismatch"


@check("attention")
def _():
    from scipy.special import softmax as ref_softmax

    B, L, D = 2, 6, 8
    Q, K, V = (RNG.normal(0, 1, (B, L, D)) for _ in range(3))

    out, attn = drills.attention(Q, K, V)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D)
    ref_w = ref_softmax(scores, axis=-1)
    assert close(attn, ref_w), "attention weights mismatch (check the 1/sqrt(d) scale)"
    assert close(out, ref_w @ V), "output mismatch"

    mask = np.tril(np.ones((L, L), bool))[None].repeat(B, 0)
    _, attn_m = drills.attention(Q, K, V, mask)
    assert close(attn_m[:, 0, 1:], 0.0), "masked positions must receive zero weight"
    assert close(attn_m.sum(-1), 1.0), "masked rows must still sum to 1"


@check("mha")
def _():
    B, L, D, H = 2, 5, 8, 2
    X = RNG.normal(0, 1, (B, L, D))
    Ws = [RNG.normal(0, 0.3, (D, D)) for _ in range(4)]

    out = drills.multihead_attention(X, *Ws, n_heads=H)
    assert out.shape == (B, L, D), f"expected {(B, L, D)}, got {out.shape}"

    # single head must reduce to plain attention
    o1 = drills.multihead_attention(X, *Ws, n_heads=1)
    a, _ = drills.attention(X @ Ws[0], X @ Ws[1], X @ Ws[2])
    assert close(o1, a @ Ws[3], 1e-5), "n_heads=1 must equal single-head attention"

    # causality: changing a later token must not alter earlier outputs
    oc = drills.multihead_attention(X, *Ws, n_heads=H, causal=True)
    X2 = X.copy(); X2[:, -1] += 10.0
    oc2 = drills.multihead_attention(X2, *Ws, n_heads=H, causal=True)
    assert close(oc[:, :-1], oc2[:, :-1], 1e-5), "causal mask leaks future information"


@check("tree")
def _():
    assert close(drills.gini_impurity(np.array([0, 0, 0, 0])), 0.0), "pure node -> 0"
    assert close(drills.gini_impurity(np.array([0, 0, 1, 1])), 0.5), "50/50 -> 0.5"
    assert close(drills.gini_impurity(np.array([], int)), 0.0), "empty -> 0"

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    f, t, g = drills.best_split(X, y)
    assert f == 0 and close(g, 0.0), "should find the perfect split"
    assert 2.0 < t < 3.0, f"threshold should be the 2/3 midpoint, got {t}"

    Xc = np.ones((6, 2))
    fc, _, gc = drills.best_split(Xc, np.array([0, 1, 0, 1, 0, 1]))
    assert fc == -1, "constant features admit no split -> feature index -1"


@check("mlp")
def _():
    N, D, H, C = 12, 5, 7, 3
    X = RNG.normal(0, 1, (N, D))
    y = RNG.integers(0, C, N)
    W1 = RNG.normal(0, 0.3, (D, H)); b1 = RNG.normal(0, 0.1, H)
    W2 = RNG.normal(0, 0.3, (H, C)); b2 = RNG.normal(0, 0.1, C)

    loss, grads = drills.mlp_forward_backward(X, y, W1, b1, W2, b2)
    for k, ref in (("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)):
        assert grads[k].shape == ref.shape, f"grad {k} shape mismatch"

    h = 1e-5
    for name, mat in (("W2", W2), ("W1", W1)):
        idx = (0, 0)
        p = [W1, b1, W2, b2]
        i = ["W1", "b1", "W2", "b2"].index(name)
        up = [a.copy() for a in p]; up[i][idx] += h
        dn = [a.copy() for a in p]; dn[i][idx] -= h
        num = (drills.mlp_forward_backward(X, y, *up)[0]
               - drills.mlp_forward_backward(X, y, *dn)[0]) / (2 * h)
        assert close(grads[name][idx], num, 1e-4), \
            f"grad {name}{idx} disagrees with finite differences"


@check("vae")
def _():
    N, D, Hd, Z = 10, 6, 8, 3
    x = RNG.random((N, D))
    eps = RNG.normal(0, 1, (N, Z))
    W_enc = RNG.normal(0, .3, (D, Hd)); b_enc = np.zeros(Hd)
    W_mu = RNG.normal(0, .3, (Hd, Z)); b_mu = np.zeros(Z)
    W_lv = RNG.normal(0, .3, (Hd, Z)); b_lv = np.zeros(Z)
    W_dec = RNG.normal(0, .3, (Z, D)); b_dec = np.zeros(D)

    xh, mu, lv, loss = drills.vae_forward(x, W_enc, b_enc, W_mu, b_mu,
                                          W_lv, b_lv, W_dec, b_dec, eps)
    assert xh.shape == (N, D) and mu.shape == (N, Z), "shape mismatch"
    assert (xh >= 0).all() and (xh <= 1).all(), "decoder output must be sigmoid-bounded"

    h = np.maximum(x @ W_enc + b_enc, 0)
    ref_mu = h @ W_mu + b_mu
    assert close(mu, ref_mu), "mu path wrong"

    z = ref_mu + np.exp(0.5 * (h @ W_lv + b_lv)) * eps
    ref_xh = 1 / (1 + np.exp(-(z @ W_dec + b_dec)))
    assert close(xh, ref_xh), "reparameterization or decoder wrong"

    kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv), axis=1)
    rec = -np.sum(x * np.log(ref_xh + 1e-12) + (1 - x) * np.log(1 - ref_xh + 1e-12), axis=1)
    assert close(loss, np.mean(rec + kl), 1e-4), "ELBO mismatch"


# ---------------------------------------------------------------- tier 3

@check("adam")
def _():
    p = np.array([1.0, 2.0]); g = np.array([0.1, -0.2])
    m = np.zeros(2); v = np.zeros(2)
    p1, m1, v1 = drills.adam_step(p, g, m, v, t=1, lr=0.1)
    assert close(m1, 0.1 * g), "m update wrong"
    assert close(v1, 0.001 * g ** 2), "v update wrong"
    mh, vh = m1 / 0.1, v1 / 0.001
    assert close(p1, p - 0.1 * mh / (np.sqrt(vh) + 1e-8), 1e-9), \
        "missing bias correction (step 1 should be ~ -lr * sign(g))"


@check("reservoir")
def _():
    counts = np.zeros(10)
    trials = 4000
    for s in range(trials):
        out = drills.reservoir_sample(range(10), 3, np.random.default_rng(s))
        assert len(out) == 3 and len(set(out)) == 3, "must return k distinct items"
        counts[list(out)] += 1
    freq = counts / trials
    assert np.abs(freq - 0.3).max() < 0.04, f"non-uniform: {freq.round(3)}"
    assert len(drills.reservoir_sample(range(2), 5, np.random.default_rng(0))) == 2, \
        "fewer items than k -> return all"


@check("stratified")
def _():
    y = np.concatenate([np.zeros(50), np.ones(30), np.full(4, 2)]).astype(int)
    idx = drills.stratified_sample(y, 5, np.random.default_rng(1))
    idx = np.asarray(idx)
    assert np.all(np.diff(idx) > 0), "indices must be sorted and unique"
    assert (y[idx] == 0).sum() == 5 and (y[idx] == 1).sum() == 5, "wrong per-class count"
    assert (y[idx] == 2).sum() == 4, "small class -> take all available"


# ---------------------------------------------------------------- helpers

def _source_of(fn_name):
    import inspect
    return inspect.getsource(getattr(drills, fn_name))


def _assert_no_point_loop(fn_name):
    src = _source_of(fn_name)
    body = [l for l in src.splitlines() if l.strip().startswith(("for ", "while "))]
    hint = ("iteration loops are fine; a loop over samples/points is not — "
            f"found {len(body)} loop(s), check they're over iterations only")
    if len(body) > 2:
        raise AssertionError(hint)


def _assert_no_output_pixel_loop(fn_name):
    src = _source_of(fn_name)
    n = sum(1 for l in src.splitlines() if l.strip().startswith("for "))
    if n >= 2:
        raise AssertionError(
            "nested loop over output pixels — use sliding_window_view or im2col")


# ---------------------------------------------------------------- runner

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        print("\n".join(CHECKS))
        return

    names = args or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"unknown drill(s): {unknown}\navailable: {list(CHECKS)}")
        return

    width = max(len(n) for n in names)
    passed = todo = failed = 0

    for n in names:
        t0 = time.perf_counter()
        try:
            CHECKS[n]()
        except NotImplementedError:
            print(f"  {n:<{width}}  ·  not started")
            todo += 1
        except AssertionError as e:
            print(f"  {n:<{width}}  ✗  {e}")
            failed += 1
        except Exception:
            print(f"  {n:<{width}}  ✗  crashed:")
            print("      " + traceback.format_exc().strip().replace("\n", "\n      "))
            failed += 1
        else:
            print(f"  {n:<{width}}  ✓  ({(time.perf_counter()-t0)*1000:.0f} ms)")
            passed += 1

    print(f"\n  {passed} passed · {failed} failed · {todo} not started")


if __name__ == "__main__":
    main()
