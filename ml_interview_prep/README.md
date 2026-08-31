# ML Interview Prep — Drill Notes

Notes and derivations worked out while implementing `drills.py`. One section
per function; add new sections here as more drills get worked through.

## `cross_entropy_loss`

### Shapes

`logits`: `[N, C]` — N = batch size (number of samples), C = number of classes.
`labels`: `[N]` int, each entry a class index in `[0, C)`.

### Pitfalls found in the initial implementation

```python
def cross_entropy_loss(logits, labels):
    d_class = logits.shape[0]
    score = np.exp((logits - np.max(logits, axis=-1, keepdims=True)))
    log_sum_exp = np.log(np.sum(score, axis=-1, keepdims=True))
    return -np.mean(score[np.arange(d_class), labels] - log_sum_exp)
```

1. **Misleading name**: `d_class = logits.shape[0]` is actually `N` (batch
   size), not the number of classes `C`.
2. **Indexing the wrong array**: `score` holds `exp(shifted_logits)`, not
   the shifted logits themselves. The code should index into
   `logits - max` (before exponentiating), then subtract `log_sum_exp`
   — not index into the already-exponentiated `score`.
3. **Broadcasting bug**: `log_sum_exp` keeps shape `[N, 1]`
   (`keepdims=True`) while `score[np.arange(d_class), labels]` is shape
   `[N]`. Subtracting `[N] - [N, 1]` broadcasts to `[N, N]` instead of
   `[N]`, so `np.mean` silently averages over N² garbage entries instead
   of N per-sample losses.

Verified with a runnable example: buggy version gave `-0.247` on a 2-sample
batch where the correct mean CE loss is `0.753`.

### Correct implementation

```python
def cross_entropy_loss(logits, labels):
    n = logits.shape[0]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)   # [N, C]
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))       # [N], no keepdims
    return -np.mean(shifted[np.arange(n), labels] - log_sum_exp)
```

### Why not just call `softmax` + `log`?

The docstring requires log-sum-exp, not `log(softmax(x))`. They are
algebraically identical, but differ in floating-point behavior: for a
confidently-wrong prediction (true class logit far below the max),
`softmax` can legitimately underflow to exactly `0.0`, and `log(0.0)`
gives `-inf` instead of the correct large-but-finite loss. The
log-sum-exp form never computes that division — it stays in log-space
throughout — so it can't hit that failure mode.

```python
softmax(np.array([0., -1000.]))  # -> [1., 0.]   (underflowed)
np.log(0.)                        # -> -inf
```

### Derivation — why `shifted[label] - log_sum_exp` equals cross entropy

Cross entropy, full definition, with `p` the true (one-hot) distribution
and `q` the predicted (softmax) distribution:

```
H(p, q) = -sum_c p_c * log(q_c)
```

**One-hot collapse**: since `p_c = 1` only at `c = label` and `0`
elsewhere, every term but `c = label` vanishes:

```
H(p, q) = -log(q_label)
```

This is why the code only needs `labels` (one index per sample) rather
than a full one-hot vector — indexing `[..., label]` *is* the one-hot dot
product, done by gather instead of multiply-and-sum-zeros.

**Expand `q_label` as softmax and take the log:**

```
q_label   = exp(z_label) / sum_c exp(z_c)
log(q_label) = z_label - log(sum_c exp(z_c))
             = z_label - LSE(z)
```

where `LSE(z)` = `log_sum_exp` in the code, and `z` = `logits` (the
max-shift cancels in the ratio, so working with `shifted` instead of raw
`logits` doesn't change the result — it's only there for numerical
stability).

**Final loss:**

```
loss = -log(q_label) = LSE(z) - z_label
```

which is `log_sum_exp - shifted[label]`, i.e. exactly what the fixed code
computes (negated and averaged over the batch).

### What `log_sum_exp` actually represents

`log_sum_exp` is **not** label-specific — it's the same value for every
class within a sample, computed from all `C` logits via `axis=-1`. That's
intentional: it's the log of the total exponential mass (the softmax
normalizer / partition function), and normalization is a property of the
whole distribution, not any single class.

Concretely, since `shifted` has its max pinned to `0`:
- One class totally dominates → `sum(exp(shifted)) ≈ 1` → `log_sum_exp ≈ 0`
  (confident prediction, small offset).
- Many classes tied near the max → `sum(exp(shifted)) ≈ C` →
  `log_sum_exp ≈ log(C)` (uncertain prediction, larger offset).

It matters because `shifted[label]` alone is just a raw, arbitrarily-scaled
logit — not a probability. Two samples can have the same `shifted[label]`
value but different losses depending on how much the other classes
"compete":

- Sample A: `shifted = [0, -100, -100]`, label=0 → `log_sum_exp ≈ 0` →
  loss ≈ 0 (very confident, correct).
- Sample B: `shifted = [0, 0, 0]`, label=0 → `log_sum_exp ≈ log(3) ≈ 1.10`
  → loss ≈ 1.10 (only 33% confident, despite identical raw score for the
  true class).

### Equivalence of the three formulations

These are all the same value, mathematically — just different
implementations of the identical computation:

1. `-log(softmax(logits)[label])` — compute softmax, then log, then index.
2. `-sum(onehot(label) * log(softmax(logits)), axis=-1)` — full one-hot
   multiply-and-sum (most terms are multiplied by zero and wasted).
3. `log_sum_exp - shifted[label]` — stay in log-space, gather one value.

All three require summing `exp()` over all `C` classes somewhere to
normalize — there's no way to get a valid probability for one class
without touching every class. Formulation 3 is preferred because it:
- never materializes actual softmax probabilities, so it can't underflow
  a confident-wrong prediction to `0.0` and then hit `log(0) = -inf`,
  and
- never allocates an `[N, C]` one-hot array or wastes a multiply on the
  `C - 1` zero terms per row — it gathers the needed value directly.

## `cross_entropy_grad`

### Implementation

```python
def cross_entropy_grad(logits, labels):
    n = logits.shape[0]
    probs = softmax(logits, axis=-1)                # [N, C]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return (probs - onehot) / n
```

Verified against a numerical (finite-difference) gradient of
`cross_entropy_loss`; max abs diff ~8.6e-11.

Unlike `cross_entropy_loss`, this is safe to build on top of `softmax`
directly — there's no `log()` afterward, so there's no underflow-to-zero
risk to avoid.

### Derivation — why `probs - onehot`, divided by `n`

Start from the per-sample loss already derived above:

```
L_i = LSE(z_i) - z_i[label_i]
```

We need `dL_i/dz_i[c]` for every class `c` (one column per class of the
gradient row for sample i).

**Term 1 — `LSE(z_i)`:** this is the standard log-sum-exp derivative —
the derivative of `log(sum(exp(x)))` w.r.t. `x[c]` is `softmax(x)[c]`:

```
d/dz_i[c] of LSE(z_i) = exp(z_i[c]) / sum_k exp(z_i[k]) = probs[i, c]
```

So this term's gradient, by itself, is the entire `probs` matrix.

**Term 2 — `z_i[label_i]`:**

```
d/dz_i[c] of z_i[label_i] = 1 if c == label_i else 0 = onehot[i, c]
```

1 only at the true-class column, 0 elsewhere — the definition of `onehot`.

**Combine:**

```
dL_i/dz_i[c] = probs[i, c] - onehot[i, c]
```

`probs - onehot` isn't a separate trick — it drops straight out of
differentiating the two pieces of the loss.

**Why divide by `n`:** the actual loss is a *mean* over the batch,
`loss = (1/N) * sum_i L_i`. By the chain rule this just scales every
per-sample gradient by the same constant:

```
d(loss)/dz_i[c] = (1/N) * dL_i/dz_i[c] = (1/N) * (probs[i,c] - onehot[i,c])
```

Sample `i`'s logits only affect `L_i`, not any other sample's loss, so
each row of the gradient only carries its own `(probs - onehot)` row,
scaled by the `1/N` the mean introduces everywhere — hence dividing the
whole `[N, C]` matrix by `n` at the end.

**Intuition:** `probs - onehot` is a "prediction minus target" residual —
push every class's probability down proportional to how much mass it
wrongly holds, except the true class, which gets pushed toward 1. It's
the same residual-style gradient as squared-error in linear regression;
softmax + cross-entropy is the categorical analogue where it drops out
just as cleanly.
