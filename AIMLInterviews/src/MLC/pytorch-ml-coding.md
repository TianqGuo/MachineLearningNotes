# PyTorch ML Coding Problems: Interview Questions and Reference Responses

[Back to Chapter 2: ML/Data Coding](./ml-coding.md)

This guide covers Python and PyTorch ML utilities through a timed mock interview, practice problems, coding challenges, and concise reference responses. The answers are representative, not the only acceptable solutions. In a live interview, a strong candidate should state assumptions, discuss trade-offs, handle edge cases, and explain how the code would be tested.

## Contents

- [Mock interview: 60-minute session](#mock-interview-60-minute-session)
- [Practice question bank with reference responses](#practice-question-bank-with-reference-responses)
- [Supplemental PyTorch core and advanced questions](#supplemental-pytorch-core-and-advanced-questions)
- [Quick drill: general Python and PyTorch ML utilities](#quick-drill-general-python-and-pytorch-ml-utilities)
- [Broader discussion questions with reference responses](#broader-discussion-questions-with-reference-responses)

## Mock interview: 60-minute session

### 1. Tensor fundamentals — 10 minutes

#### Question

Implement the following function without Python loops:

```python
def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize each row to zero mean and unit variance."""
    ...
```

Requirements:

- `x` has shape `[batch_size, num_features]`.
- Preserve `x`'s device and dtype.
- Avoid division by zero.
- Do not modify `x` in place.

#### Reference response

```python
import torch


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, received shape {tuple(x.shape)}")
    if not (x.is_floating_point() or x.is_complex()):
        raise TypeError("x must have a floating-point or complex dtype")

    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std
```

The reduction uses `keepdim=True`, so the `[B, 1]` mean and standard deviation broadcast correctly across `[B, D]`. `clamp_min` makes constant rows safe; after centering, those rows become zeros. The operations are out of place and naturally remain on the input device with the input dtype.

#### Follow-up responses

1. **What should happen when a row has zero variance?** After subtracting its mean, a constant row contains only zeros. Clamping its standard deviation to `eps` therefore produces a finite all-zero output.
2. **How would this change for `[B, C, H, W]`?** For per-sample normalization, reduce over `(1, 2, 3)`. For per-channel dataset normalization, reduce over `(0, 2, 3)`. The correct dimensions depend on the intended semantics.
3. **`view`, `reshape`, `squeeze`, and `unsqueeze`?** `view` requires compatible strides, while `reshape` may return a view or allocate a copy. `squeeze` removes size-one dimensions, and `unsqueeze` inserts a size-one dimension.
4. **When can broadcasting be wrong?** A tensor shaped `[B]` may align with the last dimension of `[B, D]` instead of the batch dimension, sometimes without an error when dimensions happen to match. Make intended broadcast dimensions explicit, such as `[B, 1]`.

### 2. Batch preprocessing — 10 minutes

#### Question

Write a reusable preprocessing component that replaces non-finite values, standardizes features using training statistics, works on CPU or GPU, saves its state, and prevents validation leakage.

#### Reference response

```python
import torch
from torch import nn


class FeatureStandardizer(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("scale", torch.ones(num_features))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "FeatureStandardizer":
        if x.ndim != 2 or x.shape[1] != self.mean.numel():
            raise ValueError("Expected [batch, num_features] with the configured width")

        # Accumulate statistics in float32 for low-precision input.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        work = torch.nan_to_num(work)
        mean = work.mean(dim=0)
        scale = work.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.scale.copy_(scale.to(device=self.scale.device, dtype=self.scale.dtype))
        self.is_fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted):
            raise RuntimeError("FeatureStandardizer must be fitted before use")
        clean = torch.nan_to_num(x)
        return (clean - self.mean.to(dtype=clean.dtype)) / self.scale.to(dtype=clean.dtype)
```

Fit the module on training data only, then call `transform`/`forward` unchanged for validation, test, and inference. Buffers are appropriate because the statistics are state but are not optimized parameters; they move with `.to(device)` and are included in `state_dict()`.

Follow-ups:

- A function is appropriate for stateless preprocessing; a class or `nn.Module` is preferable when fitted state must be saved and moved across devices.
- Statistics should be buffers, not parameters, because an optimizer should not update them.
- Low-precision inputs should use higher-precision accumulation for statistics, then cast deliberately for transformation.
- A leakage test can fit on a known training subset, alter validation values drastically, and verify that stored statistics remain unchanged.

### 3. Classification metrics — 10 minutes

#### Question

Implement binary precision, recall, and F1 from logits. Inputs may be on GPU, targets may contain only one class, no autograd graph should be created, and the result should contain Python floats.

#### Reference response

```python
import torch


@torch.inference_mode()
def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    predictions = torch.sigmoid(logits) >= threshold
    actual = targets.bool()

    tp = (predictions & actual).sum()
    fp = (predictions & ~actual).sum()
    fn = (~predictions & actual).sum()

    def divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator.float() / denominator, 0.0)

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }
```

Follow-ups:

- Logits are unbounded scores. Apply `sigmoid` before comparing them with a probability threshold; alternatively, convert the probability threshold to its equivalent logit threshold.
- Accuracy can hide failure on a rare positive class. Precision, recall, F1, PR-AUC, and cost-sensitive metrics often provide more useful information.
- For multiclass metrics, form class predictions with `argmax`, compute per-class TP/FP/FN, and expose micro, macro, and weighted aggregation.
- Aggregate sufficient statistics globally. Averaging a nonlinear metric such as F1 across batches generally does not equal dataset-level F1.

### 4. Training and evaluation loops — 20 minutes

#### Question

Implement one training epoch and a complete evaluation pass with correct modes, gradient handling, device placement, sample-weighted loss, empty-loader behavior, and final predictions.

#### Reference response

```python
import torch


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataloader")
    return {"loss": total_loss / total_examples}


@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    targets_out = []

    try:
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            predictions.append(outputs.detach().cpu())
            targets_out.append(targets.detach().cpu())
    finally:
        model.train(previous_mode)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataloader")

    return {
        "loss": total_loss / total_examples,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets_out),
    }
```

This assumes `loss_fn` returns a batch mean. Multiplying by batch size and dividing by the total number of examples prevents the smaller final batch from receiving equal weight.

Follow-ups:

- `model.eval()` changes module behavior such as Dropout and BatchNorm; `inference_mode()` disables autograd bookkeeping. They solve different problems.
- Clear gradients before the forward/backward pass for the next optimizer update.
- Clip gradients after `backward()` and, with AMP, after unscaling but before `optimizer.step()`.
- AMP uses `autocast` around the forward/loss calculation and `GradScaler` for scaled backward and optimizer steps on CUDA.
- For accumulation, divide loss by the accumulation count, call `backward()` for each microbatch, and step/clear gradients only at the accumulation boundary.
- Exact resumption may require model, optimizer, scheduler, scaler, epoch/step, sampler position, configuration, and Python/NumPy/PyTorch RNG states.

### 5. Code review and debugging — 10 minutes

#### Question

Review the following code:

```python
def evaluate(model, loader):
    model.train()
    losses = []

    for x, y in loader:
        prediction = model(x)
        loss = torch.nn.CrossEntropyLoss()(prediction, y)
        losses.append(loss)

    return torch.tensor(losses).mean().item()
```

#### Reference response

Problems include:

- `model.train()` enables training behavior during evaluation.
- Inputs are not moved to the model's device.
- Autograd remains enabled, and storing losses retains computation graphs.
- `torch.tensor(losses)` attempts to construct a new tensor from tensors and can cause device or conversion errors.
- Each batch is weighted equally even when batch sizes differ.
- The loss object is recreated every iteration.
- Empty loaders are not handled.
- The model's prior mode is not restored.

A corrected version is:

```python
@torch.inference_mode()
def evaluate(model, loader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0

    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    finally:
        model.train(previous_mode)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return total_loss / total_examples
```

## Practice question bank with reference responses

### Python and clean utility design

1. **Write a function that splits a sequence into batches while optionally retaining the final incomplete batch.**
   Validate that `batch_size > 0`, iterate over `range(0, len(items), batch_size)`, and yield slices. Skip the last slice only when `drop_last=True` and its length is smaller than `batch_size`.

2. **Implement a reproducible train/validation split without scikit-learn.**
   Create indices, shuffle them with a local seeded generator, and split at a documented boundary. Validate the ratio and avoid global RNG state so unrelated random calls do not change the result.

3. **Write a generator that lazily reads and transforms records from a large dataset.**
   Open the source in a context manager, parse and transform one record at a time, and `yield` it. Decide whether malformed rows should raise, be skipped with logging, or be sent to an error stream.

4. **Flatten a nested dictionary while preserving a reversible mapping.**
   Traverse recursively and store keys as tuples such as `("model", "hidden_size")`. Tuple paths avoid delimiter-escaping problems and can be replayed to reconstruct the nested dictionaries.

5. **Merge metric dictionaries and compute weighted averages.**
   Require each record to include a weight such as sample count, sum `metric * weight`, and divide by total weight per key. Define behavior for missing keys and zero total weight explicitly.

6. **Implement an execution-time decorator.**
   Use `functools.wraps`, capture `time.perf_counter()` before and in a `finally` block after the call, and return the original result unchanged. A `finally` block also records timing when the function raises.

7. **Design a registry for losses, optimizers, or preprocessing functions.**
   Use an explicit mapping from validated names to constructors, for example `REGISTRY = {"adamw": torch.optim.AdamW}`. Avoid `eval`; produce an error listing supported names and keep object construction separate from lookup.

8. **Refactor a monolithic training script into testable modules.**
   Separate configuration, data loading, preprocessing, model construction, training, evaluation, metrics, checkpointing, and logging. Pass dependencies explicitly so each component can be tested with small fakes or synthetic tensors.

9. **Define a typed configuration object.**
   Use nested dataclasses or a validated settings model with explicit defaults and types. Validate cross-field constraints, serialize the resolved configuration with checkpoints, and prefer immutability when runtime mutation is unnecessary.

10. **Make a utility deterministic without mutable global state.**
    Accept a seed or generator as an argument and use local objects such as `random.Random(seed)` or `torch.Generator().manual_seed(seed)`. Determinism should be controlled by the caller rather than hidden module state.

11. **When should you use a dataclass, function, class, or abstract base class?**
    Use a function for stateless behavior, a dataclass for structured data, a class for stateful behavior or lifecycle, and an abstract base class only when multiple implementations need a stable enforced contract.

12. **What is wrong with mutable default arguments?**
    The object is created once and shared across calls. Use `None` as the default and create the list or dictionary inside the function.

13. **Explain shallow versus deep copies with an ML configuration.**
    A shallow copy creates a new outer dictionary but shares nested objects, so changing `copy["optimizer"]["lr"]` can affect the original. A deep copy recursively duplicates nested mutable values, though immutable objects and external resources need special consideration.

14. **Design errors for an invalid preprocessing shape or dtype.**
    Name the argument, state the expected contract, report the received shape/dtype, and suggest the likely fix. For example: `Expected features with shape [B, 32] and floating dtype; received [32] torch.int64`.

15. **Accept either NumPy arrays or PyTorch tensors without unclear behavior.**
    Document one canonical internal representation and a clear return policy. For example, convert inputs with `torch.as_tensor`, preserve device only for tensor inputs, and either always return a tensor or explicitly restore the original backend.

### Tensor operations

16. **Compute the masked mean of `[B, T, D]` using a `[B, T]` mask.**
    Convert the mask to `x.dtype`, expand it with `mask.unsqueeze(-1)`, sum `x * mask` over time, and divide by `mask.sum(dim=1, keepdim=True).clamp_min(1)`. Define whether an all-padding sequence returns zeros or raises.

17. **Compute pairwise cosine similarity for `[N, D]` and `[M, D]`.**
    Normalize both tensors along the feature dimension with `F.normalize`, then compute `a_normalized @ b_normalized.T`, producing `[N, M]`.

18. **Determine whether the true label is in the top-k predictions.**
    Compute `topk_indices = logits.topk(k, dim=1).indices`, compare with `targets[:, None]`, and call `.any(dim=1)`. Validate that `1 <= k <= num_classes`.

19. **Implement one-hot encoding without `F.one_hot`.**
    Allocate zeros of shape `[B, C]` on the label device and use `scatter_(1, labels[:, None], 1)`. Validate label range; use an appropriate floating dtype if the result will feed a loss.

20. **Create label-smoothed targets.**
    Fill a `[B, C]` tensor with `smoothing / (C - 1)` and scatter `1 - smoothing` into the true-class column. State whether smoothing mass includes or excludes the true class, because both conventions exist.

21. **Compute a batched confusion matrix without per-example loops.**
    Encode each pair as `targets * C + predictions`, apply `torch.bincount(..., minlength=C*C)`, and reshape to `[C, C]`. Accumulate these matrices across batches.

22. **Pad variable-length tensors and create a mask.**
    Use `pad_sequence(sequences, batch_first=True)`, compute lengths, and compare `torch.arange(max_len)[None, :] < lengths[:, None]`. Keep the lengths or mask with the padded batch.

23. **Gather one class score per example from `[B, C]`.**
    Use `logits.gather(1, labels[:, None]).squeeze(1)`. This makes the batch-wise indexing intent explicit.

24. **Advanced indexing versus `gather` versus `index_select`?**
    Advanced indexing is flexible and can select using multiple index tensors. `gather` takes per-position indices along one dimension, while `index_select` applies one shared 1D index list to an entire dimension.

25. **Diagnose CPU targets with GPU predictions.**
    PyTorch operations generally require participating tensors on the same device. Move the complete batch recursively to the model's device before the forward/loss computation, while avoiding device transfers hidden deep inside utilities.

26. **Explain contiguous tensors and `.contiguous()`.**
    Transposes and some slices produce views with nonstandard strides. `view` may reject them; `.contiguous()` creates standard contiguous storage, while `reshape` may handle the case by copying when required.

27. **Find the `squeeze()` batch-size-one bug.**
    Calling `squeeze()` without a dimension removes every size-one axis, possibly deleting the batch dimension. Use `squeeze(-1)` or another explicitly intended dimension.

28. **How do in-place operations affect autograd?**
    They change tensor storage that a backward formula may need and can trigger version-counter errors or silently complicate reasoning. Avoid them on leaf tensors requiring gradients and on intermediates needed for backward unless their safety is well understood.

29. **Compare `detach()`, `clone()`, and `torch.no_grad()`.**
    `detach()` returns a tensor sharing storage but disconnected from the current graph. `clone()` copies data while retaining gradient history, and `no_grad()` disables graph recording for operations executed inside its context.

30. **Why can `.item()` slow a GPU loop?**
    Converting a GPU scalar to a Python value forces synchronization so the CPU waits for queued GPU work. Aggregate on device and transfer less frequently when possible.

### Preprocessing and datasets

31. **Implement `fit`, `transform`, and `inverse_transform` standardization.**
    `fit` stores training-only means and safe scales, `transform` applies `(x - mean) / scale`, and `inverse_transform` applies `z * scale + mean`. Validate feature width and serialize the fitted state.

32. **Normalize images with per-channel statistics.**
    For `[B, C, H, W]`, reshape mean and std to `[1, C, 1, 1]` and compute `(x - mean) / std.clamp_min(eps)`. Confirm the expected input range, color order, and dtype.

33. **Implement a custom `Dataset` for numerical and categorical features.**
    Store immutable references or indexed arrays, implement `__len__`, and have `__getitem__` return a documented structure such as `{"numeric": ..., "categorical": ..., "target": ...}`. Fit encoders outside the dataset on training data only.

34. **Write a `collate_fn` for variable-length sequences.**
    Extract sequences, record lengths, pad them with `pad_sequence`, create a validity mask, and stack labels and identifiers. Keep the output structure consistent with the training loop.

35. **Handle missing values without leakage.**
    Fit numerical imputation values and categorical vocabularies on training data only. Store a missing indicator or sentinel category when useful, and reuse the frozen preprocessing state for validation and inference.

36. **Handle an unseen category at inference.**
    Map it to a reserved unknown token, use hashing, or reject it with a clear contract. Never extend a learned embedding vocabulary silently unless the model and deployment process explicitly support it.

37. **Where should preprocessing live?**
    Per-example parsing often belongs in the dataset, padding and batch assembly in the collator, differentiable/exportable transforms in the model, and request validation plus production parity in the serving layer. Choose one owner for every transform to avoid duplication.

38. **Make stochastic augmentation reproducible with DataLoader workers.**
    Seed the DataLoader generator and derive distinct worker seeds in `worker_init_fn`; also seed libraries used inside workers. If augmentations should change by epoch, incorporate the epoch into the seed deliberately.

39. **Why might multiple DataLoader workers be slower?**
    Startup, serialization, interprocess communication, duplicated memory, small datasets, or storage contention can exceed the work saved. Profile worker counts, batch size, persistent workers, prefetching, and the actual transform cost.

40. **Pinned memory and non-blocking transfers?**
    Page-locked host memory enables faster asynchronous host-to-GPU copies. `non_blocking=True` is useful when the source is pinned and computation/copy overlap is organized correctly; it is not automatically faster in every pipeline.

### Metrics

41. **Implement binary accuracy from logits.**
    Convert logits to predictions with `logits >= 0` or `sigmoid(logits) >= 0.5`, compare with Boolean targets, and average the equality tensor in floating point.

42. **Implement multiclass top-k accuracy.**
    Take `logits.topk(k, dim=1).indices`, compare against `targets[:, None]`, reduce with `any(dim=1)`, and average. Report both the numerator and denominator for distributed aggregation.

43. **Micro, macro, and weighted F1?**
    Micro aggregates TP/FP/FN before computing F1, macro averages class F1 equally, and weighted F1 weights each class by support. State how zero-support classes are handled.

44. **Masked mean squared error?**
    Compute squared error, select or multiply by a Boolean mask, sum valid error, and divide by the valid count. Raise or return a documented sentinel when no target is valid.

45. **MAPE and its failure cases?**
    MAPE averages `abs(prediction - target) / abs(target)`, so zero or near-zero targets make it undefined or unstable. Alternatives include MAE, WAPE, sMAPE, or a domain-specific denominator rule.

46. **Confusion matrix without storing all predictions?**
    Maintain a `[C, C]` integer tensor and add each batch's `bincount`-based matrix. In distributed evaluation, sum the matrices across workers before deriving metrics.

47. **Design a streaming metric.**
    `update` accumulates sufficient statistics, `compute` derives the final metric without mutating state, and `reset` clears it. Make device, dtype, distributed reduction, and empty-state behavior explicit.

48. **Why is averaging batch F1 wrong?**
    F1 is a nonlinear ratio of aggregate counts. Compute it once from dataset-level TP/FP/FN rather than averaging ratios calculated from differently sized or distributed batches.

49. **Metrics for imbalanced fraud detection?**
    Use precision, recall, PR-AUC, F-beta, recall at a fixed precision, and expected business cost. Evaluate calibration and threshold behavior; plain accuracy and sometimes ROC-AUC can obscure rare-positive performance.

50. **How should a classification threshold be selected?**
    Choose it on a validation set using a predeclared cost or metric objective, then lock it before the final test evaluation. Revisit it when prevalence, costs, or calibration drift in production.

### Training and evaluation

51. **Implement early stopping.**
    Track the best validation value and checkpoint, count evaluations without an improvement larger than `min_delta`, and stop after `patience` failures. Define metric direction, ties, warm-up, and whether patience counts epochs or evaluations.

52. **Add gradient clipping.**
    Call `loss.backward()`, unscale gradients first when using AMP, then call `clip_grad_norm_` or `clip_grad_value_` before the optimizer step. Log the pre-clipping norm when diagnosing instability.

53. **Add gradient accumulation.**
    Divide each microbatch loss by `accumulation_steps`, call backward for every microbatch, and step/clear gradients at each boundary. Handle the final partial group and DDP synchronization deliberately.

54. **Add automatic mixed precision.**
    Run forward and loss computation inside the device-appropriate `autocast` context, then use `GradScaler.scale(loss).backward()`, `step`, and `update` on CUDA. Keep numerically sensitive operations in higher precision when needed.

55. **Save and resume a checkpoint exactly.**
    Save model, optimizer, scheduler, AMP scaler, epoch/global step, best metric, configuration, data/sampler position when relevant, and Python/NumPy/PyTorch RNG states. Restore objects before continuing the next batch.

56. **When should a learning-rate scheduler step?**
    Follow the scheduler's semantics: OneCycle and many warm-up schedules step per optimizer update, StepLR commonly steps per epoch, and ReduceLROnPlateau steps after a monitored validation metric. Document the order relative to `optimizer.step()`.

57. **Freeze and later unfreeze a backbone.**
    Set `requires_grad=False` for frozen parameters and build the optimizer from trainable parameters. On unfreezing, set selected parameters true and rebuild or extend optimizer groups, often with a smaller learning rate.

58. **Why does validation differ between `train()` and `eval()`?**
    Dropout is stochastic in training mode, while BatchNorm uses batch statistics and updates running estimates. Evaluation mode disables Dropout and uses stored BatchNorm statistics.

59. **Prevent gradients during validation.**
    Use both `model.eval()` and `torch.inference_mode()` or `torch.no_grad()`. Do not call backward or optimizer methods, and detach any tensors retained after the loop.

60. **Calculate epoch loss with a smaller final batch.**
    If the criterion returns a batch mean, accumulate `loss.item() * batch_size` and divide by total examples. For token-level losses, weight by the number of valid tokens instead.

61. **Handle a non-finite loss.**
    Detect it with `torch.isfinite`, log the batch and relevant diagnostics, and follow an explicit policy: stop, skip, reduce scale, or recover from a checkpoint. In distributed training, ensure all ranks make the same decision.

62. **Support classification and regression in one loop.**
    Keep the loop task-agnostic and inject a task adapter that prepares targets, computes loss, converts outputs to predictions, and updates metrics. Avoid scattered `if task == ...` branches throughout the loop.

63. **Add callback hooks cleanly.**
    Define a small lifecycle interface such as `on_train_start`, `on_batch_end`, `on_validation_end`, and `on_exception`. Pass read-only context or controlled state so logging and checkpoint callbacks do not own core optimization logic.

64. **Move nested batches to a device.**
    Use a recursive tree-map that calls `.to(device)` on tensors, preserves mappings/sequences, and leaves strings or identifiers unchanged. Centralize it so every loop uses identical behavior.

65. **Preserve input order and sample identifiers during prediction.**
    Include stable IDs in dataset outputs and return them with predictions. Disable shuffling for inference or sort collected outputs back to a recorded source index; account for distributed sampler padding.

### Testing and debugging

66. **Test a standardization utility, including constant features.**
    Verify fitted means/scales, near-zero transformed mean, near-unit variance for varying features, finite all-zero output for constant features, inverse round-trip, shape errors, state serialization, and CPU/GPU behavior.

67. **Verify evaluation does not update model state.**
    Snapshot parameters and BatchNorm buffers, run evaluation, and compare the complete state afterward. Also assert `model.training` is restored as intended and outputs do not require gradients.

68. **Compare CPU and GPU metric implementations.**
    Run identical fixed inputs on both devices and compare outputs with dtype-appropriate tolerances. Avoid nondeterministic operations or document expected tolerance when exact equality is unrealistic.

69. **Test empty input, batch size one, and unexpected shapes.**
    Parameterize cases and assert either a defined output or a clear exception. Include scalar/singleton dimensions, zero-length axes, noncontiguous tensors, wrong dtypes, and mismatched devices where relevant.

70. **Use a tiny dataset to verify overfitting.**
    Train on one small batch with augmentation and regularization disabled. A sufficiently expressive model should drive training loss very low; failure suggests a bug in data, labels, loss, gradients, or optimizer steps.

71. **Diagnose a loss that never changes.**
    Check label correctness, learning rate, model mode, `requires_grad`, nonzero gradients, optimizer parameter groups, `zero_grad`/`backward`/`step` order, detached outputs, loss compatibility, and whether parameters actually change.

72. **Diagnose validation changes across identical runs.**
    Check seeds, evaluation mode, stochastic transforms, DataLoader worker seeds, sample order, nondeterministic GPU kernels, uninitialized state, data races, and floating-point reduction order.

73. **Find a memory leak from retained graphs.**
    Look for lists or logs storing `loss`, outputs, or hidden states without detaching. Store `.item()` for scalars or `.detach().cpu()` for tensors, and avoid `retain_graph=True` unless it is genuinely required.

74. **Check gradients numerically.**
    Use `torch.autograd.gradcheck` with double-precision inputs requiring gradients, small well-conditioned values, and a function returning differentiable outputs. Avoid nondifferentiable boundaries in the test points.

75. **Verify identical predictions after checkpoint restoration.**
    Save a model, load it into a fresh instance, put both in evaluation mode, and compare outputs on the same fixed input. Also test device mapping and use appropriate floating-point tolerances.

## Supplemental PyTorch core and advanced questions

The following 50-question supplement is based on the GitHub-formatted excerpt supplied with this guide and the [full question index published by Devinterview](https://devinterview.io/questions/machine-learning-and-data-science/pytorch-interview-questions/). The responses below are newly written, condensed, and updated for current PyTorch practices rather than copied from the source.

### PyTorch fundamentals

1. **What is PyTorch, and how does it differ from frameworks such as TensorFlow?**
   PyTorch is an open-source tensor and deep-learning framework with eager execution, automatic differentiation, accelerator support, distributed training, compilation, and deployment tooling. PyTorch is strongly integrated with Python and research workflows; TensorFlow has a different ecosystem and deployment history. Modern versions of both support eager execution and graph optimization, so the choice should consider team expertise, required runtimes, libraries, and production constraints rather than the outdated dynamic-versus-static distinction alone.

2. **What are tensors in PyTorch?**
   A tensor is a typed, multidimensional array with a shape, strides, dtype, layout, and device. Unlike a plain NumPy array, it can participate in autograd, run on accelerators, and use dense, sparse, quantized, or other specialized representations.

3. **What is the difference between a Tensor and a Variable?**
   There is no practical distinction in modern PyTorch. `Variable` was merged into `torch.Tensor` in PyTorch 0.4; tensors now carry `requires_grad`, `grad`, and `grad_fn` behavior directly. New code should not wrap tensors in `Variable`.

4. **How do you convert a NumPy array to a tensor?**
   `torch.from_numpy(array)` and usually `torch.as_tensor(array)` share compatible CPU memory with the NumPy array, while `torch.tensor(array)` copies and infers or accepts a dtype. Use `.clone()` when independent storage is required, and move to an accelerator explicitly after conversion.

5. **What is the purpose of `.grad`?**
   After `backward()`, `.grad` stores the accumulated gradient for leaf tensors whose `requires_grad` flag is true, including model parameters. Non-leaf tensors do not retain `.grad` by default; call `retain_grad()` if it is needed for debugging. Gradients accumulate until cleared or set to `None`.

6. **What is CUDA, and how does it relate to PyTorch?**
   CUDA is NVIDIA's GPU computing platform. A CUDA-enabled PyTorch build can allocate tensors and execute supported operations on NVIDIA GPUs. Models and participating tensors must be placed on compatible devices explicitly, and performance depends on batching, memory movement, kernel efficiency, and synchronization—not merely calling `.cuda()`.

7. **How does autograd work?**
   When grad mode is enabled, PyTorch records operations involving tensors that require gradients into a dynamic graph. Calling `backward()` applies reverse-mode automatic differentiation and the chain rule, accumulating derivatives into leaf tensors. Saved intermediates are normally released after backward unless the graph is retained.

### Neural-network design

8. **What are the main steps for creating a neural-network model?**
   Define the data and metric contracts, create an `nn.Module`, initialize it deliberately, choose a compatible loss and optimizer, build train/validation loops, verify it can overfit a tiny dataset, train with monitoring, evaluate on protected data, and package the model with preprocessing and configuration for inference.

9. **How does `nn.Sequential` differ from subclassing `nn.Module`?**
   `nn.Sequential` composes modules in a single linear chain. A custom `nn.Module` supports branching, residual connections, multiple inputs or outputs, conditional computation, shared layers, and other nonsequential behavior. `Sequential` itself is also an `nn.Module`.

10. **How do you implement a custom layer?**
    Subclass `nn.Module`, create child modules, `nn.Parameter` values, or registered buffers in `__init__`, and implement tensor operations in `forward`. Initialize parameters explicitly when defaults are unsuitable and test shapes, gradients, devices, dtypes, state serialization, and compilation behavior.

11. **What is the role of `forward`?**
    `forward` defines the module's tensor computation. Call `module(inputs)` rather than invoking `forward` directly so PyTorch can run hooks and other `nn.Module.__call__` behavior around the computation.

### Training and optimization

12. **What are optimizers, and how are they used?**
    Optimizers update selected parameters using their gradients and optimizer state such as momentum or adaptive moments. A normal update is `zero_grad`, forward, loss, backward, and `step`. Save optimizer state in checkpoints when training must resume faithfully.

13. **Why and when do you call `zero_grad()`?**
    PyTorch accumulates gradients, so they must be cleared before beginning a logically new optimizer update unless intentional accumulation is being used. `optimizer.zero_grad(set_to_none=True)` often saves work and makes missing gradients distinguishable from explicit zeros; it does not perform backward itself.

14. **How do you implement learning-rate scheduling?**
    Create a scheduler tied to the optimizer and step it according to its contract: per optimizer update, per epoch, or after a validation metric. For example, OneCycle-style schedules usually step per update, StepLR commonly steps per epoch, and ReduceLROnPlateau consumes a monitored metric.

15. **Describe backpropagation in PyTorch.**
    The forward pass creates outputs and a scalar or explicitly seeded loss. `loss.backward()` traverses the recorded graph in reverse and accumulates parameter gradients. The optimizer then reads those gradients to update parameters; backpropagation calculates gradients, while the optimizer performs the update.

16. **How does gradient clipping work, and why is it useful?**
    Gradient clipping limits gradient values or the global norm to reduce unstable updates, especially in recurrent, deep, or mixed-precision training. Apply it after backward, and after unscaling when using AMP, but before the optimizer step. It mitigates symptoms and does not replace diagnosing bad objectives, data, or learning rates.

### Debugging and model improvement

17. **How do you check whether a model is using the GPU?**
    Inspect `next(model.parameters()).device` and representative input/output devices, and verify accelerator utilization and memory with the PyTorch profiler or system tools. GPU allocation alone does not demonstrate useful utilization; profile kernels, input stalls, transfers, and synchronization.

18. **How do you monitor and reduce overfitting?**
    Compare training and validation curves on representative splits, inspect slice metrics, and check for leakage first. Possible controls include more or better data, augmentation, weight decay, dropout, early stopping, smaller capacity, transfer learning, label smoothing, and improved validation design.

19. **What is batch normalization, and how does it affect training?**
    BatchNorm normalizes activations using batch statistics during training, then applies learned scale and bias while updating running estimates. It can improve optimization and permit larger learning rates, but small or nonrepresentative batches can make estimates noisy. `train()` and `eval()` modes are therefore behaviorally different.

20. **How does PyTorch initialize neural-network weights?**
    Standard modules define default `reset_parameters` methods. Custom initialization can be applied under `torch.no_grad()` or with `model.apply`, using methods such as Xavier or Kaiming initialization chosen for the activation and architecture. Initialization should also address biases, normalization parameters, embeddings, and reproducibility.

21. **What common training problems occur, and how do you debug them?**
    Common failures include device or shape mismatches, incorrect targets or loss functions, missing or exploding gradients, non-finite values, data leakage, wrong model mode, excessive synchronization, input bottlenecks, and failure to update parameters. Start with assertions and a tiny overfit test, inspect gradients and parameter deltas, then profile only after correctness is established.

### Data handling and preprocessing

22. **How do you build a DataLoader for a custom dataset?**
    Implement a map-style `Dataset` with `__len__` and `__getitem__`, or an `IterableDataset` for streams, then pass it to `DataLoader` with deliberate batch, shuffle or sampler, worker, collation, and memory settings. Use a custom `collate_fn` for variable-length or nested samples.

23. **What are `torchvision` transforms used for?**
    Transforms decode, convert, resize, normalize, and augment image-like inputs. Stochastic training transforms should be separated from deterministic validation/inference preprocessing. Their expected range, dtype, channel order, and target transformations must remain consistent with the model.

24. **How do you preprocess time-series data for RNNs?**
    Split chronologically to prevent future leakage, fit normalization on training data, build input/target windows, retain lengths and masks, and pad or pack variable-length sequences. Decide whether hidden state resets between sequences and avoid mixing unrelated entities across a continuous state.

25. **What is data augmentation, and how is it implemented?**
    Augmentation samples label-preserving variations to improve generalization and robustness. It can live in dataset or transform pipelines and may include geometric, photometric, temporal, mixing, or domain-specific operations. Validate that each transformation preserves task semantics and seed workers when reproducibility is required.

### Advanced topics

26. **How do you use GPUs for distributed PyTorch training?**
    `DistributedDataParallel` normally runs one process per GPU, partitions input with a distributed sampler, and synchronizes gradients with collective communication. Choose an appropriate backend, set devices before constructing the model, checkpoint carefully, and ensure all ranks follow compatible control flow. Use sharded approaches such as FSDP when model or optimizer state does not fit per device.

27. **How do you implement transfer learning?**
    Load a pretrained model, replace its task-specific head, apply the required preprocessing, and first train the new head with the backbone frozen when appropriate. Then selectively unfreeze layers, use smaller learning rates for pretrained weights, and monitor catastrophic forgetting and domain mismatch.

28. **Compare RNNs, LSTMs, and GRUs.**
    A basic RNN has a simple recurrent state but is vulnerable to vanishing or exploding gradients. LSTMs add input, forget, and output gates plus a cell state; GRUs use a simpler gated state with fewer parameters. The best choice depends on sequence length, data, latency, and whether newer convolutional or attention architectures are more appropriate.

29. **What is TorchScript, and how does it help deployment?**
    TorchScript historically converted PyTorch programs into a serializable representation runnable without Python through scripting or tracing. Current PyTorch documentation marks TorchScript as deprecated and directs new export workflows toward `torch.export`; existing TorchScript deployments may still require maintenance and migration planning.

### Coding challenges

30. **Implement a DataLoader for a CSV dataset.**

    ```python
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset


    class CsvDataset(Dataset):
        def __init__(self, path: str, feature_cols: list[str], target_col: str):
            frame = pd.read_csv(path)
            self.x = torch.tensor(frame[feature_cols].to_numpy(), dtype=torch.float32)
            self.y = torch.tensor(frame[target_col].to_numpy(), dtype=torch.long)

        def __len__(self) -> int:
            return self.y.shape[0]

        def __getitem__(self, index: int):
            return self.x[index], self.y[index]


    loader = DataLoader(
        CsvDataset("train.csv", ["f1", "f2"], "label"),
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )
    ```

    For large files, avoid loading the entire CSV per worker; preprocess to a columnar or sharded format or use an iterable pipeline.

31. **Demonstrate slicing, indexing, concatenation, and transposition.**

    ```python
    x = torch.arange(12).reshape(3, 4)
    first_two_columns = x[:, :2]
    selected_rows = x[torch.tensor([0, 2])]
    stacked_rows = torch.cat([x, x], dim=0)
    transposed = x.transpose(0, 1)
    ```

    Explain resulting shapes and remember that transposed tensors may be noncontiguous.

32. **Create a feedforward MNIST model.**

    ```python
    from torch import nn


    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10),
    )

    loss_fn = nn.CrossEntropyLoss()
    ```

    Feed raw logits to `CrossEntropyLoss`; do not add softmax in the model for this training objective.

33. **Manually compute linear-regression gradients.**

    ```python
    def linear_regression_gradients(x, y, weight, bias):
        prediction = x @ weight + bias
        residual = prediction - y
        count = y.numel()
        grad_weight = (2.0 / count) * x.transpose(0, 1) @ residual
        grad_bias = (2.0 / count) * residual.sum(dim=0)
        return grad_weight, grad_bias
    ```

    Compare the result with autograd in a unit test using the same mean-squared-error convention.

34. **Implement a CNN for image classification.**

    ```python
    class SmallCnn(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x).flatten(1)
            return self.classifier(x)
    ```

    Training still requires normalized inputs, a compatible loss, optimizer, validation, and device handling.

35. **Save and load a trained model.**

    ```python
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        "checkpoint.pt",
    )

    checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    ```

    Reconstruct the architecture and configuration before loading, validate artifact provenance, and call `eval()` for inference.

### Case studies and scenarios

36. **How do you handle imbalanced classes?**
    Use stratified evaluation, class-weighted or focal losses, balanced sampling, targeted augmentation, threshold tuning, and metrics such as per-class recall, macro F1, and PR-AUC. Choose methods based on the cost of errors and avoid evaluating with accuracy alone.

37. **How can PyTorch support real-time inference?**
    Use an inference or no-grad context, evaluation mode, controlled batching, compilation or export, lower precision, quantization, and architecture optimization. Measure end-to-end tail latency, warm-up, memory, input processing, concurrency, device transfers, numerical drift, and fallback behavior on target hardware.

38. **When would you convert a model to ONNX?**
    ONNX can be useful when a non-Python runtime, cross-framework interoperability, or a deployment engine expects ONNX. Validate supported operators, dynamic shapes, numerical parity, preprocessing, and performance on the target runtime; conversion is not automatically an optimization.

39. **How would you deploy a model as a REST API?**
    Package a versioned model and preprocessing pipeline behind a service such as FastAPI, load the model once at startup, validate requests, batch when useful, run inference without gradients, and return a stable schema. Add authentication, timeouts, concurrency controls, observability, health checks, canary rollout, and rollback.

40. **How do you fine-tune a pretrained model?**
    Match the pretrained preprocessing and tokenizer or feature contract, replace the task head, establish a frozen-backbone baseline, then unfreeze deliberately with smaller learning rates for pretrained layers. Monitor domain shift, overfitting, catastrophic forgetting, and calibration.

### Advanced topics and research

41. **What are GNNs, and how are they implemented in PyTorch?**
    Graph neural networks update node, edge, or graph representations through message passing over connectivity. Implement the primitives with tensor indexing and sparse operations or use libraries such as PyTorch Geometric. Define batching, neighborhood sampling, aggregation, and permutation invariance carefully.

42. **What are important NAS directions, and how can PyTorch support them?**
    Neural architecture search may use reinforcement learning, evolutionary search, Bayesian optimization, differentiable relaxation, or weight-sharing supernets. Current practical work emphasizes hardware-aware multi-objective search, lower search cost, reliable rankings, and reproducibility. PyTorch can express candidate modules and training loops while an orchestration layer manages trials and measured latency.

43. **How are GANs implemented, and what makes them difficult?**
    Train a generator to fool a discriminator and a discriminator to distinguish real from generated samples, using separate objectives and optimizer steps. Common problems include mode collapse, oscillation, unstable gradients, sensitivity to architecture and hyperparameters, and difficult evaluation. Alternatives or improvements include Wasserstein objectives, gradient penalties, normalization, and careful update ratios.

44. **What is model quantization, and when is it useful?**
    Quantization represents weights and sometimes activations with lower-precision formats to reduce memory, bandwidth, power, or latency. Post-training quantization is cheaper, while quantization-aware training can recover quality by simulating quantization during training. Current PyTorch development centralizes many quantization workflows in `torchao`; always benchmark accuracy and actual target-hardware performance.

45. **What role does PyTorch play in reinforcement learning?**
    PyTorch supplies differentiable policies, value functions, distributions, optimizers, vectorized tensor computation, and accelerator support. For example, an actor-critic implementation samples actions from a policy distribution, estimates returns or advantages, and updates policy and value losses while an environment library handles interaction.

### Practical implementation and contribution

46. **How do you create a custom C++ or CUDA operation?**
    Define and register the operator schema and device implementations, build it with the supported extension tooling, and register autograd or a decomposition when gradients or compilation are required. Test CPU/CUDA parity, shapes, dtypes, noncontiguous tensors, numerical gradients, errors, streams, and performance before choosing a custom kernel over composable PyTorch operations.

47. **How should you discuss open-source contributions or community tools?**
    Use a specific example: the problem, why the tool or contribution was selected, your technical work, review or compatibility challenges, tests and documentation, and measurable impact. If you have not contributed code upstream, describe responsible evaluation, issue reporting, internal extensions, or maintenance of community dependencies honestly.

48. **How should you discuss a project where PyTorch was central?**
    Structure the response around the product problem, data and constraints, architecture and training choices, evaluation design, debugging, deployment, and quantified outcome. Separate your personal decisions from the team's work and explain one trade-off or failure that changed the final system.

49. **How do you improve experiment reproducibility?**
    Record code, environment, configuration, data and artifact versions; seed Python, NumPy, PyTorch, and DataLoader workers; control samplers; and request deterministic algorithms when required. Exact equality is not guaranteed across PyTorch releases, platforms, or CPU/GPU execution, and deterministic kernels can be slower, so define the intended reproducibility boundary.

50. **How can PyTorch Lightning simplify a PyTorch workflow?**
    Lightning can standardize training hooks, device and distributed setup, precision, logging, checkpointing, and common loop mechanics while keeping models in PyTorch. It can reduce boilerplate and enforce team conventions, but candidates should still understand raw PyTorch semantics and weigh framework abstraction, debugging, customization, dependency, and migration costs.

## Quick drill: general Python and PyTorch ML utilities

1. **How would you normalize each row of a tensor without using a Python loop?**

   Compute the mean and standard deviation across the feature dimension and retain that dimension for broadcasting:

   ```python
   def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
       mean = x.mean(dim=1, keepdim=True)
       std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
       return (x - mean) / std
   ```

   `keepdim=True` produces `[B, 1]` statistics that broadcast across `[B, D]`. Clamping protects constant rows from division by zero, and the operations preserve the tensor's device and floating-point dtype.

2. **What is the difference between `view`, `reshape`, and `squeeze`?**

   `view` returns a different shape over the same storage and requires compatible memory strides, so it often fails after a transpose. `reshape` has the same shape semantics but may create a contiguous copy when a view is impossible. `squeeze(dim)` removes a specified dimension only when its size is one. Avoid bare `squeeze()` when batch size can be one because it may accidentally remove the batch dimension.

3. **How would you design a reusable preprocessing component?**

   Give it a small contract such as `fit`, `transform`, and `state_dict`. Fit statistics only on training data, store non-trainable state such as means and scales as registered buffers, validate input shape and dtype, and make device behavior explicit. Keep stochastic training augmentation separate from deterministic validation and inference preprocessing.

4. **What is data leakage, and how do you prevent it during preprocessing?**

   Leakage occurs when information unavailable at training time influences the model or evaluation—for example, fitting a scaler on the full dataset before splitting. Split first, fit preprocessing only on the training set, then reuse the frozen transformation for validation, test, and serving. For temporal data, split chronologically and ensure features do not contain future information.

5. **Explain precision, recall, and F1. When would you use each?**

   Precision is `TP / (TP + FP)` and answers, “Of the predicted positives, how many were correct?” Recall is `TP / (TP + FN)` and answers, “Of the real positives, how many did we find?” F1 is their harmonic mean. Favor precision when false positives are expensive, recall when false negatives are expensive, and F1 when both matter and the classes are imbalanced. The final choice should reflect the product's error costs.

6. **Why is averaging F1 across batches usually incorrect?**

   F1 is a nonlinear function of true positives, false positives, and false negatives. The mean of per-batch F1 scores generally differs from the F1 calculated over the full dataset, especially when batch sizes or class distributions differ. Accumulate TP, FP, and FN across all batches—reducing them across workers in distributed evaluation—and compute F1 once at the end.

7. **What are the essential steps in a PyTorch training loop?**

   Put the model in training mode, move the batch to the correct device, clear old gradients, run the forward pass, compute loss, backpropagate, and update the parameters:

   ```python
   model.train()
   for x, y in loader:
       x, y = x.to(device), y.to(device)
       optimizer.zero_grad(set_to_none=True)
       logits = model(x)
       loss = loss_fn(logits, y)
       loss.backward()
       optimizer.step()
   ```

   Production loops should also handle sample-weighted metrics, non-finite loss checks, logging, checkpointing, and mixed precision when appropriate.

8. **What is the difference between `model.eval()` and `torch.no_grad()` or `torch.inference_mode()`?**

   `model.eval()` changes layer behavior—for example, it disables Dropout and tells BatchNorm to use stored running statistics. `no_grad()` disables gradient recording, while `inference_mode()` additionally removes more autograd overhead and is preferred for pure evaluation when its stricter semantics are acceptable. Evaluation normally needs both `model.eval()` and an inference/no-grad context.

9. **What happens if you forget `optimizer.zero_grad()`?**

   PyTorch accumulates gradients into each parameter's `.grad`. Without clearing them, the next update uses the sum of gradients from multiple backward passes. That is useful only when deliberate gradient accumulation is implemented with the correct loss scaling and optimizer-step schedule; otherwise, it silently changes the optimization behavior.

10. **How would you refactor a monolithic ML script into clean modules?**

    Separate configuration, dataset and preprocessing code, model construction, training, evaluation, metrics, checkpointing, and logging. Keep tensor transformations and metrics as pure functions where possible, make orchestration loops thin, and pass dependencies explicitly instead of relying on global state. Use stable interfaces based on tensors and small result dictionaries rather than tightly coupling every component to one framework abstraction.

11. **How would you write a utility that works on both CPU and GPU?**

    Derive new tensors from existing tensors or create them with the input's `device` and `dtype`; do not call `.cuda()` or create hidden CPU constants inside the function. Move the complete batch at a clear boundary, keep participating tensors on compatible devices, and return results on the input device unless the contract explicitly says otherwise. Test CPU/GPU parity within an appropriate numerical tolerance.

12. **What tests would you write for a tensor preprocessing or metric utility?**

    Test the normal case plus singleton batches, constant values, missing or non-finite values, empty inputs where supported, incorrect ranks, incompatible dtypes, and noncontiguous tensors. Verify expected shapes, device and dtype preservation, numerical output, no unintended mutation, serialization of fitted state, and CPU/accelerator parity. For preprocessing, add a leakage test; for metrics, include all-positive, all-negative, and zero-denominator cases.

## Broader discussion questions with reference responses

1. **What makes an ML utility reusable without excessive abstraction?**
   It should have a small explicit contract, useful defaults, clear shape/device/dtype behavior, composable inputs and outputs, and focused tests. Add abstraction only after multiple real use cases reveal a stable common interface.

2. **Which validation belongs at module boundaries?**
   Validate assumptions that would otherwise fail later or produce a silent wrong result: rank, feature width, dtype family, value range, fitted state, required keys, and incompatible configuration combinations. Avoid repeating expensive checks inside tight inner loops when upstream contracts already guarantee them.

3. **When should a utility return tensors versus Python numbers?**
   Return tensors when callers may need device-side composition, batching, distributed reduction, or gradients. Return Python numbers at reporting or serialization boundaries where synchronization is intentional.

4. **How do you keep preprocessing consistent across training, evaluation, and inference?**
   Implement one versioned preprocessing component, fit it only on training data, serialize its state with the model artifact, and reuse it in evaluation and serving. Test parity with golden examples.

5. **How should an ML codebase be structured for independent testing?**
   Separate pure transformations and metrics from stateful orchestration. Inject the model, loss, optimizer, data source, and callbacks into thin loops so each boundary can be tested with synthetic data and simple fakes.

6. **What can PyTorch control about reproducibility?**
   It can seed its RNGs, request deterministic algorithms, and control generator use, but hardware, library versions, distributed scheduling, data pipelines, and floating-point reduction order can still affect results. Reproducibility is an end-to-end system property.

7. **How do you balance readable tensor code and vectorization?**
   Start with the clearest correct tensor expression, benchmark the actual bottleneck, and optimize only where the gain matters. Preserve readability with named intermediate shapes, assertions, comments about semantics, and equivalence tests.

8. **What should a training loop log?**
   Log resolved configuration, code/data/model versions, epoch and global step, losses, key metrics, learning rate, throughput, gradient norms when useful, validation results, checkpoint identity, timing, and notable warnings such as skipped non-finite batches.

9. **How do you review correctness across devices, dtypes, and shapes?**
   Define the supported matrix, parameterize tests across it, include singleton/empty/noncontiguous cases, compare CPU and accelerator outputs within tolerance, and test mixed precision plus serialization. Unsupported combinations should fail clearly.

10. **What changes before moving notebook code into production?**
    Extract configuration and reusable modules, remove hidden state, add validation and tests, make preprocessing reproducible, add logging/checkpointing/error handling, pin dependencies, profile resource use, define artifact/version contracts, and document deployment and rollback behavior.
