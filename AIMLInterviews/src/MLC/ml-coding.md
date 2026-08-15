# <a name="ml-coding"></a> 2. ML/Data Coding :robot:

ML coding rounds vary by company. Some focus on implementing classical algorithms from scratch, while others test practical Python and PyTorch skills such as tensor operations, preprocessing, metrics, and training loops. In either format, interviewers evaluate correctness, numerical stability, code quality, edge-case handling, complexity, and your ability to explain design choices.

## How to use this chapter

- Use [`solutions/ml_algorithms.py`](./solutions/ml_algorithms.py) as the canonical, executable NumPy reference for core interview problems.
- Run [`solutions/test_ml_algorithms.py`](./solutions/test_ml_algorithms.py) to verify the implementations and study useful edge cases.
- Use the older [notebooks](./notebooks/) as supplementary, exploratory material. Some predate the canonical solutions and may be less complete.
- Practice writing each priority problem without looking at the reference, then compare correctness, complexity, and edge-case handling.

Run the reference tests from the repository root:

```bash
uv run --with numpy python src/MLC/solutions/test_ml_algorithms.py
```

## PyTorch ML Coding

Modern ML coding interviews may test practical PyTorch skills rather than only asking candidates to implement algorithms from scratch. The [PyTorch ML Coding Problems](./pytorch-ml-coding.md) guide includes:

- A 60-minute mock interview covering tensors, preprocessing, metrics, training/evaluation loops, and debugging
- Python utility and clean-code questions for ML workflows
- Tensor operations, datasets, batching, device handling, and autograd
- Training, optimization, mixed precision, checkpointing, and reproducibility
- Testing, debugging, deployment, and advanced PyTorch questions
- Coding challenges and concise reference responses

## Priority ML coding problems

The following set combines classic questions that remain common with modern primitives increasingly expected in AI/ML interviews.

| Problem | Canonical solution | Supplemental notebook | What a strong solution should cover |
| --- | --- | --- | --- |
| Numerically stable softmax and cross-entropy | `softmax`, `cross_entropy_from_logits` | — | Max subtraction, log-sum-exp, shapes, class-index validation |
| Linear regression with gradient descent | `linear_regression_gradient_descent` | [Linear regression](./notebooks/linear_regression_md.ipynb) | Vectorized gradients, bias, MSE scaling, convergence |
| Logistic regression with gradient descent | `logistic_regression_gradient_descent` | [Logistic regression](./notebooks/logistic_regression_md.ipynb) | Stable sigmoid, binary cross-entropy gradient, thresholds |
| k-nearest neighbors | `knn_predict` | [k-NN](./notebooks/k_nearest_neighbors.ipynb) | Pairwise distances, top-k selection, ties, complexity |
| k-means clustering | `kmeans` | [k-means](./notebooks/k_means_2.ipynb) | Initialization, vectorized assignment, convergence, empty clusters |
| Decision-tree split | `gini_impurity`, `best_gini_split` | [Decision tree](./notebooks/decision_tree.ipynb) | Candidate thresholds, weighted impurity, stopping conditions |
| Principal component analysis | `principal_component_analysis` | — | Centering, SVD/eigendecomposition, component ordering, variance |
| 2D convolution | `conv2d_valid` | [Convolution](./notebooks/convolution.ipynb) | Output shape, stride, cross-correlation vs convolution |
| Scaled dot-product attention | `scaled_dot_product_attention` | — | Q/K/V shapes, `1/sqrt(d_k)`, masking before stable softmax |
| Binary metrics and ROC-AUC | `binary_classification_metrics`, `roc_auc` | — | Zero denominators, class imbalance, ties, rank interpretation |
| Reservoir sampling | `reservoir_sample` | — | Unknown stream length, uniform probability, O(k) memory |
| TF-IDF | `tfidf` | — | Token counts, document frequency, smoothing, sparse scaling |

All canonical functions are in [`solutions/ml_algorithms.py`](./solutions/ml_algorithms.py).

## Additional classic algorithms

These are useful follow-up exercises, especially when they match the target team's domain:

- Linear SVM and hinge loss ([notebook](./notebooks/svm.ipynb))
- Perceptron learning rule ([notebook](./notebooks/perceptron.ipynb))
- Feedforward neural network and backpropagation ([notebook](./notebooks/feedforward.ipynb))
- Multiclass or multilabel extensions of metrics and losses
- Naive Bayes for text classification
- Matrix factorization for recommendation systems
- Gradient boosting: explain the training loop and implement a simple residual-fitting step

## Data and sampling questions

- Implement train/validation/test splitting without leakage
- Standardize features using training-only statistics
- Handle missing values and unseen categories consistently
- Implement uniform, stratified, weighted, and reservoir sampling
- Build mini-batches and pad variable-length sequences
- Aggregate sample-weighted losses and streaming metrics correctly

## What to explain during the interview

1. State input shapes, dtypes, assumptions, and expected outputs before coding.
2. Start with a correct baseline, then vectorize or optimize the bottleneck.
3. Discuss time and space complexity, including the cost of pairwise matrices.
4. Handle numerical stability, empty inputs, ties, constant features, and invalid labels.
5. Write small tests for normal cases and at least one failure or boundary case.
6. Explain how the implementation would change for large datasets, GPUs, distributed training, or production libraries.
