# Machine Learning from Scratch
### A Graduate-Level Perspective

> *"In God we trust. All others must bring data."* — W. Edwards Deming

---

## What Does "From Scratch" Mean?

Building machine learning **from scratch** means implementing algorithms without relying on high-level libraries (like scikit-learn or PyTorch) to do the heavy lifting. Instead, you build the core mechanics yourself — using only fundamental tools like NumPy — so that you truly understand what happens beneath the API surface.

This approach forces you to confront the mathematics directly: gradient descent isn't a black box, it's a loop you write; a neural network isn't a `.fit()` call, it's a chain of matrix multiplications and partial derivatives you derive by hand.

For a master's student, this is the difference between **using** machine learning and **understanding** it.

---

## Why Math and Statistics Are Non-Negotiable

Machine learning is, at its core, applied mathematics. The abstractions provided by modern frameworks are beautiful — but they can hide the assumptions, limitations, and failure modes that only a mathematically literate practitioner will catch.

### 🔢 Linear Algebra
The language of data and models.

- Data lives in **vector spaces** — every sample is a point in ℝⁿ
- Model parameters are **matrices** that transform these spaces
- Operations like dot products, matrix decomposition (SVD, eigendecomposition), and projections appear throughout: PCA, regression, attention mechanisms, recommendation systems
- Understanding matrix rank, null spaces, and conditioning helps you diagnose numerical instability

### 📐 Calculus & Optimization
The engine of learning.

- **Gradient descent** is the workhorse of modern ML — you cannot implement it without understanding partial derivatives and the chain rule
- Backpropagation is *just* the chain rule applied recursively through a computational graph
- Convexity determines whether your optimizer is guaranteed to find a global minimum
- Concepts like the Hessian, Jacobian, and Lagrange multipliers appear in second-order methods and constrained optimization

### 📊 Probability & Statistics
The foundation of inference and uncertainty.

- Every supervised learning problem is a statistical estimation problem
- **Maximum Likelihood Estimation (MLE)** and **MAP estimation** derive the loss functions you minimize (e.g., cross-entropy from MLE under a Bernoulli model)
- Bayesian reasoning lets you quantify and propagate uncertainty — critical in scientific and high-stakes applications
- Hypothesis testing, confidence intervals, and p-values are essential for evaluating whether your model is actually better — not just luckier
- Distributions (Gaussian, Bernoulli, Multinomial, Dirichlet...) define the generative stories behind models like Naive Bayes, GMMs, and VAEs

### 📉 Information Theory
The geometry of uncertainty.

- **Entropy** measures unpredictability; **KL divergence** measures how far two distributions are from each other
- These concepts directly underlie cross-entropy loss, variational inference, and model compression
- Understanding mutual information connects feature selection to the data-processing inequality

---

## Core Algorithms to Implement from Scratch

| Algorithm | Key Math |
|---|---|
| Linear Regression | Least squares, normal equations, gradient descent |
| Logistic Regression | Sigmoid, MLE, cross-entropy loss |
| Neural Networks | Backprop, chain rule, weight initialization |
| PCA | Eigendecomposition, covariance matrix, SVD |
| k-Means Clustering | Expectation-Maximization, Euclidean geometry |
| Decision Trees | Information gain, entropy, Gini impurity |
| SVM | Convex optimization, duality, kernel trick |
| Naive Bayes | Bayes' theorem, conditional independence |
| Gaussian Processes | Covariance kernels, multivariate Gaussians |

---

## Recommended Mathematical Background

For a master's student, you should be comfortable with — or actively studying — the following:

```
Mathematics for ML
├── Linear Algebra
│   ├── Gilbert Strang — Introduction to Linear Algebra
│   └── 3Blue1Brown — Essence of Linear Algebra (visual intuition)
├── Calculus & Optimization
│   ├── Boyd & Vandenberghe — Convex Optimization (free online)
│   └── Nocedal & Wright — Numerical Optimization
├── Probability & Statistics
│   ├── Bishop — Pattern Recognition and Machine Learning
│   └── Murphy — Probabilistic Machine Learning (2022, free online)
└── Information Theory
    └── Cover & Thomas — Elements of Information Theory
```

---

## Project Structure (Typical "From Scratch" Repo)

```
ml-from-scratch/
├── README.md
├── linear_models/
│   ├── linear_regression.py       # Normal equations + gradient descent
│   └── logistic_regression.py     # Binary + multiclass (softmax)
├── neural_networks/
│   ├── layers.py                  # Dense, activation layers
│   ├── losses.py                  # MSE, cross-entropy
│   └── network.py                 # Forward pass + backprop
├── unsupervised/
│   ├── kmeans.py
│   └── pca.py
├── trees/
│   └── decision_tree.py
├── utils/
│   ├── metrics.py                 # Accuracy, F1, AUC-ROC
│   └── data_utils.py              # Train/test split, normalization
└── notebooks/
    └── experiments/               # Jupyter notebooks with derivations
```

---

## The Graduate Mindset

At the master's level, the goal is not just to get models working — it's to understand **why** they work, **when** they fail, and **how** to fix them. This requires:

1. **Deriving before implementing** — prove the update rule before coding it
2. **Reading primary literature** — seminal papers (Rosenblatt 1958, Rumelhart 1986, Vaswani 2017) reveal the original mathematical reasoning
3. **Questioning assumptions** — i.i.d. data? Gaussian noise? Convexity? These break in practice
4. **Connecting theory to behavior** — if your model underfits, is that high bias? Why? What does the bias-variance tradeoff tell you to do?

Building from scratch is the most direct path to that depth.

---

## Getting Started

```bash
# Recommended environment
python >= 3.10
numpy          # Core linear algebra and array ops
matplotlib     # Visualization
jupyter        # For derivation notebooks

# Optional (for comparison and validation)
scikit-learn   # Verify your implementations match
```

---

*Built for learners who want to understand machine learning, not just use it.*
