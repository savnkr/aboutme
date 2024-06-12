---
title: "Vector Valued Kernel Ridge Regression"
date: 2024-06-10
draft: false
summary: "Vector valued Kernel ridge regression"
categories: [Blog]
tags: [machine learning, gaussian processes]
image:
    caption: 'Image credit: [**Unsplash**](https://unsplash.com)'
author:
    -admin
math: true
commentable: true
---

## Introduction

Kernel Ridge Regression (KRR) is a powerful regression technique that combines ridge regression with kernel methods to handle non-linear relationships. In vector-valued KRR, we extend this concept to handle vector-valued functions, which are crucial in multi-output regression problems.

## Mathematical Foundation

**Basic Ridge Regression:**

Given a dataset $\{(x_i, y_i)\}_{i=1}^n$, the objective of ridge regression is to minimize:
$$ \sum_{i=1}^n \|y_i - \langle w, \phi(x_i) \rangle \|^2 + \lambda \|w\|^2 $$
where $\phi(x_i)$ is the feature map and $\lambda$ is the regularization parameter.

**Kernel Methods:**

In kernel methods, we replace the dot product $\langle \phi(x_i), \phi(x_j) \rangle$ with a kernel function $k(x_i, x_j)$.

**Vector Valued Functions:**

For vector-valued functions $ f: \mathbb{R}^d \rightarrow \mathbb{R}^m $, we need a matrix-valued kernel function $ K(x, x') $.

**Combining Ridge Regression with Kernel Methods:**

The objective function for vector valued KRR becomes:
$$ \sum_{i=1}^n \|y_i - f(x_i)\|^2 + \lambda \|f\|_{\mathcal{H}_K}^2 $$

## Formulation of Vector Valued KRR

**Objective Function:**
$$ J(f) = \sum_{i=1}^n \|y_i - f(x_i)\|^2 + \lambda \|f\|_{\mathcal{H}_K}^2 $$

**Regularization Term:**
$$ \|f\|_{\mathcal{H}_K}^2 $$

**Kernel Function:**
$$ K(x, x') $$

**Solution:**
The solution to the optimization problem is given by:
$$ f(x) = \sum_{i=1}^n K(x, x_i) \alpha_i $$
where $\alpha_i$ are the coefficients to be determined.

## Implementation in Python

### Preparing the Data

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Sample data generation
X = np.random.rand(100, 10)  # 100 samples, 10 features
Y = np.random.rand(100, 5)   # 100 samples, 5 outputs

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### Defining the kernel

```python
def rbf_kernel(X1, X2, gamma=1.0):
    K = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)
    return K
```

### Solving the optimisation problem

```python
def vector_valued_krr(X_train, Y_train, lambda_param, kernel_func):
    K = kernel_func(X_train, X_train)
    n = K.shape[0]
    I = np.eye(n)
    
    A = K + lambda_param * I
    alpha = np.linalg.solve(A, Y_train)
    return alpha, X_train

alpha, X_train_fit = vector_valued_krr(X_train, Y_train, lambda_param=0.1, kernel_func=rbf_kernel)
```

### Prediction

```python
def predict(X_test, X_train_fit, alpha, kernel_func):
    K_test = kernel_func(X_test, X_train_fit)
    return K_test @ alpha

Y_pred = predict(X_test, X_train_fit, alpha, rbf_kernel)
```

### Evaluate the model

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")
```

### Visualising the results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(Y_test[:, 0], label='True')
plt.plot(Y_pred[:, 0], label='Predicted')
plt.legend()
plt.show()
```

## Conclusion

In this blog, we explored the mathematics behind vector-valued KRR and demonstrated its implementation in Python. Vector-valued KRR is a powerful technique for multi-output regression problems, combining the flexibility of kernel methods with the robustness of ridge regression.

## References and Further Reading

1. Scholkopf, B., & Smola, A. J. (2001). Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.
2. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
3. Bishop, C. M. (2006). Pattern Recognition and Machine Learning.

