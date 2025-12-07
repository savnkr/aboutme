---
title: "Gaussian Identities"
date: 2025-12-07
summary: "Key identities and properties of Gaussian distributions."
tags: ['math', 'probability', 'gaussian']
---

# Gaussian Identities

The Gaussian distribution is fundamental to machine learning and statistics. Here are some key identities.

## 1. The Multivariate Gaussian Distribution

A random vector $ \mathbf{x} \in \mathbb{R}^n $ has a multivariate Gaussian distribution if its probability density function is given by:

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

where $ \boldsymbol{\mu} $ is the mean vector and $ \boldsymbol{\Sigma} $ is the covariance matrix.

## 2. Product of Two Gaussians

The product of two Gaussian PDFs is another Gaussian PDF (up to a scaling factor).

Given:
$$
\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \quad \text{and} \quad \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)
$$

Their product is proportional to $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$ where:

$$
\boldsymbol{\Sigma}_c = (\boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1})^{-1}
$$
$$
\boldsymbol{\mu}_c = \boldsymbol{\Sigma}_c (\boldsymbol{\Sigma}_1^{-1} \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_2^{-1} \boldsymbol{\mu}_2)
$$

## 3. Conditional Distribution

If $ \mathbf{x} $ and $ \mathbf{y} $ are jointly Gaussian:

$$
\begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix} \boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy} \end{bmatrix} \right)
$$

Then the conditional distribution $ p(\mathbf{x} | \mathbf{y}) $ is also Gaussian:

$$
p(\mathbf{x} | \mathbf{y}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_{x|y}, \boldsymbol{\Sigma}_{x|y})
$$

where:

$$
\boldsymbol{\mu}_{x|y} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} (\mathbf{y} - \boldsymbol{\mu}_y)
$$
$$
\boldsymbol{\Sigma}_{x|y} = \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}
$$

These identities are crucial for understanding Gaussian Processes, Kalman Filters, and Variational Inference.
