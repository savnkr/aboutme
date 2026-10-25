---
title: "The Gaussian toolkit: identities, linear-algebra moves and tricks"
description: "The identities and moves that experts use without thinking and beginners struggle with: whitening, low-rank updates, Schur complements and spectra; moments, conditioning in both forms, KL, Stein, Bonnet and Price; total variance, the ELBO and Fisher information; and how all of it runs reinforcement learning. Each with a step-by-step proof."
date: 2025-12-07
updated: 2026-10-26
cover: /blog/thumbs/gaussian-identities.svg
tags: [probability, gaussian, reference]
---

The Gaussian is the one distribution that stays closed under almost everything we want to do: linear maps, sums, conditioning, marginalising, multiplying densities. That closure is why Gaussian processes, Kalman filters, variational inference, diffusion models and most continuous-control reinforcement learning are tractable. It is also why derivations in these areas feel either effortless or hopeless: with the right handful of moves they collapse to a few lines, and without them they sprawl.

This page collects those moves. Section 2 is the linear algebra underneath (whitening, low-rank updates, Schur complements, spectra). Sections 3 to 9 are the Gaussian identities proper. Section 10 covers the probabilistic moves that recur everywhere (total variance, moment matching, the ELBO, Fisher information), and Section 11 shows the same tools running reinforcement learning. The core results are proved step by step in [Gaussian processes, from first principles](/aboutme/blog/understanding-gaussian-processes/) and cited as "GP Lemma 5" and so on; everything else is proved here. Proofs, notes, remarks and examples are folded; the notes marked *why experts reach for it* are the point of the page.

Two pictures carry most of the intuition. **Conditioning is a slice**: fix $x_1 = a$ and read off the Gaussian along that line. **Marginalising is a shadow**: project the whole cloud onto the $x_2$ axis. Move $\rho$ and $a$: the slice is always narrower than the shadow, and its centre moves along the dashed regression line.

<figure class="fig" data-gauss2d>
  <canvas data-figure="gauss2d" style="height:340px" aria-label="A 2-D Gaussian with a conditional slice and a marginal shadow."></canvas>
  <figcaption>
    <label class="slider">ρ <input type="range" min="-0.95" max="0.95" step="0.01" value="0.7" data-rho /></label>
    <label class="slider">slice at x₁ = a <input type="range" min="-2.5" max="2.5" step="0.01" value="1" data-a /></label>
    <span class="mono" data-out></span>
  </figcaption>
</figure>

## Notation

Plain italic letters are scalars, bold italic lower-case letters are vectors ($\boldsymbol{x}, \boldsymbol{\mu} \in \mathbb{R}^D$), bold upright capitals are matrices ($\boldsymbol{\Sigma}$, $\mathbf{P} = \boldsymbol{\Sigma}^{-1}$, $\mathbf{A}$). $\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ is the density

$$
\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = (2\pi)^{-D/2}\lvert\boldsymbol{\Sigma}\rvert^{-1/2}\exp\!\Big(-\tfrac12(\boldsymbol{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\Big).
$$

## 1. Where the constant comes from

:::proposition[The Gaussian integral]
For symmetric positive definite $\mathbf{P} \in \mathbb{R}^{D\times D}$, $\;\displaystyle\int_{\mathbb{R}^D} \exp\!\big(-\tfrac12\boldsymbol{x}^\top\mathbf{P}\boldsymbol{x}\big)\,d\boldsymbol{x} = (2\pi)^{D/2}\,\lvert\mathbf{P}\rvert^{-1/2}$.
:::

:::proof
**Step 1. Rotate to the eigenbasis.** Write $\mathbf{P} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with $\mathbf{Q}$ orthogonal and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_D)$, $\lambda_i > 0$. Substitute $\boldsymbol{y} = \mathbf{Q}^\top\boldsymbol{x}$; since $\lvert\det\mathbf{Q}\rvert = 1$, $d\boldsymbol{x} = d\boldsymbol{y}$.

**Step 2. The exponent separates.** $\boldsymbol{x}^\top\mathbf{P}\boldsymbol{x} = \boldsymbol{y}^\top\boldsymbol{\Lambda}\boldsymbol{y} = \sum_i\lambda_iy_i^2$, so the integrand is a product of one-dimensional factors:

$$
\int\exp\!\Big(-\tfrac12\sum_i\lambda_iy_i^2\Big)d\boldsymbol{y} = \prod_{i=1}^D\int_{-\infty}^{\infty}e^{-\lambda_iy_i^2/2}\,dy_i .
$$

**Step 3. Each factor.** Substituting $y_i = s/\sqrt{\lambda_i}$ and using $\int e^{-s^2/2}ds = \sqrt{2\pi}$, each factor is $\sqrt{2\pi/\lambda_i}$.

**Step 4. Multiply.** $\prod_i\sqrt{2\pi/\lambda_i} = (2\pi)^{D/2}\big(\prod_i\lambda_i\big)^{-1/2} = (2\pi)^{D/2}\lvert\mathbf{P}\rvert^{-1/2}$, since the determinant is the product of the eigenvalues.
:::

With $\mathbf{P} = \boldsymbol{\Sigma}^{-1}$ (so $\lvert\mathbf{P}\rvert^{-1/2} = \lvert\boldsymbol{\Sigma}\rvert^{1/2}$) and a shift by $\boldsymbol{\mu}$, this is exactly the normaliser in the density.

## 2. Five linear-algebra moves

Most "Gaussian algebra" is linear algebra in disguise, and most of the difficulty beginners meet is choosing which way to write a matrix expression. Five moves cover nearly every derivation in probabilistic machine learning. Beginners expand and compute; experts rewrite with one of these first.

### Move 1: a scalar is its own trace

$\boldsymbol{a}^\top\mathbf{M}\boldsymbol{b} = \operatorname{tr}(\boldsymbol{a}^\top\mathbf{M}\boldsymbol{b}) = \operatorname{tr}(\mathbf{M}\boldsymbol{b}\boldsymbol{a}^\top)$, because a $1\times1$ matrix equals its trace and $\operatorname{tr}(\mathbf{U}\mathbf{V}) = \operatorname{tr}(\mathbf{V}\mathbf{U})$. It turns a quadratic form into a linear function of an outer product, which can then be averaged or summed.

:::note[Why experts reach for it: sufficient statistics appear by themselves]
**Step 1.** The data-dependent part of a Gaussian log-likelihood is $-\tfrac12\sum_{n=1}^N(\boldsymbol{x}_n - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}_n - \boldsymbol{\mu})$.

**Step 2.** Each term is $\operatorname{tr}\big(\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}_n - \boldsymbol{\mu})(\boldsymbol{x}_n - \boldsymbol{\mu})^\top\big)$ by Move 1; the trace is linear, so the sum is $\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{S}_{\boldsymbol{\mu}})$ with $\mathbf{S}_{\boldsymbol{\mu}} = \sum_n(\boldsymbol{x}_n - \boldsymbol{\mu})(\boldsymbol{x}_n - \boldsymbol{\mu})^\top$.

**Step 3.** The data now enter only through one $D\times D$ matrix. Expanding $\mathbf{S}_{\boldsymbol{\mu}}$ shows that it depends on the data only through $\sum_n\boldsymbol{x}_n$ and $\sum_n\boldsymbol{x}_n\boldsymbol{x}_n^\top$: those two sums are all a Gaussian model ever needs to remember, however large $N$ is.
:::

### Move 2: never invert; factor and whiten

:::lemma[Whitening with a Cholesky factor]
Let $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ with $\mathbf{L}$ lower triangular (Cholesky). Then

1. $\boldsymbol{x} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{z}$ with $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ has $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$;
2. $(\boldsymbol{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) = \lVert\boldsymbol{w}\rVert^2$, where $\boldsymbol{w}$ solves the triangular system $\mathbf{L}\boldsymbol{w} = \boldsymbol{x} - \boldsymbol{\mu}$;
3. $\log\lvert\boldsymbol{\Sigma}\rvert = 2\sum_i\log L_{ii}$.
:::

:::proof
**Step 1. Sampling.** By GP Lemma 2, $\boldsymbol{\mu} + \mathbf{L}\boldsymbol{z}$ is Gaussian with mean $\boldsymbol{\mu}$ and covariance $\mathbf{L}\mathbf{I}\mathbf{L}^\top = \boldsymbol{\Sigma}$.

**Step 2. The inverse factorises.** $\boldsymbol{\Sigma}^{-1} = (\mathbf{L}\mathbf{L}^\top)^{-1} = \mathbf{L}^{-\top}\mathbf{L}^{-1}$.

**Step 3. Mahalanobis distance.** With $\boldsymbol{w} = \mathbf{L}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})$, the quadratic form is $(\boldsymbol{x} - \boldsymbol{\mu})^\top\mathbf{L}^{-\top}\mathbf{L}^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) = \boldsymbol{w}^\top\boldsymbol{w}$.

**Step 4. Determinant.** $\lvert\boldsymbol{\Sigma}\rvert = \lvert\mathbf{L}\rvert\,\lvert\mathbf{L}^\top\rvert = \lvert\mathbf{L}\rvert^2$, and the determinant of a triangular matrix is the product of its diagonal. Take logs.
:::

Every Gaussian is a standard normal seen through a linear map. Expert practice is to move to $\boldsymbol{z}$-space, where every coordinate is independent with unit variance, do the work there, and map back at the end. Numerically, a triangular solve costs $O(D^2)$ and is stable, while forming $\boldsymbol{\Sigma}^{-1}$ costs $O(D^3)$ and amplifies rounding error by the condition number. In code, `inv(S)` is almost always a bug.

### Move 3: update, don't recompute

:::lemma[Sherman–Morrison and the matrix determinant lemma]
For invertible $\mathbf{A}$ and vectors $\boldsymbol{u}, \boldsymbol{v}$ with $1 + \boldsymbol{v}^\top\mathbf{A}^{-1}\boldsymbol{u} \ne 0$:

$$
(\mathbf{A} + \boldsymbol{u}\boldsymbol{v}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\boldsymbol{u}\boldsymbol{v}^\top\mathbf{A}^{-1}}{1 + \boldsymbol{v}^\top\mathbf{A}^{-1}\boldsymbol{u}},
\qquad
\lvert\mathbf{A} + \boldsymbol{u}\boldsymbol{v}^\top\rvert = \lvert\mathbf{A}\rvert\,\big(1 + \boldsymbol{v}^\top\mathbf{A}^{-1}\boldsymbol{u}\big).
$$
:::

:::proof
**Step 1. Multiply the candidate inverse by $\mathbf{A} + \boldsymbol{u}\boldsymbol{v}^\top$.** Write $\gamma = \boldsymbol{v}^\top\mathbf{A}^{-1}\boldsymbol{u}$, a scalar. Then

$$
(\mathbf{A} + \boldsymbol{u}\boldsymbol{v}^\top)\Big(\mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\boldsymbol{u}\boldsymbol{v}^\top\mathbf{A}^{-1}}{1 + \gamma}\Big)
= \mathbf{I} + \boldsymbol{u}\boldsymbol{v}^\top\mathbf{A}^{-1} - \frac{\boldsymbol{u}\boldsymbol{v}^\top\mathbf{A}^{-1} + \boldsymbol{u}\,\gamma\,\boldsymbol{v}^\top\mathbf{A}^{-1}}{1 + \gamma},
$$

using $\boldsymbol{v}^\top\mathbf{A}^{-1}\boldsymbol{u} = \gamma$ in the last product.

**Step 2. Simplify.** The fraction is $\boldsymbol{u}(1 + \gamma)\boldsymbol{v}^\top\mathbf{A}^{-1}/(1 + \gamma) = \boldsymbol{u}\boldsymbol{v}^\top\mathbf{A}^{-1}$, which cancels the middle term, leaving $\mathbf{I}$.

**Step 3. Determinant: factor out $\mathbf{A}$.** $\lvert\mathbf{A} + \boldsymbol{u}\boldsymbol{v}^\top\rvert = \lvert\mathbf{A}\rvert\,\lvert\mathbf{I} + \boldsymbol{b}\boldsymbol{v}^\top\rvert$ with $\boldsymbol{b} = \mathbf{A}^{-1}\boldsymbol{u}$.

**Step 4. Eigenvalues of $\mathbf{I} + \boldsymbol{b}\boldsymbol{v}^\top$.** The rank-one matrix $\boldsymbol{b}\boldsymbol{v}^\top$ maps every vector to a multiple of $\boldsymbol{b}$; it has eigenvalue $\boldsymbol{v}^\top\boldsymbol{b}$ (eigenvector $\boldsymbol{b}$) and eigenvalue $0$ on the $(D-1)$-dimensional subspace orthogonal to $\boldsymbol{v}$. Adding $\mathbf{I}$ adds $1$ to each, so the determinant is $1 + \boldsymbol{v}^\top\boldsymbol{b}$.
:::

:::note[Why experts reach for it: one new data point costs O(D²), not O(D³)]
In Bayesian linear regression the posterior precision gains one rank-one term per observation, $\mathbf{A} \leftarrow \mathbf{A} + \sigma_n^{-2}\boldsymbol{\phi}\boldsymbol{\phi}^\top$ (GP Section 2). Sherman–Morrison updates $\mathbf{A}^{-1}$, and the determinant lemma updates $\log\lvert\mathbf{A}\rvert$, in $O(M^2)$ per point. This is recursive least squares, the Kalman filter's update step, and the reason a GP can absorb one new observation by extending its Cholesky factor by one row in $O(N^2)$. The general rank-$k$ version is the Woodbury identity (GP Lemma 6).
:::

### Move 4: blocks, Schur complements, and the chain rule

:::lemma[Block factorisation and determinant]
For $\mathbf{M} = \begin{bmatrix}\mathbf{A} & \mathbf{B}\\\mathbf{C} & \mathbf{D}\end{bmatrix}$ with $\mathbf{A}$ invertible, and the Schur complement $\mathbf{S} = \mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}$,

$$
\mathbf{M} = \begin{bmatrix}\mathbf{I} & \mathbf{0}\\\mathbf{C}\mathbf{A}^{-1} & \mathbf{I}\end{bmatrix}\begin{bmatrix}\mathbf{A} & \mathbf{0}\\\mathbf{0} & \mathbf{S}\end{bmatrix}\begin{bmatrix}\mathbf{I} & \mathbf{A}^{-1}\mathbf{B}\\\mathbf{0} & \mathbf{I}\end{bmatrix},
\qquad
\lvert\mathbf{M}\rvert = \lvert\mathbf{A}\rvert\,\lvert\mathbf{S}\rvert .
$$
:::

:::proof
**Step 1. Multiply the last two factors.** $\begin{bmatrix}\mathbf{A} & \mathbf{0}\\\mathbf{0} & \mathbf{S}\end{bmatrix}\begin{bmatrix}\mathbf{I} & \mathbf{A}^{-1}\mathbf{B}\\\mathbf{0} & \mathbf{I}\end{bmatrix} = \begin{bmatrix}\mathbf{A} & \mathbf{B}\\\mathbf{0} & \mathbf{S}\end{bmatrix}$.

**Step 2. Multiply by the first factor.** The top block row is unchanged; the bottom block row becomes $\big[\mathbf{C}\mathbf{A}^{-1}\mathbf{A},\;\; \mathbf{C}\mathbf{A}^{-1}\mathbf{B} + \mathbf{S}\big] = [\mathbf{C},\;\; \mathbf{D}]$. That is $\mathbf{M}$.

**Step 3. Determinants.** The outer factors are block triangular with identity diagonal blocks, so their determinants are $1$. The middle factor is block diagonal, with determinant $\lvert\mathbf{A}\rvert\,\lvert\mathbf{S}\rvert$.
:::

:::note[Why experts reach for it: the determinant obeys the chain rule of probability]
For a Gaussian joint covariance, $\mathbf{A} = \boldsymbol{\Sigma}_{aa}$ and $\mathbf{S} = \boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}$ is the conditional covariance of $\boldsymbol{b}$ given $\boldsymbol{a}$ (GP Lemma 5). So $\log\lvert\boldsymbol{\Sigma}\rvert = \log\lvert\boldsymbol{\Sigma}_{aa}\rvert + \log\lvert\boldsymbol{\Sigma}_{b\mid a}\rvert$: the log-determinant splits exactly as $\log p(\boldsymbol{a}, \boldsymbol{b}) = \log p(\boldsymbol{a}) + \log p(\boldsymbol{b} \mid \boldsymbol{a})$. Applied one variable at a time, it is why a GP's marginal likelihood is a sum of one-step-ahead predictive terms (GP Section 7), and why a Cholesky factor's diagonal entries are conditional standard deviations.
:::

### Move 5: think in eigenvalues

A function of a symmetric matrix acts on its eigenvalues and leaves its eigenvectors alone: if $\mathbf{K} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$, then $(\mathbf{K} + \sigma^2\mathbf{I})^{-1} = \mathbf{U}(\boldsymbol{\Lambda} + \sigma^2\mathbf{I})^{-1}\mathbf{U}^\top$. Many formulas that look opaque become one-line statements about numbers.

:::note[Why experts reach for it: a GP is a spectral filter]
**Step 1.** The GP posterior mean at the training inputs is $\mathbf{K}(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y}$ (GP Proposition 5).

**Step 2.** In the eigenbasis this is $\mathbf{U}\,\boldsymbol{\Lambda}(\boldsymbol{\Lambda} + \sigma_n^2\mathbf{I})^{-1}\,\mathbf{U}^\top\boldsymbol{y}$, that is,

$$
\sum_i\frac{\lambda_i}{\lambda_i + \sigma_n^2}\,(\boldsymbol{u}_i^\top\boldsymbol{y})\,\boldsymbol{u}_i .
$$

**Step 3.** Each direction of the data is multiplied by $\lambda_i/(\lambda_i + \sigma_n^2) \in (0, 1)$: kept where the prior's signal variance $\lambda_i$ dominates the noise, suppressed where it does not. Smooth kernels have rapidly decaying $\lambda_i$, so the GP keeps a few smooth directions and filters the rest. The sum $\sum_i\lambda_i/(\lambda_i + \sigma_n^2)$ is the model's *effective number of parameters*.
:::

## 3. Moments

:::proposition[First and second moments, and quadratic forms]
For $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and any matrix $\mathbf{A}$:

$$
\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top] = \boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}^\top,
\qquad
\mathbb{E}[\boldsymbol{x}^\top\mathbf{A}\boldsymbol{x}] = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma}) + \boldsymbol{\mu}^\top\mathbf{A}\boldsymbol{\mu}.
$$

(These hold for any distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$.)
:::

:::proof
**Step 1. Second moment.** Expand the definition of covariance: $\boldsymbol{\Sigma} = \mathbb{E}[(\boldsymbol{x} - \boldsymbol{\mu})(\boldsymbol{x} - \boldsymbol{\mu})^\top] = \mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top] - \boldsymbol{\mu}\boldsymbol{\mu}^\top$ (the two cross terms each equal $-\boldsymbol{\mu}\boldsymbol{\mu}^\top$, and $+\boldsymbol{\mu}\boldsymbol{\mu}^\top$ remains). Rearrange.

**Step 2. The trace trick.** A scalar is its own trace, and $\operatorname{tr}(\mathbf{U}\mathbf{V}) = \operatorname{tr}(\mathbf{V}\mathbf{U})$: $\boldsymbol{x}^\top\mathbf{A}\boldsymbol{x} = \operatorname{tr}(\boldsymbol{x}^\top\mathbf{A}\boldsymbol{x}) = \operatorname{tr}(\mathbf{A}\boldsymbol{x}\boldsymbol{x}^\top)$.

**Step 3. Expectation passes through the trace** (both are linear): $\mathbb{E}[\boldsymbol{x}^\top\mathbf{A}\boldsymbol{x}] = \operatorname{tr}(\mathbf{A}\,\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top]) = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma}) + \operatorname{tr}(\mathbf{A}\boldsymbol{\mu}\boldsymbol{\mu}^\top)$ by Step 1, and $\operatorname{tr}(\mathbf{A}\boldsymbol{\mu}\boldsymbol{\mu}^\top) = \boldsymbol{\mu}^\top\mathbf{A}\boldsymbol{\mu}$ by Step 2 read backwards.
:::

Fourth moments are where the Gaussian is special: they are fixed entirely by the covariance.

:::proposition[Isserlis (Wick) theorem]
For $\boldsymbol{x} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$, $\;\mathbb{E}[x_ix_jx_kx_l] = \Sigma_{ij}\Sigma_{kl} + \Sigma_{ik}\Sigma_{jl} + \Sigma_{il}\Sigma_{jk}$: a sum over the three ways to pair four indices.
:::

:::proof
**Step 1. The moment generating function.** $M(\boldsymbol{t}) = \mathbb{E}[e^{\boldsymbol{t}^\top\boldsymbol{x}}] = \exp(q)$ with $q = \tfrac12\boldsymbol{t}^\top\boldsymbol{\Sigma}\boldsymbol{t}$ (complete the square in the integral, as in GP Lemma 1, then use Proposition 1). Moments are derivatives at zero: $\mathbb{E}[x_ix_jx_kx_l] = \partial_i\partial_j\partial_k\partial_l M\,\big|_{\boldsymbol{t} = \mathbf{0}}$.

**Step 2. Derivatives of $q$.** Write $s_i = \partial_iq = (\boldsymbol{\Sigma}\boldsymbol{t})_i$. Then $\partial_js_i = \Sigma_{ij}$, all higher derivatives of $q$ vanish, and $s_i = 0$ at $\boldsymbol{t} = \mathbf{0}$.

**Step 3. Differentiate four times**, using $\partial_iM = s_iM$ and the product rule:

$$
\begin{aligned}
\partial_j\partial_iM &= (\Sigma_{ij} + s_is_j)\,M, \\
\partial_k\partial_j\partial_iM &= (\Sigma_{ij}s_k + \Sigma_{ik}s_j + \Sigma_{jk}s_i + s_is_js_k)\,M .
\end{aligned}
$$

**Step 4. Differentiate once more and set $\boldsymbol{t} = \mathbf{0}$.** Every term still containing an $s$ vanishes. What survives is each $s$ in Step 3 differentiated by $\partial_l$, with $M(\mathbf{0}) = 1$: $\Sigma_{ij}\Sigma_{kl} + \Sigma_{ik}\Sigma_{jl} + \Sigma_{jk}\Sigma_{il}$.
:::

:::example[Where Isserlis is used: the Hutchinson trace estimator]
Expensive GP computations need $\operatorname{tr}(\mathbf{A})$ for a matrix $\mathbf{A}$ we can only multiply by (for example $\mathbf{A} = \mathbf{K}_y^{-1}\partial\mathbf{K}_y$ in the gradient of the marginal likelihood, GP Proposition 8). Draw $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Step 1. Unbiased.** By Proposition 2 with $\boldsymbol{\Sigma} = \mathbf{I}$, $\boldsymbol{\mu} = \mathbf{0}$: $\mathbb{E}[\boldsymbol{z}^\top\mathbf{A}\boldsymbol{z}] = \operatorname{tr}\mathbf{A}$.

**Step 2. Second moment by Isserlis** (symmetric $\mathbf{A}$, $\Sigma_{ij} = \delta_{ij}$):

$$
\mathbb{E}[(\boldsymbol{z}^\top\mathbf{A}\boldsymbol{z})^2] = \sum_{ijkl}A_{ij}A_{kl}(\delta_{ij}\delta_{kl} + \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}) = (\operatorname{tr}\mathbf{A})^2 + 2\operatorname{tr}(\mathbf{A}^2).
$$

**Step 3. Variance.** Subtract the squared mean: $\operatorname{Var}[\boldsymbol{z}^\top\mathbf{A}\boldsymbol{z}] = 2\operatorname{tr}(\mathbf{A}^2) = 2\lVert\mathbf{A}\rVert_F^2$. Averaging $S$ independent probes divides it by $S$.
:::

## 4. The closure toolkit

| Operation | Result | Proof |
|---|---|---|
| Linear map $\mathbf{C}\boldsymbol{x} + \boldsymbol{d}$ | $\mathcal{N}(\mathbf{C}\boldsymbol{\mu} + \boldsymbol{d},\, \mathbf{C}\boldsymbol{\Sigma}\mathbf{C}^\top)$ | GP Lemma 2 |
| Sum of independent Gaussians (convolution) | means add, covariances add | GP Lemma 3 |
| Product of densities in the same variable | precisions add; normaliser $\mathcal{N}(\boldsymbol{\mu}_1 \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)$ | GP Lemma 4 |
| Conditioning, moment form | $\boldsymbol{\mu}_b + \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}(\boldsymbol{a} - \boldsymbol{\mu}_a)$, Schur complement | GP Lemma 5 |
| Likelihood $\mathcal{N}(\boldsymbol{y} \mid \mathbf{C}\boldsymbol{x} + \boldsymbol{d}, \mathbf{S})$ times prior on $\boldsymbol{x}$ | evidence times posterior, two forms | GP Lemma 8 |
| Push-through and Woodbury | $\mathbf{A}^{-1} = \boldsymbol{\Sigma}_w - \boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}_w$ | GP Lemma 6 |
| Sampling a conditional by correcting a prior sample | Matheron's rule | GP Proposition 6 |
| Mixture of Gaussians | not Gaussian | GP Note on Lemma 4 |

## 5. Conditioning and marginalising, in both coordinates

The moment form makes marginalising trivial (read off a block) and conditioning costly (a Schur complement). The information form swaps the roles. Here is the information-form half, which the GP post only sketched.

:::proposition[Information form: slice and shadow]
Let $\begin{bmatrix}\boldsymbol{a}\\\boldsymbol{b}\end{bmatrix}$ be Gaussian with mean $\begin{bmatrix}\boldsymbol{\mu}_a\\\boldsymbol{\mu}_b\end{bmatrix}$ and precision $\mathbf{P} = \begin{bmatrix}\mathbf{P}_{aa} & \mathbf{P}_{ab}\\\mathbf{P}_{ba} & \mathbf{P}_{bb}\end{bmatrix}$. Then

$$
\begin{aligned}
\text{(slice)}\quad & \boldsymbol{b} \mid \boldsymbol{a} \sim \mathcal{N}\big(\boldsymbol{\mu}_b - \mathbf{P}_{bb}^{-1}\mathbf{P}_{ba}(\boldsymbol{a} - \boldsymbol{\mu}_a),\;\; \mathbf{P}_{bb}^{-1}\big), \\
\text{(shadow)}\quad & \boldsymbol{a} \sim \mathcal{N}\big(\boldsymbol{\mu}_a,\;\; (\mathbf{P}_{aa} - \mathbf{P}_{ab}\mathbf{P}_{bb}^{-1}\mathbf{P}_{ba})^{-1}\big).
\end{aligned}
$$
:::

:::proof
Write $\boldsymbol{\alpha} = \boldsymbol{a} - \boldsymbol{\mu}_a$ and $\boldsymbol{\beta} = \boldsymbol{b} - \boldsymbol{\mu}_b$.

**Step 1. Expand the joint exponent in blocks.** Using $\mathbf{P}_{ab}^\top = \mathbf{P}_{ba}$,

$$
-\tfrac12\begin{bmatrix}\boldsymbol{\alpha}\\\boldsymbol{\beta}\end{bmatrix}^\top\mathbf{P}\begin{bmatrix}\boldsymbol{\alpha}\\\boldsymbol{\beta}\end{bmatrix}
= -\tfrac12\boldsymbol{\alpha}^\top\mathbf{P}_{aa}\boldsymbol{\alpha} - \boldsymbol{\beta}^\top\mathbf{P}_{ba}\boldsymbol{\alpha} - \tfrac12\boldsymbol{\beta}^\top\mathbf{P}_{bb}\boldsymbol{\beta}.
$$

**Step 2. Slice: fix $\boldsymbol{a}$.** As a function of $\boldsymbol{\beta}$ this is $-\tfrac12\boldsymbol{\beta}^\top\mathbf{P}_{bb}\boldsymbol{\beta} + \boldsymbol{\beta}^\top\boldsymbol{h} + \text{const}$ with $\boldsymbol{h} = -\mathbf{P}_{ba}\boldsymbol{\alpha}$. GP Lemma 1 gives $\boldsymbol{\beta} \mid \boldsymbol{a} \sim \mathcal{N}(\mathbf{P}_{bb}^{-1}\boldsymbol{h}, \mathbf{P}_{bb}^{-1})$. Add back $\boldsymbol{\mu}_b$.

**Step 3. Shadow: complete the square in $\boldsymbol{\beta}$.** By GP Lemma 1, the two $\boldsymbol{\beta}$ terms equal $-\tfrac12(\boldsymbol{\beta} - \mathbf{P}_{bb}^{-1}\boldsymbol{h})^\top\mathbf{P}_{bb}(\boldsymbol{\beta} - \mathbf{P}_{bb}^{-1}\boldsymbol{h}) + \tfrac12\boldsymbol{h}^\top\mathbf{P}_{bb}^{-1}\boldsymbol{h}$.

**Step 4. Integrate $\boldsymbol{b}$ out.** By Proposition 1 the integral of the first term over $\boldsymbol{\beta}$ is $(2\pi)^{\dim\boldsymbol{b}/2}\lvert\mathbf{P}_{bb}\rvert^{-1/2}$, which does not depend on $\boldsymbol{a}$. What remains in $\boldsymbol{\alpha}$ is

$$
-\tfrac12\boldsymbol{\alpha}^\top\mathbf{P}_{aa}\boldsymbol{\alpha} + \tfrac12\boldsymbol{\alpha}^\top\mathbf{P}_{ab}\mathbf{P}_{bb}^{-1}\mathbf{P}_{ba}\boldsymbol{\alpha}
= -\tfrac12\boldsymbol{\alpha}^\top\big(\mathbf{P}_{aa} - \mathbf{P}_{ab}\mathbf{P}_{bb}^{-1}\mathbf{P}_{ba}\big)\boldsymbol{\alpha},
$$

using $\boldsymbol{h}^\top\mathbf{P}_{bb}^{-1}\boldsymbol{h} = \boldsymbol{\alpha}^\top\mathbf{P}_{ab}\mathbf{P}_{bb}^{-1}\mathbf{P}_{ba}\boldsymbol{\alpha}$. Read off the precision.
:::

:::remark[A free corollary: the block inverse]
The marginal of $\boldsymbol{a}$ also has covariance $\boldsymbol{\Sigma}_{aa}$ (moment form). Comparing with the shadow, $\boldsymbol{\Sigma}_{aa} = (\mathbf{P}_{aa} - \mathbf{P}_{ab}\mathbf{P}_{bb}^{-1}\mathbf{P}_{ba})^{-1}$. Comparing the slice with GP Lemma 5, $\mathbf{P}_{bb}^{-1} = \boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}$. The block-inverse formulas fall out of probability with no matrix algebra. In the figure, with unit variances, $\mathbf{P}_{22} = 1/(1 - \rho^2)$: the slice's variance $1 - \rho^2$ is $1/\mathbf{P}_{22}$, while the shadow's is $\Sigma_{22} = 1$.
:::

## 6. Entropy and KL divergence

:::proposition[Entropy]
$\;H\big[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})\big] = \tfrac12\log\big((2\pi e)^D\lvert\boldsymbol{\Sigma}\rvert\big)$.
:::

:::proof
**Step 1.** $H = -\mathbb{E}[\log p(\boldsymbol{x})] = \tfrac D2\log 2\pi + \tfrac12\log\lvert\boldsymbol{\Sigma}\rvert + \tfrac12\mathbb{E}\big[(\boldsymbol{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\big]$.

**Step 2.** By Proposition 2 applied to $\boldsymbol{x} - \boldsymbol{\mu}$ (mean $\mathbf{0}$), the expectation is $\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}) = D$.

**Step 3.** $\tfrac D2\log 2\pi + \tfrac D2 + \tfrac12\log\lvert\boldsymbol{\Sigma}\rvert = \tfrac12\log\big((2\pi e)^D\lvert\boldsymbol{\Sigma}\rvert\big)$.
:::

Entropy depends only on $\lvert\boldsymbol{\Sigma}\rvert$, the volume of the uncertainty ellipsoid. Among all distributions with a given covariance, the Gaussian has the largest entropy, which is one reason it is the default "least committed" choice.

:::proposition[KL divergence between Gaussians]
For $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ in $\mathbb{R}^D$,

$$
\operatorname{KL}(p_0 \,\|\, p_1) = \tfrac12\Big[\operatorname{tr}(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\Sigma}_0) + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_1^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) - D + \log\frac{\lvert\boldsymbol{\Sigma}_1\rvert}{\lvert\boldsymbol{\Sigma}_0\rvert}\Big].
$$
:::

:::proof
**Step 1. Definition.** $\operatorname{KL}(p_0 \| p_1) = \mathbb{E}_0[\log p_0(\boldsymbol{x})] - \mathbb{E}_0[\log p_1(\boldsymbol{x})]$, expectations under $p_0$.

**Step 2. The first term** is minus the entropy of $p_0$: $-\tfrac D2\log 2\pi - \tfrac12\log\lvert\boldsymbol{\Sigma}_0\rvert - \tfrac D2$ (Proposition 5, Steps 1–2).

**Step 3. The second term.** $\mathbb{E}_0[\log p_1] = -\tfrac D2\log 2\pi - \tfrac12\log\lvert\boldsymbol{\Sigma}_1\rvert - \tfrac12\mathbb{E}_0\big[(\boldsymbol{x} - \boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_1^{-1}(\boldsymbol{x} - \boldsymbol{\mu}_1)\big]$. Under $p_0$, $\boldsymbol{x} - \boldsymbol{\mu}_1$ has mean $\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1$ and covariance $\boldsymbol{\Sigma}_0$, so by Proposition 2 the expectation is $\operatorname{tr}(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\Sigma}_0) + (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_1^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$.

**Step 4. Subtract.** The $\log 2\pi$ terms cancel; collecting the rest gives the formula.
:::

:::note[Where the KL appears]
It is the regulariser in every variational autoencoder (with $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ it becomes $\tfrac12\sum_i(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2)$ for a diagonal $p_0$), and the term that keeps inducing-point posteriors close to the prior in sparse and [deep GPs](/aboutme/blog/deep-gaussian-processes/). Its three parts read cleanly: a shape mismatch (trace), a location mismatch (Mahalanobis distance of the means), and a volume mismatch (log-determinants).
:::

## 7. Stein, Bonnet and Price: differentiating through a Gaussian

Variational inference and reinforcement learning need gradients of $\mathbb{E}_{\boldsymbol{x}\sim\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})}[f(\boldsymbol{x})]$ with respect to $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$. Three identities turn them into expectations of derivatives of $f$.

:::lemma[Stein's lemma]
For $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and differentiable $g$ with $\mathbb{E}\lvert\nabla g\rvert < \infty$: $\;\mathbb{E}\big[g(\boldsymbol{x})(\boldsymbol{x} - \boldsymbol{\mu})\big] = \boldsymbol{\Sigma}\,\mathbb{E}\big[\nabla g(\boldsymbol{x})\big]$.
:::

:::proof
**Step 1. One dimension, the key fact.** For $\varphi(z) = \mathcal{N}(z \mid 0, 1)$, differentiating gives $\varphi'(z) = -z\,\varphi(z)$.

**Step 2. Integrate by parts.** For a scalar function $h$ of $z \sim \mathcal{N}(0, 1)$,

$$
\mathbb{E}[h(z)\,z] = \int h(z)\,z\,\varphi(z)\,dz = -\int h(z)\,\varphi'(z)\,dz = \int h'(z)\,\varphi(z)\,dz = \mathbb{E}[h'(z)],
$$

since the boundary term $h\varphi$ vanishes at $\pm\infty$.

**Step 3. Whiten.** Write $\boldsymbol{x} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{z}$ with $\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}$ and $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (GP Lemma 2). Let $h(\boldsymbol{z}) = g(\boldsymbol{\mu} + \mathbf{L}\boldsymbol{z})$; by the chain rule $\nabla_{\boldsymbol{z}}h = \mathbf{L}^\top\nabla g$.

**Step 4. Coordinate by coordinate.** The $z_j$ are independent, so Step 2 applies to each with the others held fixed: $\mathbb{E}[h(\boldsymbol{z})\,\boldsymbol{z}] = \mathbb{E}[\nabla_{\boldsymbol{z}}h] = \mathbf{L}^\top\mathbb{E}[\nabla g]$.

**Step 5. Undo the whitening.** $\mathbb{E}[g(\boldsymbol{x})(\boldsymbol{x} - \boldsymbol{\mu})] = \mathbf{L}\,\mathbb{E}[h(\boldsymbol{z})\,\boldsymbol{z}] = \mathbf{L}\mathbf{L}^\top\mathbb{E}[\nabla g] = \boldsymbol{\Sigma}\,\mathbb{E}[\nabla g]$.
:::

:::proposition[Bonnet and Price]
For smooth $f$ with integrable derivatives,

$$
\nabla_{\boldsymbol{\mu}}\,\mathbb{E}[f(\boldsymbol{x})] = \mathbb{E}[\nabla f(\boldsymbol{x})] \;\;\text{(Bonnet)},
\qquad
\nabla_{\boldsymbol{\Sigma}}\,\mathbb{E}[f(\boldsymbol{x})] = \tfrac12\,\mathbb{E}[\nabla^2 f(\boldsymbol{x})] \;\;\text{(Price)}.
$$
:::

:::proof
*Bonnet.*

**Step 1. Reparameterise.** $\mathbb{E}[f(\boldsymbol{x})] = \mathbb{E}_{\boldsymbol{z}}[f(\boldsymbol{\mu} + \mathbf{L}\boldsymbol{z})]$, and now $\boldsymbol{\mu}$ appears only inside $f$.

**Step 2. Differentiate under the integral.** $\nabla_{\boldsymbol{\mu}}f(\boldsymbol{\mu} + \mathbf{L}\boldsymbol{z}) = \nabla f(\boldsymbol{x})$, so $\nabla_{\boldsymbol{\mu}}\mathbb{E}[f] = \mathbb{E}[\nabla f]$. This is the *reparameterisation trick*.

*Price, in one dimension* (the multivariate case follows the same steps entrywise).

**Step 3. The Gaussian density solves the heat equation in its variance.** For $p_v(x) = \mathcal{N}(x \mid \mu, v)$, direct differentiation gives

$$
\partial_vp_v = p_v\Big(-\frac{1}{2v} + \frac{(x - \mu)^2}{2v^2}\Big),
\qquad
\partial_x^2p_v = p_v\Big(-\frac1v + \frac{(x - \mu)^2}{v^2}\Big),
$$

so $\partial_vp_v = \tfrac12\partial_x^2p_v$.

**Step 4. Move the derivative onto $f$.** $\partial_v\mathbb{E}[f] = \int f\,\partial_vp_v\,dx = \tfrac12\int f\,\partial_x^2p_v\,dx = \tfrac12\int f''\,p_v\,dx$, integrating by parts twice (boundary terms vanish).
:::

:::remark[Two different gradient estimators]
Bonnet's identity is the reparameterisation gradient behind VAEs and Gaussian variational inference: low variance, but it needs $\nabla f$. The *score-function* identity $\nabla_{\boldsymbol{\mu}}\mathbb{E}[f] = \mathbb{E}\big[f(\boldsymbol{x})\,\boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\big]$ needs only values of $f$ (it is Stein's lemma read in reverse, combined with Bonnet), which is why it powers policy gradients in reinforcement learning, at the price of higher variance. Price's Step 3 is the same heat equation that the [operator post](/aboutme/blog/vector-valued-kernel-ridge-regression/) learns: adding Gaussian noise is running heat flow.
:::

## 8. The matrix normal

A random matrix $\mathbf{X} \in \mathbb{R}^{n\times p}$ whose rows are correlated by one covariance and whose columns are correlated by another appears whenever data have two axes: inputs and outputs in a multi-output GP, space and time in a weather field.

:::definition[Matrix normal distribution]
$\mathbf{X} \sim \mathcal{MN}_{n\times p}(\mathbf{M}, \boldsymbol{\Omega}, \boldsymbol{\Sigma})$, with mean $\mathbf{M} \in \mathbb{R}^{n\times p}$, row covariance $\boldsymbol{\Omega} \in \mathbb{R}^{n\times n}$ and column covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{p\times p}$ (both positive definite), if

$$
\operatorname{vec}(\mathbf{X}) \sim \mathcal{N}\big(\operatorname{vec}(\mathbf{M}),\; \boldsymbol{\Sigma} \otimes \boldsymbol{\Omega}\big),
$$

where $\operatorname{vec}$ stacks the columns. Entrywise, $\operatorname{Cov}[X_{ij}, X_{kl}] = \Omega_{ik}\,\Sigma_{jl}$.
:::

Conventions differ between books (some stack rows, which swaps the order of the Kronecker factors); the entrywise covariance is the unambiguous statement.

:::lemma[Two vec identities]
1. $\operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B}) = (\mathbf{B}^\top \otimes \mathbf{A})\operatorname{vec}(\mathbf{X})$.
2. $\operatorname{vec}(\mathbf{A})^\top\operatorname{vec}(\mathbf{B}) = \operatorname{tr}(\mathbf{A}^\top\mathbf{B})$.
:::

:::proof
**Step 1. One column of the product.** Column $j$ of $\mathbf{A}\mathbf{X}\mathbf{B}$ is $\mathbf{A}\mathbf{X}\boldsymbol{b}_j = \sum_kB_{kj}\,\mathbf{A}\boldsymbol{x}_k$, where $\boldsymbol{x}_k$ is column $k$ of $\mathbf{X}$.

**Step 2. Stack the columns.** Block $j$ of $\operatorname{vec}(\mathbf{A}\mathbf{X}\mathbf{B})$ is $\sum_k(B_{kj}\mathbf{A})\boldsymbol{x}_k$, which is block row $j$ of the matrix with blocks $B_{kj}\mathbf{A} = (\mathbf{B}^\top)_{jk}\mathbf{A}$ applied to $\operatorname{vec}(\mathbf{X})$. That matrix is $\mathbf{B}^\top \otimes \mathbf{A}$.

**Step 3. Inner product of vecs.** $\operatorname{vec}(\mathbf{A})^\top\operatorname{vec}(\mathbf{B}) = \sum_{ij}A_{ij}B_{ij} = \sum_j(\mathbf{A}^\top\mathbf{B})_{jj} = \operatorname{tr}(\mathbf{A}^\top\mathbf{B})$.
:::

:::proposition[Matrix normal density and characteristic function]
With $\mathbf{E} = \mathbf{X} - \mathbf{M}$,

$$
p(\mathbf{X}) = (2\pi)^{-np/2}\,\lvert\boldsymbol{\Sigma}\rvert^{-n/2}\,\lvert\boldsymbol{\Omega}\rvert^{-p/2}\exp\!\Big(-\tfrac12\operatorname{tr}\big(\boldsymbol{\Sigma}^{-1}\mathbf{E}^\top\boldsymbol{\Omega}^{-1}\mathbf{E}\big)\Big),
$$

and $\mathbb{E}\big[e^{i\operatorname{tr}(\mathbf{Z}^\top\mathbf{X})}\big] = \exp\!\big(i\operatorname{tr}(\mathbf{Z}^\top\mathbf{M}) - \tfrac12\operatorname{tr}(\mathbf{Z}^\top\boldsymbol{\Omega}\mathbf{Z}\boldsymbol{\Sigma})\big)$.
:::

:::proof
**Step 1. Start from the vector density** of $\operatorname{vec}(\mathbf{X})$ in $\mathbb{R}^{np}$, with covariance $\boldsymbol{\Sigma} \otimes \boldsymbol{\Omega}$.

**Step 2. The determinant.** By the mixed-product property, the eigenvalues of $\boldsymbol{\Sigma} \otimes \boldsymbol{\Omega}$ are all products $\sigma_i\omega_j$ of eigenvalues of $\boldsymbol{\Sigma}$ ($p$ of them) and $\boldsymbol{\Omega}$ ($n$ of them). So $\lvert\boldsymbol{\Sigma} \otimes \boldsymbol{\Omega}\rvert = \prod_{i,j}\sigma_i\omega_j = \big(\prod_i\sigma_i\big)^n\big(\prod_j\omega_j\big)^p = \lvert\boldsymbol{\Sigma}\rvert^n\lvert\boldsymbol{\Omega}\rvert^p$.

**Step 3. The quadratic form.** $(\boldsymbol{\Sigma} \otimes \boldsymbol{\Omega})^{-1} = \boldsymbol{\Sigma}^{-1} \otimes \boldsymbol{\Omega}^{-1}$, and by Lemma 5, part 1 (with $\mathbf{B} = \boldsymbol{\Sigma}^{-1}$, which is symmetric),

$$
\operatorname{vec}(\mathbf{E})^\top(\boldsymbol{\Sigma}^{-1} \otimes \boldsymbol{\Omega}^{-1})\operatorname{vec}(\mathbf{E}) = \operatorname{vec}(\mathbf{E})^\top\operatorname{vec}(\boldsymbol{\Omega}^{-1}\mathbf{E}\boldsymbol{\Sigma}^{-1}) = \operatorname{tr}(\mathbf{E}^\top\boldsymbol{\Omega}^{-1}\mathbf{E}\boldsymbol{\Sigma}^{-1}),
$$

the last step by part 2. Cycle the trace to $\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{E}^\top\boldsymbol{\Omega}^{-1}\mathbf{E})$.

**Step 4. The characteristic function.** $\operatorname{tr}(\mathbf{Z}^\top\mathbf{X}) = \operatorname{vec}(\mathbf{Z})^\top\operatorname{vec}(\mathbf{X})$ (part 2), so this is the Gaussian characteristic function (GP Lemma 2, Step 4) at $\boldsymbol{t} = \operatorname{vec}(\mathbf{Z})$. Its quadratic term is $\operatorname{vec}(\mathbf{Z})^\top(\boldsymbol{\Sigma} \otimes \boldsymbol{\Omega})\operatorname{vec}(\mathbf{Z}) = \operatorname{tr}(\mathbf{Z}^\top\boldsymbol{\Omega}\mathbf{Z}\boldsymbol{\Sigma})$, by the same two identities.
:::

:::remark[A multi-output GP is a matrix normal]
For a separable kernel $k(\boldsymbol{x}, \boldsymbol{x}')\mathbf{B}$, the $N\times p$ matrix $\mathbf{F}$ of function values (rows: inputs, columns: outputs) has $\operatorname{Cov}[F_{ia}, F_{jb}] = k(\boldsymbol{x}_i, \boldsymbol{x}_j)\,B_{ab}$. That is $\mathbf{F} \sim \mathcal{MN}(\mathbf{0}, \mathbf{K}, \mathbf{B})$: the row covariance is the Gram matrix and the column covariance is the output covariance. The density's determinant $\lvert\mathbf{B}\rvert^{N}\lvert\mathbf{K}\rvert^{p}$ and its trace form are why the marginal likelihood of such models costs $O(N^3 + p^3)$ rather than $O(N^3p^3)$, the same saving as the Kronecker trick in the [operator post](/aboutme/blog/vector-valued-kernel-ridge-regression/).
:::

## 9. Derivatives of the log-density

:::proposition[Gradients, and the maximum-likelihood estimates]
For $\ell(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \log\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\boldsymbol{r} = \boldsymbol{x} - \boldsymbol{\mu}$ (treating the entries of $\boldsymbol{\Sigma}$ as free),

$$
\nabla_{\boldsymbol{\mu}}\ell = \boldsymbol{\Sigma}^{-1}\boldsymbol{r},
\qquad
\nabla_{\boldsymbol{\Sigma}}\ell = \tfrac12\big(\boldsymbol{\Sigma}^{-1}\boldsymbol{r}\boldsymbol{r}^\top\boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\big).
$$

Summing over data $\boldsymbol{x}_1, \dots, \boldsymbol{x}_N$ and setting both to zero gives the sample mean and the (biased) sample covariance.
:::

:::proof
**Step 1. Write the log-density.** $\ell = -\tfrac D2\log 2\pi - \tfrac12\log\lvert\boldsymbol{\Sigma}\rvert - \tfrac12\boldsymbol{r}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{r}$.

**Step 2. Gradient in $\boldsymbol{\mu}$.** $\nabla_{\boldsymbol{\mu}}\big(-\tfrac12\boldsymbol{r}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{r}\big) = \boldsymbol{\Sigma}^{-1}\boldsymbol{r}$, since $\nabla_{\boldsymbol{r}}(\boldsymbol{r}^\top\mathbf{A}\boldsymbol{r}) = 2\mathbf{A}\boldsymbol{r}$ for symmetric $\mathbf{A}$ and $\partial\boldsymbol{r}/\partial\boldsymbol{\mu} = -\mathbf{I}$.

**Step 3. Gradient in $\boldsymbol{\Sigma}$, entry by entry.** Let $\mathbf{E}_{ij}$ be the matrix with a single $1$ in position $(i, j)$. By GP Lemma 7, $\partial\log\lvert\boldsymbol{\Sigma}\rvert/\partial\Sigma_{ij} = \operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{E}_{ij}) = (\boldsymbol{\Sigma}^{-1})_{ji}$ and

$$
\frac{\partial\,\boldsymbol{r}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{r}}{\partial\Sigma_{ij}} = -\boldsymbol{r}^\top\boldsymbol{\Sigma}^{-1}\mathbf{E}_{ij}\boldsymbol{\Sigma}^{-1}\boldsymbol{r} = -(\boldsymbol{\Sigma}^{-1}\boldsymbol{r})_i(\boldsymbol{\Sigma}^{-1}\boldsymbol{r})_j .
$$

Collecting entries (and $\boldsymbol{\Sigma}^{-1}$ symmetric) gives the stated matrix.

**Step 4. Maximum likelihood.** Summing Step 2 over data, $\boldsymbol{\Sigma}^{-1}\sum_n(\boldsymbol{x}_n - \boldsymbol{\mu}) = \mathbf{0}$ gives $\hat{\boldsymbol{\mu}} = \tfrac1N\sum_n\boldsymbol{x}_n$. Summing Step 3 and multiplying by $\boldsymbol{\Sigma}$ on both sides, $\sum_n\boldsymbol{r}_n\boldsymbol{r}_n^\top = N\boldsymbol{\Sigma}$, so $\hat{\boldsymbol{\Sigma}} = \tfrac1N\sum_n(\boldsymbol{x}_n - \hat{\boldsymbol{\mu}})(\boldsymbol{x}_n - \hat{\boldsymbol{\mu}})^\top$.
:::

## 10. Probabilistic moves

### Splitting uncertainty: the law of total variance

:::proposition[Law of total covariance]
For random vectors $\boldsymbol{x}$ and $\boldsymbol{y}$ with $\boldsymbol{m}(\boldsymbol{y}) = \mathbb{E}[\boldsymbol{x} \mid \boldsymbol{y}]$,

$$
\operatorname{Cov}[\boldsymbol{x}] = \underbrace{\mathbb{E}\big[\operatorname{Cov}[\boldsymbol{x} \mid \boldsymbol{y}]\big]}_{\text{average spread}} + \underbrace{\operatorname{Cov}\big[\boldsymbol{m}(\boldsymbol{y})\big]}_{\text{spread of the averages}} .
$$
:::

:::proof
**Step 1. Second moment by the tower rule.** $\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top] = \mathbb{E}\big[\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top \mid \boldsymbol{y}]\big] = \mathbb{E}\big[\operatorname{Cov}[\boldsymbol{x} \mid \boldsymbol{y}] + \boldsymbol{m}(\boldsymbol{y})\boldsymbol{m}(\boldsymbol{y})^\top\big]$, using Proposition 2, Step 1 conditionally on $\boldsymbol{y}$.

**Step 2. Mean by the tower rule.** $\mathbb{E}[\boldsymbol{x}] = \mathbb{E}[\boldsymbol{m}(\boldsymbol{y})]$.

**Step 3. Subtract.** $\operatorname{Cov}[\boldsymbol{x}] = \mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top] - \mathbb{E}[\boldsymbol{x}]\mathbb{E}[\boldsymbol{x}]^\top = \mathbb{E}[\operatorname{Cov}[\boldsymbol{x} \mid \boldsymbol{y}]] + \big(\mathbb{E}[\boldsymbol{m}\boldsymbol{m}^\top] - \mathbb{E}[\boldsymbol{m}]\mathbb{E}[\boldsymbol{m}]^\top\big)$, and the bracket is $\operatorname{Cov}[\boldsymbol{m}(\boldsymbol{y})]$.
:::

:::proposition[Moment matching is the best Gaussian fit in KL(p ‖ q)]
For any distribution $p$ with mean $\boldsymbol{\mu}_p$ and covariance $\boldsymbol{\Sigma}_p$, the Gaussian $q = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$ minimising $\operatorname{KL}(p \,\|\, q)$ has $\boldsymbol{m} = \boldsymbol{\mu}_p$ and $\mathbf{S} = \boldsymbol{\Sigma}_p$.
:::

:::proof
**Step 1. Drop what does not depend on $q$.** $\operatorname{KL}(p \| q) = \mathbb{E}_p[\log p] - \mathbb{E}_p[\log q]$, so minimise $-\mathbb{E}_p[\log q] = \tfrac12\log\lvert\mathbf{S}\rvert + \tfrac12\mathbb{E}_p\big[(\boldsymbol{x} - \boldsymbol{m})^\top\mathbf{S}^{-1}(\boldsymbol{x} - \boldsymbol{m})\big] + \text{const}$.

**Step 2. Expectation of the quadratic form** (Proposition 2): $\operatorname{tr}(\mathbf{S}^{-1}\boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}_p - \boldsymbol{m})^\top\mathbf{S}^{-1}(\boldsymbol{\mu}_p - \boldsymbol{m})$.

**Step 3. The mean.** Only the second term involves $\boldsymbol{m}$; it is non-negative and zero at $\boldsymbol{m} = \boldsymbol{\mu}_p$.

**Step 4. The covariance.** With $\boldsymbol{m} = \boldsymbol{\mu}_p$, the objective is $\tfrac12\log\lvert\mathbf{S}\rvert + \tfrac12\operatorname{tr}(\mathbf{S}^{-1}\boldsymbol{\Sigma}_p)$. Its gradient in $\mathbf{S}$ (Proposition 9, Step 3) is $\tfrac12\mathbf{S}^{-1} - \tfrac12\mathbf{S}^{-1}\boldsymbol{\Sigma}_p\mathbf{S}^{-1}$, which vanishes at $\mathbf{S} = \boldsymbol{\Sigma}_p$.
:::

:::example[Ensembles: aleatoric plus epistemic]
A deep ensemble of $M$ networks predicts $\mathcal{N}(\mu_j, \sigma_j^2)$, $j = 1, \dots, M$, each with weight $1/M$.

**Step 1.** Let $J$ be a uniformly random member. By Proposition 10 with $\boldsymbol{y} = J$: $\operatorname{Var}[y] = \tfrac1M\sum_j\sigma_j^2 + \tfrac1M\sum_j(\mu_j - \bar\mu)^2$, with $\bar\mu = \tfrac1M\sum_j\mu_j$.

**Step 2.** The first term is the noise each member believes is in the data (aleatoric). The second is the members' disagreement (epistemic): it shrinks where data constrain the model and grows away from them.

**Step 3.** By Proposition 11, $\mathcal{N}(\bar\mu, \operatorname{Var}[y])$ is the best single Gaussian summary of the ensemble in $\operatorname{KL}(p \| q)$. Moment matching is also the core step of expectation propagation, assumed-density filtering, and PILCO's uncertainty propagation.
:::

:::note[The other direction of KL behaves differently]
Minimising $\operatorname{KL}(q \| p)$ instead, as variational inference does, penalises $q$ for putting mass where $p$ has none. Fitted to a two-mode $p$, it locks onto one mode and underestimates the spread, while $\operatorname{KL}(p \| q)$ (Proposition 11) covers both modes and overestimates it. Knowing which direction an algorithm minimises predicts how its uncertainty will be wrong.
:::

### The evidence lower bound in three lines

:::proposition[ELBO]
For any density $q(\boldsymbol{z})$, $\;\log p(\boldsymbol{y}) = \underbrace{\mathbb{E}_q\big[\log p(\boldsymbol{y}, \boldsymbol{z}) - \log q(\boldsymbol{z})\big]}_{\text{ELBO}(q)} + \operatorname{KL}\big(q(\boldsymbol{z}) \,\|\, p(\boldsymbol{z} \mid \boldsymbol{y})\big) \;\ge\; \text{ELBO}(q)$.
:::

:::proof
**Step 1.** $\log p(\boldsymbol{y})$ does not depend on $\boldsymbol{z}$, so it equals its own average under $q$: $\log p(\boldsymbol{y}) = \mathbb{E}_q[\log p(\boldsymbol{y})]$.

**Step 2.** Bayes' rule, $p(\boldsymbol{y}) = p(\boldsymbol{y}, \boldsymbol{z})/p(\boldsymbol{z} \mid \boldsymbol{y})$, and multiplying and dividing by $q(\boldsymbol{z})$: $\log p(\boldsymbol{y}) = \mathbb{E}_q\Big[\log\dfrac{p(\boldsymbol{y}, \boldsymbol{z})}{q(\boldsymbol{z})} + \log\dfrac{q(\boldsymbol{z})}{p(\boldsymbol{z} \mid \boldsymbol{y})}\Big]$.

**Step 3.** The second expectation is a KL divergence, which is non-negative.
:::

So maximising the ELBO over $q$ is the same as minimising the KL to the true posterior, without ever computing that posterior. With Gaussian $q$, the KL to a Gaussian prior is Proposition 6 in closed form and the expected log-likelihood has Bonnet and Price gradients (Proposition 7): this is how VAEs, sparse GPs and Bayesian neural networks are trained.

:::note[log-sum-exp: never exponentiate first]
Mixtures, softmaxes and importance weights need $\log\sum_ie^{a_i}$ with $a_i$ of size $\pm 1000$, where $e^{a_i}$ overflows or underflows. With $m = \max_ia_i$, $\;\log\sum_ie^{a_i} = m + \log\sum_ie^{a_i - m}$: every exponent is now at most $0$ and the largest term is exactly $1$. Work with log-densities throughout and exponentiate only at the very end, if at all.
:::

### Fisher information: the geometry inside the KL

:::proposition[For Gaussians, the KL is exactly the Fisher metric]
For fixed $\boldsymbol{\Sigma}$: $\;\operatorname{KL}\big(\mathcal{N}(\boldsymbol{\mu} + \boldsymbol{\delta}, \boldsymbol{\Sigma}) \,\|\, \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})\big) = \tfrac12\boldsymbol{\delta}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\delta}$, and the Fisher information of the mean is $\mathbf{F} = \mathbb{E}\big[\nabla_{\boldsymbol{\mu}}\log p\,\nabla_{\boldsymbol{\mu}}\log p^\top\big] = \boldsymbol{\Sigma}^{-1}$.
:::

:::proof
**Step 1. KL.** In Proposition 6 with $\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}$: the trace is $\operatorname{tr}(\mathbf{I}) = D$, cancelling $-D$, and the log-determinant ratio is $\log 1 = 0$. Only the Mahalanobis term remains.

**Step 2. Fisher.** By Proposition 9, $\nabla_{\boldsymbol{\mu}}\log p = \boldsymbol{\Sigma}^{-1}\boldsymbol{r}$ with $\boldsymbol{r} = \boldsymbol{x} - \boldsymbol{\mu}$, so $\mathbf{F} = \boldsymbol{\Sigma}^{-1}\mathbb{E}[\boldsymbol{r}\boldsymbol{r}^\top]\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Sigma}^{-1}$.
:::

In general the KL between nearby distributions is $\tfrac12\boldsymbol{\delta}^\top\mathbf{F}\boldsymbol{\delta}$ to second order; for a Gaussian mean it is exact. The *natural gradient* $\mathbf{F}^{-1}\nabla$ (Amari, 1998) here is $\boldsymbol{\Sigma}\nabla$: it measures steps in units of the distribution's own spread, so it does not care how the parameters happen to be scaled.

## 11. Gaussians in reinforcement learning

Continuous-control policies are almost always Gaussian, $\pi_{\boldsymbol{\theta}}(\boldsymbol{a} \mid \boldsymbol{s}) = \mathcal{N}\big(\boldsymbol{a} \mid \boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{s}), \boldsymbol{\Sigma}\big)$, and the standard algorithms are the identities above wearing different names. When an RL derivation gets messy, it is usually one of the following.

:::proposition[The score of a Gaussian policy, and why baselines are free]
With $\mathbf{J}_{\boldsymbol{\theta}}(\boldsymbol{s}) = \partial\boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{s})/\partial\boldsymbol{\theta}$,

$$
\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(\boldsymbol{a} \mid \boldsymbol{s}) = \mathbf{J}_{\boldsymbol{\theta}}(\boldsymbol{s})^\top\boldsymbol{\Sigma}^{-1}\big(\boldsymbol{a} - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{s})\big),
\qquad
\mathbb{E}_{\boldsymbol{a}\sim\pi}\big[\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(\boldsymbol{a} \mid \boldsymbol{s})\big] = \mathbf{0} .
$$

Hence subtracting any baseline $b(\boldsymbol{s})$ from the return leaves the policy gradient $\mathbb{E}\big[\nabla\log\pi\;(Q(\boldsymbol{s}, \boldsymbol{a}) - b(\boldsymbol{s}))\big]$ unchanged.
:::

:::proof
**Step 1. Log-density.** $\log\pi = -\tfrac12(\boldsymbol{a} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{a} - \boldsymbol{\mu}) + (\text{terms without } \boldsymbol{\theta})$.

**Step 2. Chain rule.** By Proposition 9, $\nabla_{\boldsymbol{\mu}}\log\pi = \boldsymbol{\Sigma}^{-1}(\boldsymbol{a} - \boldsymbol{\mu})$, and $\nabla_{\boldsymbol{\theta}} = \mathbf{J}^\top\nabla_{\boldsymbol{\mu}}$.

**Step 3. Zero mean.** Under $\pi$, $\mathbb{E}[\boldsymbol{a} - \boldsymbol{\mu}] = \mathbf{0}$, and $\mathbf{J}^\top\boldsymbol{\Sigma}^{-1}$ does not depend on $\boldsymbol{a}$.

**Step 4. The baseline term vanishes.** $\mathbb{E}_{\boldsymbol{a}}\big[\nabla\log\pi\;b(\boldsymbol{s})\big] = b(\boldsymbol{s})\,\mathbb{E}_{\boldsymbol{a}}[\nabla\log\pi] = \mathbf{0}$ by Step 3.
:::

:::remark[Why REINFORCE gets noisier as the policy converges, and the fix]
**Step 1.** With $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$ write $\boldsymbol{a} = \boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The score of Proposition 14 is $\mathbf{J}^\top\boldsymbol{\varepsilon}/\sigma$: its size grows like $1/\sigma$, its variance like $1/\sigma^2$.

**Step 2.** As training shrinks the exploration noise $\sigma$, the score-function gradient estimate becomes ever noisier.

**Step 3. Reparameterise instead** (Bonnet, Proposition 7): $\nabla_{\boldsymbol{\theta}}\mathbb{E}\big[Q(\boldsymbol{s}, \boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{s}) + \sigma\boldsymbol{\varepsilon})\big] = \mathbb{E}\big[\mathbf{J}^\top\nabla_{\boldsymbol{a}}Q(\boldsymbol{s}, \boldsymbol{a})\big]$. No $1/\sigma$ appears, and at $\sigma \to 0$ it becomes the deterministic policy gradient (Silver et al., 2014). This is why DDPG, TD3 and SAC differentiate through a learned $Q$.
:::

:::proposition[Trust regions for Gaussian policies are Mahalanobis balls]
For $\pi_{\text{old}} = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma})$ and $\pi_{\text{new}} = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma})$, $\;\operatorname{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) = \tfrac12(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$; for diagonal $\boldsymbol{\Sigma}$, $\;\sum_i(\Delta\mu_i)^2/(2\sigma_i^2)$.
:::

:::proof
**Step 1.** This is Proposition 13 with $\boldsymbol{\delta} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$.

**Step 2.** For diagonal $\boldsymbol{\Sigma}$, $\boldsymbol{\Sigma}^{-1} = \operatorname{diag}(1/\sigma_i^2)$ and the quadratic form is a sum over action dimensions.
:::

A TRPO or PPO constraint $\operatorname{KL} \le \delta$ therefore lets the mean of action dimension $i$ move by about $\sigma_i\sqrt{2\delta}$: dimensions the policy is still exploring may change a lot, settled ones only a little. Natural policy gradients are Proposition 13 applied to the whole policy.

:::note[Entropy bonuses act only on the covariance; and the tanh correction]
**Step 1.** By Proposition 5, $H[\pi(\cdot \mid \boldsymbol{s})] = \tfrac12\log\big((2\pi e)^D\lvert\boldsymbol{\Sigma}\rvert\big)$ does not involve the mean: a maximum-entropy bonus (SAC) pushes only on $\boldsymbol{\Sigma}$.

**Step 2.** SAC bounds actions by squashing, $\boldsymbol{a} = \tanh(\boldsymbol{u})$ elementwise with $\boldsymbol{u} \sim \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}\boldsymbol{\sigma}^2)$. The change-of-variables formula for an invertible map gives $p_{\boldsymbol{a}}(\boldsymbol{a}) = p_{\boldsymbol{u}}(\boldsymbol{u})\,\lvert\det\partial\boldsymbol{a}/\partial\boldsymbol{u}\rvert^{-1}$.

**Step 3.** The Jacobian is diagonal with entries $1 - \tanh^2(u_i)$, so $\log\pi(\boldsymbol{a}) = \log\mathcal{N}(\boldsymbol{u} \mid \boldsymbol{\mu}, \operatorname{diag}\boldsymbol{\sigma}^2) - \sum_i\log\big(1 - \tanh^2(u_i)\big)$. Forgetting the second term is a classic bug: the entropy of the squashed policy is then wrong, increasingly so as actions saturate.
:::

:::proposition[Linear-quadratic control: noise does not change the best action]
For dynamics $\boldsymbol{x}' = \mathbf{A}\boldsymbol{x} + \mathbf{B}\boldsymbol{u} + \boldsymbol{w}$ with $\boldsymbol{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{W})$ independent of $\boldsymbol{x}, \boldsymbol{u}$, and a quadratic cost-to-go $V(\boldsymbol{x}') = \boldsymbol{x}'^\top\mathbf{P}\boldsymbol{x}'$,

$$
\mathbb{E}[V(\boldsymbol{x}')] = (\mathbf{A}\boldsymbol{x} + \mathbf{B}\boldsymbol{u})^\top\mathbf{P}(\mathbf{A}\boldsymbol{x} + \mathbf{B}\boldsymbol{u}) + \operatorname{tr}(\mathbf{P}\mathbf{W}),
$$

so the action minimising $\boldsymbol{u}^\top\mathbf{R}\boldsymbol{u} + \mathbb{E}[V(\boldsymbol{x}')]$ is the same as for $\mathbf{W} = \mathbf{0}$.
:::

:::proof
**Step 1.** Given $\boldsymbol{x}$ and $\boldsymbol{u}$, $\boldsymbol{x}'$ has mean $\mathbf{A}\boldsymbol{x} + \mathbf{B}\boldsymbol{u}$ and covariance $\mathbf{W}$.

**Step 2.** Proposition 2 with $\mathbf{A} \to \mathbf{P}$ gives the formula.

**Step 3.** The noise enters only through $\operatorname{tr}(\mathbf{P}\mathbf{W})$, which does not depend on $\boldsymbol{u}$; its gradient in $\boldsymbol{u}$ is zero, so the minimiser is unchanged. This is *certainty equivalence*: plan as if there were no noise, and pay a constant for it.
:::

:::note[The Kalman filter is two identities]
**Predict.** $\boldsymbol{x}_{t+1} = \mathbf{A}\boldsymbol{x}_t + \boldsymbol{w}$: an affine map plus independent noise (GP Lemmas 2 and 3), so $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \mapsto \mathcal{N}(\mathbf{A}\boldsymbol{\mu}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top + \mathbf{W})$. Covariances add: uncertainty grows.

**Update.** Observe $\boldsymbol{y} = \mathbf{C}\boldsymbol{x} + \boldsymbol{v}$: the linear-Gaussian model, GP Lemma 8. Its moment form is the Kalman gain $\boldsymbol{\Sigma}\mathbf{C}^\top(\mathbf{C}\boldsymbol{\Sigma}\mathbf{C}^\top + \mathbf{V})^{-1}$; its information form is the information filter, where precisions add and uncertainty shrinks. For a scalar observation, Sherman–Morrison (Lemma 2) makes the update $O(D^2)$.
:::

## Questions

:::question
In the figure, why is the slice never wider than the shadow?
:::

:::answer
The slice's variance is $1/\mathbf{P}_{22} = \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$ (Remark after Proposition 4), which is $\Sigma_{22}$ minus a non-negative quantity. Observing $x_1$ can only remove uncertainty about $x_2$; for Gaussians it never adds any, whatever value $a$ is observed.
:::

:::question
$\operatorname{KL}(p_0 \| p_1) \ne \operatorname{KL}(p_1 \| p_0)$. Which term of Proposition 6 makes the asymmetry obvious?
:::

:::answer
The Mahalanobis term measures the distance between the means in units of $\boldsymbol{\Sigma}_1$. Swapping the arguments measures it in units of $\boldsymbol{\Sigma}_0$, which differs whenever the covariances do.
:::

:::question
How many Hutchinson probes give a trace estimate with relative standard deviation $1\%$ for $\mathbf{A} = \mathbf{I}_{1000}$, and for a rank-one $\mathbf{A} = \boldsymbol{v}\boldsymbol{v}^\top$?
:::

:::answer
With $S$ probes the variance is $2\lVert\mathbf{A}\rVert_F^2/S$. For $\mathbf{I}_{1000}$: $\operatorname{tr} = 1000$, $\lVert\mathbf{A}\rVert_F^2 = 1000$, so the relative s.d. is $\sqrt{2/(1000S)}$, and $S = 20$ suffices. For $\boldsymbol{v}\boldsymbol{v}^\top$: $\operatorname{tr} = \lVert\boldsymbol{v}\rVert^2$ and $\lVert\mathbf{A}\rVert_F^2 = \lVert\boldsymbol{v}\rVert^4$, so the relative s.d. is $\sqrt{2/S}$ and $S = 20\,000$. Hutchinson is good for "spread out" spectra and poor for low-rank ones.
:::


:::question
Your Gaussian-policy gradient estimates get noisier as training converges. Why, and what are two fixes?
:::

:::answer
The score is $\mathbf{J}^\top\boldsymbol{\varepsilon}/\sigma$ (Remark after Proposition 14), so its variance grows like $1/\sigma^2$ as exploration shrinks. Fixes: use the reparameterised gradient through a learned $Q$ (Bonnet), subtract a baseline or use advantages (Proposition 14), and keep $\sigma$ from collapsing (an entropy bonus).
:::

:::question
You add one data point to a GP with $N = 10\,000$ observations. What does it cost to update the posterior, and why?
:::

:::answer
$O(N^2)$, not $O(N^3)$. The new Gram matrix is the old one bordered by one row and column, so its Cholesky factor gains one row, computed by a triangular solve (Lemma 3 with a $1\times1$ block $\mathbf{D}$: the new diagonal entry is the square root of the Schur complement, the predictive variance of the new point).
:::

## References

- C. E. Rasmussen, C. K. I. Williams. [*Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/). MIT Press, 2006. Appendix A.
- K. B. Petersen, M. S. Pedersen. *The Matrix Cookbook*. 2012.
- C. M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006. Section 2.3.
- A. K. Gupta, D. K. Nagar. *Matrix Variate Distributions*. Chapman & Hall/CRC, 1999.
- L. Isserlis. *On a formula for the product-moment coefficient of any order of a normal frequency distribution in any number of variables*. Biometrika, 1918.
- C. M. Stein. *Estimation of the mean of a multivariate normal distribution*. Annals of Statistics, 1981.
- M. Opper, C. Archambeau. *The variational Gaussian approximation revisited*. Neural Computation, 2009. (Bonnet's and Price's theorems.)
- M. F. Hutchinson. *A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines*. Communications in Statistics, 1989.
- G. H. Golub, C. F. Van Loan. *Matrix Computations*. Johns Hopkins University Press, 4th ed., 2013.
- S. Amari. *Natural gradient works efficiently in learning*. Neural Computation, 1998.
- B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and scalable predictive uncertainty estimation using deep ensembles*. NeurIPS 2017.
- R. J. Williams. *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning, 1992.
- D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, M. Riedmiller. *Deterministic policy gradient algorithms*. ICML 2014.
- J. Schulman, S. Levine, P. Abbeel, M. Jordan, P. Moritz. *Trust region policy optimization*. ICML 2015.
- T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. *Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor*. ICML 2018.
- R. E. Kalman. *A new approach to linear filtering and prediction problems*. Journal of Basic Engineering, 1960.
