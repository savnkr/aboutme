---
title: "Deep Gaussian processes: stacking uncertainty"
description: "What happens when you stack Gaussian processes like neural network layers? You get deep GPs: expressive, uncertainty-aware, and surprisingly elegant once you know four Gaussian tricks."
date: 2026-02-07
updated: 2026-10-23
cover: /blog/deep-gp/cover.png
tags: [gaussian-processes, deep-gp, variational-inference, bayesian]
---

A single Gaussian process is a beautiful thing. Hand it some data, pick a kernel, and it gives you a posterior with calibrated uncertainty bars — no hyperparameter hacking, no dropout tricks, no ensembles. Just Bayes' rule applied to functions.

But there is a catch.

A standard GP is, in a precise sense, *shallow*. The kernel you choose — say, an RBF — determines what functions the GP considers plausible. If the true function has structure that your kernel cannot express (sharp transitions, compositional patterns, hierarchical features), the GP will struggle. You can engineer fancier kernels, but you are still making one modelling choice upfront and hoping it is enough.

What if, instead of designing a better kernel, we just *stacked* GPs?

---

## The idea: GPs as layers

Think of a neural network. Each layer transforms its input nonlinearly, and the composition of layers gives the network its expressiveness. A deep GP does the same thing, but each "layer" is a full Gaussian process.

For an input $\mathbf{x}$, a deep GP with $L$ layers defines:

$$
\begin{aligned}
f_1(\cdot) &\sim \mathcal{GP}(\mu_1(\cdot),\, k_1(\cdot,\cdot)) \\
f_2(\cdot) &\sim \mathcal{GP}(\mu_2(\cdot),\, k_2(\cdot,\cdot)) \\
&\;\;\vdots \\
f_L(\cdot) &\sim \mathcal{GP}(\mu_L(\cdot),\, k_L(\cdot,\cdot))
\end{aligned}
$$

The output of layer $\ell$ feeds as input to layer $\ell+1$:

$$
\mathbf{h}_1 = f_1(\mathbf{x}), \quad \mathbf{h}_2 = f_2(\mathbf{h}_1), \quad \ldots, \quad \mathbf{y} = f_L(\mathbf{h}_{L-1})
$$

Each layer uses a simple kernel (say, RBF), but the *composition* of layers can represent complex, hierarchical functions. The first layer might learn low-level features, the second layer combines them — exactly the deep learning playbook, except every layer carries full uncertainty.

**Why is this interesting?** Because a two-layer deep GP with RBF kernels can approximate functions that no single-layer GP with any stationary kernel can represent well. The depth buys expressiveness without hand-engineering.

---

## The problem: inference is hard

Here is where things get honest.   

In a standard (single-layer) GP, the posterior is available in closed form. You condition on data, invert a matrix, and you are done. In a deep GP, the intermediate layers $\mathbf{h}_1, \ldots, \mathbf{h}_{L-1}$ are *latent*. You need to marginalise over all possible hidden representations, which means integrating over function compositions — and that integral is analytically intractable.

So we need an approximation. The standard approach is **variational inference with inducing points**, and this is where the derivation gets genuinely satisfying.

---

## The toolkit: four Gaussian identities

Before diving into the deep GP derivation, let me give you the complete toolkit. Every calculation we will do — and most calculations in GPs, diffusion models, Kalman filters, and linear VAEs — reduces to four identities.

### 1. Affine transform with independent noise

If $\mathbf{u} \sim \mathcal{N}(\mathbf{m}, S)$ and $\mathbf{f} \mid \mathbf{u} = A\mathbf{u} + \mathbf{b} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, C)$ independent of $\mathbf{u}$, then:

$$
\mathbf{f} \sim \mathcal{N}(A\mathbf{m} + \mathbf{b},\; ASA^\top + C)
$$

**When to use it:** Whenever you need to integrate out a Gaussian variable from a linear-Gaussian model. This is the workhorse identity.

### 2. Block-Gaussian conditioning (Schur complement)

If $[\mathbf{x}; \mathbf{y}]$ is jointly Gaussian with covariance blocks $\Sigma_{xx}, \Sigma_{xy}, \Sigma_{yy}$, then:

$$
\mathbf{x} \mid \mathbf{y} \sim \mathcal{N}\big(\boldsymbol{\mu}_x + \Sigma_{xy}\Sigma_{yy}^{-1}(\mathbf{y} - \boldsymbol{\mu}_y),\; \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}\big)
$$

**When to use it:** Conditioning one subset of a joint Gaussian on another. This is the standard GP posterior formula.

### 3. Product of Gaussians

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1)\;\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_2, \Sigma_2) \propto \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma)
$$

where $\Sigma^{-1} = \Sigma_1^{-1} + \Sigma_2^{-1}$ and $\boldsymbol{\mu} = \Sigma(\Sigma_1^{-1}\boldsymbol{\mu}_1 + \Sigma_2^{-1}\boldsymbol{\mu}_2)$.

**When to use it:** Combining two sources of Gaussian information about the same variable. Precisions (inverse covariances) add. This is Bayesian updating in its purest form.

### 4. Linear Gaussian chain

If $\mathbf{x}_t = A_t \mathbf{x}_{t-1} + \boldsymbol{\epsilon}_t$ with $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, Q_t)$, then means and covariances propagate as:

$$
\mathbf{m}_t = A_t \mathbf{m}_{t-1}, \quad P_t = A_t P_{t-1} A_t^\top + Q_t
$$

**When to use it:** Propagating uncertainty through sequential transformations — Kalman filters, diffusion forward processes, autoregressive models.

These four identities, plus basic matrix algebra, are all you need. No completing the square, no brute-force integrals.

---

## The confusion: product vs convolution vs mixture

These three operations all combine Gaussians, but they do completely different things. Mixing them up is one of the most common mistakes in probabilistic ML — I have done it myself. Let me be explicit.

### Product of Gaussians (same variable)

You have two Gaussian densities *over the same variable* $\mathbf{x}$ and you **multiply** them pointwise:

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1) \cdot \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_2, \Sigma_2) \propto \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma)
$$

Result: **precisions add, covariances shrink.** The result is *narrower* than either input — you are combining two sources of information, so uncertainty decreases.

$$
\Sigma^{-1} = \Sigma_1^{-1} + \Sigma_2^{-1}, \quad \boldsymbol{\mu} = \Sigma(\Sigma_1^{-1}\boldsymbol{\mu}_1 + \Sigma_2^{-1}\boldsymbol{\mu}_2)
$$

**When it appears:** Bayes' rule with Gaussian prior and Gaussian likelihood. The posterior precision is the sum of prior and likelihood precisions.

### What if the variables are different?

If you have $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1) \cdot \mathcal{N}(\mathbf{y} \mid \boldsymbol{\mu}_2, \Sigma_2)$ where $\mathbf{x}$ and $\mathbf{y}$ are **different variables**, this is just the joint density of two independent Gaussians:

$$
p(\mathbf{x}, \mathbf{y}) = \mathcal{N}\!\left(\begin{bmatrix}\mathbf{x} \\ \mathbf{y}\end{bmatrix} \;\middle|\; \begin{bmatrix}\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2\end{bmatrix},\; \begin{bmatrix}\Sigma_1 & 0 \\ 0 & \Sigma_2\end{bmatrix}\right)
$$

No precision addition happens — the covariance is block-diagonal. The product-of-Gaussians identity **only gives precision addition when both factors are over the same variable.** This is the critical point. In Bayesian inference, the prior $p(\mathbf{x})$ and the likelihood $p(\mathbf{y} \mid \mathbf{x})$ involve different variables ($\mathbf{x}$ vs $\mathbf{y}$), but we rewrite the likelihood as a function of $\mathbf{x}$ (for fixed observed $\mathbf{y}$), making it a Gaussian in $\mathbf{x}$. Only *then* does the product identity apply.

### Convolution of Gaussians (sum of independent variables)

You have two *independent* Gaussian random variables and you **add** them: $\mathbf{z} = \mathbf{x} + \mathbf{y}$.

$$
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1), \quad \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2), \quad \mathbf{x} \perp \mathbf{y}
$$

$$
\mathbf{z} = \mathbf{x} + \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2, \; \Sigma_1 + \Sigma_2)
$$

Result: **means add, covariances add.** The result is *wider* than either input — combining independent sources of randomness increases uncertainty.

Equivalently, this is the convolution of the two densities:

$$
p(\mathbf{z}) = \int \mathcal{N}(\mathbf{z} - \mathbf{y} \mid \boldsymbol{\mu}_1, \Sigma_1)\;\mathcal{N}(\mathbf{y} \mid \boldsymbol{\mu}_2, \Sigma_2)\; d\mathbf{y}
$$

**When it appears:** The affine transform identity is a generalisation of this. When you write $\mathbf{f} = A\mathbf{u} + \boldsymbol{\varepsilon}$, the covariance $A S A^\top + C$ has the "$+C$" term precisely because you are *adding* independent noise — that is a convolution.

### Mixture of Gaussians (weighted sum of densities)

You take a **weighted sum of density functions** (not of random variables):

$$
p(\mathbf{x}) = \pi_1 \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1) + \pi_2 \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_2, \Sigma_2), \quad \pi_1 + \pi_2 = 1
$$

Result: **not Gaussian.** The resulting density is multimodal in general. There is no closed-form simplification — you cannot reduce a mixture to a single Gaussian (only approximate it, e.g. by moment matching).

**When it appears:** Gaussian mixture models, marginalising over discrete latent variables.

### The comparison at a glance

| Operation | Formula | Result | Variance |
|---|---|---|---|
| **Product** (same variable) | $p_1(\mathbf{x}) \cdot p_2(\mathbf{x})$ | Gaussian | **Shrinks** (precisions add) |
| **Convolution** (independent sum) | $\mathbf{z} = \mathbf{x} + \mathbf{y}$ | Gaussian | **Grows** (covariances add) |
| **Mixture** (weighted densities) | $\pi_1 p_1(\mathbf{x}) + \pi_2 p_2(\mathbf{x})$ | **Not** Gaussian | Can be anything |
| **Product** (different variables) | $p_1(\mathbf{x}) \cdot p_2(\mathbf{y})$ | Joint Gaussian (independent) | Block-diagonal |

The key intuition: **product sharpens** (more information), **convolution blurs** (more noise), **mixture creates bumps** (multiple modes).

---

## Deriving the variational posterior, step by step

Now let us do the core deep GP calculation. We will work through one layer — once you see how it works for one, the deep case is just iteration.

### Setup

We have a GP prior over function values at training points $X$ and inducing points $Z$:

$$
\begin{bmatrix} \mathbf{f} \\ \mathbf{u} \end{bmatrix}
\sim \mathcal{N}\!\left(
\begin{bmatrix} \mathbf{m}_f \\ \mathbf{m}_u \end{bmatrix},\;
\begin{bmatrix}
K_{ff} & K_{fu} \\
K_{uf} & K_{uu}
\end{bmatrix}
\right)
$$

where $\mathbf{f} = f(X) \in \mathbb{R}^N$ are the function values we care about, and $\mathbf{u} = f(Z) \in \mathbb{R}^M$ are the inducing variables — a small set of "summary" points that make inference tractable.

Our variational approximation places a free Gaussian over the inducing variables:

$$
q(\mathbf{u}) = \mathcal{N}(\mathbf{u} \mid \mathbf{m}, S)
$$

The question is: **what does the implied distribution over $\mathbf{f}$ look like?**

$$
q(\mathbf{f}) = \int p(\mathbf{f} \mid \mathbf{u})\, q(\mathbf{u})\, d\mathbf{u}
$$

### Step 1: Write $p(\mathbf{f} \mid \mathbf{u})$ as a linear-Gaussian

Using **identity 2** (block conditioning):

$$
\mathbf{f} \mid \mathbf{u} \sim \mathcal{N}\big(\mathbf{m}_f + K_{fu}K_{uu}^{-1}(\mathbf{u} - \mathbf{m}_u),\; K_{ff} - K_{fu}K_{uu}^{-1}K_{uf}\big)
$$

This is a linear function of $\mathbf{u}$ plus independent noise. Define:

$$
A = K_{fu}K_{uu}^{-1}, \quad \mathbf{b} = \mathbf{m}_f - A\mathbf{m}_u, \quad C = K_{ff} - K_{fu}K_{uu}^{-1}K_{uf}
$$

Now the conditional reads: $\mathbf{f} \mid \mathbf{u} = A\mathbf{u} + \mathbf{b} + \boldsymbol{\varepsilon}$, with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, C)$.

### Step 2: Integrate out $\mathbf{u}$ with the affine transform

We  have $\mathbf{u} \sim \mathcal{N}(\mathbf{m}, S)$ and $\mathbf{f} \mid \mathbf{u}$ is linear-Gaussian. Apply **identity 1** directly:

$$
q(\mathbf{f}) = \mathcal{N}(\tilde{\boldsymbol{\mu}},\, \tilde{\Sigma})
$$

**Mean:**
$$
\tilde{\boldsymbol{\mu}} = A\mathbf{m} + \mathbf{b} = \mathbf{m}_f + K_{fu}K_{uu}^{-1}(\mathbf{m} - \mathbf{m}_u)
$$

**Covariance:**
$$
\tilde{\Sigma} = ASA^\top + C = K_{fu}K_{uu}^{-1} S\, K_{uu}^{-1}K_{uf} + K_{ff} - K_{fu}K_{uu}^{-1}K_{uf}
$$

Factor it:

$$
\boxed{\tilde{\Sigma} = K_{ff} - K_{fu}K_{uu}^{-1}(K_{uu} - S)K_{uu}^{-1}K_{uf}}
$$

That is it. Two identities, a few lines of algebra, no integrals. The mean is the prior mean corrected by how far the inducing variables deviate from their prior. The covariance is the prior covariance minus a correction that depends on how "certain" the variational distribution $q(\mathbf{u})$ is (the $K_{uu} - S$ term).

When $S = K_{uu}$ (maximum uncertainty), $\tilde{\Sigma} = K_{ff}$ — we recover the prior. When $S \to 0$ (perfect knowledge of inducing points), we get the maximum possible variance reduction. The variational parameters $\mathbf{m}$ and $S$ are what we optimise.

---

## Why this matters beyond GPs

Here is the surprising thing: the exact same four identities appear everywhere in probabilistic ML.

**Diffusion models.** The DDPM forward process $x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$ is a linear Gaussian chain (identity 4). The closed-form $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)I)$ falls out from a three-line induction using the chain rule for means and covariances.

**Diffusion posterior.** The reverse step $q(x_{t-1} \mid x_t, x_0)$ is a product of two Gaussians in $x_{t-1}$ (identity 3). Precisions add, means combine — no completing the square needed.

**Kalman filters.** The prediction step is identity 4. The update step is identity 3 (or equivalently, identity 2). Every Kalman filter derivation in every textbook is these two identities applied alternately.

**Linear VAEs.** The encoder-decoder structure with Gaussian assumptions uses identity 1 (marginalising the latent) and identity 3 (posterior over the latent given the observation).

Once you internalise these four identities, a large fraction of probabilistic ML becomes pattern matching.

---

## Practical notes

A few things that matter when you actually implement deep GPs:

**Never invert $K_{uu}$ explicitly.** Use the Cholesky decomposition: $K_{uu} = LL^\top$, then solve triangular systems. This is numerically stable and $O(M^3)$ instead of $O(N^3)$.

**Add jitter.** In practice, $K_{uu} + \epsilon I$ with $\epsilon \sim 10^{-6}$ prevents numerical issues from near-singular kernel matrices.

**The ELBO.** The variational parameters $\mathbf{m}$ and $S$ are optimised by maximising the evidence lower bound:

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

The KL term is between two Gaussians, so it is closed form. The expected log-likelihood depends on the observation model — Gaussian likelihoods give a closed-form expectation, non-Gaussian ones require further approximation (quadrature or sampling).

**Depth in practice.** Two to three layers is typical. Going deeper introduces optimisation challenges (vanishing signal through layers) without clear empirical gains for most problems. The sweet spot is often a two-layer model where the first layer acts as a learned input warping.

---

## Takeaway

Deep GPs are what you get when you take the GP framework seriously and ask: *what if the kernel could learn?* By stacking GP layers, the model learns hierarchical representations while maintaining calibrated uncertainty — something deep neural networks struggle with.

The derivation looks intimidating at first, but it reduces entirely to four Gaussian identities applied in sequence. The same identities power diffusion models, Kalman filters, and variational autoencoders. Learning to see these patterns — rather than re-deriving from scratch each time — is what makes probabilistic ML tractable.

The next time you see a nasty-looking integral over Gaussians, ask: *is this an affine transform, a block conditioning, a product, or a chain?* It is almost always one of the four.

---

*Further reading:*
- Damianou & Lawrence, [Deep Gaussian Processes](https://arxiv.org/abs/1211.0358) (2013) — the original paper.
- Salimbeni & Deisenroth, [Doubly Stochastic Variational Inference for Deep GPs](https://arxiv.org/abs/1705.08933) (2017) — the practical inference method.
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* — the GP bible, free online.

---

## Appendix: Full derivations

Every identity used in this post is derived from scratch below. No steps are skipped.

### A.1 Affine transform with independent noise

**Claim.** If $\mathbf{u} \sim \mathcal{N}(\mathbf{m}, S)$ and $\mathbf{f} \mid \mathbf{u} = A\mathbf{u} + \mathbf{b} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, C)$, $\boldsymbol{\varepsilon} \perp \mathbf{u}$, then $\mathbf{f} \sim \mathcal{N}(A\mathbf{m} + \mathbf{b},\; ASA^\top + C)$.

**Proof (via moments).**

The mean:

$$
\mathbb{E}[\mathbf{f}] = \mathbb{E}[A\mathbf{u} + \mathbf{b} + \boldsymbol{\varepsilon}] = A\,\mathbb{E}[\mathbf{u}] + \mathbf{b} + \mathbb{E}[\boldsymbol{\varepsilon}] = A\mathbf{m} + \mathbf{b}
$$

The covariance. Since $\mathbf{u}$ and $\boldsymbol{\varepsilon}$ are independent:

$$
\mathrm{Cov}(\mathbf{f}) = \mathrm{Cov}(A\mathbf{u} + \boldsymbol{\varepsilon}) = A\,\mathrm{Cov}(\mathbf{u})\,A^\top + \mathrm{Cov}(\boldsymbol{\varepsilon}) = ASA^\top + C
$$

The cross-term $\mathrm{Cov}(A\mathbf{u}, \boldsymbol{\varepsilon}) = A\,\mathbb{E}[\mathbf{u}\boldsymbol{\varepsilon}^\top] = 0$ vanishes by independence. Since $\mathbf{f}$ is a linear function of jointly Gaussian variables $(\mathbf{u}, \boldsymbol{\varepsilon})$, it is Gaussian. $\square$

**Proof (via characteristic function).**

The characteristic function of $\mathbf{u}$ is $\phi_{\mathbf{u}}(\mathbf{t}) = \exp(i\mathbf{t}^\top \mathbf{m} - \tfrac{1}{2}\mathbf{t}^\top S \mathbf{t})$.

For $\mathbf{f} = A\mathbf{u} + \mathbf{b} + \boldsymbol{\varepsilon}$:

$$
\phi_{\mathbf{f}}(\mathbf{t}) = \mathbb{E}[e^{i\mathbf{t}^\top \mathbf{f}}] = e^{i\mathbf{t}^\top \mathbf{b}}\;\mathbb{E}[e^{i\mathbf{t}^\top A\mathbf{u}}]\;\mathbb{E}[e^{i\mathbf{t}^\top \boldsymbol{\varepsilon}}]
$$

where the factorisation uses independence. Substituting:

$$
\phi_{\mathbf{f}}(\mathbf{t}) = e^{i\mathbf{t}^\top \mathbf{b}} \cdot e^{i(A^\top\mathbf{t})^\top \mathbf{m} - \frac{1}{2}(A^\top\mathbf{t})^\top S (A^\top\mathbf{t})} \cdot e^{-\frac{1}{2}\mathbf{t}^\top C \mathbf{t}}
$$

$$
= \exp\!\Big(i\mathbf{t}^\top(A\mathbf{m} + \mathbf{b}) - \tfrac{1}{2}\mathbf{t}^\top(ASA^\top + C)\mathbf{t}\Big)
$$

This is the characteristic function of $\mathcal{N}(A\mathbf{m} + \mathbf{b},\; ASA^\top + C)$. $\square$

The characteristic function proof is more general — it works even when $A$ is not square (e.g., projecting to a lower dimension), where a Jacobian-based density argument would fail.

---

### A.2 Block-Gaussian conditioning (Schur complement)

**Claim.** If

$$
\begin{bmatrix}\mathbf{x} \\ \mathbf{y}\end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix}\boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y\end{bmatrix},\; \begin{bmatrix}\Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}\end{bmatrix}\right)
$$

then $\mathbf{x} \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_{x|y},\; \Sigma_{x|y})$ where

$$
\boldsymbol{\mu}_{x|y} = \boldsymbol{\mu}_x + \Sigma_{xy}\Sigma_{yy}^{-1}(\mathbf{y} - \boldsymbol{\mu}_y), \quad \Sigma_{x|y} = \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}
$$

**Proof.** Write the joint log-density (dropping constants that do not depend on $\mathbf{x}$):

$$
\log p(\mathbf{x}, \mathbf{y}) = -\frac{1}{2}\begin{bmatrix}\mathbf{x} - \boldsymbol{\mu}_x \\ \mathbf{y} - \boldsymbol{\mu}_y\end{bmatrix}^\top \begin{bmatrix}\Lambda_{xx} & \Lambda_{xy} \\ \Lambda_{yx} & \Lambda_{yy}\end{bmatrix} \begin{bmatrix}\mathbf{x} - \boldsymbol{\mu}_x \\ \mathbf{y} - \boldsymbol{\mu}_y\end{bmatrix} + \text{const}
$$

where $\Lambda = \Sigma^{-1}$ is the joint precision. Expand the quadratic form and collect terms in $\mathbf{x}$:

$$
Q = (\mathbf{x} - \boldsymbol{\mu}_x)^\top \Lambda_{xx} (\mathbf{x} - \boldsymbol{\mu}_x) + 2(\mathbf{x} - \boldsymbol{\mu}_x)^\top \Lambda_{xy}(\mathbf{y} - \boldsymbol{\mu}_y) + \text{terms in } \mathbf{y} \text{ only}
$$

Complete the square in $\mathbf{x}$. Define $\mathbf{m} = \boldsymbol{\mu}_x - \Lambda_{xx}^{-1}\Lambda_{xy}(\mathbf{y} - \boldsymbol{\mu}_y)$. Then:

$$
Q = (\mathbf{x} - \mathbf{m})^\top \Lambda_{xx} (\mathbf{x} - \mathbf{m}) + \text{terms not involving } \mathbf{x}
$$

So the conditional is $\mathbf{x} \mid \mathbf{y} \sim \mathcal{N}(\mathbf{m},\; \Lambda_{xx}^{-1})$.

**Remaining step:** express $\Lambda_{xx}^{-1}$ and $\mathbf{m}$ in terms of covariance blocks. From the block matrix inverse (A.6):

$$
\Lambda_{xx} = (\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx})^{-1}
$$

so $\Lambda_{xx}^{-1} = \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx} = \Sigma_{x|y}$.

For the mean, the block inverse also gives $\Lambda_{xy} = -\Lambda_{xx}\Sigma_{xy}\Sigma_{yy}^{-1}$, so:

$$
\mathbf{m} = \boldsymbol{\mu}_x - \Lambda_{xx}^{-1}(-\Lambda_{xx}\Sigma_{xy}\Sigma_{yy}^{-1})(\mathbf{y} - \boldsymbol{\mu}_y) = \boldsymbol{\mu}_x + \Sigma_{xy}\Sigma_{yy}^{-1}(\mathbf{y} - \boldsymbol{\mu}_y) = \boldsymbol{\mu}_{x|y}
$$

$\square$

**Key insight.** The conditional covariance $\Sigma_{x|y}$ does not depend on the observed value $\mathbf{y}$ — only the mean shifts. This is a fundamental property of Gaussians and the reason GP posterior variance is data-independent (it only depends on input locations, not output values).

---

### A.3 Product of Gaussians (same variable) — with normalisation constant

**Claim.** $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1)\;\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_2, \Sigma_2) = c \cdot \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma)$ where $\Sigma^{-1} = \Sigma_1^{-1} + \Sigma_2^{-1}$, $\boldsymbol{\mu} = \Sigma(\Sigma_1^{-1}\boldsymbol{\mu}_1 + \Sigma_2^{-1}\boldsymbol{\mu}_2)$, and $c = \mathcal{N}(\boldsymbol{\mu}_1 \mid \boldsymbol{\mu}_2,\; \Sigma_1 + \Sigma_2)$.

**Proof.** Take the log of the product and keep only terms involving $\mathbf{x}$:

$$
\log[\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1) \cdot \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_2, \Sigma_2)]
$$

$$
= -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^\top \Sigma_1^{-1}(\mathbf{x} - \boldsymbol{\mu}_1) - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_2)^\top \Sigma_2^{-1}(\mathbf{x} - \boldsymbol{\mu}_2) + \text{const}
$$

Expand both quadratic forms:

$$
= -\frac{1}{2}\mathbf{x}^\top\Sigma_1^{-1}\mathbf{x} + \mathbf{x}^\top\Sigma_1^{-1}\boldsymbol{\mu}_1 - \frac{1}{2}\mathbf{x}^\top\Sigma_2^{-1}\mathbf{x} + \mathbf{x}^\top\Sigma_2^{-1}\boldsymbol{\mu}_2 + \text{const}
$$

Collect terms:

$$
= -\frac{1}{2}\mathbf{x}^\top \underbrace{(\Sigma_1^{-1} + \Sigma_2^{-1})}_{\Sigma^{-1}} \mathbf{x} + \mathbf{x}^\top\underbrace{(\Sigma_1^{-1}\boldsymbol{\mu}_1 + \Sigma_2^{-1}\boldsymbol{\mu}_2)}_{\Sigma^{-1}\boldsymbol{\mu}} + \text{const}
$$

This is a quadratic form in $\mathbf{x}$ with precision $\Sigma^{-1} = \Sigma_1^{-1} + \Sigma_2^{-1}$ and precision-weighted mean $\Sigma^{-1}\boldsymbol{\mu} = \Sigma_1^{-1}\boldsymbol{\mu}_1 + \Sigma_2^{-1}\boldsymbol{\mu}_2$. Solving: $\boldsymbol{\mu} = \Sigma(\Sigma_1^{-1}\boldsymbol{\mu}_1 + \Sigma_2^{-1}\boldsymbol{\mu}_2)$.

**The normalisation constant.** To find $c$, note that both sides must integrate to the same value over $\mathbf{x}$. The left side integrates to $\int \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1)\,\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_2, \Sigma_2)\,d\mathbf{x}$. The right side integrates to $c$ (since the Gaussian integrates to 1). Computing the left integral by collecting all the constant terms we dropped:

$$
c = \mathcal{N}(\boldsymbol{\mu}_1 \mid \boldsymbol{\mu}_2,\; \Sigma_1 + \Sigma_2)
$$

This factor measures how "compatible" the two Gaussians are — it is large when their means are close relative to their combined spread, and small when they disagree. It appears in GP marginal likelihoods and is easy to forget. $\square$

**Same variable vs different variables.** The precision-addition structure only holds when both factors are functions of the **same** $\mathbf{x}$. If you have $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_1, \Sigma_1) \cdot \mathcal{N}(\mathbf{y} \mid \boldsymbol{\mu}_2, \Sigma_2)$ with $\mathbf{x} \neq \mathbf{y}$, this is just the joint of two independent Gaussians — the covariance is block-diagonal with no interaction. In Bayesian inference, the likelihood $p(\mathbf{y} \mid \mathbf{x})$ involves a different variable $\mathbf{y}$, but we fix $\mathbf{y}$ at the observed value and treat the likelihood as a function of $\mathbf{x}$ alone. Only then does the product identity apply.

---

### A.4 Convolution of Gaussians (sum of independent variables)

**Claim.** If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ and $\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$ are independent, then $\mathbf{z} = \mathbf{x} + \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2,\; \Sigma_1 + \Sigma_2)$.

**Proof (via characteristic function).**

$$
\phi_{\mathbf{z}}(\mathbf{t}) = \mathbb{E}[e^{i\mathbf{t}^\top(\mathbf{x} + \mathbf{y})}] = \mathbb{E}[e^{i\mathbf{t}^\top\mathbf{x}}]\;\mathbb{E}[e^{i\mathbf{t}^\top\mathbf{y}}]
$$

by independence. Substituting the Gaussian characteristic functions:

$$
= \exp\!\big(i\mathbf{t}^\top\boldsymbol{\mu}_1 - \tfrac{1}{2}\mathbf{t}^\top\Sigma_1\mathbf{t}\big) \cdot \exp\!\big(i\mathbf{t}^\top\boldsymbol{\mu}_2 - \tfrac{1}{2}\mathbf{t}^\top\Sigma_2\mathbf{t}\big)
$$

$$
= \exp\!\big(i\mathbf{t}^\top(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2) - \tfrac{1}{2}\mathbf{t}^\top(\Sigma_1 + \Sigma_2)\mathbf{t}\big)
$$

This is the characteristic function of $\mathcal{N}(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2, \Sigma_1 + \Sigma_2)$. $\square$

**Why the name "convolution"?** The density of $\mathbf{z} = \mathbf{x} + \mathbf{y}$ is the convolution of the individual densities:

$$
p_{\mathbf{z}}(\mathbf{z}) = \int p_{\mathbf{x}}(\mathbf{z} - \mathbf{y})\,p_{\mathbf{y}}(\mathbf{y})\,d\mathbf{y} = (p_{\mathbf{x}} * p_{\mathbf{y}})(\mathbf{z})
$$

The characteristic function proof sidesteps evaluating this integral directly.

**Contrast with the product identity:** In the product, *precisions* add (the result gets sharper). In the convolution, *covariances* add (the result gets wider). Product combines **information**. Convolution combines **randomness**.

The affine transform (A.1) is a generalisation: $A\mathbf{u} + \boldsymbol{\varepsilon}$ is a convolution of $A\mathbf{u}$ (covariance $ASA^\top$) with $\boldsymbol{\varepsilon}$ (covariance $C$), giving total covariance $ASA^\top + C$.

---

### A.5 Linear Gaussian chain (recursive propagation)

**Claim.** If $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{m}_0, P_0)$ and $\mathbf{x}_t = A_t \mathbf{x}_{t-1} + \boldsymbol{\epsilon}_t$, $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, Q_t)$, $\boldsymbol{\epsilon}_t \perp \mathbf{x}_{t-1}$, then $\mathbf{x}_t \sim \mathcal{N}(\mathbf{m}_t, P_t)$ with $\mathbf{m}_t = A_t \mathbf{m}_{t-1}$ and $P_t = A_t P_{t-1} A_t^\top + Q_t$.

**Proof.** By induction. Base case: $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{m}_0, P_0)$ by assumption.

Inductive step: assume $\mathbf{x}_{t-1} \sim \mathcal{N}(\mathbf{m}_{t-1}, P_{t-1})$. Then $\mathbf{x}_t = A_t \mathbf{x}_{t-1} + \boldsymbol{\epsilon}_t$ is an affine transform with independent noise. By identity A.1 with $A = A_t$, $\mathbf{b} = \mathbf{0}$, $C = Q_t$:

$$
\mathbf{x}_t \sim \mathcal{N}(A_t \mathbf{m}_{t-1},\; A_t P_{t-1} A_t^\top + Q_t)
$$

$\square$

This is the **Kalman prediction step**. The Kalman update step (incorporating an observation $\mathbf{y}_t = H_t \mathbf{x}_t + \boldsymbol{\eta}_t$) is the product identity (A.3) applied to the predicted prior and the observation likelihood, both viewed as Gaussians in $\mathbf{x}_t$.

---

### A.6 Block matrix inverse (Schur complement)

This identity is used in the proof of A.2 and is worth stating explicitly.

**Claim.** If $\Sigma_{yy}$ is invertible, then

$$
\begin{bmatrix}\Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}\end{bmatrix}^{-1} = \begin{bmatrix} S^{-1} & -S^{-1}\Sigma_{xy}\Sigma_{yy}^{-1} \\ -\Sigma_{yy}^{-1}\Sigma_{yx}S^{-1} & \Sigma_{yy}^{-1} + \Sigma_{yy}^{-1}\Sigma_{yx}S^{-1}\Sigma_{xy}\Sigma_{yy}^{-1}\end{bmatrix}
$$

where $S = \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}$ is the Schur complement.

**Proof.** We need blocks $E, F, G, H$ such that:

$$
\begin{bmatrix}\Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}\end{bmatrix}\begin{bmatrix}E & F \\ G & H\end{bmatrix} = \begin{bmatrix}I & 0 \\ 0 & I\end{bmatrix}
$$

This gives four equations:

| | |
|---|---|
| (1) $\Sigma_{xx}E + \Sigma_{xy}G = I$ | (2) $\Sigma_{xx}F + \Sigma_{xy}H = 0$ |
| (3) $\Sigma_{yx}E + \Sigma_{yy}G = 0$ | (4) $\Sigma_{yx}F + \Sigma_{yy}H = I$ |

From (3): $G = -\Sigma_{yy}^{-1}\Sigma_{yx}E$.

Substitute into (1): $(\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx})E = I$, so $E = S^{-1}$.

Then $G = -\Sigma_{yy}^{-1}\Sigma_{yx}S^{-1}$.

From (2): $F = -\Sigma_{xx}^{-1}\Sigma_{xy}H$, but to avoid assuming $\Sigma_{xx}$ invertible, substitute $G$ into the structure. From (2) and using $E = S^{-1}$:

$$
\Sigma_{xx}F + \Sigma_{xy}H = 0 \implies F = -S^{-1}\Sigma_{xy}\Sigma_{yy}^{-1}
$$

(by multiplying (2) by $S^{-1}$ from the left after rewriting in terms of the Schur complement).

From (4): $\Sigma_{yx}F + \Sigma_{yy}H = I$, substituting $F$:

$$
H = \Sigma_{yy}^{-1}(I - \Sigma_{yx}F) = \Sigma_{yy}^{-1} + \Sigma_{yy}^{-1}\Sigma_{yx}S^{-1}\Sigma_{xy}\Sigma_{yy}^{-1}
$$

$\square$

---

### A.7 KL divergence between two Gaussians

This appears in every variational GP method — both Titsias and Hensman use it.

**Claim.** For $q = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ and $p = \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)$, both in $\mathbb{R}^D$:

$$
\mathrm{KL}(q \| p) = \frac{1}{2}\Big[\mathrm{tr}(\Sigma_0^{-1}\Sigma_1) + (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^\top \Sigma_0^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1) - D + \log\frac{|\Sigma_0|}{|\Sigma_1|}\Big]
$$

**Proof.** By definition:

$$
\mathrm{KL}(q \| p) = \mathbb{E}_q[\log q(\mathbf{x}) - \log p(\mathbf{x})]
$$

Compute each term. The log-density of a Gaussian $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma)$ is:

$$
\log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})
$$

Therefore:

$$
\mathrm{KL}(q \| p) = \frac{1}{2}\log\frac{|\Sigma_0|}{|\Sigma_1|} + \frac{1}{2}\mathbb{E}_q\big[(\mathbf{x} - \boldsymbol{\mu}_0)^\top\Sigma_0^{-1}(\mathbf{x} - \boldsymbol{\mu}_0) - (\mathbf{x} - \boldsymbol{\mu}_1)^\top\Sigma_1^{-1}(\mathbf{x} - \boldsymbol{\mu}_1)\big]
$$

Use the trace trick (A.8 below) for the expected quadratic forms:

$$
\mathbb{E}_q[(\mathbf{x} - \boldsymbol{\mu}_0)^\top\Sigma_0^{-1}(\mathbf{x} - \boldsymbol{\mu}_0)] = \mathrm{tr}(\Sigma_0^{-1}\Sigma_1) + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^\top\Sigma_0^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)
$$

$$
\mathbb{E}_q[(\mathbf{x} - \boldsymbol{\mu}_1)^\top\Sigma_1^{-1}(\mathbf{x} - \boldsymbol{\mu}_1)] = \mathrm{tr}(\Sigma_1^{-1}\Sigma_1) = D
$$

Substituting gives the result. $\square$

---

### A.8 The trace trick (expected quadratic forms)

This identity is used constantly — in KL computations, ELBO derivations, and expected log-likelihoods.

**Claim.** If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ and $A$ is a fixed matrix, then:

$$
\mathbb{E}[(\mathbf{x} - \mathbf{c})^\top A (\mathbf{x} - \mathbf{c})] = (\boldsymbol{\mu} - \mathbf{c})^\top A (\boldsymbol{\mu} - \mathbf{c}) + \mathrm{tr}(A\Sigma)
$$

for any constant vector $\mathbf{c}$.

**Proof.** Let $\tilde{\mathbf{x}} = \mathbf{x} - \mathbf{c}$. Then $\tilde{\mathbf{x}} \sim \mathcal{N}(\boldsymbol{\mu} - \mathbf{c}, \Sigma)$. Write $\tilde{\mathbf{x}} = (\boldsymbol{\mu} - \mathbf{c}) + \boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} = \mathbf{x} - \boldsymbol{\mu} \sim \mathcal{N}(\mathbf{0}, \Sigma)$. Expand:

$$
\tilde{\mathbf{x}}^\top A \tilde{\mathbf{x}} = [(\boldsymbol{\mu} - \mathbf{c}) + \boldsymbol{\epsilon}]^\top A [(\boldsymbol{\mu} - \mathbf{c}) + \boldsymbol{\epsilon}]
$$

$$
= (\boldsymbol{\mu} - \mathbf{c})^\top A (\boldsymbol{\mu} - \mathbf{c}) + 2(\boldsymbol{\mu} - \mathbf{c})^\top A \boldsymbol{\epsilon} + \boldsymbol{\epsilon}^\top A \boldsymbol{\epsilon}
$$

Take expectations. The cross-term vanishes: $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$. For the quadratic term:

$$
\mathbb{E}[\boldsymbol{\epsilon}^\top A \boldsymbol{\epsilon}] = \mathbb{E}[\mathrm{tr}(A \boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top)] = \mathrm{tr}(A\,\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top]) = \mathrm{tr}(A\Sigma)
$$

where we used the cyclic property of trace: $\boldsymbol{\epsilon}^\top A \boldsymbol{\epsilon} = \mathrm{tr}(\boldsymbol{\epsilon}^\top A \boldsymbol{\epsilon}) = \mathrm{tr}(A \boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top)$. $\square$

**Special cases:**
- $\mathbf{c} = \mathbf{0}$: $\mathbb{E}[\mathbf{x}^\top A \mathbf{x}] = \boldsymbol{\mu}^\top A \boldsymbol{\mu} + \mathrm{tr}(A\Sigma)$
- $\mathbf{c} = \boldsymbol{\mu}$: $\mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})^\top A (\mathbf{x} - \boldsymbol{\mu})] = \mathrm{tr}(A\Sigma)$
- $A = I$: $\mathbb{E}[\|\mathbf{x} - \mathbf{c}\|^2] = \|\boldsymbol{\mu} - \mathbf{c}\|^2 + \mathrm{tr}(\Sigma)$

---

### A.9 Woodbury identity and matrix determinant lemma

These reduce $N \times N$ inversions to $M \times M$ when $M \ll N$ — the computational backbone of all sparse GP methods.

**Woodbury identity.** For invertible $A$ ($N \times N$) and $C$ ($M \times M$), with $U$ ($N \times M$) and $V$ ($M \times N$):

$$
(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
$$

**Proof.** Multiply both sides on the right by $(A + UCV)$ and verify the result is $I$:

$$
\big[A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}\big](A + UCV)
$$

$$
= I + A^{-1}UCV - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}V - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}UCV
$$

Factor the last two terms. Let $W = (C^{-1} + VA^{-1}U)^{-1}$. Then:

$$
= I + A^{-1}UCV - A^{-1}UW\big[V + VA^{-1}UCV\big]
$$

$$
= I + A^{-1}UCV - A^{-1}UW\big[I + VA^{-1}UC\big]V
$$

Note that $W(I + VA^{-1}UC) = W(C^{-1} + VA^{-1}U)C = C$ (since $W$ is the inverse of $C^{-1} + VA^{-1}U$). So:

$$
= I + A^{-1}UCV - A^{-1}UCV = I
$$

$\square$

**GP application.** In a GP with $N$ observations and $M$ inducing points, the marginal covariance is $K_{ff} + \sigma^2 I = \sigma^2 I + K_{fu}K_{uu}^{-1}K_{uf} + (K_{ff} - K_{fu}K_{uu}^{-1}K_{uf})$. The Woodbury identity inverts the $N \times N$ matrix using only $M \times M$ inversions, reducing cost from $O(N^3)$ to $O(NM^2)$.

**Matrix determinant lemma.** Under the same conditions:

$$
|A + UCV| = |C^{-1} + VA^{-1}U| \cdot |C| \cdot |A|
$$

This is used to compute log marginal likelihoods without forming the full $N \times N$ matrix.

---

## Appendix B: Titsias vs Hensman — Sparse GP Inference

This section derives both approaches from first principles, making the mathematical distinction precise.

### B.1 The shared starting point

Both methods approximate the exact GP posterior using inducing variables $\mathbf{u} = f(Z) \in \mathbb{R}^M$, where $Z$ are $M$ inducing locations. The exact GP model is:

$$
\mathbf{f} \sim \mathcal{N}(\mathbf{0}, K_{ff}), \quad \mathbf{y} \mid \mathbf{f} \sim \mathcal{N}(\mathbf{f}, \sigma^2 I)
$$

The exact marginal likelihood $p(\mathbf{y})$ requires inverting $K_{ff} + \sigma^2 I$, which is $O(N^3)$.

Both methods introduce the joint prior over $(\mathbf{f}, \mathbf{u})$ and the conditional $p(\mathbf{f} \mid \mathbf{u})$ from identity A.2. The divergence begins in *how they handle the variational distribution*.

---

### B.2 Titsias (2009): Variational free energy

**Reference:** Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, AISTATS 2009.

**Key idea.** Titsias does not parameterise $q(\mathbf{u})$ directly. Instead, he works with the variational distribution over *all* function values:

$$
q(\mathbf{f}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})\, q(\mathbf{u})
$$

The crucial constraint is that the conditional $p(\mathbf{f} \mid \mathbf{u})$ is kept *exact* (from the GP prior). Only the marginal over $\mathbf{u}$ is free.

**Step 1: Write the evidence lower bound (ELBO).**

$$
\log p(\mathbf{y}) \geq \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})] - \mathrm{KL}(q(\mathbf{f}, \mathbf{u}) \| p(\mathbf{f}, \mathbf{u}))
$$

Since $q(\mathbf{f}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})$ and $p(\mathbf{f}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})p(\mathbf{u})$, the KL simplifies:

$$
\mathrm{KL}(q(\mathbf{f}, \mathbf{u}) \| p(\mathbf{f}, \mathbf{u})) = \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

This is because the $p(\mathbf{f} \mid \mathbf{u})$ factors cancel in the log-ratio. The ELBO becomes:

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

where $q(\mathbf{f}) = \int p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})\,d\mathbf{u}$.

**Step 2: Find the optimal $q(\mathbf{u})$ using variational calculus.**

This is where Titsias' approach uses a technique most people skip. We want to maximise $\mathcal{L}$ over *all* distributions $q(\mathbf{u})$, not just Gaussians. This requires functional optimisation.

For the Gaussian likelihood $p(\mathbf{y} \mid \mathbf{f}) = \mathcal{N}(\mathbf{y} \mid \mathbf{f}, \sigma^2 I)$, expand the expected log-likelihood. Using $q(\mathbf{f}) = \mathcal{N}(\tilde{\boldsymbol{\mu}}, \tilde{\Sigma})$ from the affine transform identity (with $\tilde{\boldsymbol{\mu}}$ and $\tilde{\Sigma}$ depending on $q(\mathbf{u})$):

$$
\mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\mathbb{E}_{q(\mathbf{f})}[\|\mathbf{y} - \mathbf{f}\|^2]
$$

By the trace trick (A.8):

$$
\mathbb{E}_{q(\mathbf{f})}[\|\mathbf{y} - \mathbf{f}\|^2] = \|\mathbf{y} - \tilde{\boldsymbol{\mu}}\|^2 + \mathrm{tr}(\tilde{\Sigma})
$$

Now, take the **functional derivative** of $\mathcal{L}$ with respect to $q(\mathbf{u})$ and set it to zero. The standard variational calculus result (see B.4 below) gives:

$$
\log q^*(\mathbf{u}) = \mathbb{E}_{p(\mathbf{f}|\mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})] + \log p(\mathbf{u}) + \text{const}
$$

For the Gaussian likelihood, $\mathbb{E}_{p(\mathbf{f}|\mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})]$ is a quadratic function of $\mathbf{u}$ (because $p(\mathbf{f} \mid \mathbf{u})$ is linear-Gaussian in $\mathbf{u}$, and the log-likelihood is quadratic in $\mathbf{f}$). Added to $\log p(\mathbf{u})$ (also quadratic in $\mathbf{u}$), the result is quadratic in $\mathbf{u}$. Therefore $q^*(\mathbf{u})$ is Gaussian — **this is not assumed, it is derived**.

**Step 3: Compute the optimal parameters and the collapsed bound.**

After computing $q^*(\mathbf{u})$ and substituting back, the optimised ELBO (the "collapsed bound") is:

$$
\mathcal{L}_{\text{Titsias}} = \log \mathcal{N}(\mathbf{y} \mid \mathbf{0},\; K_{fu}K_{uu}^{-1}K_{uf} + \sigma^2 I) - \frac{1}{2\sigma^2}\mathrm{tr}(K_{ff} - K_{fu}K_{uu}^{-1}K_{uf})
$$

The first term is the log marginal likelihood under the Nystr&ouml;m approximation $\tilde{K} = K_{fu}K_{uu}^{-1}K_{uf}$. The second term is the **trace penalty**:

$$
\frac{1}{2\sigma^2}\mathrm{tr}(K_{ff} - Q_{ff}), \quad \text{where } Q_{ff} = K_{fu}K_{uu}^{-1}K_{uf}
$$

This trace penalty is always non-negative (since $K_{ff} - Q_{ff}$ is positive semidefinite — it is the conditional covariance of $\mathbf{f} \mid \mathbf{u}$). It measures how much information the inducing approximation loses. If $Z$ contains all training points, $Q_{ff} = K_{ff}$ and the penalty vanishes — we recover the exact marginal likelihood.

**The optimal variational posterior is:**

$$
q^*(\mathbf{u}) = \mathcal{N}(\mathbf{u} \mid \boldsymbol{\mu}^*, \Sigma^*)
$$

$$
\Sigma^* = (K_{uu}^{-1} + \sigma^{-2}K_{uu}^{-1}K_{uf}K_{fu}K_{uu}^{-1})^{-1} = K_{uu}(K_{uu} + \sigma^{-2}K_{uf}K_{fu})^{-1}K_{uu}
$$

$$
\boldsymbol{\mu}^* = \sigma^{-2}\Sigma^* K_{uu}^{-1}K_{uf}\mathbf{y} = K_{uu}(K_{uu} + \sigma^{-2}K_{uf}K_{fu})^{-1}\sigma^{-2}K_{uf}\mathbf{y}
$$

---

### B.3 Hensman et al. (2013): Stochastic variational inference

**Reference:** Hensman, Fusi, and Lawrence, *Gaussian Processes for Big Data*, UAI 2013.

**Key idea.** Hensman keeps the same structural assumption $q(\mathbf{f}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})$, but instead of optimising $q(\mathbf{u})$ analytically, parameterises it as:

$$
q(\mathbf{u}) = \mathcal{N}(\mathbf{u} \mid \mathbf{m}, S)
$$

with **free parameters** $\mathbf{m} \in \mathbb{R}^M$ and $S \in \mathbb{R}^{M \times M}$ (SPD). These are optimised via gradient methods.

**Why?** Because the ELBO can then be decomposed over data points, enabling mini-batch training.

**Step 1: Write the ELBO (same starting point as Titsias).**

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

**Step 2: Decompose the expected log-likelihood over data points.**

Since $p(\mathbf{y} \mid \mathbf{f}) = \prod_{n=1}^N p(y_n \mid f_n)$:

$$
\mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] = \sum_{n=1}^N \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]
$$

Each term depends only on the marginal $q(f_n)$, which is a univariate Gaussian:

$$
q(f_n) = \mathcal{N}(f_n \mid \tilde{\mu}_n, \tilde{\sigma}_n^2)
$$

where (from the affine transform identity, applied to one data point):

$$
\tilde{\mu}_n = \mathbf{k}_n^\top K_{uu}^{-1}\mathbf{m}, \quad \tilde{\sigma}_n^2 = k_{nn} - \mathbf{k}_n^\top K_{uu}^{-1}(K_{uu} - S)K_{uu}^{-1}\mathbf{k}_n
$$

with $\mathbf{k}_n = k(Z, x_n) \in \mathbb{R}^M$ and $k_{nn} = k(x_n, x_n)$.

**This decomposition is the key.** Each term in the sum depends on $\mathbf{m}$ and $S$ only through $\tilde{\mu}_n$ and $\tilde{\sigma}_n^2$. So we can form unbiased estimates of the sum using mini-batches:

$$
\mathcal{L} \approx \frac{N}{|\mathcal{B}|}\sum_{n \in \mathcal{B}} \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)] - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

The KL term is $O(M^3)$ (closed form from A.7), and each expected log-likelihood term is $O(M^2)$. Total cost per step: $O(|\mathcal{B}|M^2 + M^3)$.

**Step 3: For Gaussian likelihood, compute the expected log-likelihood in closed form.**

$$
\mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)] = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\big[(y_n - \tilde{\mu}_n)^2 + \tilde{\sigma}_n^2\big]
$$

This follows directly from the trace trick (A.8) with $A = \sigma^{-2}$, $c = y_n$.

---

### B.4 Variational calculus: deriving the optimal $q^*(\mathbf{u})$

This is the functional optimisation step that Titsias uses and most presentations skip.

**Setup.** We want to maximise $\mathcal{L}[q] = \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})] - \mathrm{KL}(q(\mathbf{f}, \mathbf{u}) \| p(\mathbf{f}, \mathbf{u}))$ over all distributions $q(\mathbf{u})$, with $q(\mathbf{f}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})$.

**Rewrite the ELBO.** Using $q(\mathbf{f}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})q(\mathbf{u})$:

$$
\mathcal{L} = \int q(\mathbf{u}) \Big[\int p(\mathbf{f} \mid \mathbf{u}) \log p(\mathbf{y} \mid \mathbf{f})\,d\mathbf{f}\Big]\,d\mathbf{u} - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

Define the "expected log-likelihood as a function of $\mathbf{u}$":

$$
\ell(\mathbf{u}) := \mathbb{E}_{p(\mathbf{f}|\mathbf{u})}[\log p(\mathbf{y} \mid \mathbf{f})] = \int p(\mathbf{f} \mid \mathbf{u}) \log p(\mathbf{y} \mid \mathbf{f})\,d\mathbf{f}
$$

Then:

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{u})}[\ell(\mathbf{u})] - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))
$$

$$
= \int q(\mathbf{u})\,\ell(\mathbf{u})\,d\mathbf{u} - \int q(\mathbf{u}) \log\frac{q(\mathbf{u})}{p(\mathbf{u})}\,d\mathbf{u}
$$

$$
= \int q(\mathbf{u})\Big[\ell(\mathbf{u}) + \log p(\mathbf{u}) - \log q(\mathbf{u})\Big]\,d\mathbf{u}
$$

**Take the functional derivative.** To maximise over $q(\mathbf{u})$ subject to $\int q(\mathbf{u})\,d\mathbf{u} = 1$, introduce a Lagrange multiplier $\lambda$ and take the functional derivative:

$$
\frac{\delta}{\delta q(\mathbf{u})}\Big[\mathcal{L} - \lambda\big(\int q(\mathbf{u})\,d\mathbf{u} - 1\big)\Big] = 0
$$

$$
\ell(\mathbf{u}) + \log p(\mathbf{u}) - \log q(\mathbf{u}) - 1 - \lambda = 0
$$

Solving for $q$:

$$
\log q^*(\mathbf{u}) = \ell(\mathbf{u}) + \log p(\mathbf{u}) + \text{const}
$$

$$
q^*(\mathbf{u}) \propto p(\mathbf{u})\,\exp(\ell(\mathbf{u}))
$$

**Interpretation.** The optimal variational distribution is the prior $p(\mathbf{u})$ reweighted by $\exp(\ell(\mathbf{u}))$, which plays the role of a "pseudo-likelihood" — it summarises what the data says about $\mathbf{u}$ through the inducing-point approximation.

**For Gaussian likelihood:** $\ell(\mathbf{u})$ is quadratic in $\mathbf{u}$ (because $p(\mathbf{f} \mid \mathbf{u})$ is linear in $\mathbf{u}$ and $\log p(\mathbf{y} \mid \mathbf{f})$ is quadratic in $\mathbf{f}$). So $\exp(\ell(\mathbf{u}))$ is Gaussian in $\mathbf{u}$, and $p(\mathbf{u}) \exp(\ell(\mathbf{u}))$ is a product of Gaussians in $\mathbf{u}$ — which is Gaussian by identity A.3. This is why $q^*(\mathbf{u})$ turns out to be Gaussian *without assuming it*.

---

### B.5 The mathematical distinction: Titsias vs Hensman

| | Titsias (2009) | Hensman (2013) |
|---|---|---|
| **Variational family** | $q(\mathbf{u})$ is *any* distribution; optimal form derived via variational calculus | $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, S)$ assumed upfront |
| **Optimisation** | Analytical: $q^*(\mathbf{u})$ in closed form, then plug back to get collapsed bound | Numerical: optimise $\mathbf{m}, S$ via gradient ascent on ELBO |
| **Data requirement** | Full dataset needed to compute $\mathcal{L}_{\text{Titsias}}$ (the trace term $\mathrm{tr}(K_{ff} - Q_{ff})$ touches all $N$ points) | Mini-batches: ELBO decomposes over data points |
| **Cost per iteration** | $O(NM^2)$ — must see all data | $O(|\mathcal{B}|M^2 + M^3)$ — mini-batch |
| **Gaussian likelihood** | **At convergence, Hensman recovers Titsias' bound exactly** | Same bound, but reached iteratively |
| **Non-Gaussian likelihood** | $q^*(\mathbf{u})$ is no longer Gaussian — the variational calculus still gives a formula, but it is intractable | $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, S)$ remains tractable; expected log-likelihood approximated via quadrature |
| **Inducing point optimisation** | $Z$ optimised by gradient ascent on the collapsed bound | $Z$ optimised jointly with $\mathbf{m}, S$ via gradients on the ELBO |

The deepest difference: **Titsias proves the optimal form and collapses the bound; Hensman parameterises and optimises.** For Gaussian likelihoods they are equivalent at convergence. For non-Gaussian likelihoods, Hensman's approach is the practical one — Titsias' variational calculus gives a non-Gaussian $q^*(\mathbf{u})$ that cannot be computed.

---

## Appendix C: Essential GP Identities

Identities that, once internalised, make most GP derivations feel routine.

### C.1 GP marginal likelihood and its gradients

For $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K + \sigma^2 I)$ where $K$ depends on hyperparameters $\boldsymbol{\theta}$:

$$
\log p(\mathbf{y} \mid \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^\top(K + \sigma^2 I)^{-1}\mathbf{y} - \frac{1}{2}\log|K + \sigma^2 I| - \frac{N}{2}\log(2\pi)
$$

**Gradient with respect to any hyperparameter $\theta_j$:**

$$
\frac{\partial}{\partial \theta_j}\log p(\mathbf{y} \mid \boldsymbol{\theta}) = \frac{1}{2}\mathrm{tr}\!\Big[(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - (K + \sigma^2 I)^{-1})\frac{\partial K}{\partial \theta_j}\Big]
$$

where $\boldsymbol{\alpha} = (K + \sigma^2 I)^{-1}\mathbf{y}$.

**Proof.** This uses two matrix calculus identities:

$$
\frac{\partial}{\partial \theta}\log|A(\theta)| = \mathrm{tr}\!\Big(A^{-1}\frac{\partial A}{\partial \theta}\Big)
$$

$$
\frac{\partial}{\partial \theta}\mathbf{y}^\top A(\theta)^{-1}\mathbf{y} = -\mathbf{y}^\top A^{-1}\frac{\partial A}{\partial \theta}A^{-1}\mathbf{y}
$$

Applying both to $A = K + \sigma^2 I$:

$$
\frac{\partial \mathcal{L}}{\partial \theta_j} = \frac{1}{2}\mathbf{y}^\top A^{-1}\frac{\partial K}{\partial \theta_j}A^{-1}\mathbf{y} - \frac{1}{2}\mathrm{tr}\!\Big(A^{-1}\frac{\partial K}{\partial \theta_j}\Big)
$$

$$
= \frac{1}{2}\mathrm{tr}\!\Big[(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - A^{-1})\frac{\partial K}{\partial \theta_j}\Big]
$$

using $\mathbf{y}^\top A^{-1}\frac{\partial K}{\partial \theta_j}A^{-1}\mathbf{y} = \mathrm{tr}(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top \frac{\partial K}{\partial \theta_j})$. $\square$

**Insight.** The $\boldsymbol{\alpha}\boldsymbol{\alpha}^\top$ term is the "data fit" gradient; the $A^{-1}$ term is the "complexity" gradient. They trade off: increasing kernel flexibility improves fit but increases complexity.

---

### C.2 GP predictive distribution

Given training data $(\mathbf{X}, \mathbf{y})$ and test points $\mathbf{X}_*$, the predictive distribution is:

$$
\mathbf{f}_* \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)
$$

$$
\boldsymbol{\mu}_* = K_{*f}(K_{ff} + \sigma^2 I)^{-1}\mathbf{y} = K_{*f}\boldsymbol{\alpha}
$$

$$
\Sigma_* = K_{**} - K_{*f}(K_{ff} + \sigma^2 I)^{-1}K_{f*}
$$

**Proof.** Write the joint of $\mathbf{f}_*$ and $\mathbf{y}$:

$$
\begin{bmatrix}\mathbf{f}_* \\ \mathbf{y}\end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \begin{bmatrix}K_{**} & K_{*f} \\ K_{f*} & K_{ff} + \sigma^2 I\end{bmatrix}\right)
$$

The joint for $\mathbf{y}$ follows from the convolution identity (A.4): $\mathbf{y} = \mathbf{f} + \boldsymbol{\epsilon}$, so $\mathrm{Cov}(\mathbf{y}) = K_{ff} + \sigma^2 I$. The cross-covariance $\mathrm{Cov}(\mathbf{f}_*, \mathbf{y}) = \mathrm{Cov}(\mathbf{f}_*, \mathbf{f} + \boldsymbol{\epsilon}) = K_{*f}$ (since $\boldsymbol{\epsilon}$ is independent of $\mathbf{f}_*$).

Apply block conditioning (A.2) with $\mathbf{x} = \mathbf{f}_*$ and $\mathbf{y} = \mathbf{y}$. $\square$

---

### C.3 The FITC and VFE approximations via Woodbury

In the sparse setting, we replace $K_{ff}$ with the Nystr&ouml;m approximation $Q_{ff} = K_{fu}K_{uu}^{-1}K_{uf}$. The approximate marginal likelihood is:

$$
p(\mathbf{y}) \approx \mathcal{N}(\mathbf{y} \mid \mathbf{0},\; Q_{ff} + D)
$$

where $D$ differs between methods:

| Method | $D$ | Interpretation |
|---|---|---|
| **FITC** (Snelson & Ghahramani) | $\mathrm{diag}(K_{ff} - Q_{ff}) + \sigma^2 I$ | Diagonal correction per data point |
| **VFE** (Titsias) | $\sigma^2 I$ | No diagonal correction; trace penalty added |

Both can be inverted efficiently using Woodbury (A.9):

$$
(Q_{ff} + D)^{-1} = D^{-1} - D^{-1}K_{fu}(K_{uu} + K_{uf}D^{-1}K_{fu})^{-1}K_{uf}D^{-1}
$$

This is $O(NM^2)$ since $D$ is diagonal (or $\sigma^2 I$), so $D^{-1}$ is trivial.

---

### C.4 Derivatives of log-determinant and inverse

These are used everywhere in GP hyperparameter optimisation.

**Derivative of log-determinant:**

$$
\frac{\partial}{\partial \theta}\log|A(\theta)| = \mathrm{tr}\!\Big(A^{-1}\frac{\partial A}{\partial \theta}\Big)
$$

**Proof.** For invertible $A$, $|A + \epsilon B| = |A||I + \epsilon A^{-1}B|$. For small $\epsilon$:

$$
\log|I + \epsilon A^{-1}B| = \epsilon\,\mathrm{tr}(A^{-1}B) + O(\epsilon^2)
$$

using $\log|I + \epsilon M| = \epsilon\,\mathrm{tr}(M) + O(\epsilon^2)$ (from $|I + \epsilon M| = \exp(\mathrm{tr}(\log(I + \epsilon M))) \approx \exp(\epsilon\,\mathrm{tr}(M))$). $\square$

**Derivative of matrix inverse:**

$$
\frac{\partial}{\partial \theta}A(\theta)^{-1} = -A^{-1}\frac{\partial A}{\partial \theta}A^{-1}
$$

**Proof.** Differentiate $A A^{-1} = I$:

$$
\frac{\partial A}{\partial \theta}A^{-1} + A\frac{\partial A^{-1}}{\partial \theta} = 0
$$

$$
\frac{\partial A^{-1}}{\partial \theta} = -A^{-1}\frac{\partial A}{\partial \theta}A^{-1}
$$

$\square$

---

### C.5 The reparameterisation for $S$: Cholesky parameterisation

In Hensman's approach, $S$ must remain SPD during optimisation. The standard trick is to parameterise via the Cholesky factor:

$$
S = LL^\top, \quad L \text{ lower triangular with positive diagonal}
$$

Optimise over $L$ (unconstrained except for the positivity of diagonal entries, which is enforced by parameterising diagonal entries as $\exp(\ell_{ii})$). Then $S = LL^\top$ is automatically SPD.

**Gradient propagation.** If $\mathcal{L}$ depends on $S$, the gradient with respect to $L$ is:

$$
\frac{\partial \mathcal{L}}{\partial L} = \frac{\partial \mathcal{L}}{\partial S}L + \Big(\frac{\partial \mathcal{L}}{\partial S}\Big)^\top L
$$

where $\frac{\partial \mathcal{L}}{\partial S}$ is the gradient with respect to $S$ treated as a symmetric matrix. In practice, automatic differentiation handles this, but understanding the formula prevents debugging surprises.

---

### C.6 Summary: the complete sparse GP inference recipe

**Titsias (batch, Gaussian likelihood):**
1. Compute $Q_{ff} = K_{fu}K_{uu}^{-1}K_{uf}$
2. Evaluate collapsed bound: $\log \mathcal{N}(\mathbf{y} \mid \mathbf{0}, Q_{ff} + \sigma^2 I) - \frac{1}{2\sigma^2}\mathrm{tr}(K_{ff} - Q_{ff})$
3. Optimise $Z$ and kernel hyperparameters via gradients (C.1)
4. Recover $q^*(\mathbf{u})$ from the analytical formula (B.2)

**Hensman (mini-batch, any likelihood):**
1. Parameterise $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, LL^\top)$
2. For each mini-batch $\mathcal{B}$, compute: $\frac{N}{|\mathcal{B}|}\sum_{n \in \mathcal{B}} \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)] - \mathrm{KL}(q(\mathbf{u}) \| p(\mathbf{u}))$
3. Optimise $\mathbf{m}, L, Z$, and kernel hyperparameters via stochastic gradient ascent
4. For Gaussian likelihood at convergence, this equals Titsias' bound
