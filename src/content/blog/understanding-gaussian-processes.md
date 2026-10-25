---
title: "Gaussian processes, from first principles"
description: "From Bayesian linear regression to kernels, exact conditioning, the marginal likelihood and a working implementation. Every step derived, nothing skipped; every idea interactive."
date: 2024-06-07
updated: 2026-10-26
cover: /blog/thumbs/gp.jpg
tags: [gaussian-processes, kernels, bayesian, tutorial]
---

A Gaussian process (GP) is a prior over functions that stays tractable after conditioning on data. It returns a full predictive distribution at every input, so it reports *where* it is uncertain, not only *what* it predicts. Click to add data, drag a point, or switch the kernel:

<figure class="fig" data-lab>
  <canvas data-figure="lab" aria-label="Interactive Gaussian process. Click to add data, drag a point, choose a kernel."></canvas>
  <div class="controls">
    <div class="seg" role="group" aria-label="Kernel">
      <button type="button" data-kernel="rbf" class="on">RBF</button>
      <button type="button" data-kernel="matern52">Matérn 5/2</button>
      <button type="button" data-kernel="matern12">Matérn 1/2</button>
      <button type="button" data-kernel="periodic">Periodic</button>
      <button type="button" data-kernel="gibbs">Non-stationary</button>
    </div>
    <label class="slider">length-scale <input type="range" min="0.03" max="0.3" step="0.005" value="0.09" data-ell /></label>
    <button type="button" class="ghost" data-clear>Clear</button>
    <button type="button" class="ghost" data-reset>Reset</button>
  </div>
  <div class="insight" data-insight aria-live="polite"></div>
  <template data-for="rbf"><strong>The band</strong> is ±2 posterior standard deviations. It is narrow near data and returns to the prior away from it. Drag a point vertically and watch the band.</template>
  <template data-for="matern52"><strong>Matérn 5/2:</strong> sample paths are twice differentiable. Same data, a rougher belief than RBF.</template>
  <template data-for="matern12"><strong>Matérn 1/2:</strong> the Ornstein–Uhlenbeck process; continuous, nowhere differentiable.</template>
  <template data-for="periodic"><strong>Periodic:</strong> the prior encodes repetition, so the model extrapolates with confidence.</template>
  <template data-for="gibbs"><strong>Non-stationary:</strong> the length-scale grows from left to right; the function is rough on the left and smooth on the right.</template>
  <template data-for="drag"><strong>The band did not move.</strong> The posterior variance depends on the observation locations, not their values (Section 5).</template>
  <template data-for="prior"><strong>Prior:</strong> no data; the samples show what the kernel assumes before observing anything.</template>
</figure>

The post builds the GP from Bayesian linear regression, derives the posterior and the marginal likelihood, and ends with an implementation and its failure modes. Everything rests on eight facts about Gaussians, proved in Section 1; later derivations cite them by number. Proofs are folded and written as numbered steps, each with its justification, so that no step is left to the reader.

## Notation

Plain italic letters ($a$, $s$, $\sigma_n$) are scalars. Bold italic lower-case letters ($\boldsymbol{x}$, $\boldsymbol{y}$, $\boldsymbol{\mu}$) are column vectors; the dimension is stated where each is introduced. Bold upright capitals ($\mathbf{K}$, $\mathbf{X}$, $\boldsymbol{\Sigma}$, $\boldsymbol{\Phi}$) are matrices. $\mathbf{I}$ is the identity and $\mathbf{0}$ a zero vector or matrix.

| Symbol | Meaning |
|---|---|
| $\mathbf{X} \in \mathbb{R}^{N\times D}$, $\boldsymbol{y} \in \mathbb{R}^N$ | training inputs (one per row), noisy observations |
| $\boldsymbol{x}_i \in \mathbb{R}^D$, $y_i \in \mathbb{R}$ | the $i$-th input and observation |
| $\boldsymbol{x}_* \in \mathbb{R}^D$, $f_* = f(\boldsymbol{x}_*)$ | a test input, the latent value there |
| $\boldsymbol{f}_{\mathbf{X}} = f(\mathbf{X}) \in \mathbb{R}^N$ | latent function values at the training inputs |
| $\sigma_n^2$ | observation-noise variance |
| $k(\boldsymbol{x}, \boldsymbol{x}')$, $\mathbf{K} = k(\mathbf{X}, \mathbf{X})$ | kernel, Gram matrix with entries $K_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ |
| $\boldsymbol{k}_* = k(\mathbf{X}, \boldsymbol{x}_*) \in \mathbb{R}^N$, $k_{**} = k(\boldsymbol{x}_*, \boldsymbol{x}_*)$ | cross-covariance vector, prior variance at $\boldsymbol{x}_*$ |
| $\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$ | covariance of the observations |
| $\mu_*$, $\sigma_*^2$ | posterior mean and variance of $f_*$ |
| $\sigma_f^2$, $\ell$, $\boldsymbol{\theta}$ | kernel amplitude, length-scale, all hyperparameters |
| $\boldsymbol{\phi}(\boldsymbol{x}) \in \mathbb{R}^M$, $\boldsymbol{\Phi} \in \mathbb{R}^{N\times M}$ | feature map, design matrix with rows $\boldsymbol{\phi}(\boldsymbol{x}_i)^\top$ |
| $\boldsymbol{w} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_w)$ | weights and their prior |
| $\mathbf{P}$, $\boldsymbol{h}$ | a precision matrix and a linear coefficient (information form) |

All covariance matrices are symmetric and positive definite unless stated. $\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ is the Gaussian density in $\boldsymbol{a} \in \mathbb{R}^D$:

$$
\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = (2\pi)^{-D/2}\,\lvert\boldsymbol{\Sigma}\rvert^{-1/2}\exp\!\Big(-\tfrac12(\boldsymbol{a} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{a} - \boldsymbol{\mu})\Big).
$$

## 1. Gaussian facts you will need

Every result in this post is a sequence of the eight facts below.

### Reading a Gaussian off its exponent

Expand the exponent of $\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$, term by term:

$$
\begin{aligned}
-\tfrac12(\boldsymbol{a} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{a} - \boldsymbol{\mu})
&= -\tfrac12\big(\boldsymbol{a}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{a} - \boldsymbol{a}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} - \boldsymbol{\mu}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{a} + \boldsymbol{\mu}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}\big) \\
&= -\tfrac12\,\boldsymbol{a}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{a} + \boldsymbol{a}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} - \tfrac12\,\boldsymbol{\mu}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} .
\end{aligned}
$$

The first line multiplies out the product. The second uses $\boldsymbol{\mu}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{a} = \boldsymbol{a}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$, which holds because $\boldsymbol{\mu}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{a}$ is a scalar, a scalar equals its own transpose, and $(\boldsymbol{\Sigma}^{-1})^\top = \boldsymbol{\Sigma}^{-1}$. So the coefficient of the quadratic term is the **precision** $\boldsymbol{\Sigma}^{-1}$, and the coefficient of the linear term is $\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$. The converse is the workhorse of every Bayesian calculation below.

:::lemma[Completing the square]
Let $\mathbf{P}$ be symmetric positive definite and $\boldsymbol{h}$ a vector. For all $\boldsymbol{a}$,

$$
-\tfrac12\,\boldsymbol{a}^\top\mathbf{P}\boldsymbol{a} + \boldsymbol{a}^\top \boldsymbol{h}
= -\tfrac12\,(\boldsymbol{a} - \mathbf{P}^{-1}\boldsymbol{h})^\top\mathbf{P}\,(\boldsymbol{a} - \mathbf{P}^{-1}\boldsymbol{h}) + \tfrac12\,\boldsymbol{h}^\top\mathbf{P}^{-1}\boldsymbol{h}.
$$

Consequently, a density with $p(\boldsymbol{a}) \propto \exp\!\big(-\tfrac12\boldsymbol{a}^\top\mathbf{P}\boldsymbol{a} + \boldsymbol{a}^\top \boldsymbol{h}\big)$ is $\mathcal{N}(\mathbf{P}^{-1}\boldsymbol{h},\, \mathbf{P}^{-1})$.
:::

:::proof
**Step 1. Name the centre.** Let $\boldsymbol{a}_0 = \mathbf{P}^{-1}\boldsymbol{h}$. Then

$$
\mathbf{P}\boldsymbol{a}_0 = \boldsymbol{h}, \qquad \boldsymbol{a}_0^\top\mathbf{P} = (\mathbf{P}\boldsymbol{a}_0)^\top = \boldsymbol{h}^\top, \qquad \boldsymbol{a}_0^\top\mathbf{P}\boldsymbol{a}_0 = \boldsymbol{h}^\top\mathbf{P}^{-1}\boldsymbol{h} .
$$

The middle identity uses $\mathbf{P}^\top = \mathbf{P}$.

**Step 2. Expand the square on the right-hand side.**

$$
\begin{aligned}
-\tfrac12(\boldsymbol{a} - \boldsymbol{a}_0)^\top\mathbf{P}(\boldsymbol{a} - \boldsymbol{a}_0)
&= -\tfrac12\big(\boldsymbol{a}^\top\mathbf{P}\boldsymbol{a} - \boldsymbol{a}^\top\mathbf{P}\boldsymbol{a}_0 - \boldsymbol{a}_0^\top\mathbf{P}\boldsymbol{a} + \boldsymbol{a}_0^\top\mathbf{P}\boldsymbol{a}_0\big) \\
&= -\tfrac12\big(\boldsymbol{a}^\top\mathbf{P}\boldsymbol{a} - \boldsymbol{a}^\top \boldsymbol{h} - \boldsymbol{h}^\top \boldsymbol{a} + \boldsymbol{h}^\top\mathbf{P}^{-1}\boldsymbol{h}\big) \\
&= -\tfrac12\,\boldsymbol{a}^\top\mathbf{P}\boldsymbol{a} + \boldsymbol{a}^\top \boldsymbol{h} - \tfrac12\,\boldsymbol{h}^\top\mathbf{P}^{-1}\boldsymbol{h} .
\end{aligned}
$$

Reasons, line by line: (1) multiply out; (2) substitute the three identities of Step 1; (3) $\boldsymbol{h}^\top \boldsymbol{a} = \boldsymbol{a}^\top \boldsymbol{h}$, the same scalar.

**Step 3. Rearrange.** Add $\tfrac12\boldsymbol{h}^\top\mathbf{P}^{-1}\boldsymbol{h}$ to both sides of Step 2. This is the identity.

**Step 4. The consequence.** The term $\tfrac12\boldsymbol{h}^\top\mathbf{P}^{-1}\boldsymbol{h}$ does not contain $\boldsymbol{a}$, so it is absorbed into the proportionality constant:

$$
p(\boldsymbol{a}) \propto \exp\!\big(-\tfrac12(\boldsymbol{a} - \boldsymbol{a}_0)^\top\mathbf{P}(\boldsymbol{a} - \boldsymbol{a}_0)\big).
$$

**Step 5. Identify the density.** The right-hand side of Step 4 is the density of $\mathcal{N}(\boldsymbol{a}_0, \mathbf{P}^{-1})$ without its constant. Two densities that are proportional and both integrate to one are equal, so $p(\boldsymbol{a}) = \mathcal{N}(\boldsymbol{a} \mid \mathbf{P}^{-1}\boldsymbol{h}, \mathbf{P}^{-1})$.
:::

### Linear maps and sums: convolution

:::lemma[Affine transformation]
If $\boldsymbol{a} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\boldsymbol{b} = \mathbf{C}\boldsymbol{a} + \boldsymbol{d}$ for a fixed matrix $\mathbf{C}$ and vector $\boldsymbol{d}$, then $\boldsymbol{b} \sim \mathcal{N}(\mathbf{C}\boldsymbol{\mu} + \boldsymbol{d},\; \mathbf{C}\boldsymbol{\Sigma}\mathbf{C}^\top)$.
:::

:::proof
**Step 1. The mean.** Expectation is linear, and $\mathbf{C}$, $\boldsymbol{d}$ are constants:

$$
\mathbb{E}[\boldsymbol{b}] = \mathbb{E}[\mathbf{C}\boldsymbol{a} + \boldsymbol{d}] = \mathbf{C}\,\mathbb{E}[\boldsymbol{a}] + \boldsymbol{d} = \mathbf{C}\boldsymbol{\mu} + \boldsymbol{d} .
$$

**Step 2. The centred variable.** Subtract the mean:

$$
\boldsymbol{b} - \mathbb{E}[\boldsymbol{b}] = \mathbf{C}\boldsymbol{a} + \boldsymbol{d} - \mathbf{C}\boldsymbol{\mu} - \boldsymbol{d} = \mathbf{C}(\boldsymbol{a} - \boldsymbol{\mu}).
$$

**Step 3. The covariance.**

$$
\begin{aligned}
\operatorname{Cov}[\boldsymbol{b}] &= \mathbb{E}\big[(\boldsymbol{b} - \mathbb{E}[\boldsymbol{b}])(\boldsymbol{b} - \mathbb{E}[\boldsymbol{b}])^\top\big]
&& \text{definition} \\
&= \mathbb{E}\big[\mathbf{C}(\boldsymbol{a} - \boldsymbol{\mu})(\boldsymbol{a} - \boldsymbol{\mu})^\top\mathbf{C}^\top\big]
&& \text{Step 2, and } (\mathbf{C}\boldsymbol{u})^\top = \boldsymbol{u}^\top\mathbf{C}^\top \\
&= \mathbf{C}\,\mathbb{E}\big[(\boldsymbol{a} - \boldsymbol{\mu})(\boldsymbol{a} - \boldsymbol{\mu})^\top\big]\,\mathbf{C}^\top
&& \text{linearity} \\
&= \mathbf{C}\boldsymbol{\Sigma}\mathbf{C}^\top .
\end{aligned}
$$

**Step 4. The distribution is Gaussian.** A distribution is determined by its characteristic function $\varphi(\boldsymbol{t}) = \mathbb{E}[e^{i \boldsymbol{t}^\top(\cdot)}]$, and for $\boldsymbol{a} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ it is $\varphi_a(\boldsymbol{t}) = \exp\!\big(i \boldsymbol{t}^\top\boldsymbol{\mu} - \tfrac12\boldsymbol{t}^\top\boldsymbol{\Sigma}\boldsymbol{t}\big)$. Then

$$
\begin{aligned}
\varphi_b(\boldsymbol{t}) &= \mathbb{E}\big[e^{i \boldsymbol{t}^\top(\mathbf{C}\boldsymbol{a} + \boldsymbol{d})}\big]
= e^{i \boldsymbol{t}^\top \boldsymbol{d}}\;\mathbb{E}\big[e^{i (\mathbf{C}^\top \boldsymbol{t})^\top \boldsymbol{a}}\big]
&& \boldsymbol{t}^\top\mathbf{C}\boldsymbol{a} = (\mathbf{C}^\top \boldsymbol{t})^\top \boldsymbol{a} \\
&= e^{i \boldsymbol{t}^\top \boldsymbol{d}}\;\varphi_a(\mathbf{C}^\top \boldsymbol{t})
= \exp\!\big(i \boldsymbol{t}^\top(\mathbf{C}\boldsymbol{\mu} + \boldsymbol{d}) - \tfrac12\,\boldsymbol{t}^\top\mathbf{C}\boldsymbol{\Sigma}\mathbf{C}^\top \boldsymbol{t}\big),
\end{aligned}
$$

which is the characteristic function of $\mathcal{N}(\mathbf{C}\boldsymbol{\mu} + \boldsymbol{d}, \mathbf{C}\boldsymbol{\Sigma}\mathbf{C}^\top)$.
:::

:::lemma[Sum of independent Gaussians (convolution)]
If $\boldsymbol{a} \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $\boldsymbol{b} \sim \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ are independent, then $\boldsymbol{a} + \boldsymbol{b} \sim \mathcal{N}(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2,\; \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)$. Equivalently, the convolution of two Gaussian densities is a Gaussian density:

$$
\int \mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)\;\mathcal{N}(\boldsymbol{z} - \boldsymbol{a} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)\;da = \mathcal{N}(\boldsymbol{z} \mid \boldsymbol{\mu}_1 + \boldsymbol{\mu}_2,\; \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2).
$$
:::

:::proof
**Step 1. Stack the two variables.** Independent Gaussians are jointly Gaussian, with zero cross-covariance:

$$
\begin{bmatrix} \boldsymbol{a} \\ \boldsymbol{b} \end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix}\boldsymbol{\mu}_1\\\boldsymbol{\mu}_2\end{bmatrix},\; \begin{bmatrix}\boldsymbol{\Sigma}_1 & \mathbf{0}\\\mathbf{0} & \boldsymbol{\Sigma}_2\end{bmatrix}\right).
$$

**Step 2. Write the sum as a linear map.** With $\mathbf{C} = [\,\mathbf{I}\;\;\mathbf{I}\,]$,

$$
\boldsymbol{a} + \boldsymbol{b} = \begin{bmatrix}\mathbf{I} & \mathbf{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{a} \\ \boldsymbol{b} \end{bmatrix}.
$$

**Step 3. Apply Lemma 2 for the mean.** $\mathbf{C}\begin{bmatrix}\boldsymbol{\mu}_1\\\boldsymbol{\mu}_2\end{bmatrix} = \boldsymbol{\mu}_1 + \boldsymbol{\mu}_2$.

**Step 4. Apply Lemma 2 for the covariance.** Multiply the blocks:

$$
\begin{bmatrix}\mathbf{I} & \mathbf{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{\Sigma}_1 & \mathbf{0}\\\mathbf{0} & \boldsymbol{\Sigma}_2\end{bmatrix}\begin{bmatrix}\mathbf{I}\\\mathbf{I}\end{bmatrix}
= \begin{bmatrix}\boldsymbol{\Sigma}_1 & \boldsymbol{\Sigma}_2\end{bmatrix}\begin{bmatrix}\mathbf{I}\\\mathbf{I}\end{bmatrix}
= \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 .
$$

**Step 5. Why this is a convolution.** The event $\boldsymbol{a} + \boldsymbol{b} = \boldsymbol{z}$ happens for every pair $(\boldsymbol{a},\, \boldsymbol{b} = \boldsymbol{z} - \boldsymbol{a})$. Summing the probability of all such pairs, and using independence to factor the joint density,

$$
p_{a+b}(\boldsymbol{z}) = \int p_a(\boldsymbol{a})\,p_b(\boldsymbol{z} - \boldsymbol{a})\,da ,
$$

which is the convolution integral in the statement. Steps 3 and 4 evaluate it.
:::

:::example[The convolution integral, computed by hand in one dimension]
Take independent scalars $a \sim \mathcal{N}(\mu_1, s_1^2)$ and $b \sim \mathcal{N}(\mu_2, s_2^2)$. We compute $p(z) = \int \mathcal{N}(a \mid \mu_1, s_1^2)\,\mathcal{N}(z - a \mid \mu_2, s_2^2)\,da$ directly, without Lemma 2.

**Step 1. Write out the integrand.** Let $r = z - \mu_2$, so that $(z - a) - \mu_2 = r - a$:

$$
\mathcal{N}(a \mid \mu_1, s_1^2)\,\mathcal{N}(z - a \mid \mu_2, s_2^2) = \frac{1}{2\pi s_1 s_2}\exp\!\Big(-\frac12\Big[\frac{(a - \mu_1)^2}{s_1^2} + \frac{(r - a)^2}{s_2^2}\Big]\Big).
$$

**Step 2. Expand both squares.**

$$
\frac{(a - \mu_1)^2}{s_1^2} + \frac{(r - a)^2}{s_2^2}
= \frac{a^2 - 2a\mu_1 + \mu_1^2}{s_1^2} + \frac{a^2 - 2ar + r^2}{s_2^2} .
$$

**Step 3. Collect powers of $a$.**

$$
= p\,a^2 - 2q\,a + c,
\qquad\text{where}\qquad
\begin{aligned}
p &= \frac{1}{s_1^2} + \frac{1}{s_2^2} = \frac{s_1^2 + s_2^2}{s_1^2 s_2^2}, \\
q &= \frac{\mu_1}{s_1^2} + \frac{r}{s_2^2} = \frac{\mu_1 s_2^2 + r s_1^2}{s_1^2 s_2^2}, \\
c &= \frac{\mu_1^2}{s_1^2} + \frac{r^2}{s_2^2}.
\end{aligned}
$$

**Step 4. Complete the square in $a$** (Lemma 1 in one dimension):

$$
p\,a^2 - 2q\,a = p\Big(a - \frac{q}{p}\Big)^2 - \frac{q^2}{p} .
$$

**Step 5. Integrate over $a$.** Only the first term depends on $a$, and it is an unnormalised Gaussian with variance $1/p$:

$$
\int \exp\!\Big(-\frac{p}{2}\Big(a - \frac{q}{p}\Big)^2\Big)\,da = \sqrt{\frac{2\pi}{p}} .
$$

So $p(z) = \frac{1}{2\pi s_1 s_2}\sqrt{\frac{2\pi}{p}}\,\exp\!\big(\tfrac12(\tfrac{q^2}{p} - c)\big)$.

**Step 6. Put $\tfrac{q^2}{p} - c$ over one denominator.** The common denominator is $s_1^2 s_2^2(s_1^2 + s_2^2)$:

$$
\frac{q^2}{p} = \frac{(\mu_1 s_2^2 + r s_1^2)^2}{s_1^2 s_2^2(s_1^2 + s_2^2)},
\qquad
c = \frac{(\mu_1^2 s_2^2 + r^2 s_1^2)(s_1^2 + s_2^2)}{s_1^2 s_2^2(s_1^2 + s_2^2)} .
$$

**Step 7. Expand the two numerators.**

$$
\begin{aligned}
(\mu_1 s_2^2 + r s_1^2)^2 &= \mu_1^2 s_2^4 + 2\mu_1 r\,s_1^2 s_2^2 + r^2 s_1^4, \\
(\mu_1^2 s_2^2 + r^2 s_1^2)(s_1^2 + s_2^2) &= \mu_1^2 s_1^2 s_2^2 + \mu_1^2 s_2^4 + r^2 s_1^4 + r^2 s_1^2 s_2^2 .
\end{aligned}
$$

**Step 8. Subtract.** The terms $\mu_1^2 s_2^4$ and $r^2 s_1^4$ cancel:

$$
2\mu_1 r\,s_1^2 s_2^2 - \mu_1^2 s_1^2 s_2^2 - r^2 s_1^2 s_2^2 = -\,s_1^2 s_2^2\,(r - \mu_1)^2 .
$$

Dividing by the common denominator, $\tfrac12\big(\tfrac{q^2}{p} - c\big) = -\dfrac{(r - \mu_1)^2}{2(s_1^2 + s_2^2)} = -\dfrac{(z - \mu_1 - \mu_2)^2}{2(s_1^2 + s_2^2)}$.

**Step 9. The constant.** With $1/p = s_1^2 s_2^2/(s_1^2 + s_2^2)$,

$$
\frac{1}{2\pi s_1 s_2}\sqrt{\frac{2\pi s_1^2 s_2^2}{s_1^2 + s_2^2}} = \frac{1}{\sqrt{2\pi}\,\sqrt{s_1^2 + s_2^2}} = \frac{1}{\sqrt{2\pi(s_1^2 + s_2^2)}} .
$$

**Step 10. Combine Steps 8 and 9.** $p(z) = \mathcal{N}(z \mid \mu_1 + \mu_2,\; s_1^2 + s_2^2)$, as Lemma 3 says. **Convolution adds variances: it blurs.**
:::

### Multiplying densities: product

:::lemma[Product of two Gaussian densities in the same variable]
For any $\boldsymbol{a}$,

$$
\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)\;\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) = Z\;\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c),
$$

with

$$
\begin{aligned}
\boldsymbol{\Sigma}_c &= \big(\boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1}\big)^{-1}, \\
\boldsymbol{\mu}_c &= \boldsymbol{\Sigma}_c\big(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_2^{-1}\boldsymbol{\mu}_2\big), \\
Z &= \mathcal{N}(\boldsymbol{\mu}_1 \mid \boldsymbol{\mu}_2,\; \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2).
\end{aligned}
$$
:::

:::proof
**Step 1. Add the exponents.** A product of exponentials is the exponential of the sum. Expanding each exponent as in "Reading a Gaussian off its exponent",

$$
\begin{aligned}
&\log\big[\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)\,\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)\big] \\
&= \big(-\tfrac12\boldsymbol{a}^\top\boldsymbol{\Sigma}_1^{-1}\boldsymbol{a} + \boldsymbol{a}^\top\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1\big) \\
&\quad + \big(-\tfrac12\boldsymbol{a}^\top\boldsymbol{\Sigma}_2^{-1}\boldsymbol{a} + \boldsymbol{a}^\top\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\mu}_2\big) + \text{const} \\
&= -\tfrac12\,\boldsymbol{a}^\top\big(\boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1}\big)\boldsymbol{a} + \boldsymbol{a}^\top\big(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_2^{-1}\boldsymbol{\mu}_2\big) + \text{const},
\end{aligned}
$$

where "const" collects every term without $\boldsymbol{a}$.

**Step 2. Read off the information form.** Step 1 is $-\tfrac12\boldsymbol{a}^\top\mathbf{P}\boldsymbol{a} + \boldsymbol{a}^\top \boldsymbol{h} + \text{const}$ with

$$
\mathbf{P} = \boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1}, \qquad \boldsymbol{h} = \boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_2^{-1}\boldsymbol{\mu}_2 .
$$

**Step 3. Complete the square** (Lemma 1). As a function of $\boldsymbol{a}$, the product is proportional to $\mathcal{N}(\boldsymbol{a} \mid \mathbf{P}^{-1}\boldsymbol{h}, \mathbf{P}^{-1}) = \mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$. Call the constant of proportionality $Z$.

**Step 4. Find $Z$ by integrating both sides over $\boldsymbol{a}$.** The right-hand side integrates to $Z$, since a density integrates to one:

$$
Z = \int \mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)\;\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)\;da .
$$

**Step 5. Swap the arguments of the second factor.** The Gaussian density depends on its two arguments only through $(\boldsymbol{a} - \boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{a} - \boldsymbol{\mu}_2)$, which is unchanged when $\boldsymbol{a}$ and $\boldsymbol{\mu}_2$ swap. So $\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) = \mathcal{N}(\boldsymbol{\mu}_2 \mid \boldsymbol{a}, \boldsymbol{\Sigma}_2)$ and

$$
Z = \int \mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)\;\mathcal{N}(\boldsymbol{\mu}_2 \mid \boldsymbol{a}, \boldsymbol{\Sigma}_2)\;da .
$$

**Step 6. Recognise a convolution.** Consider the two-stage experiment $\boldsymbol{a} \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$, then $\boldsymbol{v} = \boldsymbol{a} + \boldsymbol{u}$ with independent $\boldsymbol{u} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_2)$. Given $\boldsymbol{a}$, $\boldsymbol{v} \sim \mathcal{N}(\boldsymbol{a}, \boldsymbol{\Sigma}_2)$, so the integral in Step 5 is $\int p(\boldsymbol{a})\,p(\boldsymbol{v} \mid \boldsymbol{a})\,da = p(\boldsymbol{v})$ evaluated at $\boldsymbol{v} = \boldsymbol{\mu}_2$.

**Step 7. Apply Lemma 3.** $\boldsymbol{v} = \boldsymbol{a} + \boldsymbol{u} \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)$. Therefore

$$
Z = \mathcal{N}(\boldsymbol{\mu}_2 \mid \boldsymbol{\mu}_1,\; \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2) = \mathcal{N}(\boldsymbol{\mu}_1 \mid \boldsymbol{\mu}_2,\; \boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2),
$$

using the symmetry of Step 5 once more.
:::

:::note[Product, convolution and mixture are different operations]
- **Product** of two densities *in the same variable* (Lemma 4): precisions add, so the result is *narrower* than either factor. This is Bayes' rule combining two sources of information. **It sharpens.**
- **Convolution**, the density of a *sum of independent variables* (Lemma 3): covariances add, so the result is *wider*. This is adding noise. **It blurs.**
- **Mixture** $\pi_1\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) + \pi_2\mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$, a weighted sum of densities: **not Gaussian** in general (it can have two modes).

Steps 4 to 7 of Lemma 4 link the first two: the normaliser of a product is a convolution evaluated at a point.
:::

### Conditioning

:::lemma[Gaussian conditioning]
If

$$
\begin{bmatrix} \boldsymbol{a} \\ \boldsymbol{b} \end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix}\boldsymbol{\mu}_a\\\boldsymbol{\mu}_b\end{bmatrix}, \begin{bmatrix}\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab}\\\boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}\end{bmatrix}\right)
$$

with $\boldsymbol{\Sigma}_{aa}$ invertible, then

$$
\boldsymbol{b} \mid \boldsymbol{a} \sim \mathcal{N}\big(\boldsymbol{\mu}_b + \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}(\boldsymbol{a} - \boldsymbol{\mu}_a),\;\; \boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}\big).
$$
:::

:::proof
**Step 1. Remove from $\boldsymbol{b}$ the part predictable from $\boldsymbol{a}$.** Define

$$
\mathbf{G} = \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}, \qquad \boldsymbol{r} = \boldsymbol{b} - \mathbf{G}\boldsymbol{a} .
$$

**Step 2. The residual is uncorrelated with $\boldsymbol{a}$.** Covariance is linear in each argument:

$$
\operatorname{Cov}[\boldsymbol{r}, \boldsymbol{a}] = \operatorname{Cov}[\boldsymbol{b}, \boldsymbol{a}] - \mathbf{G}\operatorname{Cov}[\boldsymbol{a}, \boldsymbol{a}] = \boldsymbol{\Sigma}_{ba} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{aa} = \mathbf{0} .
$$

**Step 3. Uncorrelated implies independent here.** The pair $(\boldsymbol{r}, \boldsymbol{a})$ is a linear map of $(\boldsymbol{a}, \boldsymbol{b})$:

$$
\begin{bmatrix} \boldsymbol{r} \\ \boldsymbol{a} \end{bmatrix} = \begin{bmatrix} -\mathbf{G} & \mathbf{I} \\ \mathbf{I} & \mathbf{0} \end{bmatrix}\begin{bmatrix} \boldsymbol{a} \\ \boldsymbol{b} \end{bmatrix},
$$

so it is jointly Gaussian (Lemma 2). By Step 2 its covariance is block diagonal, so its density factorises into a function of $\boldsymbol{r}$ times a function of $\boldsymbol{a}$. Hence $\boldsymbol{r}$ and $\boldsymbol{a}$ are independent, and conditioning on $\boldsymbol{a}$ does not change the distribution of $\boldsymbol{r}$.

**Step 4. The mean of $\boldsymbol{r}$.** $\mathbb{E}[\boldsymbol{r}] = \boldsymbol{\mu}_b - \mathbf{G}\boldsymbol{\mu}_a$.

**Step 5. The covariance of $\boldsymbol{r}$.** Expand $\operatorname{Cov}[\boldsymbol{b} - \mathbf{G}\boldsymbol{a}]$ by bilinearity, then substitute $\mathbf{G}$:

$$
\begin{aligned}
\operatorname{Cov}[\boldsymbol{r}]
&= \boldsymbol{\Sigma}_{bb} - \mathbf{G}\boldsymbol{\Sigma}_{ab} - \boldsymbol{\Sigma}_{ba}\mathbf{G}^\top + \mathbf{G}\boldsymbol{\Sigma}_{aa}\mathbf{G}^\top
&& \text{bilinearity} \\
&= \boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab} + \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{aa}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}
&& \mathbf{G}^\top = \boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab} \\
&= \boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}
&& \boldsymbol{\Sigma}_{aa}\boldsymbol{\Sigma}_{aa}^{-1} = \mathbf{I} .
\end{aligned}
$$

Here $\mathbf{G}^\top = (\boldsymbol{\Sigma}_{aa}^{-1})^\top\boldsymbol{\Sigma}_{ba}^\top = \boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}$, because a covariance matrix is symmetric, which makes $\boldsymbol{\Sigma}_{aa}^{-1}$ symmetric and $\boldsymbol{\Sigma}_{ba}^\top = \boldsymbol{\Sigma}_{ab}$.

**Step 6. Put $\boldsymbol{b}$ back.** $\boldsymbol{b} = \boldsymbol{r} + \mathbf{G}\boldsymbol{a}$. Given $\boldsymbol{a}$, the term $\mathbf{G}\boldsymbol{a}$ is a constant, and adding a constant to a Gaussian shifts its mean only. So $\boldsymbol{b} \mid \boldsymbol{a}$ is Gaussian with the covariance of Step 5 and mean

$$
\boldsymbol{\mu}_b - \mathbf{G}\boldsymbol{\mu}_a + \mathbf{G}\boldsymbol{a} = \boldsymbol{\mu}_b + \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}(\boldsymbol{a} - \boldsymbol{\mu}_a).
$$
:::

### Matrix identities

:::lemma[Push-through and Woodbury]
Let $\boldsymbol{\Sigma}_w$ be $M\times M$ and positive definite, $\boldsymbol{\Phi}$ be $N\times M$, $\sigma_n^2 > 0$, and write

$$
\mathbf{A} = \sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \boldsymbol{\Sigma}_w^{-1} \;\;(M\times M),
\qquad
\mathbf{G} = \boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top + \sigma_n^2\mathbf{I} \;\;(N\times N).
$$

Then

$$
\begin{aligned}
\text{(push-through)}\quad & \sigma_n^{-2}\,\mathbf{A}^{-1}\boldsymbol{\Phi}^\top = \boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}, \\
\text{(Woodbury)}\quad & \mathbf{A}^{-1} = \boldsymbol{\Sigma}_w - \boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}_w .
\end{aligned}
$$
:::

:::proof
*Push-through.*

**Step 1. Multiply $\boldsymbol{\Phi}^\top$ by $\mathbf{G}$ and expand.**

$$
\sigma_n^{-2}\boldsymbol{\Phi}^\top\mathbf{G} = \sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top + \boldsymbol{\Phi}^\top .
$$

**Step 2. Insert an identity.** Write $\boldsymbol{\Phi}^\top = \boldsymbol{\Sigma}_w^{-1}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top$ in the second term, then factor $\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top$ out on the right:

$$
\sigma_n^{-2}\boldsymbol{\Phi}^\top\mathbf{G} = \big(\sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \boldsymbol{\Sigma}_w^{-1}\big)\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top = \mathbf{A}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top .
$$

**Step 3. Multiply by the inverses.** Both $\mathbf{A}$ and $\mathbf{G}$ are positive definite, hence invertible. Multiply Step 2 by $\mathbf{A}^{-1}$ on the left and $\mathbf{G}^{-1}$ on the right:

$$
\sigma_n^{-2}\mathbf{A}^{-1}\boldsymbol{\Phi}^\top = \boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1} .
$$

*Woodbury.*

**Step 4. Name the candidate.** Let $\mathbf{R} = \boldsymbol{\Sigma}_w - \boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}_w$. For square matrices, $\mathbf{A}\mathbf{R} = \mathbf{I}$ implies $\mathbf{R} = \mathbf{A}^{-1}$, so it suffices to compute $\mathbf{A}\mathbf{R}$.

**Step 5. Multiply out.**

$$
\begin{aligned}
\mathbf{A}\mathbf{R}
&= \mathbf{A}\boldsymbol{\Sigma}_w - \mathbf{A}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}_w \\
&= \big(\sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}_w + \mathbf{I}\big) - \mathbf{A}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}_w
&& \text{definition of } \mathbf{A} \\
&= \sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}_w + \mathbf{I} - \sigma_n^{-2}\boldsymbol{\Phi}^\top\mathbf{G}\mathbf{G}^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}_w
&& \text{Step 2: } \mathbf{A}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top = \sigma_n^{-2}\boldsymbol{\Phi}^\top\mathbf{G} \\
&= \sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}_w + \mathbf{I} - \sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}_w
&& \mathbf{G}\mathbf{G}^{-1} = \mathbf{I} \\
&= \mathbf{I} .
\end{aligned}
$$
:::

:::lemma[Derivatives of the inverse and the log-determinant]
For a symmetric positive definite $\mathbf{K}(\theta)$ that depends smoothly on a scalar $\theta$,

$$
\frac{\partial\mathbf{K}^{-1}}{\partial\theta} = -\mathbf{K}^{-1}\frac{\partial\mathbf{K}}{\partial\theta}\mathbf{K}^{-1},
\qquad
\frac{\partial}{\partial\theta}\log\lvert\mathbf{K}\rvert = \operatorname{tr}\!\Big(\mathbf{K}^{-1}\frac{\partial\mathbf{K}}{\partial\theta}\Big).
$$
:::

:::proof
Write $\partial$ for $\partial/\partial\theta$.

*Inverse.*

**Step 1. Differentiate the identity $\mathbf{K}\mathbf{K}^{-1} = \mathbf{I}$** with the product rule; the right-hand side is constant:

$$
(\partial\mathbf{K})\,\mathbf{K}^{-1} + \mathbf{K}\,(\partial\mathbf{K}^{-1}) = \mathbf{0} .
$$

**Step 2. Solve for $\partial\mathbf{K}^{-1}$.** Move the first term across and multiply on the left by $\mathbf{K}^{-1}$:

$$
\partial\mathbf{K}^{-1} = -\mathbf{K}^{-1}(\partial\mathbf{K})\,\mathbf{K}^{-1} .
$$

*Log-determinant.*

**Step 3. Perturb and factor.** For a small $\epsilon$ and $\mathbf{E} = \partial\mathbf{K}$, $\;\mathbf{K}(\theta + \epsilon) = \mathbf{K} + \epsilon\mathbf{E} + O(\epsilon^2) = \mathbf{K}\big(\mathbf{I} + \epsilon\mathbf{K}^{-1}\mathbf{E}\big) + O(\epsilon^2)$.

**Step 4. Take the log-determinant.** The determinant of a product is the product of determinants:

$$
\log\lvert\mathbf{K} + \epsilon\mathbf{E}\rvert = \log\lvert\mathbf{K}\rvert + \log\lvert\mathbf{I} + \epsilon\mathbf{K}^{-1}\mathbf{E}\rvert .
$$

**Step 5. Use eigenvalues.** Let $\lambda_1, \dots, \lambda_N$ be the eigenvalues of $\mathbf{K}^{-1}\mathbf{E}$ (they are real: the matrix is similar to the symmetric $\mathbf{K}^{-1/2}\mathbf{E}\mathbf{K}^{-1/2}$). The eigenvalues of $\mathbf{I} + \epsilon\mathbf{K}^{-1}\mathbf{E}$ are $1 + \epsilon\lambda_i$, and a determinant is the product of the eigenvalues:

$$
\log\lvert\mathbf{I} + \epsilon\mathbf{K}^{-1}\mathbf{E}\rvert = \sum_{i=1}^N \log(1 + \epsilon\lambda_i) = \epsilon\sum_{i=1}^N \lambda_i + O(\epsilon^2) = \epsilon\operatorname{tr}\big(\mathbf{K}^{-1}\mathbf{E}\big) + O(\epsilon^2),
$$

using $\log(1 + u) = u + O(u^2)$ and that the trace is the sum of the eigenvalues.

**Step 6. Take the limit.** Subtract $\log\lvert\mathbf{K}\rvert$, divide by $\epsilon$ and let $\epsilon \to 0$: $\;\partial\log\lvert\mathbf{K}\rvert = \operatorname{tr}(\mathbf{K}^{-1}\partial\mathbf{K})$.
:::

### Putting it together: the linear-Gaussian model

Lemma 4 multiplies two densities *in the same variable*. The product that appears most often in practice has two different variables: a prior over a hidden $\boldsymbol{x}$, times a likelihood for an observation $\boldsymbol{y}$ whose mean depends linearly on $\boldsymbol{x}$. Bayesian linear regression, the GP and every step of a Kalman filter are this product.

:::lemma[Linear-Gaussian model]
Let $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)$ and $\boldsymbol{y} \mid \boldsymbol{x} \sim \mathcal{N}(\mathbf{C}\boldsymbol{x} + \boldsymbol{d},\, \mathbf{S})$. Then

$$
\begin{aligned}
&\mathcal{N}(\boldsymbol{y} \mid \mathbf{C}\boldsymbol{x} + \boldsymbol{d}, \mathbf{S})\;\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x) \\
&\quad= \underbrace{\mathcal{N}(\boldsymbol{y} \mid \mathbf{C}\boldsymbol{\mu}_x + \boldsymbol{d},\; \mathbf{C}\boldsymbol{\Sigma}_x\mathbf{C}^\top + \mathbf{S})}_{p(\boldsymbol{y})}\;
\underbrace{\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_{x \mid y}, \boldsymbol{\Sigma}_{x \mid y})}_{p(\boldsymbol{x} \mid \boldsymbol{y})},
\end{aligned}
$$

where the posterior can be written in two equivalent forms:

$$
\begin{aligned}
\text{(moment form)}\quad
\boldsymbol{\Sigma}_{x \mid y} &= \boldsymbol{\Sigma}_x - \boldsymbol{\Sigma}_x\mathbf{C}^\top(\mathbf{C}\boldsymbol{\Sigma}_x\mathbf{C}^\top + \mathbf{S})^{-1}\mathbf{C}\boldsymbol{\Sigma}_x, \\
\boldsymbol{\mu}_{x \mid y} &= \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_x\mathbf{C}^\top(\mathbf{C}\boldsymbol{\Sigma}_x\mathbf{C}^\top + \mathbf{S})^{-1}(\boldsymbol{y} - \mathbf{C}\boldsymbol{\mu}_x - \boldsymbol{d}); \\[4pt]
\text{(information form)}\quad
\boldsymbol{\Sigma}_{x \mid y} &= \big(\boldsymbol{\Sigma}_x^{-1} + \mathbf{C}^\top\mathbf{S}^{-1}\mathbf{C}\big)^{-1}, \\
\boldsymbol{\mu}_{x \mid y} &= \boldsymbol{\Sigma}_{x \mid y}\big(\boldsymbol{\Sigma}_x^{-1}\boldsymbol{\mu}_x + \mathbf{C}^\top\mathbf{S}^{-1}(\boldsymbol{y} - \boldsymbol{d})\big).
\end{aligned}
$$
:::

:::proof
*Moment form: build the joint, then factor it the other way.*

**Step 1. Write the model as a sum.** Draw $\boldsymbol{x}$, then add independent noise:

$$
\boldsymbol{y} = \mathbf{C}\boldsymbol{x} + \boldsymbol{d} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{S}) \text{ independent of } \boldsymbol{x}.
$$

**Step 2. Stack.** The pair $(\boldsymbol{x}, \boldsymbol{y})$ is a linear map of the independent pair $(\boldsymbol{x}, \boldsymbol{\varepsilon})$, plus a constant:

$$
\begin{bmatrix} \boldsymbol{x} \\ \boldsymbol{y} \end{bmatrix}
= \underbrace{\begin{bmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{C} & \mathbf{I} \end{bmatrix}}_{\mathbf{M}}
\begin{bmatrix} \boldsymbol{x} \\ \boldsymbol{\varepsilon} \end{bmatrix}
+ \begin{bmatrix} \mathbf{0} \\ \boldsymbol{d} \end{bmatrix},
\qquad
\begin{bmatrix} \boldsymbol{x} \\ \boldsymbol{\varepsilon} \end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix} \boldsymbol{\mu}_x \\ \mathbf{0} \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_x & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}\right).
$$

**Step 3. The joint mean** (Lemma 2): $\mathbf{M}\begin{bmatrix} \boldsymbol{\mu}_x \\ \mathbf{0} \end{bmatrix} + \begin{bmatrix} \mathbf{0} \\ \boldsymbol{d} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\mu}_x \\ \mathbf{C}\boldsymbol{\mu}_x + \boldsymbol{d} \end{bmatrix}$.

**Step 4. The joint covariance** (Lemma 2), one block product at a time:

$$
\begin{aligned}
\mathbf{M}\begin{bmatrix} \boldsymbol{\Sigma}_x & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}\mathbf{M}^\top
&= \begin{bmatrix} \boldsymbol{\Sigma}_x & \mathbf{0} \\ \mathbf{C}\boldsymbol{\Sigma}_x & \mathbf{S} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \mathbf{C}^\top \\ \mathbf{0} & \mathbf{I} \end{bmatrix} \\
&= \begin{bmatrix} \boldsymbol{\Sigma}_x & \boldsymbol{\Sigma}_x\mathbf{C}^\top \\ \mathbf{C}\boldsymbol{\Sigma}_x & \mathbf{C}\boldsymbol{\Sigma}_x\mathbf{C}^\top + \mathbf{S} \end{bmatrix}.
\end{aligned}
$$

**Step 5. The marginal of $\boldsymbol{y}$.** Selecting a block of a Gaussian vector is the linear map $[\,\mathbf{0}\;\;\mathbf{I}\,]$, so by Lemma 2 the marginal keeps the bottom block of the mean and the bottom-right block of the covariance: $p(\boldsymbol{y}) = \mathcal{N}(\mathbf{C}\boldsymbol{\mu}_x + \boldsymbol{d},\; \mathbf{C}\boldsymbol{\Sigma}_x\mathbf{C}^\top + \mathbf{S})$.

**Step 6. The conditional of $\boldsymbol{x}$.** Apply Lemma 5 with $\boldsymbol{a} = \boldsymbol{y}$ and $\boldsymbol{b} = \boldsymbol{x}$, reading the blocks from Step 4:

$$
\boldsymbol{\Sigma}_{aa} = \mathbf{C}\boldsymbol{\Sigma}_x\mathbf{C}^\top + \mathbf{S}, \qquad
\boldsymbol{\Sigma}_{ba} = \boldsymbol{\Sigma}_x\mathbf{C}^\top, \qquad
\boldsymbol{\Sigma}_{ab} = \mathbf{C}\boldsymbol{\Sigma}_x, \qquad
\boldsymbol{\Sigma}_{bb} = \boldsymbol{\Sigma}_x .
$$

This gives the moment form.

**Step 7. Joint equals marginal times conditional.** $p(\boldsymbol{y} \mid \boldsymbol{x})\,p(\boldsymbol{x}) = p(\boldsymbol{x}, \boldsymbol{y}) = p(\boldsymbol{y})\,p(\boldsymbol{x} \mid \boldsymbol{y})$, which is the identity in the statement.

*Information form: treat the likelihood as a function of $\boldsymbol{x}$.*

**Step 8. Expand the likelihood exponent in $\boldsymbol{x}$.** Write $\boldsymbol{e} = \boldsymbol{y} - \boldsymbol{d}$. Exactly as in Step 3 of Proposition 1 below,

$$
-\tfrac12(\boldsymbol{e} - \mathbf{C}\boldsymbol{x})^\top\mathbf{S}^{-1}(\boldsymbol{e} - \mathbf{C}\boldsymbol{x})
= -\tfrac12\,\boldsymbol{x}^\top\mathbf{C}^\top\mathbf{S}^{-1}\mathbf{C}\,\boldsymbol{x} + \boldsymbol{x}^\top\mathbf{C}^\top\mathbf{S}^{-1}\boldsymbol{e} + \text{const}.
$$

As a function of $\boldsymbol{x}$, the likelihood is a Gaussian in information form with precision $\mathbf{C}^\top\mathbf{S}^{-1}\mathbf{C}$ and linear coefficient $\mathbf{C}^\top\mathbf{S}^{-1}\boldsymbol{e}$.

**Step 9. Expand the prior exponent** ("Reading a Gaussian off its exponent"): $-\tfrac12\,\boldsymbol{x}^\top\boldsymbol{\Sigma}_x^{-1}\boldsymbol{x} + \boldsymbol{x}^\top\boldsymbol{\Sigma}_x^{-1}\boldsymbol{\mu}_x + \text{const}$.

**Step 10. Add and complete the square.** The log of the product is $-\tfrac12\boldsymbol{x}^\top\mathbf{P}\boldsymbol{x} + \boldsymbol{x}^\top\boldsymbol{h} + \text{const}$ with $\mathbf{P} = \boldsymbol{\Sigma}_x^{-1} + \mathbf{C}^\top\mathbf{S}^{-1}\mathbf{C}$ and $\boldsymbol{h} = \boldsymbol{\Sigma}_x^{-1}\boldsymbol{\mu}_x + \mathbf{C}^\top\mathbf{S}^{-1}\boldsymbol{e}$. Lemma 1 gives the information form.

**Step 11. The two forms agree.** Steps 6 and 10 describe the same density $p(\boldsymbol{x} \mid \boldsymbol{y})$, so their means and covariances are equal. (Equating the two covariances is the Woodbury identity in general form; Lemma 6 is the case $\boldsymbol{\Sigma}_x = \boldsymbol{\Sigma}_w$, $\mathbf{C} = \boldsymbol{\Phi}$, $\mathbf{S} = \sigma_n^2\mathbf{I}$.)
:::

Two special cases are worth seeing at once. With $\mathbf{C} = \mathbf{I}$ and $\boldsymbol{d} = \mathbf{0}$, Lemma 8 *is* Lemma 4: the same-variable product is the case where the observation is a noisy copy of the hidden variable. With $\mathbf{C} = \boldsymbol{\Phi}$, it is Bayesian linear regression (Section 2).

:::remark[Two coordinate systems for one Gaussian]
A Gaussian can be stored in **moment form** $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ or in **information form** $(\boldsymbol{h}, \mathbf{P}) = (\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}, \boldsymbol{\Sigma}^{-1})$. Each makes a different half of the toolkit trivial:

- *Moment form.* Marginalising is reading off blocks (Step 5 above). Sums of independent variables add covariances (Lemma 3). Conditioning needs an inverse (the Schur complement of Lemma 5).
- *Information form.* Products in the same variable add $\boldsymbol{h}$ and $\mathbf{P}$ (Lemma 4). Conditioning is reading off blocks: fix $\boldsymbol{a}$ in the joint exponent $-\tfrac12[\boldsymbol{a}; \boldsymbol{b}]^\top\mathbf{P}[\boldsymbol{a}; \boldsymbol{b}]$, and the term quadratic in $\boldsymbol{b}$ is $-\tfrac12\boldsymbol{b}^\top\mathbf{P}_{bb}\boldsymbol{b}$, so the conditional precision is the block $\mathbf{P}_{bb}$. Marginalising needs the inverse.

The same split explains why zeros mean different things: a zero in $\boldsymbol{\Sigma}$ is marginal independence, a zero in $\mathbf{P}$ is conditional independence given all other variables. Sparse precisions are what make Gaussian Markov random fields, and information filters for large spatial fields, cheap.

The characteristic function gives a third view. For independent $\boldsymbol{a}$ and $\boldsymbol{b}$, $\varphi_{\boldsymbol{a}+\boldsymbol{b}}(\boldsymbol{t}) = \mathbb{E}[e^{i\boldsymbol{t}^\top\boldsymbol{a}}]\,\mathbb{E}[e^{i\boldsymbol{t}^\top\boldsymbol{b}}] = \varphi_{\boldsymbol{a}}(\boldsymbol{t})\,\varphi_{\boldsymbol{b}}(\boldsymbol{t})$, and multiplying two Gaussian characteristic functions adds the exponents: $\exp\!\big(i\boldsymbol{t}^\top(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2) - \tfrac12\boldsymbol{t}^\top(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)\boldsymbol{t}\big)$. That is Lemma 3 in one line. Convolution of densities is multiplication of their Fourier transforms, and a Gaussian keeps its shape in both domains, which is why both operations stay Gaussian.
:::

:::note[When to think in precision, when in covariance]
**Covariance answers a global question; precision answers a local one.** $\Sigma_{ij}$ is how $x_i$ and $x_j$ co-vary when every other variable is *unknown* (marginalised out). $P_{ij}$ describes how $x_i$ and $x_j$ relate when every other variable is *known* (held fixed). Concretely, $1/P_{ii}$ is the variance of $x_i$ given all the others (the conditional precision of one variable is its diagonal block, by the remark above), and $-P_{ij}/\sqrt{P_{ii}P_{jj}}$ is their partial correlation.

**A worked example.** Take a random walk: $x_1 \sim \mathcal{N}(0, 1)$, $x_2 = x_1 + e_2$, $x_3 = x_2 + e_3$, with independent $e_2, e_3 \sim \mathcal{N}(0, 1)$. Then

$$
\boldsymbol{\Sigma} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 2 \\ 1 & 2 & 3 \end{bmatrix},
\qquad
\mathbf{P} = \boldsymbol{\Sigma}^{-1} = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{bmatrix}.
$$

(Check: multiply them; every row of $\boldsymbol{\Sigma}$ times every column of $\mathbf{P}$ gives $1$ on the diagonal and $0$ off it.) The covariance is dense: $x_1$ and $x_3$ are correlated, because $x_3$ inherits $x_1$. The precision has $P_{13} = 0$: *once $x_2$ is known*, $x_1$ tells you nothing more about $x_3$. The precision shows the model's wiring (each variable talks only to its neighbours); the covariance shows its consequences.

**Rules of thumb.**
- **Information adds in precision; noise adds in covariance.** Combining evidence (multiplying likelihoods, Bayes' rule, Lemma 4) adds precisions. Adding independent noise or propagating forward in time (sums, Lemma 3) adds covariances. A Kalman filter does both: the *predict* step adds process noise to $\boldsymbol{\Sigma}$, the *update* step adds measurement information to $\mathbf{P}$.
- **Local models are sparse in precision.** Markov chains, PDE stencils, spatial grids and graphs couple each variable to a few neighbours, so $\mathbf{P}$ is sparse even though $\boldsymbol{\Sigma}$ is dense. Matérn-$\tfrac12$ GPs on sorted 1-D inputs have tridiagonal precision, as in the example; Matérn fields in two and three dimensions have sparse precision through the SPDE link (Lindgren, Rue and Lindström, 2011). Kernels, in contrast, specify $\boldsymbol{\Sigma}$ directly, which is why GP code works in covariance form.
- **The extremes live on opposite sides.** "Know nothing" is $\mathbf{P} = \mathbf{0}$, which has no covariance ($\boldsymbol{\Sigma} = \infty$); "know exactly" is $\boldsymbol{\Sigma} = \mathbf{0}$, which has no precision. A flat prior or a likelihood that constrains only some directions of $\boldsymbol{x}$ (a singular $\mathbf{C}^\top\mathbf{S}^{-1}\mathbf{C}$ in Lemma 8) is natural in precision form; noise-free interpolation and hard constraints are natural in covariance form.
- **Invert the smaller matrix.** The moment form of Lemma 8 inverts a matrix the size of $\boldsymbol{y}$, the information form one the size of $\boldsymbol{x}$. Section 3 is this choice: function space ($N\times N$) versus weight space ($M\times M$).
:::

## 2. Bayesian linear regression

We observe $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$ with

$$
y_i = f(\boldsymbol{x}_i) + \varepsilon_i, \qquad \varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_n^2), \qquad f(\boldsymbol{x}) = \boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{w} .
$$

Stacking the $N$ equations, $\boldsymbol{y} = \boldsymbol{\Phi}\boldsymbol{w} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2\mathbf{I})$, so the likelihood is $p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{w}) = \mathcal{N}(\boldsymbol{y} \mid \boldsymbol{\Phi}\boldsymbol{w}, \sigma_n^2\mathbf{I})$. Least squares returns one $\boldsymbol{w}$ and hides how many others fit almost as well; the Bayesian treatment keeps a distribution over $\boldsymbol{w}$, starting from the prior $\boldsymbol{w} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_w)$.

:::proposition[Posterior over weights]

$$
p(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}) = \mathcal{N}\big(\boldsymbol{w} \mid \sigma_n^{-2}\mathbf{A}^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y},\; \mathbf{A}^{-1}\big),
\qquad
\mathbf{A} = \sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \boldsymbol{\Sigma}_w^{-1}.
$$
:::

:::proof
**Step 1. Bayes' rule.** The evidence $p(\boldsymbol{y} \mid \mathbf{X})$ does not depend on $\boldsymbol{w}$:

$$
p(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}) = \frac{p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{w})\,p(\boldsymbol{w})}{p(\boldsymbol{y} \mid \mathbf{X})} \;\propto\; p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{w})\,p(\boldsymbol{w}) .
$$

**Step 2. Take logs.** Keep only the terms that depend on $\boldsymbol{w}$:

$$
\log p(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}) = -\frac{1}{2\sigma_n^2}(\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w})^\top(\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w}) - \tfrac12\,\boldsymbol{w}^\top\boldsymbol{\Sigma}_w^{-1}\boldsymbol{w} + \text{const}.
$$

**Step 3. Expand the residual square.**

$$
\begin{aligned}
(\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w})^\top(\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w})
&= \boldsymbol{y}^\top \boldsymbol{y} - \boldsymbol{y}^\top\boldsymbol{\Phi}\boldsymbol{w} - \boldsymbol{w}^\top\boldsymbol{\Phi}^\top \boldsymbol{y} + \boldsymbol{w}^\top\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{w} \\
&= \boldsymbol{y}^\top \boldsymbol{y} - 2\,\boldsymbol{w}^\top\boldsymbol{\Phi}^\top \boldsymbol{y} + \boldsymbol{w}^\top\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{w} .
\end{aligned}
$$

Reasons, line by line: (1) multiply out; (2) $\boldsymbol{y}^\top\boldsymbol{\Phi}\boldsymbol{w} = \boldsymbol{w}^\top\boldsymbol{\Phi}^\top \boldsymbol{y}$, the same scalar.

**Step 4. Substitute and collect.** $\boldsymbol{y}^\top \boldsymbol{y}$ joins the constant. Grouping the quadratic and linear terms in $\boldsymbol{w}$,

$$
\log p(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}) = -\tfrac12\,\boldsymbol{w}^\top\underbrace{\big(\sigma_n^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \boldsymbol{\Sigma}_w^{-1}\big)}_{\mathbf{A}}\,\boldsymbol{w} \;+\; \boldsymbol{w}^\top\underbrace{\big(\sigma_n^{-2}\boldsymbol{\Phi}^\top \boldsymbol{y}\big)}_{h} + \text{const}.
$$

**Step 5. Complete the square** (Lemma 1 with $\mathbf{P} = \mathbf{A}$): the posterior is $\mathcal{N}(\mathbf{A}^{-1}\boldsymbol{h}, \mathbf{A}^{-1}) = \mathcal{N}(\sigma_n^{-2}\mathbf{A}^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y}, \mathbf{A}^{-1})$.
:::

:::proposition[Predictive distribution]
For $f_* = \boldsymbol{\phi}_*^\top \boldsymbol{w}$ with $\boldsymbol{\phi}_* = \boldsymbol{\phi}(\boldsymbol{x}_*)$,

$$
p(f_* \mid \boldsymbol{x}_*, \mathbf{X}, \boldsymbol{y}) = \mathcal{N}\big(\sigma_n^{-2}\boldsymbol{\phi}_*^\top\mathbf{A}^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y},\;\; \boldsymbol{\phi}_*^\top\mathbf{A}^{-1}\boldsymbol{\phi}_*\big).
$$
:::

:::proof
**Step 1. The posterior over $\boldsymbol{w}$** is $\mathcal{N}(\bar{\boldsymbol{w}}, \mathbf{A}^{-1})$ with $\bar{\boldsymbol{w}} = \sigma_n^{-2}\mathbf{A}^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y}$ (Proposition 1).

**Step 2. $f_*$ is a linear map of $\boldsymbol{w}$.** $f_* = \boldsymbol{\phi}_*^\top \boldsymbol{w}$ is Lemma 2 with the $1\times M$ matrix $\mathbf{C} = \boldsymbol{\phi}_*^\top$ and $\boldsymbol{d} = 0$.

**Step 3. The mean.** $\mathbf{C}\bar{\boldsymbol{w}} = \boldsymbol{\phi}_*^\top\bar{\boldsymbol{w}} = \sigma_n^{-2}\boldsymbol{\phi}_*^\top\mathbf{A}^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y}$.

**Step 4. The variance.** $\mathbf{C}\mathbf{A}^{-1}\mathbf{C}^\top = \boldsymbol{\phi}_*^\top\mathbf{A}^{-1}\boldsymbol{\phi}_*$.
:::

:::remark[One lemma does it all]
Propositions 1 and 2 are Lemma 8 with $\boldsymbol{x} = \boldsymbol{w}$, $\boldsymbol{\mu}_x = \mathbf{0}$, $\boldsymbol{\Sigma}_x = \boldsymbol{\Sigma}_w$, $\mathbf{C} = \boldsymbol{\Phi}$, $\boldsymbol{d} = \mathbf{0}$ and $\mathbf{S} = \sigma_n^2\mathbf{I}$. Its information form is Proposition 1 ($\mathbf{A}$ is the posterior precision). Its marginal gives, at no extra cost, the evidence $p(\boldsymbol{y} \mid \mathbf{X}) = \mathcal{N}(\mathbf{0},\, \boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top + \sigma_n^2\mathbf{I})$, which Section 7 will recognise as $\mathcal{N}(\mathbf{0}, \mathbf{K}_y)$.
:::

$\mathbf{A}$ is a precision: the prior contributes $\boldsymbol{\Sigma}_w^{-1}$ and each observation adds $\sigma_n^{-2}\boldsymbol{\phi}(\boldsymbol{x}_i)\boldsymbol{\phi}(\boldsymbol{x}_i)^\top$, so uncertainty can only shrink as data arrive. In the figure the input is a scalar $x$ and $\boldsymbol{\phi}(x) = [1, x]^\top$; each line is a posterior sample of $\boldsymbol{w}$ and the band is $\pm 2$ predictive standard deviations.

<figure class="fig" data-blr>
  <canvas data-figure="blr" aria-label="Bayesian linear regression: click to add data points."></canvas>
  <figcaption><span class="mono" data-out>N = 0 (prior)</span> · Click to add observations. <button type="button" data-reset>Reset</button></figcaption>
</figure>

The samples agree near the data and disagree away from it, which is what we want. The limitation is the model class: no amount of data makes a line fit a curve. The next section removes that limitation without losing tractability.

## 3. From weight space to function space

Rewrite the predictive distribution so that the features appear only through inner products.

:::proposition[Kernel form of the predictive]
Define $k(\boldsymbol{x}, \boldsymbol{x}') = \boldsymbol{\phi}(\boldsymbol{x})^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(\boldsymbol{x}')$. Then the predictive mean and variance of Proposition 2 are

$$
\mu_* = \boldsymbol{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y},
\qquad
\sigma_*^2 = k_{**} - \boldsymbol{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{k}_* .
$$
:::

:::proof
**Step 1. Translate features into kernels.** Row $i$ of $\boldsymbol{\Phi}$ is $\boldsymbol{\phi}(\boldsymbol{x}_i)^\top$, so entry by entry:

$$
\begin{aligned}
\big(\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\big)_{ij} &= \boldsymbol{\phi}(\boldsymbol{x}_i)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(\boldsymbol{x}_j) = k(\boldsymbol{x}_i, \boldsymbol{x}_j), \\
\big(\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\phi}_*\big)_i &= \boldsymbol{\phi}(\boldsymbol{x}_i)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(\boldsymbol{x}_*) = k(\boldsymbol{x}_i, \boldsymbol{x}_*), \\
\boldsymbol{\phi}_*^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}_* &= k(\boldsymbol{x}_*, \boldsymbol{x}_*) = k_{**}.
\end{aligned}
$$

That is, $\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top = \mathbf{K}$, $\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\phi}_* = \boldsymbol{k}_*$, and the matrix $\mathbf{G}$ of Lemma 6 is $\mathbf{K} + \sigma_n^2\mathbf{I}$.

**Step 2. One transpose we need.** Because $\boldsymbol{\Sigma}_w$ is symmetric,

$$
\boldsymbol{\phi}_*^\top\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top = \big(\boldsymbol{\Phi}\boldsymbol{\Sigma}_w^\top\boldsymbol{\phi}_*\big)^\top = \big(\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\phi}_*\big)^\top = \boldsymbol{k}_*^\top .
$$

**Step 3. The mean.**

$$
\begin{aligned}
\mu_* &= \boldsymbol{\phi}_*^\top\big(\sigma_n^{-2}\mathbf{A}^{-1}\boldsymbol{\Phi}^\top\big)\boldsymbol{y}
&& \text{Proposition 2} \\
&= \boldsymbol{\phi}_*^\top\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{G}^{-1}\boldsymbol{y}
&& \text{push-through (Lemma 6)} \\
&= \boldsymbol{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y}
&& \text{Steps 1 and 2}.
\end{aligned}
$$

**Step 4. The variance.**

$$
\begin{aligned}
\sigma_*^2 &= \boldsymbol{\phi}_*^\top\mathbf{A}^{-1}\boldsymbol{\phi}_*
&& \text{Proposition 2} \\
&= \boldsymbol{\phi}_*^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}_* - \boldsymbol{\phi}_*^\top\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\,\mathbf{G}^{-1}\,\boldsymbol{\Phi}\boldsymbol{\Sigma}_w\boldsymbol{\phi}_*
&& \text{Woodbury (Lemma 6)} \\
&= k_{**} - \boldsymbol{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{k}_*
&& \text{Steps 1 and 2}.
\end{aligned}
$$
:::

The features no longer appear; only $k$ does. This is the **kernel trick**: any feature space, including an infinite-dimensional one, can be used as long as $k$ can be evaluated. Notice also the size of the matrix being inverted: $\mathbf{A}$ is $M\times M$ (number of features), $\mathbf{K} + \sigma_n^2\mathbf{I}$ is $N\times N$ (number of data points). The function-space form is the one to use when features outnumber data.

:::example[Infinitely many Gaussian features give the RBF kernel]
In one dimension, let $\phi_c(x) = \exp\!\big(-\tfrac{(x - c)^2}{2s^2}\big)$ for every centre $c \in \mathbb{R}$, with independent weights of equal prior variance. A sum over features becomes an integral over centres:

$$
\begin{aligned}
k(x, x') &\propto \int_{-\infty}^{\infty} \exp\!\Big(-\frac{(x - c)^2}{2s^2}\Big)\exp\!\Big(-\frac{(x' - c)^2}{2s^2}\Big)\,dc \\
&= \sqrt{\pi}\,s\,\exp\!\Big(-\frac{(x - x')^2}{4s^2}\Big).
\end{aligned}
$$

This is the squared-exponential (RBF) kernel with length-scale $\ell = \sqrt{2}\,s$.
:::

:::proof[of the integral]
**Step 1. Combine the exponents.**

$$
\exp\!\Big(-\frac{(x - c)^2}{2s^2}\Big)\exp\!\Big(-\frac{(x' - c)^2}{2s^2}\Big) = \exp\!\Big(-\frac{(x - c)^2 + (x' - c)^2}{2s^2}\Big).
$$

**Step 2. Expand the numerator.**

$$
(x - c)^2 + (x' - c)^2 = 2c^2 - 2c(x + x') + x^2 + x'^2 .
$$

**Step 3. Complete the square in $c$.** Let $\bar{c} = \tfrac{x + x'}{2}$, so $2c(x + x') = 4c\bar{c}$, and

$$
2c^2 - 4c\bar{c} = 2(c - \bar{c})^2 - 2\bar{c}^2 = 2(c - \bar{c})^2 - \tfrac12(x + x')^2 .
$$

**Step 4. Simplify the leftover terms.**

$$
x^2 + x'^2 - \tfrac12(x^2 + 2xx' + x'^2) = \tfrac12(x^2 - 2xx' + x'^2) = \tfrac12(x - x')^2 .
$$

So $(x - c)^2 + (x' - c)^2 = 2(c - \bar{c})^2 + \tfrac12(x - x')^2$.

**Step 5. Divide by $2s^2$ and split the exponential.**

$$
\exp\!\Big(-\frac{(x - c)^2 + (x' - c)^2}{2s^2}\Big) = \exp\!\Big(-\frac{(x - x')^2}{4s^2}\Big)\;\exp\!\Big(-\frac{(c - \bar{c})^2}{s^2}\Big).
$$

The first factor does not depend on $c$ and moves outside the integral.

**Step 6. Do the remaining integral.** Substitute $c = \bar{c} + s\,t$, so $dc = s\,dt$, and use $\int_{-\infty}^{\infty} e^{-t^2}dt = \sqrt{\pi}$:

$$
\int_{-\infty}^{\infty} \exp\!\Big(-\frac{(c - \bar{c})^2}{s^2}\Big)dc = s\int_{-\infty}^{\infty} e^{-t^2}\,dt = s\sqrt{\pi} .
$$

**Step 7. Match the RBF form.** $\tfrac{(x - x')^2}{4s^2} = \tfrac{(x - x')^2}{2\ell^2}$ exactly when $\ell^2 = 2s^2$.
:::

Increase the number of features $M$; the random functions converge to samples from the RBF GP.

<figure class="fig" data-features>
  <canvas data-figure="features" aria-label="Random functions built from more and more Gaussian bump features."></canvas>
  <figcaption><label class="slider"><span class="mono" data-out>M = 16 features</span> <input type="range" min="0" max="7" step="1" value="4" data-m /></label> Faint curves (bottom) are the individual features, shown up to $M = 16$.</figcaption>
</figure>

## 4. Gaussian processes

The kernel form suggests placing the prior directly on functions.

:::definition[Gaussian process]
A Gaussian process is a collection of random variables, any finite subset of which is jointly Gaussian. A GP over $f : \mathcal{X} \to \mathbb{R}$ is specified by a mean function and a kernel, $f \sim \mathcal{GP}(m, k)$, where $m(\boldsymbol{x}) = \mathbb{E}[f(\boldsymbol{x})]$ and $k(\boldsymbol{x}, \boldsymbol{x}') = \operatorname{Cov}[f(\boldsymbol{x}), f(\boldsymbol{x}')]$. For any inputs $\mathbf{X}$,

$$
\boldsymbol{f}_{\mathbf{X}} = f(\mathbf{X}) \sim \mathcal{N}\big(m(\mathbf{X}), \mathbf{K}\big).
$$
:::

:::definition[Positive-definite kernel]
A symmetric function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is positive definite if, for every finite set $\{\boldsymbol{x}_i\}_{i=1}^N$ and every $\boldsymbol{c} \in \mathbb{R}^N$, $\sum_{i,j} c_i\,c_j\,k(\boldsymbol{x}_i, \boldsymbol{x}_j) \ge 0$; that is, every Gram matrix is positive semidefinite. Exactly these functions are covariance functions of GPs.
:::

We take $m \equiv 0$ from now on; a known trend can be subtracted first.

**Sampling a function on a grid** takes three steps:

1. Evaluate the Gram matrix $\mathbf{K}$ on the grid and factor it as $\mathbf{K} = \mathbf{L}\mathbf{L}^\top$ (Cholesky; $\mathbf{L}$ is lower triangular).
2. Draw $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, one independent standard normal per grid point.
3. Set $\boldsymbol{f}_{\mathbf{X}} = \mathbf{L}\boldsymbol{z}$. By Lemma 2 it is Gaussian with mean $\mathbf{L}\mathbf{0} = \mathbf{0}$ and covariance $\mathbf{L}\mathbf{I}\mathbf{L}^\top = \mathbf{K}$, as required.

:::note
Smooth kernels make $\mathbf{K}$ numerically singular on fine grids (nearby columns are almost identical). Add a small jitter, $\mathbf{K} + \epsilon\mathbf{I}$ with $\epsilon \approx 10^{-6}$, before factoring.
:::

## 5. Kernels encode assumptions

The kernel determines which functions are plausible a priori. Select a kernel to see three prior samples; the inset plots $k(0.5, \boldsymbol{x})$.

<figure class="fig" data-gallery-k>
  <canvas data-figure="gallery" aria-label="Prior samples for different kernels."></canvas>
  <div class="controls"><div class="seg" role="group" aria-label="Kernel">
    <button type="button" data-k="rbf" class="on">RBF</button>
    <button type="button" data-k="matern12">Matérn ½</button>
    <button type="button" data-k="matern32">Matérn 3⁄2</button>
    <button type="button" data-k="matern52">Matérn 5⁄2</button>
    <button type="button" data-k="periodic">Periodic</button>
    <button type="button" data-k="linear">Linear</button>
    <button type="button" data-k="sum">RBF + linear</button>
    <button type="button" data-k="locper">RBF × periodic</button>
  </div></div>
</figure>

The squared-exponential (RBF) kernel is

$$
k_{\text{RBF}}(\boldsymbol{x}, \boldsymbol{x}') = \sigma_f^2\exp\!\left(-\frac{\lVert \boldsymbol{x} - \boldsymbol{x}'\rVert^2}{2\ell^2}\right).
$$

Its sample paths are infinitely differentiable. The length-scale $\ell$ sets the distance over which values decorrelate; $\sigma_f^2$ is the prior variance at every input.

The Matérn family adds a smoothness parameter $\nu > 0$. With $r = \lVert \boldsymbol{x} - \boldsymbol{x}'\rVert$, the three common members are

$$
\begin{aligned}
k_{1/2}(r) &= \sigma_f^2\, e^{-r/\ell}, \\
k_{3/2}(r) &= \sigma_f^2\Big(1 + \frac{\sqrt{3}\,r}{\ell}\Big)e^{-\sqrt{3}\,r/\ell}, \\
k_{5/2}(r) &= \sigma_f^2\Big(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\Big)e^{-\sqrt{5}\,r/\ell}.
\end{aligned}
$$

Sample paths are $\lceil\nu\rceil - 1$ times mean-square differentiable, and $\nu \to \infty$ recovers the RBF. Physical signals are rarely infinitely smooth, so Matérn is the usual default. The **periodic** kernel encodes exact repetition, the **linear** kernel $k(\boldsymbol{x}, \boldsymbol{x}') = \boldsymbol{x}^\top \boldsymbol{x}'$ recovers Bayesian linear regression (Section 2), and **sums and products** build structure: a sum models superposed effects (a trend plus variation), a product modulates one structure by another (a slowly changing periodic pattern).

:::proposition[Closure under sums and products]
If $k_1$ and $k_2$ are positive definite, so are $k_1 + k_2$ and $k_1 k_2$.
:::

:::proof
**Step 1. Reduce to matrices.** Fix any inputs $\boldsymbol{x}_1, \dots, \boldsymbol{x}_N$ and let $\mathbf{K}_1$, $\mathbf{K}_2$ be the Gram matrices of $k_1$, $k_2$. Both are positive semidefinite by assumption. We must show that the Gram matrices of $k_1 + k_2$ and $k_1 k_2$ are too.

*Sum.*

**Step 2.** The Gram matrix of $k_1 + k_2$ is $\mathbf{K}_1 + \mathbf{K}_2$, and for any $\boldsymbol{c}$,

$$
\boldsymbol{c}^\top(\mathbf{K}_1 + \mathbf{K}_2)\boldsymbol{c} = \underbrace{\boldsymbol{c}^\top\mathbf{K}_1\boldsymbol{c}}_{\ge 0} + \underbrace{\boldsymbol{c}^\top\mathbf{K}_2\boldsymbol{c}}_{\ge 0} \ge 0 .
$$

*Product.*

**Step 3. The Gram matrix of $k_1 k_2$** has entries $k_1(\boldsymbol{x}_p, \boldsymbol{x}_q)\,k_2(\boldsymbol{x}_p, \boldsymbol{x}_q)$: it is the elementwise (Hadamard) product $\mathbf{K}_1 \circ \mathbf{K}_2$.

**Step 4. Decompose each factor.** Symmetric positive semidefinite matrices have eigendecompositions with non-negative eigenvalues:

$$
\mathbf{K}_1 = \sum_i \lambda_i\,\boldsymbol{u}_iu_i^\top, \qquad \mathbf{K}_2 = \sum_j \kappa_j\,\boldsymbol{v}_jv_j^\top, \qquad \lambda_i, \kappa_j \ge 0 .
$$

**Step 5. Hadamard product of two rank-one matrices.** Entry $(p, q)$ of $(uu^\top) \circ (vv^\top)$ is $u_pu_q\,v_pv_q = (u_pv_p)(u_qv_q)$, which is entry $(p, q)$ of $(\boldsymbol{u} \circ \boldsymbol{v})(\boldsymbol{u} \circ \boldsymbol{v})^\top$. So

$$
(uu^\top) \circ (vv^\top) = (\boldsymbol{u} \circ \boldsymbol{v})(\boldsymbol{u} \circ \boldsymbol{v})^\top .
$$

**Step 6. Expand.** The Hadamard product is linear in each factor, so by Steps 4 and 5

$$
\mathbf{K}_1 \circ \mathbf{K}_2 = \sum_{i,j}\lambda_i\kappa_j\,(\boldsymbol{u}_i \circ \boldsymbol{v}_j)(\boldsymbol{u}_i \circ \boldsymbol{v}_j)^\top .
$$

**Step 7. Conclude.** For any $\boldsymbol{c}$, $\boldsymbol{c}^\top(\mathbf{K}_1 \circ \mathbf{K}_2)\boldsymbol{c} = \sum_{i,j}\lambda_i\kappa_j\big((\boldsymbol{u}_i \circ \boldsymbol{v}_j)^\top \boldsymbol{c}\big)^2 \ge 0$, a sum of non-negative terms. (This is the Schur product theorem.)
:::

A kernel is **stationary** if $k(\boldsymbol{x}, \boldsymbol{x}') = \kappa(\boldsymbol{x} - \boldsymbol{x}')$; every example above except the linear kernel is stationary.

:::theorem[Bochner]
A continuous function $\kappa : \mathbb{R}^D \to \mathbb{R}$ defines a positive-definite stationary kernel if and only if it is the Fourier transform of a finite non-negative measure $\mu$:

$$
\kappa(\boldsymbol{\tau}) = \int_{\mathbb{R}^D} e^{i\,\boldsymbol{\omega}^\top\boldsymbol{\tau}}\,\mu(d\boldsymbol{\omega}).
$$
:::

A stationary kernel is therefore a spectrum. The RBF has a Gaussian spectral density; Matérn has heavier, Student-$\boldsymbol{t}$-like tails. Sampling frequencies from $\mu$ gives random Fourier features, and the stationarity assumption is what [From Fourier to wavelets](/aboutme/blog/fourier-to-wavelets/) relaxes.

## 6. Conditioning

We now derive the posterior directly in function space, with no features.

**Step 1. The joint prior.** By Definition 1, $\boldsymbol{f}_{\mathbf{X}}$ and $f_*$ are jointly Gaussian, with every covariance given by the kernel:

$$
\begin{bmatrix} \boldsymbol{f}_{\mathbf{X}} \\ f_* \end{bmatrix}
\sim \mathcal{N}\!\left(\mathbf{0},\;
\begin{bmatrix} \mathbf{K} & \boldsymbol{k}_* \\ \boldsymbol{k}_*^\top & k_{**} \end{bmatrix}\right).
$$

**Step 2. Add the noise.** The observations are $\boldsymbol{y} = \boldsymbol{f}_{\mathbf{X}} + \boldsymbol{\varepsilon}$, with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2\mathbf{I})$ independent of $f$. The test value receives no noise, so

$$
\begin{bmatrix} \boldsymbol{y} \\ f_* \end{bmatrix} = \begin{bmatrix} \boldsymbol{f}_{\mathbf{X}} \\ f_* \end{bmatrix} + \begin{bmatrix} \boldsymbol{\varepsilon} \\ 0 \end{bmatrix},
\qquad
\begin{bmatrix} \boldsymbol{\varepsilon} \\ 0 \end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \begin{bmatrix} \sigma_n^2\mathbf{I} & 0 \\ 0^\top & 0 \end{bmatrix}\right).
$$

**Step 3. Sum of independent Gaussians** (Lemma 3): the covariances add.

$$
\begin{bmatrix} \boldsymbol{y} \\ f_* \end{bmatrix}
\sim \mathcal{N}\!\left(\mathbf{0},\;
\begin{bmatrix} \mathbf{K} + \sigma_n^2\mathbf{I} & \boldsymbol{k}_* \\ \boldsymbol{k}_*^\top & k_{**} \end{bmatrix}\right)
= \mathcal{N}\!\left(0,\;
\begin{bmatrix} \mathbf{K}_y & \boldsymbol{k}_* \\ \boldsymbol{k}_*^\top & k_{**} \end{bmatrix}\right).
$$

Only the top-left block changes. The cross-covariance is unchanged because $\operatorname{Cov}[\boldsymbol{f}_{\mathbf{X}} + \boldsymbol{\varepsilon},\, f_*] = \operatorname{Cov}[\boldsymbol{f}_{\mathbf{X}}, f_*] + \operatorname{Cov}[\boldsymbol{\varepsilon}, f_*] = \boldsymbol{k}_* + \mathbf{0}$.

**Step 4. Condition** (Lemma 5) with

$$
\begin{aligned}
&\boldsymbol{a} = \boldsymbol{y}, && \boldsymbol{b} = f_*, && \boldsymbol{\mu}_a = \mathbf{0}, && \mu_b = 0, \\
&\boldsymbol{\Sigma}_{aa} = \mathbf{K}_y, && \boldsymbol{\Sigma}_{ab} = \boldsymbol{k}_*, && \boldsymbol{\Sigma}_{ba} = \boldsymbol{k}_*^\top, && \boldsymbol{\Sigma}_{bb} = k_{**} .
\end{aligned}
$$

:::proposition[GP posterior predictive]

$$
f_* \mid \mathbf{X}, \boldsymbol{y}, \boldsymbol{x}_* \sim \mathcal{N}(\mu_*, \sigma_*^2),
\qquad
\mu_* = \boldsymbol{k}_*^\top\mathbf{K}_y^{-1}\boldsymbol{y},
\qquad
\sigma_*^2 = k_{**} - \boldsymbol{k}_*^\top\mathbf{K}_y^{-1}\boldsymbol{k}_* .
$$

For a noisy test observation $y_* = f_* + \varepsilon_*$, Lemma 3 adds $\sigma_n^2$ to the variance.
:::

These are the formulas of Proposition 3, obtained here without any reference to features. The two derivations are two views of one model:

<figure class="sketch-fig"><svg class="sketch" viewBox="0 0 920 560" role="img" aria-label="Two roads to the same posterior: the weight-space view (prior over weights, posterior, predictive) and the function-space view (GP prior, conditioning, posterior GP), linked by the kernel trick, give the same predictive mean and variance."><text x="220" y="44" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--a1)">WEIGHT SPACE</text><text x="700" y="44" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--a1)">FUNCTION SPACE</text><text x="220" y="70" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">a prior over parameters</text><text x="700" y="70" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">a prior over functions</text><path d="M68.09424029793361 90.805721553152 C131.688860132834 94.13083045315831, 190.0943187275782 93.88931292321938, 368.517491179523 90.80491635626734 M69.4845994539166 91.85909560838616 C173.9583000916942 95.06638385762052, 280.25443964083104 95.07810131610262, 369.9403054633259 91.30382860367497 M368.3840086212382 91.68015578780323 C370.4386317865737 115.37147912085057, 371.5316083865054 137.82106451597065, 372.2539845636114 164.4888700881973 M370.10886715846135 92.52660608841106 C369.82477711322156 111.48685010718182, 371.3389134603962 131.27638442059978, 371.1025793005712 162.60541787398978 M369.3990616506291 160.57571911709812 C276.3375648005039 163.88571546493614, 184.38040783602827 164.07191742917522, 70.81524330638615 163.12042656602875 M369.42423093826585 162.9474442304958 C299.9971988300559 164.60351037927865, 230.7586319990133 164.6985996652994, 70.54589517098559 162.27922624551 M68.64250250961632 159.43864168915897 C72.0774617128931 149.1551686692983, 71.91231796675548 134.56483319979162, 70.18538303878158 91.42466502580791 M69.38340792255477 163.2838296408765 C70.40584627778829 137.0036875114776, 69.81094160170854 113.11459715133533, 68.81073918240145 93.19107370031998" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="220" y="135" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">prior:  w ~ N(0, Σ_w)</text><path d="M217.40109737422318 166.3713511267677 C222.45067888966275 178.98655779510736, 220.26331319680884 189.45889737512917, 217.9782886257395 220.37025307212025 M219.2971433748491 167.80784734142944 C219.13766433690043 186.475680159498, 221.12481637392492 205.36267448803412, 219.91859399275853 221.05062444759534" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M218.18076816195622 220.85994578873738 C220.34761872439373 219.55252090798018, 217.59836761218676 216.80032048726503, 212.49528456046045 208.25291771754672 M219.50800036239437 221.86549313900062 C216.95219347262824 217.22493565402783, 216.21513737287205 212.83069679787556, 213.85349831737375 208.72917768037928" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M218.18076816195622 220.85994578873738 C222.7838089788122 219.6629837517093, 222.47074812102366 216.91078333099415, 224.6743195155749 208.25291771754672 M219.50800036239437 221.86549313900062 C221.2083185239749 217.41074679701427, 224.72738747556537 213.016507940862, 226.0325332724882 208.72917768037928" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="236" y="200" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">observe y</text><path d="M68.09424029793361 226.80572155315198 C131.688860132834 230.1308304531583, 190.0943187275782 229.8893129232194, 368.517491179523 226.80491635626734 M69.4845994539166 227.85909560838616 C173.9583000916942 231.0663838576205, 280.25443964083104 231.0781013161026, 369.9403054633259 227.30382860367496 M368.3840086212382 227.68015578780324 C370.4386317865737 251.37147912085055, 371.5316083865054 273.82106451597065, 372.2539845636114 300.4888700881973 M370.10886715846135 228.52660608841106 C369.82477711322156 247.48685010718182, 371.3389134603962 267.2763844205998, 371.1025793005712 298.6054178739898 M369.3990616506291 296.5757191170981 C276.3375648005039 299.88571546493614, 184.38040783602827 300.07191742917524, 70.81524330638615 299.12042656602875 M369.42423093826585 298.94744423049576 C299.9971988300559 300.60351037927865, 230.7586319990133 300.6985996652994, 70.54589517098559 298.27922624551 M68.64250250961632 295.438641689159 C72.0774617128931 285.1551686692983, 71.91231796675548 270.5648331997916, 70.18538303878158 227.4246650258079 M69.38340792255477 299.2838296408765 C70.40584627778829 273.0036875114776, 69.81094160170854 249.11459715133532, 68.81073918240145 229.19107370031998" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="220" y="271" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">posterior:  p(w | X, y)</text><path d="M217.40109737422318 302.3713511267677 C222.45067888966275 314.9865577951074, 220.26331319680884 325.45889737512914, 217.9782886257395 356.37025307212025 M219.2971433748491 303.80784734142946 C219.13766433690043 322.475680159498, 221.12481637392492 341.3626744880341, 219.91859399275853 357.05062444759534" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M218.18076816195622 356.8599457887374 C220.34761872439373 355.5525209079802, 217.59836761218676 352.800320487265, 212.49528456046045 344.2529177175467 M219.50800036239437 357.8654931390006 C216.95219347262824 353.2249356540279, 216.21513737287205 348.8306967978756, 213.85349831737375 344.7291776803793" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M218.18076816195622 356.8599457887374 C222.7838089788122 355.6629837517093, 222.47074812102366 352.91078333099415, 224.6743195155749 344.2529177175467 M219.50800036239437 357.8654931390006 C221.2083185239749 353.41074679701427, 224.72738747556537 349.01650794086197, 226.0325332724882 344.7291776803793" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="236" y="336" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">f* = φ(x*)ᵀ w</text><path d="M548.0942402979337 90.805721553152 C611.688860132834 94.13083045315831, 670.0943187275782 93.88931292321938, 848.5174911795231 90.80491635626734 M549.4845994539166 91.85909560838616 C653.9583000916942 95.06638385762052, 760.254439640831 95.07810131610262, 849.9403054633259 91.30382860367497 M848.3840086212382 91.68015578780323 C850.4386317865738 115.37147912085057, 851.5316083865055 137.82106451597065, 852.2539845636114 164.4888700881973 M850.1088671584614 92.52660608841106 C849.8247771132216 111.48685010718182, 851.3389134603963 131.27638442059978, 851.1025793005713 162.60541787398978 M849.399061650629 160.57571911709812 C756.337564800504 163.88571546493614, 664.3804078360283 164.07191742917522, 550.8152433063862 163.12042656602875 M849.4242309382659 162.9474442304958 C779.9971988300559 164.60351037927865, 710.7586319990132 164.6985996652994, 550.5458951709855 162.27922624551 M548.6425025096163 159.43864168915897 C552.0774617128931 149.1551686692983, 551.9123179667555 134.56483319979162, 550.1853830387815 91.42466502580791 M549.3834079225547 163.2838296408765 C550.4058462777883 137.0036875114776, 549.8109416017086 113.11459715133533, 548.8107391824014 93.19107370031998" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="700" y="135" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">prior:  f ~ GP(0, k)</text><path d="M697.4010973742231 166.3713511267677 C702.4506788896628 178.98655779510736, 700.2633131968088 189.45889737512917, 697.9782886257395 220.37025307212025 M699.2971433748492 167.80784734142944 C699.1376643369003 186.475680159498, 701.1248163739249 205.36267448803412, 699.9185939927586 221.05062444759534" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M698.1807681619562 220.85994578873738 C700.3476187243937 219.55252090798018, 697.5983676121867 216.80032048726503, 692.4952845604604 208.25291771754672 M699.5080003623943 221.86549313900062 C696.9521934726282 217.22493565402783, 696.215137372872 212.83069679787556, 693.8534983173737 208.72917768037928" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M698.1807681619562 220.85994578873738 C702.7838089788122 219.6629837517093, 702.4707481210236 216.91078333099415, 704.6743195155749 208.25291771754672 M699.5080003623943 221.86549313900062 C701.208318523975 217.41074679701427, 704.7273874755654 213.016507940862, 706.0325332724882 208.72917768037928" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="716" y="200" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">condition (Lemma 5)</text><path d="M548.0942402979337 226.80572155315198 C611.688860132834 230.1308304531583, 670.0943187275782 229.8893129232194, 848.5174911795231 226.80491635626734 M549.4845994539166 227.85909560838616 C653.9583000916942 231.0663838576205, 760.254439640831 231.0781013161026, 849.9403054633259 227.30382860367496 M848.3840086212382 227.68015578780324 C850.4386317865738 251.37147912085055, 851.5316083865055 273.82106451597065, 852.2539845636114 300.4888700881973 M850.1088671584614 228.52660608841106 C849.8247771132216 247.48685010718182, 851.3389134603963 267.2763844205998, 851.1025793005713 298.6054178739898 M849.399061650629 296.5757191170981 C756.337564800504 299.88571546493614, 664.3804078360283 300.07191742917524, 550.8152433063862 299.12042656602875 M849.4242309382659 298.94744423049576 C779.9971988300559 300.60351037927865, 710.7586319990132 300.6985996652994, 550.5458951709855 298.27922624551 M548.6425025096163 295.438641689159 C552.0774617128931 285.1551686692983, 551.9123179667555 270.5648331997916, 550.1853830387815 227.4246650258079 M549.3834079225547 299.2838296408765 C550.4058462777883 273.0036875114776, 549.8109416017086 249.11459715133532, 548.8107391824014 229.19107370031998" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="700" y="271" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">posterior:  GP(μ, k_post)</text><path d="M697.4010973742231 302.3713511267677 C702.4506788896628 314.9865577951074, 700.2633131968088 325.45889737512914, 697.9782886257395 356.37025307212025 M699.2971433748492 303.80784734142946 C699.1376643369003 322.475680159498, 701.1248163739249 341.3626744880341, 699.9185939927586 357.05062444759534" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M698.1807681619562 356.8599457887374 C700.3476187243937 355.5525209079802, 697.5983676121867 352.800320487265, 692.4952845604604 344.2529177175467 M699.5080003623943 357.8654931390006 C696.9521934726282 353.2249356540279, 696.215137372872 348.8306967978756, 693.8534983173737 344.7291776803793" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M698.1807681619562 356.8599457887374 C702.7838089788122 355.6629837517093, 702.4707481210236 352.91078333099415, 704.6743195155749 344.2529177175467 M699.5080003623943 357.8654931390006 C701.208318523975 353.41074679701427, 704.7273874755654 349.01650794086197, 706.0325332724882 344.7291776803793" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="716" y="336" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">evaluate at x*</text><path d="M377.4010973742232 118.37135112676769 C414.2952752610669 121.63604528228193, 444.1129446258768 121.30668553034216, 537.9782886257395 118.37025307212025 M379.2971433748491 119.80784734142944 C434.71313701169566 122.0457273239605, 492.61440791981295 122.06170653350279, 539.9185939927586 119.05062444759533" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M538.1807681619562 118.85994578873739 C539.0996120329147 121.23315831686322, 535.0468074535596 122.22070161771461, 525.9785426050802 124.9486946280414 M539.5080003623943 119.8654931390006 C534.7920771350542 121.6589989012937, 531.7776592581395 123.79824687364662, 527.3367563619935 125.42495459087395" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M538.1807681619562 118.85994578873739 C539.0634270631348 118.79696806244478, 535.0106224837797 117.34832110887771, 525.9785426050802 112.76965967292695 M539.5080003623943 119.8654931390006 C534.715699656126 117.40287384994704, 531.7012817792113 115.28599677095333, 527.3367563619935 113.2459196357595" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M537.4010973742231 134.3713511267677 C510.28520514573904 134.73367164400963, 476.09280439522115 134.40431189206987, 377.97828862573954 134.37025307212025 M539.2971433748492 135.80784734142944 C482.8848992695101 133.16360275704787, 428.95793243544176 133.17958196659015, 379.91859399275853 135.05062444759534" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M378.1807681619562 134.85994578873738 C384.10672425184947 135.02564353154088, 385.097216861209 133.57699657797383, 391.1910614709551 128.76965967292696 M379.5080003623944 135.86549313900062 C383.5265482617898 133.78753215541215, 389.3229789905388 131.67065507641843, 392.54927522786846 129.24591963575952" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M378.1807681619562 134.85994578873738 C384.1429092216293 137.46183378595936, 385.1334018309888 138.44937708681073, 391.1910614709551 140.9486946280414 M379.5080003623944 135.86549313900062 C383.602925740718 138.0436572067588, 389.39935646946697 140.18290517911174, 392.54927522786846 141.42495459087397" style="stroke:var(--dot);stroke-width:1;fill:none"/><text x="460" y="108" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--dot)">kernel trick</text><text x="460" y="196" text-anchor="middle" style="font-family:Kalam,cursive;font-size:15px;font-weight:400;fill:var(--dot)">k(x, x′) = φ(x)ᵀ Σ_w φ(x′)</text><path d="M217.40109737422318 370.3713511267677 C250.46696891224906 385.0564721382931, 276.284008894851 396.32893759475644, 357.97828862573954 428.37025307212025 M219.2971433748491 371.80784734142946 C268.0876032525361 394.0094626292996, 318.9996093017668 414.29430992961295, 359.91859399275853 429.05062444759534" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M358.1807681619562 428.8599457887374 C358.8096935046429 430.16283787172875, 354.48268329937355 430.0924938833333, 344.6077301667485 429.66009032017115 M359.5080003623944 429.8654931390006 C354.27986254280154 429.7899120983354, 350.786396105719 430.08098725861055, 345.9659439236619 430.1363502830037" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M358.1807681619562 428.8599457887374 C359.7086920610289 427.95442756172093, 356.31411011226953 425.6333946783013, 349.26913799751037 418.40841624591843 M359.5080003623944 429.8654931390006 C355.83829145063265 425.9289830055661, 353.9738157107742 422.2880116553003, 350.6273517544237 418.884676208751" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M697.4010973742231 370.3713511267677 C674.4581575613372 382.5168952048049, 644.2663861930273 393.78936066126823, 557.9782886257395 428.37025307212025 M699.2971433748492 371.80784734142946 C650.2378952281236 389.73760363325107, 603.300193252942 410.0224509335644, 559.9185939927586 429.05062444759534" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M558.1807681619562 428.8599457887374 C563.4355439885521 428.12341217255954, 563.7678139673159 425.8023792891399, 567.9004660785249 418.40841624591843 M559.5080003623943 429.8654931390006 C562.3492557821837 426.2132344572491, 566.9957443738764 422.57226310698326, 569.2586798354382 418.884676208751" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M558.1807681619562 428.8599457887374 C564.4014019451862 430.4163796526001, 565.66610018046 430.3460356642046, 572.5618739092868 429.6600903201712 M559.5080003623943 429.8654931390006 C564.0488082688009 430.21639838556183, 570.3242875577178 430.50747354583706, 573.9200876662001 430.13635028300376" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M300.1546123029252 435.8221388913472 C300.1546123029252 435.8221388913472, 300.1546123029252 435.8221388913472, 300.1546123029252 435.8221388913472 M300.1546123029252 435.8221388913472 C300.1546123029252 435.8221388913472, 300.1546123029252 435.8221388913472, 300.1546123029252 435.8221388913472 M298.7615198925762 447.8685674479841 C304.2404421583961 443.7220832301789, 306.28539949163434 441.6700480934084, 310.5515749555697 435.93514147460843 M299.56039887992387 446.57616570915604 C302.9096750682853 444.0642138606003, 305.5027956035871 441.6588249890568, 308.53031712730984 436.1785135438424 M301.17442347308463 458.98747068613244 C308.6628787068372 451.1126059661875, 315.554051566179 440.6477090736046, 319.4221099287797 436.81763007904544 M299.3132870829089 457.8727805163473 C304.0456868169178 450.9821904038406, 310.2365892003625 444.907595004909, 318.69688418373136 435.84569084898715 M298.0135203117779 466.96113976950136 C308.8339515347722 459.38194423681, 319.65220023565746 448.43670690095615, 326.1292219932254 434.03288972038683 M299.22895997867863 468.4493007372351 C308.2465758027567 457.8067863100458, 317.1209778106817 450.14501161965956, 327.8343275886092 435.58508449146433 M300.71904419057137 476.8344615246843 C308.60832362302233 467.1755468880914, 314.29789826454004 457.7268025322342, 336.1634605606618 438.17157522690167 M301.45115171523827 477.19678472231357 C314.1509851748661 464.4828367756451, 327.8246121282361 448.4928685215957, 336.7133209622065 437.0830181920776 M301.1752785567305 487.2537189106655 C311.7924330178439 475.2543752026329, 325.2920688796145 462.03680082745035, 343.84914814085465 435.40555549507405 M300.8253670533299 488.8755382577155 C315.7410593456837 470.2596766663964, 331.3204691113246 453.27836292743376, 346.49528703401563 436.9760672682096 M300.09892439103055 499.87205290415216 C314.63705671046216 479.0457395654889, 331.3388225736177 457.93277515205915, 354.70533380606935 434.55259785888865 M299.69525094345516 501.1510649910465 C313.29758852522224 485.10281650672414, 325.16726563001873 470.2369296305834, 354.59272275227846 435.3044203184932 M300.319586323718 511.3743741786832 C310.6221907463762 494.8553459006778, 328.82012622739575 478.82178652904315, 363.1020431910717 438.72011808982967 M298.94869964183715 509.90335311049387 C316.04795725864784 491.5542021278893, 330.5837373751484 472.83695313684564, 365.1910742839874 437.34000422281133 M301.9557274895586 520.0946610057429 C315.68356337231967 500.74835454790787, 333.24677731922736 481.39154062233257, 374.11652849312725 435.1489303968815 M300.0594405654684 520.0285482459403 C326.82011825344136 490.6802965891431, 351.3639260200488 460.4775112043341, 375.07061201488455 435.91460914229657 M307.455745933704 520.8937751316424 C334.1759816139767 498.1955849287312, 351.8974841702015 470.18715433092717, 381.8455775009922 433.7022878463058 M310.49373369797684 519.8011443458456 C327.46660449558505 500.9296584515449, 343.19424364560507 482.05227152488044, 384.78436457845265 435.89330393910916 M317.2360995680338 519.5727566655394 C342.76310777516096 492.42965457708937, 368.6375231203726 463.7620348231476, 393.8375274789511 433.7284706825325 M320.2141215459688 520.2716535652656 C344.0618152247547 492.517607477202, 367.38119885838734 465.3050992173005, 392.2557577399528 436.92139949732734 M327.0435631568325 517.8564483518637 C357.1161185310298 487.97138156559276, 385.49767175766164 456.0368531136544, 402.28486934076005 437.97627884682316 M328.0073111247052 519.7346055382789 C355.5269170442621 486.97273090749724, 386.3666644590847 455.2717066160183, 402.0368435703678 435.9243162416113 M337.41906511385633 521.520212631877 C355.1129136305348 501.67240344645313, 374.6814088327154 483.97630650748994, 413.65626153368476 435.0576202981621 M337.0136357551691 521.239639439545 C363.81056875070465 489.71662967522326, 392.02529930083523 460.0444769248857, 411.10593933043975 435.8545474580936 M346.9297013863709 522.8525199645636 C371.9642163275387 491.9156959773959, 399.42864427456806 462.5536820584564, 421.80883906749085 437.93655143303374 M348.3246939481024 520.5384673051973 C362.0229216749998 502.3850481974008, 379.7081489761181 484.8366025548593, 420.0455614996609 436.36507471415723 M356.4810444841418 518.343989246408 C386.6139763366581 487.571782997024, 414.0930359850856 452.9290383073406, 429.8444635144439 437.37302708542256 M356.56128343299736 520.1582894526184 C378.29995411448465 497.0384627007729, 397.49158140217236 473.66621445641925, 429.2643007762734 436.93749285843506 M365.1733934794346 519.3311378113943 C384.5224858363813 500.9136435508062, 403.69257906945995 477.04419436371865, 438.1659730093095 435.96665052681277 M366.8952761580961 519.5299194804411 C389.5964107672533 493.64194426612954, 414.17668359030546 466.6297895588309, 439.1186970772694 436.1101350208097 M373.9580240070151 518.4401314310072 C396.7429249063299 495.78737780921824, 422.31859662799184 467.4427088693779, 450.7334322493533 438.0004234661889 M374.371493290391 519.4159092685481 C394.3972956505057 498.754322766434, 413.92085101900386 474.6881550287925, 448.73511453839114 437.2061976124142 M381.4934382641488 521.0305343142311 C411.4515327871144 488.6128814696659, 435.38043950006283 461.1110769229464, 457.92038249434097 436.68449605003514 M383.5295802156241 519.8386739780726 C402.70119425321343 500.7731095971014, 419.1061783685061 480.5774824106994, 458.7316165141308 437.16186532563114 M393.0016150106013 522.03905910755 C415.51132006575864 491.39029918732564, 440.98881055224183 463.0180127650444, 468.77009156653816 435.19991686233607 M392.92588153828757 521.908617051397 C414.1923551076182 495.1361999408837, 435.15302630012695 471.6353031970015, 467.43984057773065 437.4544361340935 M400.5242595686383 521.8233168949484 C421.15536549675363 501.55829133392604, 443.6322012303924 475.48576607482215, 478.25180385071036 433.9423829245759 M401.7553160816237 520.5522552121543 C422.42634694955433 497.7021594115216, 441.69502494909926 475.75236147883084, 476.22687352118265 435.530446292684 M412.5790538230254 518.605567197911 C440.7068390900183 490.14486780383504, 465.53363775276733 458.89576341831764, 483.9169902941234 435.96598969573904 M412.57325188762456 521.9403434652272 C436.0732578161055 491.62248060157316, 462.6466472276056 463.84720541134186, 485.71712635522925 435.73379533753547 M418.8159062210283 519.1164679754219 C438.4188685208174 503.343188134119, 453.53972894298295 481.72919923202755, 495.41165743553324 433.4722714920872 M419.71738121703254 520.6160000967486 C439.05648767332644 501.0469029398872, 456.396726945724 481.084072874921, 493.97026833835304 434.779161505808 M430.72926080140303 519.8841160437426 C446.9550244353849 500.5996944598365, 466.7918881856548 476.0952265676175, 501.70836128922457 437.07278980805023 M430.88565457833016 521.3681220938785 C455.92783348480776 487.407055463026, 483.73235528919963 459.31980534420194, 503.07886586081514 435.69896505658 M438.49011107843415 520.8966353973033 C456.34688768151324 506.96873695592296, 468.43906904399574 489.08966959834027, 515.0988115189534 437.7046258348896 M439.93391961177855 519.3783484656955 C462.33056275133583 495.0814174400694, 485.7467040345727 468.12399675667365, 512.8266116185982 434.7432357895113 M449.7386042158779 518.7176697273653 C467.30950417470785 503.9462170064222, 482.4364700763728 484.10342609774136, 522.8258853219855 435.89296878158996 M449.13415026336054 520.9209277618592 C471.8486126312873 493.6994056310171, 494.0657510332251 466.31278264574667, 522.0859634974446 435.32657636573487 M456.8160549109998 520.1489707429134 C476.46093944719536 502.7272105697809, 491.88282681842725 481.3804172106135, 532.9959364580866 433.44015629463547 M457.5720856688182 520.9096813935028 C484.59646179368536 490.63509406993137, 511.0073389437364 459.5586843477809, 532.1960786018465 436.10157772763347 M467.0772544235656 518.3194385904316 C487.04840393428094 495.7432769337451, 511.3357739837866 470.56895171838863, 539.2350452495219 433.66942445801055 M467.12516418264374 519.5714190530088 C490.19841330190076 494.4496027909583, 513.6408419133194 466.4218058338085, 540.5846882550458 436.68694409884 M477.94672057584097 518.9288718539042 C495.01034186027596 502.1187527420377, 510.3713518316573 481.39656420056235, 550.3452685810577 433.86865779319965 M476.0843983780796 520.37406371401 C497.52610410796274 496.51009670643623, 519.801188152813 471.82391480364873, 551.2899729990351 436.89561798423716 M485.7749477525913 522.291717554816 C502.7326497031425 496.99763468037895, 523.6641144847649 476.4352215720151, 560.1608898999588 435.934139259187 M487.0762500471175 521.4547766313897 C508.79840731745185 493.8343191142483, 532.3345251600131 469.2227840957219, 560.7824375945562 437.2629051699578 M497.2946569010825 518.380821152151 C512.1193205816471 501.8257973553342, 531.363747299399 479.61307429058604, 566.8046692159915 433.61430025245716 M495.48450520050017 521.3480823412802 C512.1274971850373 501.6131124408145, 530.7293532971163 481.4007838400281, 568.4867860211016 435.67459972338474 M502.0770455310799 520.8711765423938 C532.9996340914829 491.0330720057619, 563.8261465548305 456.73485891982205, 578.944093101421 438.75347060699437 M505.17214906771926 520.8139936610646 C534.3346197834362 486.0200114343996, 562.3566706650555 454.07680956896576, 578.2037964769133 435.69510899315077 M515.1880377148492 520.5836760595287 C540.9266213858184 492.9582956241291, 563.2221352673641 466.3419751760097, 588.2476246910132 437.3356285942831 M514.3032410970172 520.7661366893756 C537.3727958741044 489.1816661482358, 565.4034110597725 459.37574419238183, 587.8321963789836 435.3955786091388 M523.6445340871563 519.52886047504 C545.1223931310615 497.19997121879743, 565.8938477475594 469.3849862507918, 596.4409536820336 436.5281509233075 M523.2647899553864 521.4998758060959 C545.7117663593992 496.3505220876794, 567.464652627305 469.18149467427895, 595.9905373630545 435.68201748248146 M534.2706618452667 520.7506689979122 C549.0825685150655 495.22819878072914, 572.1473010573503 476.76593669427774, 604.0193053632383 435.57085316032953 M533.4886285285687 520.6204386723585 C553.9882222054362 499.19549769240683, 572.6710763114706 475.9143485421259, 605.3951293126089 436.168713225339 M541.6100837779362 520.0154796944068 C569.9512393816896 488.0607277974667, 594.3660681714692 455.1814284238753, 613.8794544989121 434.2385780497789 M543.0293479500469 519.3163316503236 C563.3469195173443 497.79377929826546, 582.7877659333794 471.94294681436133, 616.457679242908 436.1251954717896 M552.3580121494498 519.4746978089538 C576.1986428905861 492.6639688198089, 605.9618316526348 463.9657811467575, 618.7199162998775 441.6396550944765 M550.5619364850822 520.7063031240691 C565.3501496037786 504.896899656677, 579.3007408213482 485.9157646586113, 618.9511072720007 442.2590704580528 M561.7538128735634 520.7802482828148 C580.7953401050241 501.43152879474627, 598.2462779456181 476.24704359717975, 619.0577867133998 453.31255475340697 M559.4400087046573 519.3392110182554 C576.7320631302674 501.74843642561194, 593.3904202089018 484.47738682756585, 620.7920119106293 451.29705022026144 M567.6130193975429 517.947164074974 C586.1331679868954 502.3632274921368, 604.1548454247953 478.9392037582858, 622.3410514521787 463.71492886149514 M568.6934443695146 520.340986119015 C585.2725113946658 503.48471164469356, 600.1359067142677 486.90371296702097, 620.2186040912198 461.4583680142388 M579.8396417311536 520.2426265819159 C591.1719777673997 505.1724958926095, 609.3898328538214 490.315183608813, 619.2200095884795 471.83329475272535 M579.0063224591458 519.688047493731 C590.2051911285041 505.9133043099937, 602.5389887904032 493.8835601447231, 621.4277557620139 473.14883743861776 M590.2824164466613 521.229715637625 C598.0278384726469 511.25083794652113, 605.1182919895923 498.90104487584154, 617.7959367570677 483.94381819858205 M588.9387961717933 520.3354274910361 C595.5051433399263 511.39945527746124, 602.298904573489 503.74369648705886, 621.0892880900042 482.9060103732812 M597.1910566788316 522.4111595135856 C601.0681379568726 516.4261758586657, 608.7836469170563 509.97685299502115, 618.6773351552094 493.05606340804957 M597.6489679244496 521.0448967408132 C602.1739763909445 514.1584107144231, 607.3450247683105 507.9602989414015, 620.1678464609328 494.32730197936183 M606.2725021249302 523.0730849187821 C609.0989572700645 514.0350665831648, 616.1142475680657 514.0426140614495, 618.6361835426698 502.55674302761236 M606.3147643528565 519.513089154195 C613.1876931255518 514.9204197912056, 616.7492704219045 508.9127425503639, 621.4447754792916 504.2981156992426 M615.0476946877936 521.6020836876494 C617.3547456372671 519.2949225656575, 618.4479250844415 518.4613650166613, 621.0137196386477 515.6918004971805 M615.785719481168 521.2068846610024 C617.2476156444061 519.0539653461731, 618.4162674875923 517.503877790678, 620.4802519929606 515.855075537536" style="stroke:var(--dot);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M298.1808773158665 434.86001419199005 C365.61314206762137 438.1609328072444, 428.0921480673105 437.93039481399614, 618.5848869498953 434.85924559985517 M299.5080298823726 435.8655012094123 C410.9876008818457 439.1405642800283, 524.2067615895629 439.1517490555811, 619.9430192139832 435.3354769870899 M618.3840086212382 435.68015578780324 C620.6063515339196 463.88670513909307, 621.6993281338513 490.85151655245573, 622.2539845636114 522.4888700881972 M620.1088671584614 436.5266060884111 C619.8692337977819 459.38963407641273, 621.3833701449566 483.0819523590617, 621.1025793005713 520.6054178739898 M619.4263806806697 518.6404678093177 C520.1775242192028 522.0019568759584, 421.9828036673158 522.1796939834281, 300.7781818400961 521.0694912794927 M619.4504057568815 520.90437282821 C545.3960942738804 522.6601277988037, 471.52168031966795 522.7508942669115, 300.52107843796995 520.266532447224 M298.6425025096163 517.438641689159 C302.27835857688484 504.24785553719846, 302.1132148307472 486.750206935592, 300.1853830387816 435.42466502580794 M299.38340792255474 521.2838296408765 C300.33889658342224 490.1545932485722, 299.7439919073425 461.41640862552447, 298.81073918240145 437.19107370032" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="460" y="472" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:700;fill:var(--ink)">the same predictive</text><text x="460" y="502" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--ink)">μ*  and  σ*²</text></svg><figcaption>Weight space and function space meet through the kernel.</figcaption></figure>

:::remark[The mean is a weighted sum of kernels]
With $\boldsymbol{\alpha} = \mathbf{K}_y^{-1}\boldsymbol{y}$, the posterior mean at any $\boldsymbol{x}$ is $\mu(\boldsymbol{x}) = k(\mathbf{X}, \boldsymbol{x})^\top\boldsymbol{\alpha} = \sum_{i=1}^N \alpha_i\,k(\boldsymbol{x}_i, \boldsymbol{x})$: one kernel-shaped bump per observation. This is the kernel ridge regression predictor with regulariser $\sigma_n^2$, an instance of the representer theorem.
:::

:::remark[The variance does not depend on y]
$\sigma_*^2$ involves only $\mathbf{X}$ and $\boldsymbol{x}_*$, never $\boldsymbol{y}$. The uncertainty after an experiment is known before the experiment is run, which is why GPs are used to decide where to measure next (active learning, Bayesian optimisation).
:::

:::remark[The posterior is a GP]
Applying Step 4 to any finite set of test inputs at once shows $f \mid \boldsymbol{y} \sim \mathcal{GP}(\mu, k_{\mathrm{post}})$ with $k_{\mathrm{post}}(\boldsymbol{x}, \boldsymbol{x}') = k(\boldsymbol{x}, \boldsymbol{x}') - k(\mathbf{X}, \boldsymbol{x})^\top\mathbf{K}_y^{-1}\,k(\mathbf{X}, \boldsymbol{x}')$. Conditioning can be repeated as data arrive.
:::

### Sampling the posterior: Matheron's rule

Proposition 5 describes the posterior as a *distribution*. To draw a posterior sample at $N_*$ test inputs from it, the obvious route is to compute the $N_*\times N_*$ posterior covariance and factor it, at cost $O(N_*^3)$; the result is a vector of values, not a function. Matheron's rule conditions *samples* instead.

:::proposition[Matheron's rule (pathwise conditioning)]
Let $f \sim \mathcal{GP}(0, k)$ and $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2\mathbf{I})$ be independent. For observed data $\boldsymbol{y}$, the random function

$$
\big(f \mid \boldsymbol{y}\big)(\cdot) \;=\; f(\cdot) + k(\cdot, \mathbf{X})\,\mathbf{K}_y^{-1}\big(\boldsymbol{y} - \boldsymbol{f}_{\mathbf{X}} - \boldsymbol{\varepsilon}\big)
$$

is distributed as the GP posterior. In words: draw a prior function, simulate the observations it would have produced, and correct the function by the kernel-weighted mismatch with the real data.
:::

:::proof
**Step 1. Fix test inputs.** A GP is determined by its finite-dimensional distributions, so it suffices to check any finite set of test inputs $\mathbf{X}_*$. Write $\mathbf{K}_{*\mathbf{X}} = k(\mathbf{X}_*, \mathbf{X})$, $\mathbf{K}_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$, and let $\boldsymbol{b} = f(\mathbf{X}_*)$.

**Step 2. Name the simulated observation.** Let $\boldsymbol{a} = \boldsymbol{f}_{\mathbf{X}} + \boldsymbol{\varepsilon}$: what the prior sample would have produced as data. By Steps 1 to 3 of this section (with many test inputs instead of one), $(\boldsymbol{a}, \boldsymbol{b})$ is jointly Gaussian with zero mean and

$$
\boldsymbol{\Sigma}_{aa} = \mathbf{K}_y, \qquad \boldsymbol{\Sigma}_{ba} = \mathbf{K}_{*\mathbf{X}}, \qquad \boldsymbol{\Sigma}_{bb} = \mathbf{K}_{**}.
$$

**Step 3. Write the rule at the test inputs.** With $\mathbf{G} = \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1} = \mathbf{K}_{*\mathbf{X}}\mathbf{K}_y^{-1}$, the rule evaluated at $\mathbf{X}_*$ is

$$
\tilde{\boldsymbol{b}} = \boldsymbol{b} + \mathbf{G}(\boldsymbol{y} - \boldsymbol{a}).
$$

Here $\boldsymbol{y}$ is the real data, a fixed vector; $\boldsymbol{a}$ and $\boldsymbol{b}$ are random.

**Step 4. Regroup.** Collect the random terms:

$$
\tilde{\boldsymbol{b}} = \underbrace{(\boldsymbol{b} - \mathbf{G}\boldsymbol{a})}_{\boldsymbol{r}} + \mathbf{G}\boldsymbol{y}.
$$

$\boldsymbol{r}$ is exactly the residual of Lemma 5, Step 1.

**Step 5. The law of the residual.** By Lemma 5, Steps 4 and 5 (with zero means), $\boldsymbol{r} \sim \mathcal{N}\big(\mathbf{0},\; \mathbf{K}_{**} - \mathbf{K}_{*\mathbf{X}}\mathbf{K}_y^{-1}\mathbf{K}_{*\mathbf{X}}^\top\big)$.

**Step 6. Add the constant.** $\mathbf{G}\boldsymbol{y}$ is fixed, so it shifts the mean only:

$$
\tilde{\boldsymbol{b}} \sim \mathcal{N}\big(\mathbf{K}_{*\mathbf{X}}\mathbf{K}_y^{-1}\boldsymbol{y},\;\; \mathbf{K}_{**} - \mathbf{K}_{*\mathbf{X}}\mathbf{K}_y^{-1}\mathbf{K}_{*\mathbf{X}}^\top\big).
$$

**Step 7. Compare.** This is the posterior at $\mathbf{X}_*$ (Proposition 5, applied to all test inputs at once). Since $\mathbf{X}_*$ was arbitrary, the two processes have the same finite-dimensional distributions, hence the same law.
:::

The proof is the conditioning proof of Lemma 5 read backwards. There, the residual $\boldsymbol{r}$ is independent of the data, so conditioning only moves its mean; here, we build $\boldsymbol{r}$ from a prior sample and add the data's contribution $\mathbf{G}\boldsymbol{y}$ by hand.

:::remark[Why this matters]
- **The sample is a function.** The prior sample $f(\cdot)$ can be drawn once as an explicit function, for example with random Fourier features from Bochner's theorem (Section 5), and the correction $k(\cdot, \mathbf{X})\boldsymbol{v}$ with $\boldsymbol{v} = \mathbf{K}_y^{-1}(\boldsymbol{y} - \boldsymbol{f}_{\mathbf{X}} - \boldsymbol{\varepsilon})$ is a fixed weighted sum of kernels. The result can be evaluated anywhere, at cost linear in the number of test points, and differentiated. Thompson sampling in Bayesian optimisation needs exactly this: maximise one posterior sample.
- **Prior and data are decoupled.** The prior term can use a cheap approximation (random features), while the update uses the exact kernel or inducing points. Wilson et al. (2020) call this decoupled sampling; it keeps the error where the data are, and stays accurate far from them, where random-feature posteriors alone break down.
- **Uncertainty through a simulator.** Push each pathwise sample through any downstream computation (a PDE solver, a planner, a cost function) and the outputs are samples of the propagated uncertainty, with no further Gaussian algebra.

The rule comes from geostatistics, where Matheron used it for conditional simulation of ore deposits. Cosmology rediscovered it for constrained realisations of the early universe (Hoffman and Ribak, 1991). James T. Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky and Marc Deisenroth brought it to machine learning (ICML 2020; JMLR 2021).
:::

The animation uses this rule. Six prior functions are drawn once. As each observation arrives, every curve is updated by Matheron's rule, so you can follow each one as it bends towards the data; the band is the exact posterior of Proposition 5.

<video src="/aboutme/blog/gp/prior-to-posterior.mp4" autoplay muted loop playsinline controls poster="/aboutme/blog/gp/prior-to-posterior.jpg"></video>

## 7. The marginal likelihood

The hyperparameters $\boldsymbol{\theta} = (\sigma_f, \ell, \sigma_n)$ are chosen by maximising the probability of the data with $f$ integrated out.

:::proposition[Log marginal likelihood]

$$
\log p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\theta})
= \underbrace{-\tfrac12\,\boldsymbol{y}^\top\mathbf{K}_y^{-1}\boldsymbol{y}}_{\text{data fit}}
\;\underbrace{-\;\tfrac12\log\lvert\mathbf{K}_y\rvert}_{\text{complexity}}
\;-\;\tfrac{N}{2}\log 2\pi .
$$
:::

:::proof
*Route 1: as a convolution.*

**Step 1. Marginalise.** Integrate the latent values out of the joint density:

$$
p(\boldsymbol{y} \mid \mathbf{X}) = \int p(\boldsymbol{y} \mid \boldsymbol{f}_{\mathbf{X}})\,p(\boldsymbol{f}_{\mathbf{X}} \mid \mathbf{X})\,d\boldsymbol{f}_{\mathbf{X}} = \int \mathcal{N}(\boldsymbol{y} \mid \boldsymbol{f}_{\mathbf{X}}, \sigma_n^2\mathbf{I})\;\mathcal{N}(\boldsymbol{f}_{\mathbf{X}} \mid \mathbf{0}, \mathbf{K})\;d\boldsymbol{f}_{\mathbf{X}} .
$$

**Step 2. Recognise a sum.** This is the density of $\boldsymbol{y} = \boldsymbol{f}_{\mathbf{X}} + \boldsymbol{\varepsilon}$ with independent $\boldsymbol{f}_{\mathbf{X}} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$ and $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2\mathbf{I})$, the convolution of prior and noise.

**Step 3. Apply Lemma 3.** $\boldsymbol{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{K} + \sigma_n^2\mathbf{I}) = \mathcal{N}(\mathbf{0}, \mathbf{K}_y)$.

**Step 4. Take the log** of the density in Notation with $D = N$:

$$
\log\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{K}_y) = -\tfrac{N}{2}\log 2\pi - \tfrac12\log\lvert\mathbf{K}_y\rvert - \tfrac12\,\boldsymbol{y}^\top\mathbf{K}_y^{-1}\boldsymbol{y} .
$$

*Route 2: as a product (a check, and a bonus).*

**Step 5. Swap arguments in the likelihood.** By the symmetry in Step 5 of Lemma 4, $\mathcal{N}(\boldsymbol{y} \mid \boldsymbol{f}_{\mathbf{X}}, \sigma_n^2\mathbf{I}) = \mathcal{N}(\boldsymbol{f}_{\mathbf{X}} \mid \boldsymbol{y}, \sigma_n^2\mathbf{I})$, so the integrand of Step 1 is a product of two Gaussian densities *in the same variable* $\boldsymbol{f}_{\mathbf{X}}$.

**Step 6. Apply Lemma 4** with $(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) = (\boldsymbol{y}, \sigma_n^2\mathbf{I})$ and $(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) = (0, \mathbf{K})$:

$$
\mathcal{N}(\boldsymbol{f}_{\mathbf{X}} \mid \boldsymbol{y}, \sigma_n^2\mathbf{I})\;\mathcal{N}(\boldsymbol{f}_{\mathbf{X}} \mid \mathbf{0}, \mathbf{K}) = \mathcal{N}(\boldsymbol{y} \mid \mathbf{0},\, \sigma_n^2\mathbf{I} + \mathbf{K})\;\mathcal{N}(\boldsymbol{f}_{\mathbf{X}} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c).
$$

**Step 7. Integrate over $\boldsymbol{f}_{\mathbf{X}}$.** The second factor integrates to one, leaving $p(\boldsymbol{y} \mid \mathbf{X}) = \mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{K}_y)$: the same answer as Step 3. (Route 2 is Lemma 8 with $\boldsymbol{x} = \boldsymbol{f}_{\mathbf{X}}$, $\mathbf{C} = \mathbf{I}$, $\boldsymbol{d} = \mathbf{0}$.)
:::

:::remark[Bayes' rule in one identity]
Route 2 delivers more than the evidence. Prior times likelihood equals evidence times posterior, so the factor $\mathcal{N}(\boldsymbol{f}_{\mathbf{X}} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$ in Step 6 is the posterior over the latent values at the training inputs, with $\boldsymbol{\Sigma}_c = (\mathbf{K}^{-1} + \sigma_n^{-2}\mathbf{I})^{-1}$ and $\boldsymbol{\mu}_c = \sigma_n^{-2}\boldsymbol{\Sigma}_c\,\boldsymbol{y}$. The product of Gaussians *is* Bayes' rule: its shape is the posterior and its normaliser is the marginal likelihood.
:::

The two terms pull in opposite directions. A short length-scale fits the data closely (large data-fit term), but makes $\mathbf{K}_y$ close to diagonal with a large determinant (large penalty). The maximiser balances them. Move the slider:

<figure class="fig" data-evidence>
  <canvas data-figure="evidence" aria-label="Posterior fit and log marginal likelihood as the length-scale changes."></canvas>
  <figcaption><label class="slider">length-scale (log) <input type="range" min="-2.3" max="0" step="0.01" value="-1" data-ell /></label> <span class="mono" data-out></span></figcaption>
</figure>

:::remark[Why the marginal likelihood generalises]
By the chain rule of probability,

$$
\log p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\theta}) = \sum_{n=1}^{N}\log p(y_n \mid y_1, \dots, y_{n-1}, \mathbf{X}, \boldsymbol{\theta}).
$$

Each term scores the prediction of an observation that has not yet been conditioned on. The marginal likelihood is therefore a sum of one-step-ahead predictive scores, a built-in measure of generalisation: an overly flexible kernel fits each point once it is seen, but predicts the next one poorly. (This reading follows Marc Deisenroth's GP lectures.)
:::

:::proposition[Gradient]
With $\boldsymbol{\alpha} = \mathbf{K}_y^{-1}\boldsymbol{y}$,

$$
\frac{\partial}{\partial\theta_j}\log p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\theta}) = \tfrac12\operatorname{tr}\!\Big(\big(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - \mathbf{K}_y^{-1}\big)\frac{\partial\mathbf{K}_y}{\partial\theta_j}\Big).
$$
:::

:::proof
Write $\partial$ for $\partial/\partial\theta_j$. The constant $-\tfrac N2\log 2\pi$ has zero derivative.

**Step 1. Differentiate the data-fit term** (Lemma 7, inverse):

$$
\partial\big(-\tfrac12\,\boldsymbol{y}^\top\mathbf{K}_y^{-1}\boldsymbol{y}\big) = -\tfrac12\,\boldsymbol{y}^\top(\partial\mathbf{K}_y^{-1})\,\boldsymbol{y} = \tfrac12\,\boldsymbol{y}^\top\mathbf{K}_y^{-1}(\partial\mathbf{K}_y)\mathbf{K}_y^{-1}\boldsymbol{y} = \tfrac12\,\boldsymbol{\alpha}^\top(\partial\mathbf{K}_y)\,\boldsymbol{\alpha} ,
$$

using $\boldsymbol{y}^\top\mathbf{K}_y^{-1} = (\mathbf{K}_y^{-1}\boldsymbol{y})^\top = \boldsymbol{\alpha}^\top$ (symmetry of $\mathbf{K}_y^{-1}$).

**Step 2. Write it as a trace.** A scalar equals its own trace, and $\operatorname{tr}(\mathbf{U}\mathbf{V}) = \operatorname{tr}(\mathbf{V}\mathbf{U})$:

$$
\boldsymbol{\alpha}^\top(\partial\mathbf{K}_y)\,\boldsymbol{\alpha} = \operatorname{tr}\big(\boldsymbol{\alpha}^\top(\partial\mathbf{K}_y)\,\boldsymbol{\alpha}\big) = \operatorname{tr}\big(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top\,\partial\mathbf{K}_y\big).
$$

**Step 3. Differentiate the complexity term** (Lemma 7, log-determinant):

$$
\partial\big(-\tfrac12\log\lvert\mathbf{K}_y\rvert\big) = -\tfrac12\operatorname{tr}\big(\mathbf{K}_y^{-1}\,\partial\mathbf{K}_y\big).
$$

**Step 4. Add Steps 2 and 3.** The trace is linear:

$$
\tfrac12\operatorname{tr}\big(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top\partial\mathbf{K}_y\big) - \tfrac12\operatorname{tr}\big(\mathbf{K}_y^{-1}\partial\mathbf{K}_y\big) = \tfrac12\operatorname{tr}\big((\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - \mathbf{K}_y^{-1})\,\partial\mathbf{K}_y\big).
$$
:::

Optimise over $\log\ell$, $\log\sigma_f$ and $\log\sigma_n$ to keep them positive. The objective is non-convex; distinct optima correspond to distinct explanations of the data (a short length-scale with low noise, or a long length-scale with high noise), so use several restarts.

## 8. Implementation

Never form $\mathbf{K}_y^{-1}$ explicitly. Factor $\mathbf{K}_y = \mathbf{L}\mathbf{L}^\top$ once, then every quantity is a triangular solve:

1. $\boldsymbol{\alpha} = \mathbf{L}^{-\top}(\mathbf{L}^{-1}\boldsymbol{y})$, since $\mathbf{K}_y^{-1} = (\mathbf{L}\mathbf{L}^\top)^{-1} = \mathbf{L}^{-\top}\mathbf{L}^{-1}$.
2. $\mu_* = \boldsymbol{k}_*^\top\boldsymbol{\alpha}$.
3. $\boldsymbol{v} = \mathbf{L}^{-1}\boldsymbol{k}_*$, so that $\boldsymbol{k}_*^\top\mathbf{K}_y^{-1}\boldsymbol{k}_* = \boldsymbol{k}_*^\top\mathbf{L}^{-\top}\mathbf{L}^{-1}\boldsymbol{k}_* = \boldsymbol{v}^\top \boldsymbol{v}$ and $\sigma_*^2 = k_{**} - \boldsymbol{v}^\top \boldsymbol{v}$.
4. $\log\lvert\mathbf{K}_y\rvert = 2\sum_i \log L_{ii}$, since $\lvert\mathbf{L}\mathbf{L}^\top\rvert = \lvert\mathbf{L}\rvert^2$ and the determinant of a triangular matrix is the product of its diagonal.

```python
import numpy as np
from scipy.linalg import cho_solve, solve_triangular

def rbf(A, B, ell=0.2, sf=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return sf**2 * np.exp(-0.5 * d2 / ell**2)

def gp_predict(X, y, Xs, ell=0.2, sf=1.0, sn=0.1, jitter=1e-8):
    N = len(X)
    L = np.linalg.cholesky(rbf(X, X, ell, sf) + (sn**2 + jitter) * np.eye(N))
    alpha = cho_solve((L, True), y)
    Ks = rbf(X, Xs, ell, sf)                     # N x N*
    mu = Ks.T @ alpha
    V = solve_triangular(L, Ks, lower=True)
    var = sf**2 - (V**2).sum(0)                  # diagonal of the posterior covariance
    lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * N * np.log(2 * np.pi)
    return mu, var, lml
```

:::note[Practical defaults]
Adapted from Marc Deisenroth's lectures:
- Standardise inputs and targets; initialise $\ell \approx 0.5$ and $\sigma_f \approx 1$.
- Initialise the noise high, $\sigma_n \approx 0.5\,\sigma_f$, even for clean data; the optimiser moves more easily and can reduce it.
- Use random restarts.
- If Cholesky fails repeatedly, penalise large signal-to-noise ratios $\sigma_f/\sigma_n$ rather than only increasing the jitter.
:::

**Cost.** $O(N^3)$ time and $O(N^2)$ memory. Standard routes to scale:
- **Inducing points**: $M \ll N$ pseudo-inputs, $O(NM^2)$ ([deep GP post](/aboutme/blog/deep-gaussian-processes/)).
- **Random features**: an explicit $D$-dimensional approximation of $k$, returning to the weight view at $O(ND^2)$ ([From Fourier to wavelets](/aboutme/blog/fourier-to-wavelets/)).
- **Structure and iterative solvers**: Kronecker or Toeplitz structure on grids; conjugate gradients with fast matrix–vector products.

## 9. Limitations

- **Non-Gaussian likelihoods** (classification, counts) break conjugacy; the posterior requires approximation (Laplace, expectation propagation, variational inference).
- **Stationarity.** A single length-scale is often wrong for real signals; non-stationary kernels, input warping and multi-scale features address this.
- **High-dimensional inputs.** Euclidean distances between raw inputs are rarely meaningful; deep kernels and deep GPs learn a representation.
- **Function-valued inputs and outputs.** Mapping one function to another (for example, PDE coefficients to solutions) requires operator-valued kernels, the subject of the next post in this series.

## Questions

:::question
Why does the posterior variance not depend on the observed values $\boldsymbol{y}$?
:::

:::answer
In Lemma 5 the conditional covariance $\boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba}\boldsymbol{\Sigma}_{aa}^{-1}\boldsymbol{\Sigma}_{ab}$ involves only covariances, and for a GP these depend only on the inputs.
:::

:::question
You multiply two Gaussian densities in the same variable and, separately, add two independent Gaussian variables. Which result is narrower, and why?
:::

:::answer
The product is narrower: its precision is the *sum* of the two precisions (Lemma 4), so its covariance is smaller than either. The sum is wider: its covariance is the *sum* of the two covariances (Lemma 3).
:::

:::question
Which kernel reduces a GP to Bayesian linear regression with features $\boldsymbol{\phi}$?
:::

:::answer
$k(\boldsymbol{x}, \boldsymbol{x}') = \boldsymbol{\phi}(\boldsymbol{x})^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(\boldsymbol{x}')$ (Proposition 3). With $\boldsymbol{\phi}(\boldsymbol{x}) = \boldsymbol{x}$ and $\boldsymbol{\Sigma}_w = \mathbf{I}$ this is the linear kernel.
:::

:::question
The optimiser returns a very short length-scale and near-zero noise. What has happened, and what should you check?
:::

:::answer
The model interpolates the data and treats noise as signal. Compare with the optimum at a longer length-scale and larger noise, place a prior on $\sigma_n$, and use several initialisations.
:::

:::question
How do you multiply $\mathcal{N}(\boldsymbol{y} \mid \mathbf{C}\boldsymbol{x} + \boldsymbol{d}, \mathbf{S})$ by $\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)$, when the two densities are in different variables?
:::

:::answer
Read the product as a joint $p(\boldsymbol{x}, \boldsymbol{y})$, compute the joint by stacking (Lemma 2), then factor it the other way as $p(\boldsymbol{y})\,p(\boldsymbol{x} \mid \boldsymbol{y})$ (Lemma 5). Alternatively, read the likelihood as an information-form Gaussian in $\boldsymbol{x}$ and add precisions. Both routes are Lemma 8.
:::

:::question
Matheron's rule never computes the posterior covariance. Where does it come from?
:::

:::answer
From the prior sample itself. The residual $\boldsymbol{b} - \mathbf{G}\boldsymbol{a}$ of a prior draw already has the posterior covariance (Lemma 5, Step 5); the rule only adds the deterministic shift $\mathbf{G}\boldsymbol{y}$.
:::

## References

- C. E. Rasmussen, C. K. I. Williams. [*Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/). MIT Press, 2006. Chapters 2 and 5, and Appendix A for the Gaussian identities.
- M. P. Deisenroth, A. A. Faisal, C. S. Ong. [*Mathematics for Machine Learning*](https://mml-book.github.io/). Cambridge University Press, 2020. Chapters 6 and 9.
- M. P. Deisenroth. [*Gaussian processes*](https://www.deisenroth.cc/teaching/2023-24/ml-seminar/lecture_gaussian_processes.pdf). Lecture slides, UCL, 2023/24.
- K. B. Petersen, M. S. Pedersen. *The Matrix Cookbook*. 2012. For matrix identities and derivatives.
- C. M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006. Chapters 2.3 and 6.
- D. J. C. MacKay. *Introduction to Gaussian processes*. 1998.
- J. Görtler, R. Kehlbeck, O. Deussen. [*A Visual Exploration of Gaussian Processes*](https://distill.pub/2019/visual-exploration-gaussian-processes/). Distill, 2019.
- J. T. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, M. P. Deisenroth. [*Efficiently sampling functions from Gaussian process posteriors*](https://arxiv.org/abs/2002.09309). ICML 2020.
- J. T. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, M. P. Deisenroth. [*Pathwise conditioning of Gaussian processes*](https://arxiv.org/abs/2011.04026). JMLR, 2021.
- Y. Hoffman, E. Ribak. *Constrained realizations of Gaussian fields: a simple algorithm*. The Astrophysical Journal Letters, 1991.
- A. Doucet. *A note on efficient conditional simulation of Gaussian distributions*. Technical note, 2010.
- F. Lindgren, H. Rue, J. Lindström. *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*. Journal of the Royal Statistical Society B, 2011.
