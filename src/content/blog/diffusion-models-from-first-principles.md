---
title: "Diffusion models from first principles"
description: "Why adding noise and learning to remove it generates data. The forward process and its reverse step, why predicting the noise is a likelihood bound, Tweedie's formula and denoising score matching, the SDE and the probability-flow ODE, why sampling needs many steps, and what guidance really does. Every step derived, on a toy where everything is exact, next to a diffusion model trained on it."
date: 2026-10-29
updated: 2026-10-29
cover: /blog/thumbs/diffusion.png
tags: [generative-models, diffusion, score, sde, tutorial]
---

A diffusion model learns to generate data by learning to *remove noise*. Train a network to clean up slightly noisy data, start from pure noise, and clean it up a little at a time: what comes out looks like the data. Below is the series' toy, a ring of eight Gaussians. Slide $t$ to add noise; the arrows show the direction the network must learn, the *score*. Then press **Sample** to run the process backwards from pure noise:

<figure class="fig" data-difflab>
  <canvas data-figure="difflab" data-net="/aboutme/blog/diffusion/net.json" aria-label="A diffusion lab on a ring of eight Gaussians: forward noising, the score field, and sampling from noise with an ODE or SDE sampler, using the exact score or a trained network."></canvas>
  <div class="controls">
    <label class="slider">noise level t <input type="range" min="0" max="1" step="0.01" value="0.35" data-t /></label>
    <label><input type="checkbox" data-field checked /> score field</label>
  </div>
  <div class="controls">
    <div class="seg" role="group" aria-label="Sampler">
      <button type="button" data-kind="ode" class="on">ODE (DDIM)</button>
      <button type="button" data-kind="sde">SDE (DDPM)</button>
    </div>
    <div class="seg" role="group" aria-label="Score">
      <button type="button" data-src="exact" class="on">exact score</button>
      <button type="button" data-src="net">trained network</button>
    </div>
    <label class="slider">steps <input type="range" min="1" max="100" step="1" value="30" data-steps /></label>
    <span class="mono" data-nval>30</span>
    <button type="button" data-run>Sample</button>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

*What to try.* Slide $t$ up from 0. The ring blurs, shrinks towards the centre, and becomes a single Gaussian at $t = 1$; the arrows (the score, scaled by $\sigma_t$) always point back towards where data are more likely. Now sample. With **one** step, every point lands at the centre of the ring: the best one-step guess is the average (Section 4). With the exact score, 4 steps put about 72% of the samples on a mode, 8 steps about 96%, and 30 steps 99%. Switch to the **trained network**: with 30 steps it puts 98% on a mode. Compare **ODE** and **SDE**: the ODE's paths are smooth, the SDE's jitter, and both end at the same distribution (Section 5).

This is the second post of the series on deep generative models, after [the map](/aboutme/blog/generative-models-one-map/), which introduced the score, Langevin dynamics and score matching. The toy makes this post unusual. A noised Gaussian mixture is still a Gaussian mixture, so every quantity a diffusion model tries to learn has a closed form here, and each result can be checked exactly, next to a network I trained on the same data. The post answers six questions:

1. **Why does adding noise and learning to remove it generate data at all?** (Sections 1 and 2.)
2. **Why is "predict the noise" a likelihood objective?** (Section 3.)
3. **What is the network really learning?** Noise, clean data, score and velocity are one object. (Section 4.)
4. **How can a deterministic ODE sample the same distribution as a noisy process?** (Section 5.)
5. **Why does sampling take so many steps**, even with a perfect network? (Section 6.)
6. **What does guidance actually do?** (Section 7.)

Every result has a folded proof in numbered steps. Section 8 is the trained model, and Section 9 what has happened since 2022.

## Notation

Plain italic letters are scalars, bold italic lower case are vectors. Time runs from data to noise.

| Symbol | Meaning |
|---|---|
| $\boldsymbol{x}_0 \sim p_{\mathrm{data}}$ | a data point; the data are at $t = 0$ |
| $\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_0 + \sigma_t\boldsymbol{\varepsilon}$, $\ \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | the noised point at time $t \in [0, 1]$ |
| $\alpha_t$, $\sigma_t$ | signal and noise scales; $\alpha_0 = 1$, $\sigma_0 = 0$, and $\alpha_t^2 + \sigma_t^2 = 1$ (variance preserving) |
| $p_t$ | the distribution of $\boldsymbol{x}_t$; $p_0 = p_{\mathrm{data}}$ |
| $\boldsymbol{s}(\boldsymbol{x}, t) = \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$ | the score of $p_t$ |
| $\hat{\boldsymbol{x}}_0$, $\hat{\boldsymbol{\varepsilon}}$ | a network's estimates of $\boldsymbol{x}_0$ and $\boldsymbol{\varepsilon}$ from $\boldsymbol{x}_t$ |
| $\beta_k$, $\bar\alpha_k = \prod_{j \le k}(1 - \beta_j)$ | the discrete chain's step variances, and its cumulative signal |
| $\dot\alpha_t = d\alpha_t/dt$ | time derivatives |

The discrete and continuous notations match through $\alpha_t \leftrightarrow \sqrt{\bar\alpha_k}$ and $\sigma_t \leftrightarrow \sqrt{1 - \bar\alpha_k}$. The toy uses $\alpha_t = \cos(\pi t/2)$ and $\sigma_t = \sin(\pi t/2)$, so $t = 1$ is pure noise.

## 1. The forward process: adding noise

Ho, Jain and Abbeel (2020), following Sohl-Dickstein et al. (2015), add noise in $K$ small steps:

$$
q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1}) = \mathcal{N}\big(\boldsymbol{x}_k \mid \sqrt{1 - \beta_k}\,\boldsymbol{x}_{k-1},\ \beta_k\mathbf{I}\big),
$$

shrinking the signal slightly and adding a little noise each time. Nothing here is learned. The key fact is that you never need to run the chain:

:::proposition[Jumping straight to step $k$]
$$
q(\boldsymbol{x}_k \mid \boldsymbol{x}_0) = \mathcal{N}\big(\boldsymbol{x}_k \mid \sqrt{\bar\alpha_k}\,\boldsymbol{x}_0,\ (1 - \bar\alpha_k)\mathbf{I}\big), \qquad \bar\alpha_k = \prod_{j=1}^{k}(1 - \beta_j).
$$
:::

:::proof
**Step 1. Write one step as a sum.** $\boldsymbol{x}_k = \sqrt{1 - \beta_k}\,\boldsymbol{x}_{k-1} + \sqrt{\beta_k}\,\boldsymbol{\xi}_k$ with $\boldsymbol{\xi}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ independent of everything before.

**Step 2. Induction hypothesis.** Suppose $\boldsymbol{x}_{k-1} = \sqrt{\bar\alpha_{k-1}}\,\boldsymbol{x}_0 + \sqrt{1 - \bar\alpha_{k-1}}\,\boldsymbol{\eta}$ with $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ independent of $\boldsymbol{\xi}_k$. It holds for $k = 1$ with $\bar\alpha_0 = 1$.

**Step 3. Substitute.** $\boldsymbol{x}_k = \sqrt{1 - \beta_k}\sqrt{\bar\alpha_{k-1}}\,\boldsymbol{x}_0 + \big[\sqrt{(1 - \beta_k)(1 - \bar\alpha_{k-1})}\,\boldsymbol{\eta} + \sqrt{\beta_k}\,\boldsymbol{\xi}_k\big]$. The coefficient of $\boldsymbol{x}_0$ is $\sqrt{\bar\alpha_k}$.

**Step 4. Add the two independent Gaussians in brackets.** Their sum is Gaussian with mean zero and variance $(1 - \beta_k)(1 - \bar\alpha_{k-1}) + \beta_k = 1 - (1 - \beta_k)\bar\alpha_{k-1} = 1 - \bar\alpha_k$ (variances of independent terms add).
:::

A simulation of 300 steps with the usual schedule ($\beta_k$ from $10^{-4}$ to $0.02$) gives a residual variance of $0.6051$ against the formula's $0.6036$. In continuous time the same statement is simply

$$
\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_0 + \sigma_t\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

with any decreasing $\alpha_t$ from 1 to (near) 0. The distribution of $\boldsymbol{x}_t$ is the data distribution, shrunk by $\alpha_t$ and blurred by Gaussian noise of variance $\sigma_t^2$:

$$
p_t(\boldsymbol{x}) = \int\mathcal{N}\big(\boldsymbol{x} \mid \alpha_t\boldsymbol{x}_0,\ \sigma_t^2\mathbf{I}\big)\,p_{\mathrm{data}}(\boldsymbol{x}_0)\,d\boldsymbol{x}_0 .
$$

For the toy, $p_{\mathrm{data}} = \frac18\sum_k\mathcal{N}(\boldsymbol{\mu}_k, s^2\mathbf{I})$ with $s = 0.1$. Each component, shrunk and blurred, is $\mathcal{N}(\alpha_t\boldsymbol{\mu}_k, (\alpha_t^2s^2 + \sigma_t^2)\mathbf{I})$ (a scaled Gaussian plus an independent Gaussian), so $p_t$ is again a mixture of eight Gaussians. At $t = 1$, $\alpha_1 = 0$ and $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ exactly: all trace of the data is gone. That is the starting point for generation.

## 2. Reversing one step: the posterior is Gaussian

To generate, we must go backwards: from $\boldsymbol{x}_k$ to $\boldsymbol{x}_{k-1}$. In general $q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k)$ is complicated, since it depends on the whole data distribution. But if we also knew the clean point $\boldsymbol{x}_0$, the backward step would be a simple Gaussian.

:::proposition[The reverse step, given the clean point]
$$
q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0) = \mathcal{N}\big(\boldsymbol{x}_{k-1} \mid \tilde{\boldsymbol{\mu}}_k,\ \tilde\beta_k\mathbf{I}\big), \qquad
\tilde\beta_k = \frac{(1 - \bar\alpha_{k-1})\,\beta_k}{1 - \bar\alpha_k},
$$

$$
\tilde{\boldsymbol{\mu}}_k = \frac{\sqrt{1 - \beta_k}\,(1 - \bar\alpha_{k-1})}{1 - \bar\alpha_k}\,\boldsymbol{x}_k + \frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1 - \bar\alpha_k}\,\boldsymbol{x}_0 .
$$
:::

:::proof
**Step 1. Bayes' rule.** Because the chain is Markov, $q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1}, \boldsymbol{x}_0) = q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1})$, so

$$
q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0) \propto q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1})\,q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_0)
$$

as a function of $\boldsymbol{x}_{k-1}$.

**Step 2. Both factors as Gaussians in $\boldsymbol{x}_{k-1}$.** Write $c = 1 - \beta_k$. The first factor, $\exp\!\big(-\lVert\boldsymbol{x}_k - \sqrt{c}\,\boldsymbol{x}_{k-1}\rVert^2/(2\beta_k)\big)$, has precision $c/\beta_k$ and linear coefficient $\sqrt{c}\,\boldsymbol{x}_k/\beta_k$ (read off the quadratic and linear terms in $\boldsymbol{x}_{k-1}$). The second has precision $1/(1 - \bar\alpha_{k-1})$ and linear coefficient $\sqrt{\bar\alpha_{k-1}}\,\boldsymbol{x}_0/(1 - \bar\alpha_{k-1})$.

**Step 3. Precisions add** (product of Gaussians in the same variable; the primer's Lemma 4):

$$
\frac{c}{\beta_k} + \frac{1}{1 - \bar\alpha_{k-1}} = \frac{c(1 - \bar\alpha_{k-1}) + \beta_k}{\beta_k(1 - \bar\alpha_{k-1})} = \frac{1 - \bar\alpha_k}{\beta_k(1 - \bar\alpha_{k-1})},
$$

using $c(1 - \bar\alpha_{k-1}) + \beta_k = 1 - c\,\bar\alpha_{k-1} = 1 - \bar\alpha_k$. Its inverse is $\tilde\beta_k$.

**Step 4. The mean** is the variance times the sum of the linear coefficients:

$$
\tilde{\boldsymbol{\mu}}_k = \tilde\beta_k\Big(\frac{\sqrt{c}}{\beta_k}\boldsymbol{x}_k + \frac{\sqrt{\bar\alpha_{k-1}}}{1 - \bar\alpha_{k-1}}\boldsymbol{x}_0\Big),
$$

which multiplies out to the stated coefficients.
:::

So the whole difficulty of generation is in not knowing $\boldsymbol{x}_0$. A diffusion model *estimates* it from $\boldsymbol{x}_k$ and plugs the estimate in: $p_\theta(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k) = q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \hat{\boldsymbol{x}}_0(\boldsymbol{x}_k))$. Section 4 shows what the best estimate is. First, why this is a principled way to fit a likelihood.

It is convenient to write the mean through the noise instead of $\boldsymbol{x}_0$:

:::corollary[The reverse mean in terms of the noise]
If $\boldsymbol{x}_k = \sqrt{\bar\alpha_k}\,\boldsymbol{x}_0 + \sqrt{1 - \bar\alpha_k}\,\boldsymbol{\varepsilon}$, then $\tilde{\boldsymbol{\mu}}_k = \dfrac{1}{\sqrt{1 - \beta_k}}\Big(\boldsymbol{x}_k - \dfrac{\beta_k}{\sqrt{1 - \bar\alpha_k}}\,\boldsymbol{\varepsilon}\Big)$.
:::

:::proof
**Step 1. Substitute** $\boldsymbol{x}_0 = (\boldsymbol{x}_k - \sqrt{1 - \bar\alpha_k}\,\boldsymbol{\varepsilon})/\sqrt{\bar\alpha_k}$ into the mean, using $\sqrt{\bar\alpha_k} = \sqrt{1 - \beta_k}\sqrt{\bar\alpha_{k-1}}$.

**Step 2. The coefficient of $\boldsymbol{x}_k$** becomes $\frac{\sqrt{1 - \beta_k}(1 - \bar\alpha_{k-1})}{1 - \bar\alpha_k} + \frac{\beta_k}{(1 - \bar\alpha_k)\sqrt{1 - \beta_k}} = \frac{(1 - \beta_k)(1 - \bar\alpha_{k-1}) + \beta_k}{(1 - \bar\alpha_k)\sqrt{1 - \beta_k}} = \frac{1}{\sqrt{1 - \beta_k}}$.

**Step 3. The coefficient of $\boldsymbol{\varepsilon}$** is $-\frac{\beta_k\sqrt{1 - \bar\alpha_k}}{(1 - \bar\alpha_k)\sqrt{1 - \beta_k}} = -\frac{\beta_k}{\sqrt{1 - \bar\alpha_k}\sqrt{1 - \beta_k}}$.
:::

## 3. Why "predict the noise" is a likelihood objective

The model is a Markov chain run backwards: $p(\boldsymbol{x}_K) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, then $p_\theta(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k)$ for $k = K, \dots, 1$. The noisy states $\boldsymbol{x}_{1:K}$ are latent variables, and the forward chain $q$ is a natural variational posterior for them. The ELBO of [the map](/aboutme/blog/generative-models-one-map/), Section 4, then becomes a sum of simple terms.

:::theorem[The diffusion ELBO (Sohl-Dickstein et al., 2015; Ho et al., 2020)]
$$
-\log p_\theta(\boldsymbol{x}_0) \le \mathbb{E}_q\Big[\underbrace{\mathrm{KL}\big(q(\boldsymbol{x}_K \mid \boldsymbol{x}_0) \,\Vert\, p(\boldsymbol{x}_K)\big)}_{\text{no parameters}} + \sum_{k=2}^{K}\mathrm{KL}\big(q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0) \,\Vert\, p_\theta(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k)\big) - \log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1)\Big].
$$
:::

:::proof
**Step 1. The ELBO with latents $\boldsymbol{x}_{1:K}$.** $\log p_\theta(\boldsymbol{x}_0) \ge \mathbb{E}_{q(\boldsymbol{x}_{1:K} \mid \boldsymbol{x}_0)}\big[\log p_\theta(\boldsymbol{x}_{0:K}) - \log q(\boldsymbol{x}_{1:K} \mid \boldsymbol{x}_0)\big]$.

**Step 2. Factor the model.** $p_\theta(\boldsymbol{x}_{0:K}) = p(\boldsymbol{x}_K)\prod_{k=1}^{K}p_\theta(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k)$.

**Step 3. Factor $q$ backwards.** $q(\boldsymbol{x}_{1:K} \mid \boldsymbol{x}_0) = \prod_{k=1}^{K}q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1})$. By Bayes' rule and the Markov property, for $k \ge 2$,

$$
q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1}) = q(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0)\,q(\boldsymbol{x}_k \mid \boldsymbol{x}_0)}{q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_0)}.
$$

**Step 4. Telescope.** In the product over $k = 2, \dots, K$, the ratios $q(\boldsymbol{x}_k \mid \boldsymbol{x}_0)/q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_0)$ cancel down to $q(\boldsymbol{x}_K \mid \boldsymbol{x}_0)/q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0)$. With the $k = 1$ factor $q(\boldsymbol{x}_1 \mid \boldsymbol{x}_0)$,

$$
q(\boldsymbol{x}_{1:K} \mid \boldsymbol{x}_0) = q(\boldsymbol{x}_K \mid \boldsymbol{x}_0)\prod_{k=2}^{K}q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0).
$$

**Step 5. Pair the factors.** The log-ratio of Step 1 becomes $\log\frac{p(\boldsymbol{x}_K)}{q(\boldsymbol{x}_K \mid \boldsymbol{x}_0)} + \sum_{k \ge 2}\log\frac{p_\theta(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k)}{q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0)} + \log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1)$. Taking expectations under $q$, each paired ratio becomes minus a KL divergence. Multiply by $-1$.
:::

Each KL compares two Gaussians in $\boldsymbol{x}_{k-1}$. Give the model the same variance $\tilde\beta_k$ and let it predict only the mean. Then:

:::proposition[Each term is a weighted noise-prediction error]
Parametrise the model's mean like the corollary of Section 2, with a network $\hat{\boldsymbol{\varepsilon}}_\theta(\boldsymbol{x}_k, k)$ in place of the true noise. Then

$$
\mathrm{KL}\big(q(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k, \boldsymbol{x}_0) \,\Vert\, p_\theta(\boldsymbol{x}_{k-1} \mid \boldsymbol{x}_k)\big) = \lambda_k\,\big\lVert\boldsymbol{\varepsilon} - \hat{\boldsymbol{\varepsilon}}_\theta(\boldsymbol{x}_k, k)\big\rVert^2, \qquad \lambda_k = \frac{\beta_k^2}{2\tilde\beta_k(1 - \beta_k)(1 - \bar\alpha_k)}.
$$
:::

:::proof
**Step 1. KL between Gaussians with equal covariance $\tilde\beta_k\mathbf{I}$.** The trace and log-determinant terms of the Gaussian KL cancel, leaving $\lVert\tilde{\boldsymbol{\mu}}_k - \boldsymbol{\mu}_\theta\rVert^2/(2\tilde\beta_k)$.

**Step 2. The difference of means.** Both are $\frac{1}{\sqrt{1 - \beta_k}}\big(\boldsymbol{x}_k - \frac{\beta_k}{\sqrt{1 - \bar\alpha_k}}\cdot\big)$, one with $\boldsymbol{\varepsilon}$ and one with $\hat{\boldsymbol{\varepsilon}}_\theta$, so $\tilde{\boldsymbol{\mu}}_k - \boldsymbol{\mu}_\theta = \frac{\beta_k}{\sqrt{1 - \beta_k}\sqrt{1 - \bar\alpha_k}}(\hat{\boldsymbol{\varepsilon}}_\theta - \boldsymbol{\varepsilon})$.

**Step 3. Square and divide by $2\tilde\beta_k$.**
:::

Training therefore samples a data point, a step $k$ and a noise $\boldsymbol{\varepsilon}$, forms $\boldsymbol{x}_k$, and regresses $\hat{\boldsymbol{\varepsilon}}_\theta(\boldsymbol{x}_k, k)$ on $\boldsymbol{\varepsilon}$. Ho et al. (2020) dropped the weights $\lambda_k$ and found that the unweighted "simple" loss gives better samples. Kingma and Gao (2023) showed that such reweighted losses are still ELBOs, of the data with Gaussian noise added as augmentation, so the simple loss remains a likelihood-based objective with a different emphasis across noise levels. The answer to question 2 is: predicting the noise *is* fitting a likelihood bound, term by term.

## 4. What the network learns: denoising is score matching

### Tweedie's formula

What is the best guess of the clean point? For squared error, the conditional mean, and that conditional mean is tied to the score of the noisy distribution.

:::proposition[The best guess under squared error]
Over all functions $\hat{\boldsymbol{x}}_0(\cdot)$, the error $\mathbb{E}\lVert\boldsymbol{x}_0 - \hat{\boldsymbol{x}}_0(\boldsymbol{x}_t)\rVert^2$ is minimised by $\hat{\boldsymbol{x}}_0(\boldsymbol{x}_t) = \mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t]$. Likewise for $\boldsymbol{\varepsilon}$.
:::

:::proof
**Step 1. Condition on $\boldsymbol{x}_t$.** For fixed $\boldsymbol{x}_t$, write $\boldsymbol{m} = \mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t]$. Then $\mathbb{E}[\lVert\boldsymbol{x}_0 - \boldsymbol{a}\rVert^2 \mid \boldsymbol{x}_t] = \mathbb{E}[\lVert\boldsymbol{x}_0 - \boldsymbol{m}\rVert^2 \mid \boldsymbol{x}_t] + \lVert\boldsymbol{m} - \boldsymbol{a}\rVert^2$, because the cross term $2(\boldsymbol{m} - \boldsymbol{a})^\top\mathbb{E}[\boldsymbol{x}_0 - \boldsymbol{m} \mid \boldsymbol{x}_t]$ is zero.

**Step 2. Minimise for each $\boldsymbol{x}_t$.** The first term does not depend on $\boldsymbol{a}$, so the minimiser is $\boldsymbol{a} = \boldsymbol{m}$. Averaging over $\boldsymbol{x}_t$ preserves this.
:::

:::theorem[Tweedie's formula (Robbins, 1956; Efron, 2011)]
For $\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_0 + \sigma_t\boldsymbol{\varepsilon}$,

$$
\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t] = \frac{\boldsymbol{x}_t + \sigma_t^2\,\nabla\log p_t(\boldsymbol{x}_t)}{\alpha_t}, \qquad \mathbb{E}[\boldsymbol{\varepsilon} \mid \boldsymbol{x}_t] = -\sigma_t\,\nabla\log p_t(\boldsymbol{x}_t).
$$
:::

:::proof
**Step 1. Differentiate the marginal.** $p_t(\boldsymbol{x}) = \int\mathcal{N}(\boldsymbol{x} \mid \alpha_t\boldsymbol{x}_0, \sigma_t^2\mathbf{I})\,p_{\mathrm{data}}(\boldsymbol{x}_0)\,d\boldsymbol{x}_0$, and $\nabla_{\boldsymbol{x}}\mathcal{N}(\boldsymbol{x} \mid \alpha_t\boldsymbol{x}_0, \sigma_t^2\mathbf{I}) = \frac{\alpha_t\boldsymbol{x}_0 - \boldsymbol{x}}{\sigma_t^2}\mathcal{N}(\cdots)$. So

$$
\nabla p_t(\boldsymbol{x}) = \int\frac{\alpha_t\boldsymbol{x}_0 - \boldsymbol{x}}{\sigma_t^2}\,\mathcal{N}(\boldsymbol{x} \mid \alpha_t\boldsymbol{x}_0, \sigma_t^2\mathbf{I})\,p_{\mathrm{data}}(\boldsymbol{x}_0)\,d\boldsymbol{x}_0 .
$$

**Step 2. Divide by $p_t(\boldsymbol{x})$.** By Bayes' rule, $\mathcal{N}(\boldsymbol{x} \mid \alpha_t\boldsymbol{x}_0, \sigma_t^2\mathbf{I})\,p_{\mathrm{data}}(\boldsymbol{x}_0)/p_t(\boldsymbol{x}) = p(\boldsymbol{x}_0 \mid \boldsymbol{x}_t = \boldsymbol{x})$, so

$$
\nabla\log p_t(\boldsymbol{x}) = \frac{\alpha_t\,\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t = \boldsymbol{x}] - \boldsymbol{x}}{\sigma_t^2}.
$$

**Step 3. Rearrange** for the first formula. For the second, take conditional expectations of $\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_0 + \sigma_t\boldsymbol{\varepsilon}$: $\boldsymbol{x}_t = \alpha_t\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t] + \sigma_t\mathbb{E}[\boldsymbol{\varepsilon} \mid \boldsymbol{x}_t]$, and substitute.
:::

On the toy, the exact $\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t]$, computed component by component, matches $(\boldsymbol{x}_t + \sigma_t^2\nabla\log p_t)/\alpha_t$ to machine precision. The best denoiser *is* the score, rescaled. The animation shows what that means geometrically:

<figure class="fig">
<video src="/aboutme/blog/diffusion/tweedie.mp4" autoplay muted loop playsinline controls poster="/aboutme/blog/diffusion/tweedie.jpg"></video>
<figcaption>A fixed noisy point x_t. The best guess of the clean point is a posterior-weighted average of the eight modes (line thickness is the weight), and the score points from x_t to α_t times that guess. With much noise the guess is a blurred average; with little, it snaps to the nearest mode; from pure noise it is the mean of the data.</figcaption>
</figure>

Two consequences matter for the rest of the series:

- **One denoising step returns the average.** At $t = 1$, $\boldsymbol{x}_1$ is independent of $\boldsymbol{x}_0$, so $\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_1] = \mathbb{E}[\boldsymbol{x}_0]$: for the ring, its centre, where there are no data. A perfect denoiser applied once to noise produces the blurry mean. That is why diffusion models take many small steps, and why one-step generation (the last post of the series) needs a different idea.
- **The network can be trained without ever knowing the score.** That is the next result.

:::proposition[Denoising score matching (Vincent, 2011)]
For any model $\boldsymbol{s}_\theta$,

$$
\mathbb{E}\big\lVert\boldsymbol{s}_\theta(\boldsymbol{x}_t) - \nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)\big\rVert^2 = \mathbb{E}\big\lVert\boldsymbol{s}_\theta(\boldsymbol{x}_t) - \nabla\log p_t(\boldsymbol{x}_t)\big\rVert^2 + C,
$$

with $C$ independent of $\theta$, and $\nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = -\boldsymbol{\varepsilon}/\sigma_t$.
:::

:::proof
**Step 1. Expand both squares.** Each is $\mathbb{E}\lVert\boldsymbol{s}_\theta\rVert^2 - 2\,\mathbb{E}[\boldsymbol{s}_\theta^\top\boldsymbol{g}] + \mathbb{E}\lVert\boldsymbol{g}\rVert^2$, with $\boldsymbol{g}$ the respective target. The first terms are equal, and the last ones do not depend on $\theta$.

**Step 2. The cross terms are equal.** By the tower property, $\mathbb{E}[\boldsymbol{s}_\theta(\boldsymbol{x}_t)^\top\nabla\log q(\boldsymbol{x}_t \mid \boldsymbol{x}_0)] = \mathbb{E}\big[\boldsymbol{s}_\theta(\boldsymbol{x}_t)^\top\,\mathbb{E}[\nabla\log q(\boldsymbol{x}_t \mid \boldsymbol{x}_0) \mid \boldsymbol{x}_t]\big]$.

**Step 3. The inner expectation is the true score.** $\nabla\log q(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = (\alpha_t\boldsymbol{x}_0 - \boldsymbol{x}_t)/\sigma_t^2$, whose conditional mean given $\boldsymbol{x}_t$ is $(\alpha_t\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_t] - \boldsymbol{x}_t)/\sigma_t^2 = \nabla\log p_t(\boldsymbol{x}_t)$ by Step 2 of Tweedie's proof.

**Step 4. The form of the target.** $(\alpha_t\boldsymbol{x}_0 - \boldsymbol{x}_t)/\sigma_t^2 = -\sigma_t\boldsymbol{\varepsilon}/\sigma_t^2 = -\boldsymbol{\varepsilon}/\sigma_t$.
:::

So regressing $\boldsymbol{s}_\theta(\boldsymbol{x}_t)$ onto the known $-\boldsymbol{\varepsilon}/\sigma_t$ is the same as regressing it onto the unknown true score, up to a constant. On the toy, the gap between the two objectives is $3.20$ for the true score and $3.18$ for a deliberately wrong model ($\boldsymbol{s} = -\boldsymbol{x}$): the same constant, up to Monte Carlo error. Multiplying by $\sigma_t$, this is exactly noise prediction with $\hat{\boldsymbol{\varepsilon}} = -\sigma_t\boldsymbol{s}_\theta$, which closes the loop with Section 3.

### Four predictions, one object

<figure class="sketch-fig"><svg class="sketch" viewBox="0 0 960 470" role="img" aria-label="Four things a diffusion network can predict from a noisy point x_t = alpha_t x_0 + sigma_t epsilon: the noise epsilon, the clean data (the conditional mean of x_0), the score s = -epsilon / sigma_t, and the velocity v = alpha-dot x_0 + sigma-dot epsilon. Given x_t and t, each is a fixed linear function of the others; the losses differ only in how they weight noise levels."><text x="480" y="32" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--a1)">four things a diffusion network can predict</text><text x="480" y="56" text-anchor="middle" style="font-family:Kalam,cursive;font-size:15px;font-weight:400;fill:var(--muted)">given xₜ and t, each is a fixed linear function of the others</text><path d="M378.67835013122027 223.1790281992945 C378.67835013122027 223.1790281992945, 378.67835013122027 223.1790281992945, 378.67835013122027 223.1790281992945 M378.67835013122027 223.1790281992945 C378.67835013122027 223.1790281992945, 378.67835013122027 223.1790281992945, 378.67835013122027 223.1790281992945 M363.67207145461674 252.44336404343719 C374.27935401649506 235.97065128073473, 385.31074754211346 225.0698694996661, 397.9600954089083 211.26266462179262 M365.2269186769505 252.87363331704051 C375.26890426142626 237.79837437929325, 387.73414037549753 225.73141982423022, 397.2364595965717 213.03836732278938 M364.41342393171817 260.08876883930196 C376.90360456235703 248.1085339770109, 391.9842370193755 227.74321882822923, 411.9590239247714 208.39888089778208 M365.9044417663275 261.4106632684138 C382.2201931706982 245.52011794810932, 396.69433294641965 227.62421792127208, 412.64657221920197 208.0426013572079 M371.4200687808523 268.5816761368929 C386.9876498862565 249.77484935648295, 402.6526065996494 234.77373965985709, 425.07003672491044 205.87597538495194 M371.81688475115953 268.8853788365188 C391.0679507572793 244.9110764411811, 412.3803766125063 221.52517435561856, 426.73926182287033 205.14067900540996 M374.77633969428695 272.8743332857106 C392.00508281289666 255.78617767769498, 404.5318375921614 243.05234119233108, 438.77389763524855 199.52419728682509 M375.85819893846036 273.8019087795474 C399.50081716596935 246.4741553174887, 423.89970944314507 220.05079241848986, 440.77380633262084 201.51692814523489 M383.53622966097356 278.4822119799464 C402.52682413712046 254.75916405874287, 424.6111265914326 230.87162316059195, 453.932696430506 199.25916011428848 M383.0425320413026 279.8044128410074 C407.4413358123957 253.00953289258865, 431.9018637198683 224.57602552054223, 451.73978423167 198.09441010251098 M389.7842937346231 284.1623099002644 C416.28504853024106 253.3190030231574, 447.03473645092345 217.30670788566655, 464.0795189791533 199.64891878447895 M390.1260684468932 284.0694458915043 C409.7972941498637 258.7997792872011, 433.1128561839835 233.5780613969026, 464.6576822063594 197.66605924994155 M396.75714209168694 286.94435655335576 C417.51446013309493 262.8384409085128, 437.89799157182665 239.80117401367252, 476.63994030139185 196.44517546033774 M398.2394597745548 286.9012262682957 C416.9570767109799 264.2882847721176, 436.8740080425461 240.64745005350227, 474.16132570413646 197.23995134665918 M404.58463794438563 288.63556701260745 C427.6267095380543 267.50327583167046, 451.3553049853503 241.10102473332876, 483.2292815111731 199.23332371105613 M404.989960288754 290.15045201595996 C425.35159153182826 267.1478090826144, 444.4935351811138 243.68500391018566, 486.00607337557386 198.3534101986716 M414.7553315696991 292.406557337879 C443.81171934950146 257.1184334509161, 478.0518242298504 221.07648830557167, 497.96198481619865 198.8730735120754 M414.1377984280917 291.46427880617307 C441.38057491594463 261.8897878564078, 465.7678772194909 233.989851988483, 497.3391295743691 196.3276951588615 M420.45007142532864 294.8131312546118 C450.72939104672867 259.3825824329704, 482.01673744965206 228.6264413143754, 507.3806933729672 200.8126241462008 M423.09072542126535 295.8221691168177 C441.33610927791835 270.12074635022003, 462.75237532779 246.99973083409722, 507.0395617123921 199.74623652810052 M432.03926109171636 298.4463243142743 C451.30089525827157 274.1409604983681, 472.31235059487534 248.58388010746805, 516.6383346346613 201.87277424092795 M430.61220972908825 295.94999889242877 C459.17493898814655 264.2610483651987, 488.37956899672434 232.17011566346974, 515.3613211075716 199.1528080925757 M437.8558800981016 297.70128637569877 C471.38019618995554 262.48516357885006, 504.82286970137767 225.98607811045903, 523.7469050612418 199.73464257067096 M440.7553953705457 298.30300252552996 C461.809359247679 270.70375901405225, 486.28565431228026 243.58468232344669, 524.0929651939894 201.80331042601753 M451.36544921242626 300.770810802854 C470.4641951534722 277.8410579848143, 490.0353233330445 254.73148010199478, 532.2550222064199 204.11616431609397 M448.5440047127008 301.32336455440645 C471.06611440391777 275.2104993863156, 494.06507028872204 249.5831631301869, 534.8961471088527 202.5563296490316 M456.17087418042036 303.8783456868493 C481.17142090005416 277.7860996769805, 500.827572012296 248.80022951180624, 540.6012406886656 207.26742564433368 M458.03811970977955 302.987293385109 C485.08638896422883 270.75323983402467, 512.6150988235833 239.3964831323379, 541.5698691635027 204.93253450932096 M466.98241101550684 304.07152304370777 C494.00243044730206 276.09770449819456, 516.1269379025649 250.2923216968603, 549.5355452491902 209.7018695485542 M468.01508469307606 302.06261298922715 C488.3641221059213 280.9495088846723, 505.96697476994257 259.75939619365016, 550.732913401397 208.85994472124116 M481.8543326509156 303.3192987770392 C509.3126545539695 269.49138017699556, 542.9588786291622 234.49856424236137, 556.0726667229545 213.69163042816973 M478.86711052306634 301.0199653665619 C510.2175796647374 265.2194103360319, 541.6581928479297 229.36194978231111, 557.9880950691172 211.0333280460228 M489.93065811362663 301.32432019670284 C514.8777197265688 274.7718897102739, 541.1184000280318 242.9699858060142, 565.5135750966505 216.79873992839856 M489.4579412633514 301.10924056378917 C509.23289206290326 279.4936336847508, 528.9761999044242 258.93901864565123, 566.3046931743514 215.65946863132558 M502.5227046113412 302.0994222781402 C528.3258338040587 274.8844755488681, 550.2521507164971 245.19503010829857, 573.5745593637486 217.58929984137765 M500.0613437676288 300.8501354337917 C519.8805251276752 279.91001394831034, 541.2610717532198 257.0651639916596, 573.3584678409413 217.13540241235285 M514.2063444745015 299.01767182281986 C526.8113292785282 285.6571829261295, 542.76491274472 269.61472935735566, 577.0966025244977 224.47410710616222 M513.1693021217116 299.44626029610447 C537.7246512456916 271.06385853322485, 563.9458234602757 241.88944875104178, 579.4019728088819 223.80310461185124 M524.7071440403279 295.87183077912846 C547.0513808414266 276.1553435604387, 562.9228989697217 250.84851869031223, 585.7274649338112 230.1927638460523 M524.9380940275662 298.2586647578056 C549.162238833431 272.360922800865, 570.821495390996 247.58514757150965, 585.7575762822074 230.09766227743722 M540.0779157397907 297.35085250170295 C550.9694011632914 281.3228521139401, 563.1793011819037 263.59971642840696, 589.9341148374192 237.32965565215758 M538.4767803902841 295.8463964728484 C556.2539923070509 274.35423099338254, 573.1021239027083 255.6272754256671, 589.8908289651442 236.27731506516187 M553.1762701845818 292.6163780107619 C569.0012047422628 271.48800704570255, 584.9511539207491 256.91837740301844, 593.0605756037919 245.01689994161495 M552.9856949050535 291.05705990139296 C564.0254144452908 279.57861683430383, 573.2297972178606 265.70356283069793, 595.0792487940337 241.99046609772847 M571.1300501961055 284.0886514118825 C575.959704037382 275.669265980635, 580.3873503982297 268.6175648959182, 597.0691236371691 255.43909969825668 M571.1328155861495 281.87679422958274 C576.1705104544089 275.73013525398477, 583.434798331028 267.8485501162749, 595.6144563503615 253.89449820516072" style="stroke:var(--dot);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M539.8555124265077 203.77828837247426 C553.1567468246174 206.38494890414876, 565.3337037469986 212.9131533426593, 574.3365159752043 218.66059290306276 C583.33932820341 224.4080324634662, 590.2907266206774 231.76319926682595, 593.8723857957417 238.2629257348949 C597.454044970806 244.76265220296386, 597.9297698594233 251.24478713326386, 595.8264710255901 257.65895171147645 C593.7231721917568 264.07311628968904, 589.0182446735745 270.8209482896414, 581.2525927927421 276.7479132041704 C573.4869409119098 282.6748781186994, 561.432578041807 289.137276616314, 549.2325597405961 293.2207411986502 C537.0325414393852 297.3042057809864, 522.6124696457593 299.6644699775912, 508.0524829854765 301.2487006981878 C493.49249632519366 302.8329314187844, 476.71473667228696 303.8986757887858, 461.8726397788989 302.72612552222984 C447.0305428855108 301.5535752556739, 431.793772851063 297.59289726262557, 418.99990162514797 294.21339909885216 C406.20603039923293 290.83390093507876, 394.1798059485749 287.86336168090617, 385.1094124234087 282.4491365395895 C376.0390188982425 277.03491139827287, 368.23264257820495 268.3247333155288, 364.5775404741509 261.72804825095216 C360.92243837009687 255.1313631863755, 360.8639129888352 249.2533144892543, 363.1787997990844 242.86902615212963 C365.4936866093336 236.48473781500496, 370.64517590014066 229.35356551294285, 378.4668613356462 223.42231822820418 C386.2885467711517 217.4910709434655, 398.0065198363747 211.5397344251844, 410.10891241211766 207.28154244369765 C422.2113049878606 203.02335046221089, 436.68751452507854 199.55477387436474, 451.0812167901038 197.87316633928356 C465.474919055129 196.19155880420237, 480.8653615746877 195.9689640883754, 496.471126002269 197.1918972332105 C512.0768904298502 198.41483037804562, 535.8386814716463 203.2217648282721, 544.7158033555913 205.21076520829413 C553.5929252395364 207.19976558831618, 550.5217272925859 208.27414139007348, 549.7338573059392 209.1258995133428 M487.6272907796591 197.2165693890141 C501.50619910039694 196.6350043628533, 515.7351313151701 199.16225723696704, 529.2547192442297 202.21827509245375 C542.7743071732893 205.27429294794047, 558.5015356820717 210.64269468951642, 568.7448183540171 215.5526765219344 C578.9881010259625 220.46265835435236, 585.7103721576608 225.586149854896, 590.7144152759021 231.67816608696162 C595.7184583941433 237.77018231902724, 599.2999119653883 245.37462015346753, 598.7690770634646 252.10477391432804 C598.2382421615408 258.8349276751885, 594.4258979138572 265.8038869065853, 587.5294058643594 272.0590886521246 C580.6329138148617 278.3142903976639, 568.6051852062934 285.0766373265734, 557.3901247664784 289.63598438756384 C546.1750643266633 294.19533144855427, 534.4663657970738 297.43414596565077, 520.2390432254693 299.4151710180674 C506.0117206538649 301.396196070484, 487.2526196800965 301.6787786808131, 472.0261893368517 301.52213470206357 C456.79975899360693 301.36549072331405, 442.1575938375851 300.9485914071759, 428.8804611660008 298.47530714557035 C415.60332849441653 296.0020228839648, 402.47336574325465 291.97400723356157, 392.36339330734603 286.6824291324301 C382.2534208714374 281.39085103129867, 373.13315673690704 273.122119830378, 368.2206265505491 266.72583853878166 C363.3080963641912 260.32955724718533, 361.7360658953847 255.13769721915818, 362.8882121891985 248.30474138285203 C364.0403584830123 241.47178554654587, 368.9209961010024 232.06710437981724, 375.133504313432 225.72810352094473 C381.3460125258616 219.38910266207222, 389.46520073207716 214.7729937606758, 400.16326146377617 210.2707362296169 C410.86132219547517 205.76847869855797, 424.72584680640034 200.6104363295342, 439.3218687036259 198.71455833459123 C453.9178906008515 196.81868033964827, 479.92274411895926 199.0394254197909, 487.73939284712964 198.89546825995907 C495.5560415753 198.75151110012726, 486.6502782082007 197.21918803863184, 486.2217610726482 197.8508153756004" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="480" y="244" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">xₜ = αₜ x₀ + σₜ ε</text><text x="480" y="268" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">one noisy point</text><path d="M40.12525493472174 91.8559106802476 C40.12525493472174 91.8559106802476, 40.12525493472174 91.8559106802476, 40.12525493472174 91.8559106802476 M40.12525493472174 91.8559106802476 C40.12525493472174 91.8559106802476, 40.12525493472174 91.8559106802476, 40.12525493472174 91.8559106802476 M38.35376878442884 104.22823654276316 C42.45726715159495 101.22918445088918, 49.42340079031034 93.06664864965826, 53.43301616652465 92.60675605093611 M38.85546958568448 105.47261395410399 C41.82185079875871 103.92121548892021, 44.270631038159216 99.57076361881903, 52.0508419909218 92.65212552578437 M37.74342368896174 121.72032192036615 C48.23048212580751 107.17607237069001, 56.502935826566834 100.10570174381809, 64.16944976637912 91.14157761868917 M40.30941134722958 120.31057416730665 C44.714535525532824 114.21523141026982, 50.06107456855265 107.63158637916335, 63.29974671022807 91.60535415148259 M39.91412749171191 131.99377314372668 C49.47996534972293 123.90990082054816, 57.13761258010587 112.82367270383749, 76.88280745677262 91.01907561572169 M39.74714113182322 132.89519495688128 C54.21924401045835 116.85719824104477, 66.24679096293387 103.10769930611963, 74.691270585941 91.84278370241535 M40.753840834654916 145.20683699677897 C56.35227411610863 129.9621708611089, 72.10099400508274 107.66896360955106, 89.48760029349792 94.36252219444975 M39.35050382964131 146.62626798529377 C51.213441530253135 135.42644583699868, 63.223818833636486 121.9541905090581, 87.69886659185718 92.39801029428743 M38.69497901698271 161.11822481224152 C61.27878999050622 136.63007766217694, 85.35321566100231 110.07272594271726, 100.77746536151592 91.02764550428743 M39.9527432921219 161.55755005645406 C53.76039499377722 144.8875447632609, 66.75725366362039 127.31228347917983, 100.62964684955863 92.5802548133293 M39.116526344665736 175.4594680697512 C56.4390684958442 154.62757919017776, 70.84250096812046 140.11221893994826, 111.34516083953761 91.2444901008988 M39.17795948685654 173.89867476181539 C67.24928313899287 142.39183568401108, 93.46882411428518 112.21736756189004, 111.2325691837524 91.23267369264646 M39.42158821742443 186.5114146551949 C60.97505109186801 162.54102849454833, 83.6819475243185 138.790237819471, 122.7919410000239 90.75804194619792 M38.97959808456172 188.30571123970606 C61.73553419868824 160.45227100969316, 85.81743411122243 132.91766047777233, 123.4187496222705 90.93704641221936 M51.22400573752984 190.3959526760311 C73.69945592351826 166.59754356251116, 89.4206605308535 144.83070533830437, 133.9808721801952 94.22356298857162 M52.338002567879634 187.83695049065102 C69.95095580958733 166.577557959219, 87.978094067256 148.0868554598141, 136.07571586707948 92.62317257912622 M62.4941889703733 189.19921027465662 C90.61594168656096 159.68456928708207, 116.38410885608694 130.25152741531807, 147.47044389799348 91.12837784198841 M64.86718499194767 186.89016769073842 C84.76503871184441 164.745239419742, 107.40004578802589 138.42616104819842, 147.81195591024235 92.154721106652 M74.81991562591574 185.93902497548555 C99.59594045501483 163.82855168329903, 123.07344004754947 138.52142470333942, 157.8875896781392 92.16075626733513 M74.98656216584786 187.03415403869877 C107.2995440491392 149.5608391371304, 140.55782092806896 113.976819668603, 158.59087635997568 92.51217519562492 M87.85058450786809 189.34505070918186 C117.08470304552684 153.11774257411395, 147.2893397774639 121.53193494280572, 172.16129816809328 91.33169985152584 M87.02614660178723 187.5028862801375 C109.56104833620292 164.68106872135039, 130.88025660078455 139.0270298812617, 172.17535105661148 91.17868151352543 M97.85053148470077 188.95409239288247 C126.71902531229861 159.60929297173175, 152.40374508957302 126.31551093019688, 182.7759291090666 93.97027658772461 M98.89079998610731 187.8549237877579 C130.53750287286718 152.17158360585404, 160.876119602705 116.82797633939013, 182.64197454360672 90.84944727470905 M111.03264060560576 188.53189260930583 C128.77732514514338 166.11726077274702, 148.1868388090921 143.5613194275293, 194.58058833602018 91.06424587534549 M110.25478179524615 188.00925774046982 C129.23464985665004 167.13165039996397, 148.6597947286066 146.02811213247392, 194.90137456754354 92.13642774040619 M125.03136492655347 189.54201408808885 C155.4183960698885 148.75292609269394, 186.2016671209615 116.0543869623419, 207.2424437486744 91.85539310027536 M123.04570762179526 187.77825610472655 C150.04325210753169 156.66697200097425, 175.97016172405236 125.69696518391896, 206.61246554913842 92.77772329894518 M137.63633762625483 187.76762638489555 C153.92052731409626 169.00764020225017, 169.40545639325595 149.39828787405335, 217.08033642747122 90.56131631398266 M135.8390977904617 186.90351281363354 C154.6205770192672 165.60000369406194, 174.4979839152476 144.99107353122298, 219.82699719920436 91.12342864486081 M145.3456879771708 189.80684046163998 C165.18385836824726 168.83377350037955, 180.24212265373794 147.77476203134194, 233.03780145963105 93.54430871485431 M147.62263082907742 187.7152448471713 C181.65279374325476 150.8801243561805, 213.6138957186825 112.79595049028289, 231.12351284470793 91.41857776023146 M158.67541634551228 189.71333368648618 C191.4337049239054 150.99627363453683, 221.19948484446672 118.59768609430002, 241.71667905511492 90.63314532730428 M158.76645596859956 186.68697973219523 C181.95425527856213 164.10538584735917, 203.49468298619033 140.08849756783295, 243.25189804473118 91.18899309378749 M173.11271016224975 190.09168441407107 C200.9127337438802 152.65930372404543, 231.78290475635904 116.44226565029844, 254.23043051763364 92.31841758199673 M170.18744661411958 188.5949526226586 C193.0545206552323 166.2308462023947, 211.6534379694828 142.07289238185365, 253.94763406148107 91.95068264113445 M182.44955503907516 189.5191586646131 C201.63902235662582 169.63747008795218, 216.65252026262925 147.17772170156374, 267.9574542156578 91.04786789606838 M182.34374896082556 188.15395547872134 C215.89837405105678 151.17480729578367, 247.22507183676302 114.73725744941704, 266.59605133129924 92.66923702497547 M197.05575559445884 189.0145926052491 C221.6523478377778 155.21415977998453, 251.42093130161686 126.0130394556742, 279.87469669837907 90.04817635223786 M195.71874032005897 187.95028204808673 C223.8266557035668 156.3753772469337, 251.40314431706148 125.04880076911202, 277.7268780806235 92.09567867422062 M206.6125465696119 188.66017922914318 C230.6156630020031 155.501468437866, 256.84809824606606 129.8576669229669, 291.0306735217682 91.39384318014194 M207.5155777352769 186.97757704511017 C237.9136750989104 152.24592598737533, 268.0445091098692 117.32732287624081, 291.29819865204576 92.89940680532304 M220.06590879949516 185.69680293570943 C237.19281921217149 168.82118936818864, 252.74320059735902 148.30487718044176, 301.479080364536 92.62897543544474 M220.0214514530614 187.8962332310222 C236.49024451193705 165.42062141474275, 256.8798969240224 143.55017161693726, 301.5910021202727 92.50404660138827 M230.33594421281956 187.96466453061177 C260.52285420745136 150.45509499359142, 295.13102687472286 116.22784036974164, 313.43210899914334 94.06262158006025 M230.0778974231194 188.5880789139282 C257.76389351952434 157.58740011742265, 284.1783573358479 126.80806027259074, 315.3734357631365 92.64684629239669 M240.47019180305557 188.99861580598719 C261.7211261442318 166.50806403311282, 278.9972689893998 141.571691026226, 322.2652910308867 100.60178270415754 M242.04505076929252 189.41577502903147 C266.39447235853964 160.47046747396703, 290.0812416187427 134.437542294304, 321.6665863006796 97.7257508772088 M253.97723874439515 187.64994621955412 C267.6847781458593 172.50499831243033, 284.80856600601174 153.46789657148977, 322.93593541972695 110.10726508215062 M255.29619440551875 188.65866431774194 C279.51161495950714 159.1636623532292, 304.36266345332905 130.22301168212678, 321.7738549169841 112.319317179555 M265.49974121780866 189.95526537594944 C283.0049600497647 171.48397454820045, 297.6885677254576 150.3403395976942, 320.03644445129913 127.58901475292144 M267.00171736188963 188.94571630901285 C277.4318255696302 173.6470042290591, 288.7986739745786 161.0489939543285, 320.39521073118095 126.59611092825865 M278.74803552700706 187.35828970583717 C286.1288851406379 178.2587411673655, 298.36507151438605 167.051320503037, 322.41886233104515 140.91267664293844 M278.0236634006122 188.09137649844962 C288.55925793749725 175.19377007903373, 299.50447606603933 163.35387086001228, 320.8626768915833 140.2853283803546 M291.2534540694514 187.00517704613142 C300.77172495494784 176.10729772767172, 306.78515361819836 166.82877736748168, 318.41636824219495 155.7308004058113 M289.7799844870185 188.6549634285328 C298.08542855171106 178.80130982045426, 304.6877357460915 171.16379225159503, 320.5227611336672 152.72206466064395 M299.821641307362 189.13986122021896 C308.75425330946035 185.29292525731572, 311.83877987913195 177.90011177576588, 321.7011235776905 168.38578878050848 M302.25879426057503 187.95006576884052 C307.8281218686388 183.4194993773156, 311.79943775979393 175.15220636601413, 320.40524051199554 167.91432461991153 M315.13066129026765 187.13188107045812 C316.2639702528383 186.09927401200636, 317.6257123015851 182.4072890739397, 321.42849086200584 180.36215863726852 M314.12903717384967 188.33415745950285 C316.686703881765 185.75488758318915, 318.5132368090822 183.8165003577799, 321.1284043420223 180.2857525090529" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M38.74383221231048 93.65783043904298 C94.35303074089046 91.28471416670568, 150.87565385056996 89.79175066941504, 319.46976074024417 92.1431763260468 M39.90422696238126 92.1785791057567 C113.00207423225464 92.99135375165952, 187.2667268626784 93.29123740455141, 320.0074031051243 91.41574045629332 M319.58350630188363 94.03269822364673 C317.39539094623854 122.74556609159336, 319.04864542838857 159.07953753238544, 318.97302743876355 189.20749655431138 M319.11221035155467 91.10587989580817 C319.13530197559857 121.98257545870729, 318.82419876425524 154.46522695613095, 320.70515956762245 189.0574887041468 M320.56146250812947 189.2195999172442 C239.91289730459513 188.0622188871316, 159.7570694054879 190.83998391668686, 40.82004704900069 187.55105231252017 M320.6645478093987 187.5982504844687 C248.44041763230638 186.93053611829998, 178.75142957177638 186.56431308590916, 39.21309834919167 187.38037183091504 M41.91046312833205 187.1656677163206 C38.34183750807271 160.7686935520731, 40.98058755754456 133.9278022087179, 38.0976819046773 90.40322067802772 M40.8701640042942 189.0866512854118 C41.071346455019345 161.11867571217007, 40.71218692730032 134.96069965255447, 39.6124013024848 92.72327224356123" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="180" y="122" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">the noise ε</text><text x="180" y="149" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">ε = (xₜ − αₜ x₀) / σₜ</text><text x="180" y="173" text-anchor="middle" style="font-family:Kalam,cursive;font-size:13px;font-weight:400;fill:var(--muted)">DDPM (Ho et al., 2020)</text><path d="M639.7083836160108 92.33546627516932 C639.7083836160108 92.33546627516932, 639.7083836160108 92.33546627516932, 639.7083836160108 92.33546627516932 M639.7083836160108 92.33546627516932 C639.7083836160108 92.33546627516932, 639.7083836160108 92.33546627516932, 639.7083836160108 92.33546627516932 M638.6638629712032 104.04960944066471 C642.5513877400974 101.20840219573148, 649.1508827662487 93.4754735419338, 652.9494657542414 93.0397858168286 M639.1391584671296 105.22849330404023 C641.9494143532 103.75874738965561, 644.2693114221057 99.63726667061239, 651.6400375878808 93.08276742457959 M637.9826113992414 121.4451679350651 C648.4696698360871 106.90091838538896, 656.7421235368464 99.83054775851704, 664.4086374766587 90.86642363338814 M640.5485990575091 120.03542018200561 C644.9537232358123 113.94007742496878, 650.3002622788322 107.3564323938623, 663.5389344205076 91.33020016618156 M639.497256173001 132.47332873864838 C649.2373986385389 124.18710863748467, 657.0702428825671 112.89933941123711, 677.1219951670521 90.74392163042063 M639.3302698131123 133.37475055180298 C654.0434724080775 117.04975244898198, 666.3171981780777 103.01705717984933, 674.9304582962205 91.56762971711429 M640.9930285449345 144.93168301147796 C656.5914618263882 129.68701687580787, 672.3401817153624 107.39380962425004, 689.7267880037775 94.08736820914874 M639.5896915399209 146.35111399999275 C651.4526292405327 135.15129185169766, 663.463006543916 121.67903652375708, 687.9380543021368 92.12285630898641 M638.9341667272623 160.8430708269405 C661.5179777007859 136.35492367687593, 685.592403371282 109.79757195741622, 701.0166530717955 90.7524915189864 M640.1919310024015 161.28239607115304 C653.9995827040569 144.6123907779599, 666.9964413739 127.0371294938788, 700.8688345598382 92.30510082802827 M638.6996550259547 175.93902366467296 C656.0221971771332 155.10713478509948, 670.4256296494094 140.59177453486998, 710.9282895208266 91.72404569582052 M638.7610881681455 174.37823035673713 C666.8324118202819 142.8713912789328, 693.0519527955743 112.69692315681178, 710.8156978650414 91.71222928756818 M639.660775927704 186.23626066989382 C661.0489948769269 162.4527414245527, 683.5930001966965 138.88933553932057, 722.375069681313 91.23759754111961 M639.2187857948412 188.03055725440498 C661.7964307011858 160.3896137229034, 685.6924391180993 133.0688468947158, 723.0018783035596 91.41660200714105 M651.4631934478094 190.1207986907301 C673.9386436337978 166.32238957721017, 689.659848241133 144.5555513530034, 734.2200598904748 93.94840900327063 M652.5771902781593 187.56179650535003 C670.190143519867 166.30240397391802, 688.2172817775355 147.81170147451311, 736.314903577359 92.34801859382523 M662.7333766806529 188.9240562893556 C690.8551293968405 159.40941530178105, 716.6232965663664 129.97637343001705, 747.7096316082732 90.85322385668739 M665.1063727022273 186.6150137054374 C685.004226422124 164.470085434441, 707.6392334983055 138.1510070628974, 748.051143620522 91.87956712135099 M674.4030443072048 186.41858057040727 C699.362452806368 164.10624494816173, 723.0169125040072 138.59554865395123, 758.1267773884188 91.8856022820341 M674.5696908471369 187.5137096336205 C707.1284862684262 149.7433110310514, 740.6416517509919 113.86607576554054, 758.8300640702552 92.23702121032389 M687.4337131891572 189.8246063041036 C716.8946723497976 153.3268113807222, 747.329153744166 121.47659771118461, 772.4004858783728 91.0565458662248 M686.6092752830763 187.98244187505924 C709.3158640947298 164.9756915601667, 730.8009204127654 139.13086635898554, 772.414538766891 90.90352752822439 M698.0897191949803 188.6789384075814 C726.7431143106879 159.56679820019494, 752.2215418741122 126.5103282041289, 782.3590577903557 94.4498321826463 M699.1299876963868 187.57976980245684 C730.5374808124357 152.1740750214917, 760.6304295561474 117.11307644493274, 782.2251032248959 91.32900286963074 M711.2718283158854 188.2567386240048 C728.8704569986467 166.02111027619918, 748.1253223912317 143.6430714165896, 794.1637170173092 91.54380147026724 M710.4939695055258 187.7341037551688 C729.3288357859694 167.02231518514418, 748.6152171309253 146.07840609519045, 794.4845032488327 92.61598333532794 M725.270552636833 189.2668601027878 C755.6575837801681 148.4777721073929, 786.440854831241 115.77923297704088, 807.481631458954 91.58023911497435 M723.2848953320748 187.5031021194255 C750.2824398178113 156.3918180156732, 776.209349434332 125.42181119861795, 806.851653259418 92.50256931364416 M737.8755253365345 187.49247239959448 C754.1597150243759 168.7324862169491, 769.6446441035356 149.1231338887523, 817.3195241377507 90.28616232868164 M736.0782855007412 186.62835882833247 C754.8597647295468 165.3248497087609, 774.7371716255273 144.7159195459219, 820.0661849094839 90.84827465955979 M744.9288166584598 190.2863960565617 C764.8991861670758 169.14656951922584, 780.0978714417554 147.92602218051448, 833.2769891699107 93.26915472955325 M747.2057595103665 188.19480044209303 C781.5002541514739 151.07135916422038, 813.7173322603912 112.69271844135376, 831.3627005549876 91.1434237749304 M758.2585450268014 190.19288928140796 C791.0168336051944 151.47582922945858, 820.7826135257558 119.07724168922178, 841.2998077364039 91.11270092222604 M758.3495846498886 187.166535327117 C781.5373839598511 164.58494144228095, 803.0778116674794 140.5680531627547, 842.8350267260201 91.66854868870925 M773.3518978725293 189.81653042877005 C800.9155553226379 152.66376106167112, 831.5435948682405 116.72526337781281, 853.8135591989228 92.79797317691845 M770.4266343243992 188.3197986373576 C793.1277946544425 166.12943850150998, 811.5697013256147 142.15210476436346, 853.5307627427702 92.43023823605617 M782.6887427493547 189.24400467931207 C801.8782100669054 169.36231610265116, 816.8917079729088 146.9025677162627, 868.1966419259372 90.77271391076735 M782.5829366711051 187.87880149342033 C816.1375617613363 150.89965331048265, 847.4642595470425 114.46210346411601, 866.8352390415787 92.39408303967444 M797.2949433047386 188.73943861994806 C821.8915355480574 154.9390057946835, 851.6601190118964 125.73788547037319, 880.1138844086587 89.77302236693686 M795.9579280303386 187.6751280627857 C824.0658434138464 156.1002232616327, 851.6423320273411 124.773646783811, 877.9660657909031 91.82052468891962 M806.8517342798914 188.38502524384214 C830.8548507122825 155.22631445256496, 857.0872859563456 129.58251293766588, 891.2698612320478 91.11868919484093 M807.7547654455565 186.70242305980912 C838.15286280919 151.97077200207428, 868.2836968201487 117.05216889093978, 891.5373863623253 92.62425282002202 M819.6490374807843 186.1763585306312 C836.913370248112 169.13516221130277, 852.6052973873283 148.45602025994475, 901.7182680748156 92.35382145014374 M819.6045801343505 188.37578882594397 C836.2178635813947 165.732040070376, 856.7586710615226 143.6877062577031, 901.8301898305523 92.22889261608726 M829.9190728941086 188.44422012553355 C860.1059828887404 150.93465058851316, 894.714155556012 116.7073959646634, 913.0152376804324 94.54217717498202 M829.6610261044084 189.06763450884998 C857.3470222008133 158.06695571234442, 883.761486017137 127.2876158675125, 914.9565644444256 93.12640188731845 M840.7093795133352 188.72346182068617 C861.9603138545114 166.2329100478118, 879.2364566996794 141.29653704092496, 922.5044787411663 100.3266287188565 M842.2842384795721 189.14062104373045 C866.6336600688192 160.19531348866602, 890.3204293290223 134.16238830900295, 921.9057740109592 97.45059689190775 M854.2164264546747 187.37479223425308 C867.7803744251981 172.40094471195042, 884.7591131089822 153.53070296099744, 922.5190641010159 110.58682067707234 M855.5353821157984 188.3835103324409 C879.4983447122178 159.1732164213943, 904.1048036795562 130.51393381429568, 921.356983598273 112.79887277447672 M865.7389289280882 189.68011139064842 C883.2441477600443 171.2088205628994, 897.9277554357371 150.06518561239318, 920.2756321615788 127.31386076762041 M867.2409050721692 188.67056232371183 C877.6710132799096 173.37185024375805, 889.0378616848582 160.77383996902748, 920.6343984414606 126.32095694295762 M878.3311642082963 187.83784530075886 C885.712013821927 178.73829676228723, 897.9482001956751 167.53087609795872, 922.0019910123342 141.39223223786018 M877.6067920819014 188.5709320933713 C888.1423866187863 175.67332567395545, 899.0876047473284 163.833426454934, 920.4458055728724 140.76488397527635 M890.8365827507405 187.48473264105317 C900.5534722291118 176.35491013750547, 906.7695470651162 166.8432720223351, 918.6555559524746 155.45564642051028 M889.3631131683077 189.13451902345454 C897.8221811303332 179.10347801093826, 904.5729914381999 171.2951271519504, 920.7619488439468 152.44691067534293 M900.0608290176416 188.864707234918 C908.9934410197399 185.01777127201476, 912.0779675894115 177.62495779046492, 921.94031128797 168.11063479520752 M902.4979819708547 187.67491178353956 C908.0673095789185 183.14434539201463, 912.0386254700736 174.87705238071317, 920.6444282222751 167.63917063461057 M915.2662986412059 186.9673313204961 C916.2862767075196 186.0379849678895, 917.5118445513917 182.7151985236295, 920.9343452557704 180.87458113062544 M914.3648369364298 188.04938007063635 C916.6667369735537 185.728037181954, 918.310616608139 183.98348867908567, 920.6642673877851 180.8058156152314" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M638.7438322123105 93.65783043904298 C694.3530307408904 91.28471416670568, 750.87565385057 89.79175066941504, 919.4697607402442 92.1431763260468 M639.9042269623812 92.1785791057567 C713.0020742322546 92.99135375165952, 787.2667268626784 93.29123740455141, 920.0074031051244 91.41574045629332 M919.5835063018836 94.03269822364673 C917.3953909462385 122.74556609159336, 919.0486454283886 159.07953753238544, 918.9730274387636 189.20749655431138 M919.1122103515547 91.10587989580817 C919.1353019755985 121.98257545870729, 918.8241987642552 154.46522695613095, 920.7051595676224 189.0574887041468 M920.5614625081295 189.2195999172442 C839.9128973045952 188.0622188871316, 759.757069405488 190.83998391668686, 640.8200470490007 187.55105231252017 M920.6645478093986 187.5982504844687 C848.4404176323063 186.93053611829998, 778.7514295717764 186.56431308590916, 639.2130983491917 187.38037183091504 M641.910463128332 187.1656677163206 C638.3418375080727 160.7686935520731, 640.9805875575445 133.9278022087179, 638.0976819046773 90.40322067802772 M640.8701640042942 189.0866512854118 C641.0713464550194 161.11867571217007, 640.7121869273003 134.96069965255447, 639.6124013024848 92.72327224356123" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="780" y="122" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">the clean data x₀</text><text x="780" y="149" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">x₀ = E[x₀ | xₜ]</text><text x="780" y="173" text-anchor="middle" style="font-family:Kalam,cursive;font-size:13px;font-weight:400;fill:var(--muted)">the denoiser</text><path d="M40.235457037451624 339.72913766285785 C40.235457037451624 339.72913766285785, 40.235457037451624 339.72913766285785, 40.235457037451624 339.72913766285785 M40.235457037451624 339.72913766285785 C40.235457037451624 339.72913766285785, 40.235457037451624 339.72913766285785, 40.235457037451624 339.72913766285785 M38.534877363653585 352.19799040857606 C42.422402132547795 349.35678316364283, 49.021897158699225 341.62385450984505, 52.82048014669173 341.1881667847399 M39.01017285957999 353.3768742719516 C41.82042874565031 351.907128357567, 44.140325814556064 347.78564763852376, 51.51105198033114 341.23114839249087 M37.85362579169164 369.5935489029764 C48.34068422853741 355.0492993533003, 56.61313792929674 347.9789287264283, 64.27965186910902 339.01480460129943 M40.41961344995948 368.1838011499169 C44.824737628262724 362.0884583928801, 50.171276671282556 355.5048133617736, 63.40994881295798 339.47858113409285 M40.02432959444179 379.86700012633696 C49.59016745245281 371.78312780315844, 57.24781468283576 360.6968996864478, 76.99300955950251 338.892302598332 M39.8573432345531 380.7684219394916 C54.329446113188226 364.73042522365506, 66.35699306566376 350.98092628872996, 74.80147268867088 339.71601068502565 M40.8640429373848 393.08006397938925 C56.46247621883852 377.8353978437192, 72.21119610781261 355.5421905921613, 89.59780239622779 342.23574917706003 M39.4607059323712 394.49949496790407 C51.32364363298302 383.2996728196089, 63.33402093636636 369.82741749166837, 87.80906869458705 340.2712372768977 M38.805181119712586 408.9914517948518 C61.388992093236105 384.5033046447872, 85.46341776373221 357.9459529253275, 100.88766746424582 338.9008724868977 M40.06294539485178 409.4307770390643 C53.870597096507105 392.7607717458711, 66.86745576635028 375.1855104617901, 100.73984895228853 340.4534817959396 M39.22672844739562 423.3326950523615 C56.54927059857408 402.500806172788, 70.95270307085035 387.98544592255854, 111.4553629422675 339.11771708350904 M39.28816158958642 421.7719017444257 C67.35948524172275 390.2650626666214, 93.57902621701507 360.09059454450033, 111.34277128648229 339.1059006752567 M39.531790320154315 434.38464163780515 C60.92000926937732 410.601122392464, 83.46401458914687 387.0377165072319, 122.24608407376327 339.385978509031 M39.08980018729161 436.17893822231633 C61.66744509363619 408.5379946908148, 85.56345351054956 381.2172278626272, 122.87289269600987 339.56498297505243 M51.334207840259744 438.26917965864135 C73.80965802624816 414.47077054512147, 89.53086263358338 392.7039323209147, 134.0910742829251 342.0967899711819 M52.448204670609535 435.7101774732613 C70.06115791231723 414.4507849418293, 88.0882961699859 395.9600824424244, 136.18591796980937 340.49639956173655 M62.60439107310319 437.0724372572669 C90.72614378929084 407.55779626969235, 116.49431095881685 378.1247543979283, 147.5806460007234 339.0016048245987 M64.97738709467755 434.7633946733487 C84.8752408145743 412.6184664023523, 107.51024789075579 386.2993880308087, 147.92215801297226 340.0279480892623 M74.93011772864563 433.8122519580958 C99.70614255774471 411.7017786659093, 123.18364215027935 386.3946516859497, 157.99779178086908 340.0339832499454 M75.09676426857774 434.90738102130905 C107.40974615186909 397.4340661197407, 140.66802303079885 361.8500466512133, 158.70107846270557 340.38540217823515 M87.30472758160748 437.9729872720149 C116.7656867422479 401.4751923486335, 147.20016813661627 369.6249786790959, 172.27150027082317 339.2049268341361 M86.48028967552662 436.13082284297053 C109.18687848718021 413.12407252807793, 130.6719348052157 387.27924732689684, 172.28555315934136 339.0519084961357 M97.96073358743065 436.8273193754928 C126.8292274150285 407.482519954342, 152.51394719230294 374.1887379128071, 182.8861312117965 341.8435035703349 M99.0010020888372 435.7281507703682 C130.6477049755971 400.0448105884644, 160.9863217054349 364.7012033220004, 182.75217664633664 338.72267425731934 M111.14284270833565 436.4051195919161 C128.74147139109698 414.16949124411053, 147.996336783682 391.79145238450093, 194.03473140975956 339.6921824381785 M110.36498389797603 435.8824847230801 C129.1998501784198 415.1706961530555, 148.48623152337555 394.2267870631017, 194.35551764128292 340.76436430323923 M125.14156702928337 437.41524107069915 C155.52859817261842 396.6261530753042, 186.3118692236914 363.9276139449522, 207.35264585140428 339.72862008288564 M123.15590972452516 435.6514830873368 C150.15345421026157 404.54019898358456, 176.08036382678225 373.57019216652924, 206.7226676518683 340.6509502815554 M137.74653972898471 435.6408533675058 C154.03072941682615 416.88086718486045, 169.51565849598586 397.27151485666366, 217.19053853020114 338.43454329659295 M135.94929989319158 434.7767397962438 C154.73077912199707 413.47323067667224, 174.60818601797752 392.8643005138332, 219.93719930193427 338.9966556274711 M145.4558900799007 437.6800674442502 C165.29406047097717 416.7070004829897, 180.35232475646782 395.64798901395216, 233.14800356236097 341.41753569746453 M147.73283293180734 435.58847182978155 C181.76299584598468 398.75335133879065, 213.7240978214124 360.6691774728931, 231.23371494743785 339.2918047428417 M158.1295594192517 438.3412702493193 C191.1335420103712 399.3461581381895, 221.1426373878625 366.66766818331195, 241.82688115784484 338.50637230991464 M158.220599042339 435.3149162950283 C181.58442526214708 412.5460664199428, 203.29219836473592 388.33666928495984, 243.3621001474611 339.0622200763978 M173.22291226497967 437.9649113966814 C200.78656971508835 400.8121420295825, 231.41460926069078 364.87364434572413, 253.68457359137307 340.9463541448298 M170.2976487168495 436.4681796052689 C192.99880904689275 414.27781946942133, 211.4407157180651 390.30048573227486, 253.4017771352205 340.5786192039675 M182.55975714180508 437.3923856472234 C201.74922445935573 417.5106970705625, 216.76272236535914 395.0509486841741, 268.06765631838766 338.9210948786787 M182.45395106355548 436.0271824613317 C216.0085761537867 399.0480342783941, 247.3352739394929 362.61048443202736, 266.7062534340291 340.5424640075858 M197.16595769718882 436.88781958785944 C221.76254994050777 403.08738676259486, 251.5311334043468 373.8862664382845, 279.984898801109 337.92140333484815 M195.82894242278894 435.82350903069704 C223.93685780629673 404.24860422954407, 251.51334641979142 372.92202775172234, 277.83708018335346 339.9689056568309 M206.72274867234185 436.53340621175346 C230.72586510473303 403.37469542047626, 256.958300348796 377.73089390557715, 291.14087562449816 339.26707016275225 M207.62577983800685 434.85080402772047 C238.02387720164035 400.1191529699856, 268.1547112125991 365.20054985885105, 291.4084007547757 340.77263378793333 M220.1761109022251 433.57002991831973 C237.30302131490143 416.6944163507989, 252.85340270008896 396.17810416305207, 301.58928246726595 340.50220241805505 M220.13165355579133 435.7694602136325 C236.600446614667 413.29384839735303, 256.9900990267524 391.42339859954757, 301.70120422300266 340.37727358399854 M229.790087286559 436.59260109344484 C260.22372238503766 398.7927089540944, 295.0777413972993 364.2826404619373, 313.5423111018733 341.9358485626705 M229.53204049685883 437.21601547676124 C257.4277958880203 405.9769763270353, 284.05028443911675 374.95833139940015, 315.48363786586646 340.52007327500695 M240.5803939057855 436.8718427885975 C261.8313282469618 414.38129101572315, 279.1074710921298 389.4449180088363, 322.37549313361666 348.4750096867678 M242.15525287202246 437.2890020116418 C266.5046744612696 408.34369445657734, 290.19144372147264 382.31076927691424, 321.77678840340957 345.59897785981906 M254.0874408471251 435.52317320216446 C267.65138881764847 420.5493256798618, 284.6301275014326 401.67908392890877, 322.3900784934664 358.7352016449837 M255.4063965082487 436.53189130035224 C279.3693591046683 407.3215973893056, 303.9758180720067 378.662314782207, 321.2279979907235 360.94725374238806 M265.60994332053855 437.82849235855974 C283.1151621524946 419.3572015308107, 297.79876982818746 398.21356658030453, 320.146646554029 375.46224173553173 M267.1119194646195 436.8189432916231 C277.54202767236006 421.52023121166934, 288.9088760773085 408.92222093693874, 320.50541283391084 374.46933791086894 M278.85823762973695 435.23151668844747 C286.2390872433678 426.1319681499759, 298.47527361711593 414.9245474856473, 322.52906443377503 388.78590362554877 M278.13386550334207 435.96460348105995 C288.66946004022714 423.0669970616441, 299.6146781687692 411.2270978426226, 320.9728789943132 388.15855536296493 M290.70759714319075 435.6331136089645 C300.4244866215621 424.5032911054168, 306.6405614575665 414.99165299024645, 318.52657034492483 403.60402738842157 M289.2341275607579 437.2828999913658 C297.6931955227834 427.25185897884955, 304.44400583065016 419.44350811986175, 320.6329632363971 400.5952916432542 M299.93184341009186 437.01308820282924 C308.86445541219024 433.166152239926, 311.94898198186183 425.77333875837616, 321.8113256804204 416.2590157631188 M302.3689963633049 435.82329275145077 C307.9383239713687 431.29272635992584, 311.9096398625238 423.0254333486244, 320.51544261472543 415.78755160252183 M315.24086339299754 435.0051080530684 C316.3741723555682 433.97250099461667, 317.735914404315 430.28051605654997, 321.53869296473573 428.2353856198788 M314.23923927657955 436.2073844421131 C316.7969059844949 433.6281145657994, 318.62343891181206 431.68972734039016, 321.2386064447522 428.1589794916632" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M38.74383221231048 341.657830439043 C94.35303074089046 339.28471416670567, 150.87565385056996 337.791750669415, 319.46976074024417 340.1431763260468 M39.90422696238126 340.1785791057567 C113.00207423225464 340.9913537516595, 187.2667268626784 341.2912374045514, 320.0074031051243 339.41574045629335 M319.58350630188363 342.0326982236467 C317.39539094623854 370.74556609159333, 319.04864542838857 407.0795375323854, 318.97302743876355 437.2074965543114 M319.11221035155467 339.1058798958082 C319.13530197559857 369.9825754587073, 318.82419876425524 402.4652269561309, 320.70515956762245 437.0574887041468 M320.56146250812947 437.2195999172442 C239.91289730459513 436.06221888713156, 159.7570694054879 438.8399839166869, 40.82004704900069 435.5510523125202 M320.6645478093987 435.59825048446874 C248.44041763230638 434.9305361183, 178.75142957177638 434.5643130859092, 39.21309834919167 435.38037183091507 M41.91046312833205 435.1656677163206 C38.34183750807271 408.7686935520731, 40.98058755754456 381.92780220871794, 38.0976819046773 338.4032206780277 M40.8701640042942 437.0866512854118 C41.071346455019345 409.1186757121701, 40.71218692730032 382.96069965255447, 39.6124013024848 340.7232722435612" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="180" y="370" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">the score s</text><text x="180" y="397" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">s = −ε / σₜ</text><text x="180" y="421" text-anchor="middle" style="font-family:Kalam,cursive;font-size:13px;font-weight:400;fill:var(--muted)">score SDEs (Song et al., 2021)</text><path d="M639.8185857187406 340.2086932577796 C639.8185857187406 340.2086932577796, 639.8185857187406 340.2086932577796, 639.8185857187406 340.2086932577796 M639.8185857187406 340.2086932577796 C639.8185857187406 340.2086932577796, 639.8185857187406 340.2086932577796, 639.8185857187406 340.2086932577796 M638.774065073933 351.922836423275 C642.6615898428272 349.08162917834176, 649.2610848689786 341.348700524544, 653.0596678569713 340.91301279943883 M639.2493605698594 353.1017202866505 C642.0596164559298 351.6319743722659, 644.3795135248356 347.5104936532227, 651.7502396906107 340.9559944071898 M638.0928135019711 369.31839491767533 C648.5798719388168 354.7741453679992, 656.8523256395762 347.70377474112723, 664.5188395793884 338.73965061599836 M640.6588011602389 367.90864716461584 C645.063925338542 361.81330440757904, 650.410464381562 355.2296593764725, 663.6491365232374 339.2034271487918 M639.6074582757308 380.3465557212586 C649.3476007412688 372.06033562009486, 657.1804449852968 360.77256639384734, 677.232197269782 338.61714861303085 M639.4404719158421 381.2479775344132 C654.1536745108074 364.92297943159224, 666.4274002808075 350.8902841624596, 675.0406603989504 339.4408566997245 M641.1032306476643 392.8049099940882 C656.701663929118 377.56024385841806, 672.4503838180922 355.26703660686024, 689.8369901065073 341.96059519175896 M639.6998936426506 394.224340982603 C651.5628313432625 383.0245188343078, 663.5732086466458 369.5522635063673, 688.0482564048666 339.9960832915966 M639.0443688299921 408.7162978095507 C661.3868245667364 384.5060836749731, 685.2161099687836 358.23073357567466, 700.4707961455348 339.38042808181945 M640.3021331051312 409.15562305376324 C653.9583994423089 392.66516483474527, 666.8004012658912 375.2680459742445, 700.3229776335775 340.93303739086133 M638.8098571286846 423.8122506472832 C656.132399279863 402.9803617677097, 670.5358317521393 388.46500151748023, 711.0384916235564 339.5972726784308 M638.8712902708754 422.2514573393474 C666.9426139230117 390.7446182615431, 693.162154898304 360.5701501394221, 710.9258999677712 339.58545627017844 M639.7709780304339 434.109487652504 C661.1591969796568 410.32596840716286, 683.7032022994264 386.76256252193076, 722.4852717840429 339.1108245237299 M639.3289878975711 435.9037842370152 C661.9066328039157 408.26284070551367, 685.8026412208292 380.9420738773261, 723.1120804062895 339.2898289897513 M651.5733955505392 437.9940256733402 C674.0488457365276 414.19561655982034, 689.7700503438627 392.4287783356136, 734.3302619932045 341.8216359858808 M652.687392380889 435.43502348796017 C670.3003456225968 414.17563095652815, 688.3274838802653 395.68492845712325, 736.4251056800888 340.2212455764354 M662.1875197543923 437.55199285218856 C690.5093251426733 407.8031935367149, 716.4807146281288 378.13637113306316, 747.819833711003 338.7264508392976 M664.5605157759667 435.24295026827036 C684.6264825249698 412.90927958982996, 707.4308152779715 386.3954143091411, 748.1613457232518 339.7527941039612 M674.5132464099346 434.29180755301746 C699.4726549090977 411.9794719307719, 723.127114606737 386.4687756365615, 758.2369794911485 339.75882926464425 M674.6798929498667 435.3869366162307 C707.238688371156 397.61653801366157, 740.7518538537216 361.7393027481507, 758.9402661729849 340.110248192934 M687.5439152918871 437.69783328671383 C716.7780338295457 401.4705251516459, 746.9826705614828 369.8847175203377, 771.8546289521121 339.6844824290578 M686.7194773858062 435.85566885766946 C709.2543791202218 413.0338512988823, 730.5735873848034 387.37981245879365, 771.8686818406303 339.53146409105744 M698.19992129771 436.5521653901917 C726.8533164134177 407.44002518280524, 752.331743976842 374.3835551867391, 782.4692598930855 342.3230591652566 M699.2401897991166 435.4529967850671 C730.6476829151654 400.04730200410194, 760.7406316588772 364.986303427543, 782.3353053276256 339.20222985224103 M711.3820304186152 436.129965606615 C728.9806591013764 413.8943372588094, 748.2355244939615 391.51629839919985, 794.273919120039 339.41702845287745 M710.6041716082556 435.60733073777897 C729.4390378886992 414.89554216775434, 748.7254192336551 393.9516330778006, 794.5947053515624 340.48921031793816 M725.3807547395628 437.140087085398 C755.7677858828979 396.3509990900031, 786.5510569339708 363.6524599596511, 807.5918335616838 339.4534660975845 M723.3950974348046 435.3763291020357 C750.3926419205411 404.26504499828343, 776.3195515370618 373.2950381812281, 806.9618553621477 340.3757962962543 M737.3296684102737 436.1204089624276 C753.7550407258155 417.2005846001163, 769.3744901105185 397.43648436229273, 817.4297262404806 338.15938931129193 M735.5324285744805 435.25629539116557 C754.469470337227 413.78361518370104, 774.497688178729 393.0011968736709, 820.1763870122138 338.7215016421701 M745.0390187611896 438.1596230391719 C764.8771891522662 417.1865560779114, 779.9354534377569 396.12754460887385, 832.73113224365 341.8970912923862 M747.3159616130963 436.06802742470325 C781.3461245272737 399.23290693371234, 813.3072265027015 361.1487330678148, 830.8168436287268 339.7713603377634 M758.3687471295312 438.0661162640182 C791.1270357079243 399.3490562120688, 820.8928156284857 366.950468671832, 841.4100098391337 338.9859279048363 M758.4597867526185 435.0397623097272 C781.647586062581 412.4581684248912, 803.1880137702093 388.4412801453649, 842.94522882875 339.54177567131944 M773.4620999752591 437.6897574113802 C801.0257574253677 400.5369880442813, 831.6537969709702 364.598490360423, 853.9237613016526 340.67120015952867 M770.536836427129 436.1930256199677 C793.2379967571723 414.00266548412014, 811.6799034283445 390.0253317469737, 853.6409648455 340.30346521866636 M782.7989448520846 437.1172316619223 C801.9884121696352 417.2355430852614, 817.0019100756386 394.77579469887297, 868.3068440286672 338.64594089337754 M782.693138773835 435.75202847603055 C816.2477638640663 398.77288029309295, 847.5744616497725 362.33533044672623, 866.9454411443087 340.2673100222847 M797.4051454074684 436.6126656025584 C822.0017376507872 402.81223277729384, 851.7703211146262 373.6111124529835, 880.2240865113885 337.64624934954713 M796.0681301330684 435.548355045396 C824.1760455165762 403.97345024424305, 851.7525341300709 372.6468737664213, 878.0762678936329 339.69375167152987 M806.3058773536309 437.0129618066751 C830.5045840285587 403.61453339597097, 856.9401035743101 377.737110116407, 891.3800633347777 338.9919161774512 M807.208908519296 435.3303596226421 C837.849746027473 400.3161825938883, 868.2250159872162 365.116388089683, 891.6475884650552 340.49747980263226 M819.759239583514 434.0495855132414 C836.8861499961904 417.1739719457206, 852.4365313813778 396.65765975797376, 901.1724111485548 340.98175801297674 M819.7147822370803 436.2490158085542 C836.183575295956 413.7734039922747, 856.5732277080413 391.90295419446926, 901.2843329042915 340.85682917892024 M830.0292749968386 436.3174471081437 C860.2161849914703 398.80787757112336, 894.8243576587419 364.5806229472736, 913.1254397831623 342.41540415759215 M829.7712282071384 436.9408614914601 C857.4572243035433 405.94018269495456, 883.8716881198669 375.16084285012266, 915.0667665471555 340.9996288699286 M840.819581616065 436.5966888032964 C861.910759536003 414.29671848427165, 879.0240866085302 389.54764359842795, 921.9586218149055 348.9545652816895 M842.3944405823019 437.01384802634067 C866.5464545684339 408.3008770782069, 890.035696644208 382.4951809310784, 921.3599170846985 346.07853345474075 M854.3266285574045 435.2480192168634 C867.890576527928 420.2741716945607, 884.8693152117121 401.4039299436077, 922.6292662037459 358.46004765968263 M855.6455842185283 436.25673731505117 C879.6085468149478 407.04644340400455, 904.2150057822862 378.3871607969059, 921.467185701003 360.672099757087 M865.1930720018275 438.3080479534814 C882.8868320503293 419.61604626988924, 897.758219202886 398.25639574167445, 920.3858342643085 375.1870877502306 M866.6950481459086 437.29849888654473 C877.2682082484429 421.8327759153627, 888.7730293397069 409.0760462212353, 920.7446005441903 374.1941839255678 M878.4413663110261 435.71107228336916 C885.8222159246569 426.61152374489757, 898.058402298405 415.404103080569, 922.1121931150642 389.2654592204704 M877.7169941846312 436.44415907598164 C888.2525887215162 423.5465526565658, 899.1978068500583 411.7066534375443, 920.5560076756024 388.63811095788657 M890.9467848534703 435.3579596236634 C900.6636743318415 424.2281371201157, 906.879749167846 414.7164990049454, 918.7657580552044 403.32887340312055 M889.4733152710374 437.00774600606474 C897.932383233063 426.97670499354854, 904.6831935409297 419.1683541345607, 920.8721509466766 400.3201376579532 M900.1710311203714 436.73793421752816 C909.1036431224696 432.89099825462495, 912.1881696921413 425.498184773075, 922.0505133906997 415.98386177781765 M902.6081840735844 435.5481387661497 C908.1775116816483 431.01757237462476, 912.1488275728034 422.7502793633233, 920.7546303250049 415.5123976172207 M915.3765007439357 434.8405583031064 C916.3964788102493 433.9112119504998, 917.6220466541214 430.58842550623973, 921.0445473585002 428.7478081132357 M914.4750390391595 435.9226070532466 C916.7769390762834 433.60126416456427, 918.4208187108687 431.85671566169594, 920.7744694905149 428.67904259784166" style="stroke:var(--dot);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M638.7438322123105 341.657830439043 C694.3530307408904 339.28471416670567, 750.87565385057 337.791750669415, 919.4697607402442 340.1431763260468 M639.9042269623812 340.1785791057567 C713.0020742322546 340.9913537516595, 787.2667268626784 341.2912374045514, 920.0074031051244 339.41574045629335 M919.5835063018836 342.0326982236467 C917.3953909462385 370.74556609159333, 919.0486454283886 407.0795375323854, 918.9730274387636 437.2074965543114 M919.1122103515547 339.1058798958082 C919.1353019755985 369.9825754587073, 918.8241987642552 402.4652269561309, 920.7051595676224 437.0574887041468 M920.5614625081295 437.2195999172442 C839.9128973045952 436.06221888713156, 759.757069405488 438.8399839166869, 640.8200470490007 435.5510523125202 M920.6645478093986 435.59825048446874 C848.4404176323063 434.9305361183, 778.7514295717764 434.5643130859092, 639.2130983491917 435.38037183091507 M641.910463128332 435.1656677163206 C638.3418375080727 408.7686935520731, 640.9805875575445 381.92780220871794, 638.0976819046773 338.4032206780277 M640.8701640042942 437.0866512854118 C641.0713464550194 409.1186757121701, 640.7121869273003 382.96069965255447, 639.6124013024848 340.7232722435612" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="780" y="370" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">the velocity v</text><text x="780" y="397" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">v = α̇ₜ x₀ + σ̇ₜ ε</text><text x="780" y="421" text-anchor="middle" style="font-family:Kalam,cursive;font-size:13px;font-weight:400;fill:var(--muted)">flow matching: next post</text><path d="M322.3614419111051 152.16249095266684 C331.62768638110236 162.70453210096656, 341.8651904357083 173.5660145155698, 371.30835049534215 214.1867606616579 M323.875072671799 150.23294040900655 C336.8854806267928 166.2450669513654, 350.67049900826925 183.39839167907948, 372.0096566859167 213.23788588535973" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M371.0987930511078 215.18937002396677 C368.81754804091366 213.0604418700485, 367.20900489849777 210.97801499337467, 361.8489451967317 209.0495563883424 M371.93128996948946 214.1281172249536 C369.1435385777541 212.10990814984598, 367.25189330814754 211.00158554651765, 362.2346636015477 208.52767526137842" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M371.0987930511078 215.18937002396677 C370.32993723475096 211.8839897794682, 370.2535396525526 208.6524537326312, 369.5043385970893 203.30801133807415 M371.93128996948946 214.1281172249536 C371.08150340611826 210.66588032739415, 371.1948718667392 208.05379742639514, 369.89005700190535 202.78613021111016" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M323.0987930511078 151.18937002396677 C324.7633057586958 151.01331907855175, 327.06574498404945 150.95356544178404, 333.3902403481447 155.15588033948126 M323.93128996948946 150.1281172249536 C326.37959289298664 150.90931597799704, 329.60598180346307 152.4479270101386, 333.7759587529607 154.63399921251727" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M323.0987930511078 151.18937002396677 C323.2509165648585 152.18977116913203, 324.0212102299946 153.27912670252752, 325.73484694778705 160.89742538974951 M323.93128996948946 150.1281172249536 C324.4416280646225 152.35334380044887, 325.6630032448714 155.39571513026112, 326.1205653526031 160.37554426278552" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M634.3614419111051 152.16249095266684 C624.4143075135954 163.04741720467956, 615.4384327006944 173.9088996192828, 587.3083504953422 214.1867606616579 M635.875072671799 150.23294040900655 C623.7422504836034 165.4960110477533, 612.3840387218904 182.6493357754674, 588.0096566859168 213.23788588535973" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M587.0987930511078 215.18937002396677 C587.1766284820853 211.86887958889957, 587.9469221472215 208.63734354206255, 589.734846947787 203.30801133807415 M587.9312899694894 214.1281172249536 C588.1895101257461 210.6988895696376, 589.410885305995 208.0868066686386, 590.120565352603 202.78613021111016" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M587.0987930511078 215.18937002396677 C588.7285304086831 212.99064583864558, 591.0309696340368 210.90821896197176, 597.3902403481446 209.0495563883424 M587.9312899694894 214.1281172249536 C590.2615727578371 212.26238234252716, 593.4879616683136 211.1540597391988, 597.7759587529606 208.52767526137842" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M635.0987930511078 151.18937002396677 C634.4042253175242 152.20488135970066, 634.3278277353257 153.29423689309615, 633.5043385970894 160.89742538974951 M635.9312899694894 150.1281172249536 C635.3336213449946 152.32033455820542, 635.4469898056157 155.36270588801767, 633.8900570019053 160.37554426278552" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M635.0987930511078 151.18937002396677 C632.8523233909264 151.08311510995466, 631.2437802485105 151.02336147318695, 625.8489451967317 155.15588033948126 M635.9312899694894 150.1281172249536 C633.2615587129036 150.7568417853159, 631.369913443297 152.29545281745746, 626.2346636015477 154.63399921251727" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M322.3614419111051 364.1624909526669 C331.15971725359526 347.4855787053317, 341.3972213082012 331.12810772430004, 371.30835049534215 290.1867606616579 M323.875072671799 362.2329404090066 C335.29729302683694 342.62549091518036, 349.0823114083134 324.15923960670943, 372.0096566859167 289.23788588535973" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M371.0987930511078 291.1893700239668 C370.5250636395032 292.22288771041457, 370.56915783040847 293.3323999104263, 370.10637795194276 300.99813854429186 M371.93128996948946 290.1281172249536 C371.4924767711724 292.3514097160811, 371.76352454485937 295.42015869196734, 370.4920963567588 300.4762574173279" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M371.0987930511078 291.1893700239668 C368.91327116688916 291.18897426301817, 367.36383765236013 291.23613465940707, 362.1442875368933 295.69007826759224 M371.93128996948946 290.1281172249536 C369.3451497089878 290.8990572996075, 367.53085715219066 292.5775793885044, 362.5300059417093 295.16819714062825" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M323.0987930511078 363.1893700239668 C323.0557901601063 359.85087323818567, 323.70559205213874 356.5991805247324, 325.1328075929336 351.2072981835318 M323.93128996948946 362.1281172249536 C324.03065469956834 358.6678144117619, 325.09435056675125 356.0293538646889, 325.51852599774963 350.6854170565678" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M323.0987930511078 363.1893700239668 C324.6675826327203 360.88478668558207, 326.9109122301871 358.69544577575164, 333.0948980079831 356.5153584602314 M323.93128996948946 362.1281172249536 C326.17798176175296 360.12016682823554, 329.32701795941995 358.8719331681519, 333.4806164127991 355.99347733326744" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M634.3614419111051 364.1624909526669 C623.9463383860882 347.8284638090447, 614.9704635731872 331.47099282801304, 587.3083504953422 290.1867606616579 M635.875072671799 362.2329404090066 C622.1540628836475 341.8764350115683, 610.7958511219346 323.41018370309735, 588.0096566859168 289.23788588535973" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M587.0987930511078 291.1893700239668 C588.7060342788849 291.1212879918014, 590.9493638763516 291.1684483881903, 597.094898007983 295.69007826759224 M587.9312899694894 290.1281172249536 C590.3084784634518 291.0469225772161, 593.4575146611187 292.725444666113, 597.480616412799 295.16819714062825" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M587.0987930511078 291.1893700239668 C587.1307713406303 292.2120781516507, 587.7805732326628 293.32159035166245, 589.1328075929337 300.99813854429186 M587.9312899694894 290.1281172249536 C588.2851248646682 292.3750239347212, 589.3488207318511 295.44377291060744, 589.5185259977496 300.4762574173279" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M635.0987930511078 363.1893700239668 C632.8748195207246 360.95247295679883, 631.3253860061956 358.7631320469684, 626.1442875368933 356.5153584602314 M635.9312899694894 362.1281172249536 C633.2146530072889 359.9723015506269, 631.4003604504919 358.72406789054327, 626.5300059417093 355.99347733326744" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M635.0987930511078 363.1893700239668 C634.4500824589792 359.86168279694954, 634.4941766498844 356.60999008349626, 634.1063779519427 351.2072981835318 M635.9312899694894 362.1281172249536 C635.2380066060725 358.64420019312183, 635.5090543797595 356.0057396460488, 634.4920963567587 350.6854170565678" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="480" y="455" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--ink)">the losses differ only in how much weight each noise level gets</text></svg></figure>

Given $\boldsymbol{x}_t$ and $t$, a prediction of any one of these determines the others:

| predict | in terms of $\hat{\boldsymbol{x}}_0$ and $\hat{\boldsymbol{\varepsilon}}$ | where it is used |
|---|---|---|
| clean data | $\hat{\boldsymbol{x}}_0 = (\boldsymbol{x}_t - \sigma_t\hat{\boldsymbol{\varepsilon}})/\alpha_t$ | denoisers, EDM |
| noise | $\hat{\boldsymbol{\varepsilon}} = (\boldsymbol{x}_t - \alpha_t\hat{\boldsymbol{x}}_0)/\sigma_t$ | DDPM |
| score | $\hat{\boldsymbol{s}} = -\hat{\boldsymbol{\varepsilon}}/\sigma_t = (\alpha_t\hat{\boldsymbol{x}}_0 - \boldsymbol{x}_t)/\sigma_t^2$ | score SDEs |
| velocity | $\hat{\boldsymbol{v}} = \dot\alpha_t\hat{\boldsymbol{x}}_0 + \dot\sigma_t\hat{\boldsymbol{\varepsilon}}$ | flow matching (next post) |

The first three rows are rearrangements of $\boldsymbol{x}_t = \alpha_t\hat{\boldsymbol{x}}_0 + \sigma_t\hat{\boldsymbol{\varepsilon}}$ and Tweedie's formula. The velocity is the time derivative of the path $\alpha_t\boldsymbol{x}_0 + \sigma_t\boldsymbol{\varepsilon}$ with the endpoints held fixed. The choice matters only for how the loss weights different noise levels: for example $\lVert\hat{\boldsymbol{x}}_0 - \boldsymbol{x}_0\rVert^2 = (\sigma_t/\alpha_t)^2\lVert\hat{\boldsymbol{\varepsilon}} - \boldsymbol{\varepsilon}\rVert^2$, so $\boldsymbol{x}_0$-prediction puts far more weight on high noise than $\boldsymbol{\varepsilon}$-prediction does.

## 5. Continuous time: the SDE, Fokker–Planck, and an ODE with the same marginals

### The forward process as an SDE

Let the step size go to zero. The noising becomes a stochastic differential equation $d\boldsymbol{x} = f(t)\,\boldsymbol{x}\,dt + g(t)\,d\boldsymbol{W}$, whose meaning for small $h$ is $\boldsymbol{x}_{t+h} = (1 + f h)\boldsymbol{x}_t + g\sqrt{h}\,\boldsymbol{\xi}$ with $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

:::proposition[Which SDE gives the marginals $\alpha_t$, $\sigma_t$]
The SDE with $f = \dot\alpha_t/\alpha_t$ and $g^2 = \frac{d\sigma_t^2}{dt} - 2\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2$, started at $\boldsymbol{x}_0$, has $\boldsymbol{x}_t \sim \mathcal{N}(\alpha_t\boldsymbol{x}_0, \sigma_t^2\mathbf{I})$.
:::

:::proof
**Step 1. The process is Gaussian.** Each small step is linear in $\boldsymbol{x}$ plus independent Gaussian noise, so $\boldsymbol{x}_t$ stays Gaussian given $\boldsymbol{x}_0$, with some mean $\boldsymbol{m}_t$ and covariance $v_t\mathbf{I}$.

**Step 2. The mean.** Taking expectations of the small step, $\boldsymbol{m}_{t+h} = (1 + fh)\boldsymbol{m}_t$, so $\dot{\boldsymbol{m}} = f\boldsymbol{m}$. With $f = \dot\alpha/\alpha$, the solution from $\boldsymbol{m}_0 = \boldsymbol{x}_0$ is $\boldsymbol{m}_t = \alpha_t\boldsymbol{x}_0$, because $\frac{d}{dt}(\alpha_t\boldsymbol{x}_0) = \frac{\dot\alpha_t}{\alpha_t}\alpha_t\boldsymbol{x}_0$.

**Step 3. The variance.** Variances of independent terms add: $v_{t+h} = (1 + fh)^2v_t + g^2h = v_t + (2fv_t + g^2)h + O(h^2)$, so $\dot v = 2fv + g^2$.

**Step 4. Check $v = \sigma^2$.** With the stated $g^2$, $2f\sigma^2 + g^2 = 2\frac{\dot\alpha}{\alpha}\sigma^2 + \frac{d\sigma^2}{dt} - 2\frac{\dot\alpha}{\alpha}\sigma^2 = \frac{d\sigma^2}{dt}$, and $v_0 = 0 = \sigma_0^2$.
:::

### How densities move: the Fokker–Planck equation

An SDE moves samples; the density moves with them. In one dimension, for $dx = a(x, t)\,dt + g(t)\,dW$:

:::theorem[Fokker–Planck]
The density $p_t$ of $x_t$ satisfies $\partial_tp_t = -\partial_x\big(a\,p_t\big) + \tfrac12g^2\,\partial_x^2p_t$.
:::

:::proof
**Step 1. Follow a test function.** For a smooth $\varphi$ that vanishes at infinity, look at $\mathbb{E}[\varphi(x_{t+h})]$ with $x_{t+h} = x_t + \delta$ and $\delta = a\,h + g\sqrt{h}\,\xi$.

**Step 2. Taylor expansion.** $\varphi(x + \delta) = \varphi(x) + \varphi'(x)\delta + \tfrac12\varphi''(x)\delta^2 + O(\delta^3)$. Given $x_t$, $\mathbb{E}[\delta] = a\,h$, $\mathbb{E}[\delta^2] = g^2h + O(h^2)$ and $\mathbb{E}[\lvert\delta\rvert^3] = O(h^{3/2})$.

**Step 3. Divide by $h$ and let $h \to 0$.** $\frac{d}{dt}\mathbb{E}[\varphi(x_t)] = \mathbb{E}\big[a\,\varphi'(x_t) + \tfrac12g^2\varphi''(x_t)\big]$.

**Step 4. Write both sides as integrals against $p_t$.** The left side is $\int\varphi\,\partial_tp_t\,dx$. Integrating by parts once and twice on the right (boundary terms vanish), $\int a\,p_t\,\varphi'\,dx = -\int\varphi\,\partial_x(a\,p_t)\,dx$ and $\int p_t\,\varphi''\,dx = \int\varphi\,\partial_x^2p_t\,dx$.

**Step 5. True for every $\varphi$.** So the integrands agree: $\partial_tp_t = -\partial_x(a\,p_t) + \tfrac12g^2\partial_x^2p_t$.
:::

With $g = 0$ this is the *continuity equation* $\partial_tp_t = -\partial_x(a\,p_t)$ of a deterministic flow $\dot x = a(x, t)$: mass is carried along without being created or destroyed. The same equations hold in $d$ dimensions with $\partial_x$ replaced by the divergence and $\partial_x^2$ by the Laplacian.

### A deterministic flow with the same marginals

:::theorem[The probability-flow ODE (Song et al., 2021)]
The ODE

$$
\frac{d\boldsymbol{x}}{dt} = f(t)\,\boldsymbol{x} - \tfrac12g(t)^2\,\nabla\log p_t(\boldsymbol{x})
$$

moves the density through exactly the same $p_t$ as the forward SDE.
:::

:::proof
**Step 1. The ODE's density** obeys the continuity equation $\partial_tp = -\partial_x(b\,p)$ with $b = fx - \tfrac12g^2\,\partial_x\log p$ (one dimension; the general case is identical with divergences).

**Step 2. Simplify $b\,p$.** Since $p\,\partial_x\log p = \partial_xp$, $b\,p = f\,x\,p - \tfrac12g^2\,\partial_xp$.

**Step 3. Compare with Fokker–Planck.** $-\partial_x(b\,p) = -\partial_x(f\,x\,p) + \tfrac12g^2\,\partial_x^2p$, which is exactly the right-hand side of the Fokker–Planck equation of the forward SDE (drift $a = fx$).

**Step 4. Same equation, same start.** Both families of densities satisfy the same equation from the same $p_0$, so they coincide.
:::

That answers question 4. The ODE moves each point deterministically, yet the *population* of points has the same distribution at every time as under the noisy process. Run backwards from $t = 1$, it maps Gaussian noise to data one-to-one: it is a normalising flow in continuous time, which is where the next post on flow matching begins.

The noisy reverse process can be derived from the ODE in one more line, using Langevin dynamics from [the map](/aboutme/blog/generative-models-one-map/), Section 6.

:::proposition[The reverse-time SDE (Anderson, 1982)]
Running time backwards ($\tau = 1 - t$), the SDE

$$
d\boldsymbol{x} = \big[-f\,\boldsymbol{x} + g^2\,\nabla\log p_t(\boldsymbol{x})\big]\,d\tau + g\,d\boldsymbol{W}_\tau
$$

has the same marginals $p_t$, read in reverse.
:::

:::proof
**Step 1. Reverse the ODE.** In backward time the probability-flow ODE is $d\boldsymbol{x}/d\tau = -f\boldsymbol{x} + \tfrac12g^2\nabla\log p_t$, and it carries $p_1$ back through $p_t$ to $p_0$.

**Step 2. Add a Langevin step at rate $c = \tfrac12g^2$.** Langevin dynamics $d\boldsymbol{x} = c\,\nabla\log p\,d\tau + \sqrt{2c}\,d\boldsymbol{W}$ leaves the current density $p$ unchanged: its Fokker–Planck right-hand side is $-\partial_x(c\,\partial_xp) + c\,\partial_x^2p = 0$ (the map, Section 6).

**Step 3. Add the two motions.** The Fokker–Planck right-hand side is linear in the drift and in the diffusion coefficient, so the sum of the two motions changes the density exactly as the ODE alone does.

**Step 4. Collect terms.** The drift is $-f\boldsymbol{x} + \tfrac12g^2\nabla\log p + \tfrac12g^2\nabla\log p = -f\boldsymbol{x} + g^2\nabla\log p$, and the noise is $\sqrt{2c}\,d\boldsymbol{W} = g\,d\boldsymbol{W}$.
:::

So the reverse SDE is the reverse ODE plus a Langevin corrector that keeps nudging samples back towards the current distribution. That is why the noisy sampler forgives a network's errors better than the deterministic one, which Section 8 measures.

## 6. Sampling, and why it takes many steps

### DDIM is Euler's method in the right coordinates

The probability-flow ODE has to be solved numerically. The DDIM update of Song, Meng and Ermon (2021),

$$
\boldsymbol{x}_s = \alpha_s\hat{\boldsymbol{x}}_0 + \sigma_s\hat{\boldsymbol{\varepsilon}}, \qquad s < t,
$$

with $\hat{\boldsymbol{x}}_0$ and $\hat{\boldsymbol{\varepsilon}}$ predicted at $(\boldsymbol{x}_t, t)$, is one Euler step of it in well-chosen coordinates.

:::proposition[DDIM is Euler's method for the probability-flow ODE]
With $\boldsymbol{y} = \boldsymbol{x}/\alpha_t$ and $\lambda = \sigma_t/\alpha_t$, the probability-flow ODE becomes $\dfrac{d\boldsymbol{y}}{d\lambda} = \mathbb{E}[\boldsymbol{\varepsilon} \mid \boldsymbol{x}_t]$, and one Euler step in $\lambda$ is the DDIM update.
:::

:::proof
**Step 1. Rewrite $g^2$.** With $f = \dot\alpha/\alpha$, $g^2 = 2\sigma\dot\sigma - 2\frac{\dot\alpha}{\alpha}\sigma^2 = 2\sigma^2\big(\frac{\dot\sigma}{\sigma} - \frac{\dot\alpha}{\alpha}\big) = 2\sigma^2\frac{\dot\lambda}{\lambda}$, because $\frac{\dot\lambda}{\lambda} = \frac{\dot\sigma}{\sigma} - \frac{\dot\alpha}{\alpha}$.

**Step 2. Substitute the score.** $\nabla\log p_t = -\hat{\boldsymbol{\varepsilon}}/\sigma$ (Tweedie), so

$$
\frac{d\boldsymbol{x}}{dt} = \frac{\dot\alpha}{\alpha}\boldsymbol{x} + \sigma^2\frac{\dot\lambda}{\lambda}\cdot\frac{\hat{\boldsymbol{\varepsilon}}}{\sigma} = \frac{\dot\alpha}{\alpha}\boldsymbol{x} + \alpha\dot\lambda\,\hat{\boldsymbol{\varepsilon}},
$$

using $\sigma/\lambda = \alpha$.

**Step 3. Change variables to $\boldsymbol{y} = \boldsymbol{x}/\alpha$.** $\dot{\boldsymbol{y}} = \frac{1}{\alpha}\big(\dot{\boldsymbol{x}} - \frac{\dot\alpha}{\alpha}\boldsymbol{x}\big) = \dot\lambda\,\hat{\boldsymbol{\varepsilon}}$, so $d\boldsymbol{y}/d\lambda = \hat{\boldsymbol{\varepsilon}}$.

**Step 4. One Euler step from $t$ to $s$.** $\boldsymbol{y}_s = \boldsymbol{y}_t + (\lambda_s - \lambda_t)\hat{\boldsymbol{\varepsilon}}$, so $\boldsymbol{x}_s = \frac{\alpha_s}{\alpha_t}\boldsymbol{x}_t + \big(\sigma_s - \frac{\alpha_s\sigma_t}{\alpha_t}\big)\hat{\boldsymbol{\varepsilon}} = \alpha_s\frac{\boldsymbol{x}_t - \sigma_t\hat{\boldsymbol{\varepsilon}}}{\alpha_t} + \sigma_s\hat{\boldsymbol{\varepsilon}} = \alpha_s\hat{\boldsymbol{x}}_0 + \sigma_s\hat{\boldsymbol{\varepsilon}}$.
:::

The step is exact if $\hat{\boldsymbol{\varepsilon}}$ stays constant along the path. Its error comes entirely from how much the prediction changes between steps: the *curvature* of the paths.

### How many steps does a perfect network need?

On the toy the exact denoiser is available, so the network error is zero and only the discretisation error remains. Sampling 4000 points with DDIM:

| steps | modes covered | samples on a mode |
|---|---|---|
| 1 | 0 of 8 | 0% (all at the centre) |
| 2 | 0 of 8 | 5% |
| 4 | 8 of 8 | 69% |
| 8 | 8 of 8 | 96% |
| 16 | 8 of 8 | 99% |

One step lands every sample on the mean, as Tweedie predicted. Even with a perfect network, this simple two-dimensional problem needs about ten steps; images need tens. The paths are curved: early on the best guess is the blurry centre, and only later does it commit to a mode, so the direction of travel turns.

Three ways to need fewer steps define the rest of the series:

- **Better solvers.** Higher-order methods use the semi-linear structure of the ODE: Heun's method in EDM (Karras et al., 2022), DPM-Solver (Lu et al., 2022). They reduce, but cannot remove, the curvature error.
- **Straighter paths.** If the paths from noise to data were straight, one Euler step would be exact. Flow matching and rectified flow learn such paths: the next post.
- **Learn the jump directly.** Train a network to map $\boldsymbol{x}_t$ straight to the end of its ODE path: consistency models, flow maps, MeanFlow, and drifting models. The last post.

## 7. Guidance: what it really does

A conditional model, for a label or a text prompt $y$, needs the score of $p_t(\boldsymbol{x} \mid y)$.

:::proposition[The conditional score]
$\nabla\log p_t(\boldsymbol{x} \mid y) = \nabla\log p_t(\boldsymbol{x}) + \nabla\log p_t(y \mid \boldsymbol{x})$.
:::

:::proof
**Step 1. Bayes' rule.** $\log p_t(\boldsymbol{x} \mid y) = \log p_t(\boldsymbol{x}) + \log p_t(y \mid \boldsymbol{x}) - \log p(y)$.

**Step 2. Take the gradient in $\boldsymbol{x}$.** $p(y)$ does not depend on $\boldsymbol{x}$, so its term vanishes.
:::

*Classifier guidance* (Dhariwal and Nichol, 2021) adds a classifier's gradient to an unconditional score, scaled up to sharpen the effect. *Classifier-free guidance* (Ho and Salimans, 2022) trains one network both with and without the label (dropping it at random) and combines the two scores:

$$
\tilde{\boldsymbol{s}}_w = (1 + w)\,\boldsymbol{s}(\boldsymbol{x} \mid y) - w\,\boldsymbol{s}(\boldsymbol{x}) = \nabla\log\big[p_t(\boldsymbol{x})\,p_t(y \mid \boldsymbol{x})^{1 + w}\big].
$$

The second equality follows from the proposition: $\boldsymbol{s}(\boldsymbol{x} \mid y) - \boldsymbol{s}(\boldsymbol{x}) = \nabla\log p_t(y \mid \boldsymbol{x})$, so $\tilde{\boldsymbol{s}}_w = \boldsymbol{s}(\boldsymbol{x}) + (1 + w)\nabla\log p_t(y \mid \boldsymbol{x})$. Guidance raises the classifier's probability to the power $1 + w$, so it favours points that are *very* clearly of class $y$.

That is what it does at each noise level. What it does to the final samples is subtler. The tilted distributions $p_t(\boldsymbol{x})\,p_t(y \mid \boldsymbol{x})^{1 + w}$ at different times are not the marginals of any single diffusion, so the sampler does not end at $p_0(\boldsymbol{x})\,p_0(y \mid \boldsymbol{x})^{1 + w}$ (Bradley and Nakkiran, 2025; Chidambaram et al., 2024). On the toy it can be watched exactly. Let the class be the three upper modes, and raise $w$:

<figure class="fig" data-guidance>
  <canvas data-figure="guidance" aria-label="Classifier-free guidance on the ring: samples for the class 'three upper modes' as the guidance weight grows."></canvas>
  <div class="controls">
    <label class="slider">guidance weight w <input type="range" min="0" max="5" step="0.25" value="0" data-w /></label>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

At $w = 0$ the samples split evenly over the three class modes (32%, 32%, 35%). As $w$ grows, the middle mode takes over: 68% of the samples at $w = 2$, 83% at $w = 4$. The samples also move outwards, to a mean radius of 2.14 at $w = 4$ against 2.00 for the data, away from the other five modes. Guidance buys samples that are unmistakably of the class, at the price of diversity and of fidelity to the data. Autoguidance (Karras et al., 2024) guides with a weaker version of the same model instead, which its authors report improves quality without reducing variety.

## 8. A diffusion model, trained

The lab at the top runs either the exact score or a network I trained on the ring. It is a multilayer perceptron with three hidden layers of 128 units, taking the noisy point and sinusoidal features of $t$, trained for 20,000 steps on the simple noise-prediction loss of Section 3. In the browser it runs as the same weights, reproducing the PyTorch outputs to within $4 \times 10^{-5}$. Sampling 4000 points:

| steps | DDIM (ODE), exact | DDIM, network | DDPM (SDE), exact | DDPM, network |
|---|---|---|---|---|
| 4 | 69% | 62% | 68% | 68% |
| 8 | 96% | 89% | 96% | 91% |
| 16 | 99% | 96% | 99% | 97% |
| 100 | 99% | 98% | 100% | 99% |

(share of samples on a mode; every run with 8 or more steps covers all 8 modes.) Two things stand out. With enough steps the trained network is close to the exact score. And the noisy sampler is more forgiving of the network's errors (99% against 98%), as Section 5 predicted: its Langevin part keeps pulling samples back towards the current distribution. Compared with [the map's](/aboutme/blog/generative-models-one-map/) flow (75% on a mode), VAE (53%) and GAN (68%, 7 modes), the diffusion model covers every mode *and* lands on them. It pays for this with tens of network evaluations per sample, against one for the others: the trilemma of the previous post, in numbers.

## 9. What is new, 2022 to 2026

| year | work | what it changes |
|---|---|---|
| 2022 | Rombach et al. (latent diffusion) | diffuse in the latent space of an autoencoder, which makes high-resolution images affordable |
| 2022 | Karras et al. (EDM) | a clean design space: noise-level parametrisation, preconditioning, Heun sampling |
| 2022 | Lu et al. (DPM-Solver) | fast high-order ODE solvers exploiting the semi-linear structure |
| 2023 | Peebles and Xie (DiT) | transformers replace U-Nets and scale predictably |
| 2023 | Kingma and Gao | weighted diffusion losses are ELBOs with noise augmentation |
| 2024 | Karras et al. (EDM2) | better training dynamics and post-hoc averaging of weights |
| 2024 | Karras et al. (autoguidance) | guide with a weaker version of the same model instead of the unconditional one |
| 2024 | Esser et al. (Stable Diffusion 3) | rectified-flow transformers at scale: flow matching, the next post |
| 2024–25 | Chidambaram et al.; Bradley and Nakkiran | what guidance really samples, and why it is not the tilted distribution |
| 2025 | Price et al. (GenCast) | a diffusion model for medium-range weather ensembles, better than ECMWF's ensemble on 97.2% of 1320 targets |
| 2025 | Nie et al. (LLaDA) | masked diffusion for language at 8B parameters, competitive with autoregressive models |

## Questions

:::question
Why can a diffusion model not generate in one step simply by applying its denoiser to pure noise?
:::

:::answer
The best one-step guess under squared error is $\mathbb{E}[\boldsymbol{x}_0 \mid \boldsymbol{x}_1]$. At $t = 1$ the noisy point carries no information about the data, so this is $\mathbb{E}[\boldsymbol{x}_0]$, the mean of the data. For the ring that is its centre; for images it is a grey blur. Many small steps let the guess commit gradually, as the animation in Section 4 shows.
:::

:::question
The ODE sampler is deterministic. Where does the diversity of its samples come from?
:::

:::answer
Entirely from the starting noise $\boldsymbol{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The ODE is a one-to-one map from noise to data, so different starting points end at different samples, distributed as $p_0$ by the probability-flow theorem.
:::

:::question
With $\boldsymbol{x}_0$-prediction instead of $\boldsymbol{\varepsilon}$-prediction and the same network, why would the samples change, if the two are equivalent?
:::

:::answer
They are equivalent at the optimum, but the losses weight noise levels differently: $\lVert\hat{\boldsymbol{x}}_0 - \boldsymbol{x}_0\rVert^2 = (\sigma_t/\alpha_t)^2\lVert\hat{\boldsymbol{\varepsilon}} - \boldsymbol{\varepsilon}\rVert^2$. A network of finite capacity spends it where the loss puts weight, so the two parametrisations make different errors at different noise levels, which the sampler then accumulates.
:::

:::question
In the guidance figure, why does the middle of the three class modes win as $w$ grows?
:::

:::answer
Guidance multiplies the density by $p_t(y \mid \boldsymbol{x})^{1 + w}$ at every noise level. At high noise, the classifier is most confident at points whose nearest modes are all in the class, which is the region around the middle mode, far from the other five. Those regions are amplified first, and the sampler commits to them early. The end result is less diverse than $p(\boldsymbol{x} \mid y)$ and is pushed away from the other classes, even off the data.
:::

:::question
The reverse SDE adds noise while it removes noise. Why does that help?
:::

:::answer
Section 5 decomposed it into the reverse probability-flow ODE plus a Langevin corrector at rate $\tfrac12g^2$. The corrector leaves the right distribution unchanged but pulls a wrong one towards it, so errors made by an imperfect network or a coarse step are partly undone instead of being carried along. The trained model shows it: 99% of samples on a mode with the SDE, 98% with the ODE.
:::

## References

- J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli. *Deep unsupervised learning using nonequilibrium thermodynamics.* ICML, 2015.
- J. Ho, A. Jain, P. Abbeel. *Denoising diffusion probabilistic models.* NeurIPS, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. *Score-based generative modeling through stochastic differential equations.* ICLR, 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- J. Song, C. Meng, S. Ermon. *Denoising diffusion implicit models.* ICLR, 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- P. Vincent. *A connection between score matching and denoising autoencoders.* Neural Computation, 2011.
- H. Robbins. *An empirical Bayes approach to statistics.* Berkeley Symposium, 1956.
- B. Efron. *Tweedie's formula and selection bias.* JASA, 2011.
- B. D. O. Anderson. *Reverse-time diffusion equation models.* Stochastic Processes and their Applications, 1982.
- D. P. Kingma, R. Gao. *Understanding diffusion objectives as the ELBO with simple data augmentation.* NeurIPS, 2023.
- T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the design space of diffusion-based generative models.* NeurIPS, 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, J. Zhu. *DPM-Solver: a fast ODE solver for diffusion probabilistic model sampling in around 10 steps.* NeurIPS, 2022.
- P. Dhariwal, A. Nichol. *Diffusion models beat GANs on image synthesis.* NeurIPS, 2021.
- J. Ho, T. Salimans. *Classifier-free diffusion guidance.* 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- M. Chidambaram, K. Gatmiry, S. Chen, H. Lee, J. Lu. *What does guidance do? A fine-grained analysis in a simple setting.* 2024. [arXiv:2409.13074](https://arxiv.org/abs/2409.13074)
- A. Bradley, P. Nakkiran. *Classifier-free guidance is a predictor-corrector.* TMLR, 2025. [arXiv:2408.09000](https://arxiv.org/abs/2408.09000)
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. *High-resolution image synthesis with latent diffusion models.* CVPR, 2022.
- W. Peebles, S. Xie. *Scalable diffusion models with transformers.* ICCV, 2023.
- T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila, S. Laine. *Analyzing and improving the training dynamics of diffusion models.* CVPR, 2024.
- T. Karras, M. Aittala, T. Kynkäänniemi, J. Lehtinen, T. Aila, S. Laine. *Guiding a diffusion model with a bad version of itself.* NeurIPS, 2024.
- P. Esser et al. *Scaling rectified flow transformers for high-resolution image synthesis.* ICML, 2024.
- I. Price et al. *Probabilistic weather forecasting with machine learning.* Nature, 2025.
- S. Nie et al. *Large language diffusion models.* NeurIPS, 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
