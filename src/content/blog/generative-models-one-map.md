---
title: "Generative models, one map: what each family can compute"
description: "The opener of a series that ends at one-step generators. What it means to learn a distribution; maximum likelihood as a KL divergence; change of variables and normalising flows; VAEs; GANs and their divergences, including MMD; energy-based models, the score and Langevin dynamics; autoregressive models. Every result derived step by step, and a flow, a VAE and a GAN trained on the same data to show how each one fails."
date: 2026-10-29
updated: 2026-10-29
cover: /blog/thumbs/generative.png
tags: [generative-models, normalizing-flows, vae, gan, score, tutorial]
---

A generative model turns noise into data. Below, Gaussian noise on the left is pushed through an invertible map into the plane on the right. Slide $t$ from 0 to 1 to apply the map, and hover over the right panel to see the density at a point computed from the formula this post derives:

<figure class="fig" data-push>
  <canvas data-figure="push" aria-label="Gaussian noise pushed through an invertible map; the grid and the samples are carried along, and the density on the right follows the change-of-variables formula."></canvas>
  <div class="controls">
    <label class="slider">apply the map, t <input type="range" min="0" max="1" step="0.01" value="1" data-t /></label>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

*What to try.* At $t = 0$ the map is the identity and the two panels agree. As $t$ grows, the grid on the right is stretched outwards, swirled and bent, and the two colours let you follow the upper and lower halves of the noise. The heat map is the density of $\boldsymbol{x}$, computed from the change-of-variables formula: it stays high where the grid cells stay small and thins out where they grow. At the circled point, $\log p_{\boldsymbol{z}} = -2.85$ and the map expands area there by $e^{0.84} \approx 2.3$, so $\log p_{\boldsymbol{x}} = -2.85 - 0.84 = -3.69$. Hover elsewhere to see the same bookkeeping.

This is the first post of a series on deep generative models: this map, then diffusion models, flow matching, and finally the one-step generators that are the newest part of the field. Each later post builds on what is set up here, so this one answers the questions that the others take for granted:

1. **What does it mean to learn a distribution**, and how do you measure success when all you have is samples? (Section 1.)
2. **What can each family compute**, a density, samples or the score, and at what cost? (Sections 2 to 7.)
3. **Why does each family fail the way it does?** I trained a flow, a VAE and a GAN on the same data to show it. (Section 8.)
4. **Where do diffusion, flow matching and one-step generators fit?** (Section 9.)

Every result has a folded proof in numbered steps. The shared example of the series is a ring of eight Gaussians in the plane, simple enough to see everything and hard enough that every family gets something wrong.

## Notation

Plain italic letters are scalars, bold italic lower case are vectors, bold upright capitals are matrices.

| Symbol | Meaning |
|---|---|
| $\boldsymbol{x} \in \mathbb{R}^d$, $p_{\mathrm{data}}$ | a data point and the data distribution |
| $\boldsymbol{z} \in \mathbb{R}^k$, $p(\boldsymbol{z})$ | noise or a latent variable, usually $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| $p_\theta$ | the model distribution, with parameters $\theta$ |
| $f$, $G_\theta$ | an invertible map (flow); a generator (sampler) |
| $\mathbf{J}_f(\boldsymbol{z}) = \partial f/\partial\boldsymbol{z}$ | the Jacobian matrix of $f$ |
| $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$ | an encoder: an approximate posterior over $\boldsymbol{z}$ |
| $D(\boldsymbol{x})$ | a GAN discriminator |
| $E_\theta(\boldsymbol{x})$, $Z_\theta$ | an energy and its normalising constant |
| $\boldsymbol{s}(\boldsymbol{x}) = \nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$ | the score |
| $\mathrm{KL}$, $\mathrm{JS}$, $\mathrm{MMD}_k$ | Kullback–Leibler, Jensen–Shannon, maximum mean discrepancy |

Later posts add time: $t \in [0, 1]$ with **data at $t = 0$ and noise at $t = 1$**, a score $\boldsymbol{s}(\boldsymbol{x}, t)$ and a velocity $\boldsymbol{v}(\boldsymbol{x}, t)$.

## 1. What it means to learn a distribution

### Three things a model can be asked to do

We have samples $\boldsymbol{x}_1, \dots, \boldsymbol{x}_N$ from an unknown $p_{\mathrm{data}}$, and nothing else. A model $p_\theta$ can be asked for three different things:

- **Sample:** produce new $\boldsymbol{x} \sim p_\theta$. Image and audio generation need only this.
- **Evaluate:** compute $p_\theta(\boldsymbol{x})$ at a given point. Compression, anomaly detection and model comparison need this.
- **Score:** compute $\nabla_{\boldsymbol{x}}\log p_\theta(\boldsymbol{x})$, the direction in which the density grows fastest. Diffusion models are built on it.

No family does all three cheaply. That single fact organises the whole field, and this post.

### Maximum likelihood is a KL divergence

The standard way to fit $p_\theta$ is to maximise the average log-likelihood of the data. It has a precise meaning.

:::definition[Kullback–Leibler divergence]
$\mathrm{KL}[p \,\Vert\, q] = \mathbb{E}_{\boldsymbol{x} \sim p}\big[\log p(\boldsymbol{x}) - \log q(\boldsymbol{x})\big]$. It is non-negative and zero only when $p = q$ (Gibbs' inequality, proved in the [sparse GP post](/aboutme/blog/sparse-variational-gaussian-processes/), Section 2).
:::

:::proposition[Maximum likelihood minimises the forward KL]
$$
\arg\max_\theta\ \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}}\big[\log p_\theta(\boldsymbol{x})\big] = \arg\min_\theta\ \mathrm{KL}\big[p_{\mathrm{data}} \,\Vert\, p_\theta\big],
$$

and the average $\frac1N\sum_n\log p_\theta(\boldsymbol{x}_n)$ is an unbiased estimate of the expectation on the left.
:::

:::proof
**Step 1. Expand the KL.** $\mathrm{KL}[p_{\mathrm{data}} \,\Vert\, p_\theta] = \mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}(\boldsymbol{x})] - \mathbb{E}_{p_{\mathrm{data}}}[\log p_\theta(\boldsymbol{x})]$.

**Step 2. The first term does not depend on $\theta$.** It is minus the entropy of the data, a constant. So minimising the KL over $\theta$ is the same as maximising the second expectation.

**Step 3. The sample average.** Each $\boldsymbol{x}_n$ is drawn from $p_{\mathrm{data}}$, so $\mathbb{E}[\log p_\theta(\boldsymbol{x}_n)]$ equals the expectation, and so does the expectation of their average.
:::

Two consequences matter for everything that follows. Maximum likelihood needs $\log p_\theta(\boldsymbol{x})$, so it needs a model that can **evaluate**. And it minimises the KL in one particular direction, which decides what kind of mistakes the model prefers.

### Forward and reverse KL make different mistakes

$\mathrm{KL}[p \,\Vert\, q]$ averages over $p$. Wherever the data have mass and the model $q$ has almost none, $\log(p/q)$ is huge, so the fit spreads $q$ over all the data, even where that means putting mass between modes. The reverse divergence $\mathrm{KL}[q \,\Vert\, p]$ averages over the model instead: it punishes the model for putting mass where the data have none, so it prefers to sit inside one mode. Fit a single Gaussian $q$ to two modes and watch both:

<figure class="fig" data-kl>
  <canvas data-figure="kl" style="height:260px" aria-label="A two-mode density p and the single Gaussians that minimise the forward and the reverse KL divergence."></canvas>
  <div class="controls">
    <label class="slider">distance between the modes <input type="range" min="0.2" max="3" step="0.05" value="2" data-sep /></label>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

With the modes close together (separation $\pm1$), both fits are almost the same wide Gaussian. Pull them apart. At $\pm2$ the maximum-likelihood fit has mean $-0.40$ and standard deviation $2.02$: it covers both modes, and puts most of its mass in the empty middle. The reverse-KL fit sits exactly on the larger mode (mean $-2.00$, standard deviation $0.50$) and ignores the other one.

:::proposition[The forward-KL Gaussian matches moments]
Among Gaussians $q = \mathcal{N}(\mu, \sigma^2)$, $\mathrm{KL}[p \,\Vert\, q]$ is minimised by the mean and variance of $p$: $\mu = \mathbb{E}_p[x]$ and $\sigma^2 = \operatorname{Var}_p[x]$.
:::

:::proof
**Step 1. Only the cross term depends on $q$.** $\mathrm{KL}[p \,\Vert\, q] = \text{const} - \mathbb{E}_p[\log q(x)]$.

**Step 2. Write out the Gaussian log-density.** $-\mathbb{E}_p[\log q(x)] = \tfrac12\log(2\pi\sigma^2) + \frac{\mathbb{E}_p[(x - \mu)^2]}{2\sigma^2}$.

**Step 3. Split the square.** With $m = \mathbb{E}_p[x]$ and $v = \operatorname{Var}_p[x]$, $\mathbb{E}_p[(x - \mu)^2] = v + (m - \mu)^2$, because the cross term $2(m - \mu)\mathbb{E}_p[x - m]$ is zero.

**Step 4. Minimise over $\mu$.** Only $(m - \mu)^2$ depends on $\mu$, so $\mu = m$.

**Step 5. Minimise over $\sigma^2$.** The objective is now $\tfrac12\log(2\pi\sigma^2) + v/(2\sigma^2)$. Its derivative in $\sigma^2$ is $\frac{1}{2\sigma^2} - \frac{v}{2\sigma^4}$, which is zero at $\sigma^2 = v$, and the second derivative there is $\frac{1}{2v^2} > 0$.
:::

This is the root of a theme that runs through the series. Likelihood-based models (flows, VAEs, and diffusion models, whose training is a bound on the likelihood) are *mass-covering*: they rarely drop a mode, but they can put mass where there are no data. Models trained by other divergences can be *mode-seeking*: sharp, but liable to miss modes.

### When the model cannot be evaluated

If the model is only a sampler, $\log p_\theta$ is unavailable and maximum likelihood is impossible. Then the two distributions must be compared from samples alone. Three ways are standard, and Section 5 derives them: train a classifier to tell them apart (the Jensen–Shannon divergence of GANs), measure how far mass must move (the Wasserstein distance), or compare averages of kernel features (the maximum mean discrepancy).

## 2. Pushing noise forward: change of variables

The simplest way to build a sampler is to transform noise: draw $\boldsymbol{z} \sim p(\boldsymbol{z})$ and set $\boldsymbol{x} = f(\boldsymbol{z})$. The distribution of $\boldsymbol{x}$ is the *pushforward* of $p(\boldsymbol{z})$ through $f$. Sampling is one evaluation of $f$. Evaluating the density of $\boldsymbol{x}$ is possible when $f$ is invertible, and the formula is the most-used identity in generative modelling.

:::theorem[Change of variables in one dimension]
Let $f$ be differentiable and strictly increasing, and $x = f(z)$. Then

$$
p_x(x) = \frac{p_z\big(f^{-1}(x)\big)}{f'\big(f^{-1}(x)\big)}.
$$

For strictly decreasing $f$, replace $f'$ by $\lvert f'\rvert$.
:::

:::proof
**Step 1. The CDF.** For increasing $f$, $x \le a$ exactly when $z \le f^{-1}(a)$. So $P(x \le a) = P\big(z \le f^{-1}(a)\big) = F_z\big(f^{-1}(a)\big)$, with $F_z$ the CDF of $z$.

**Step 2. Differentiate in $a$.** The density is the derivative of the CDF. By the chain rule, $p_x(a) = p_z\big(f^{-1}(a)\big)\cdot\frac{d}{da}f^{-1}(a)$.

**Step 3. The derivative of the inverse.** Differentiating $f(f^{-1}(a)) = a$ gives $f'\big(f^{-1}(a)\big)\cdot\frac{d}{da}f^{-1}(a) = 1$.

**Step 4. Decreasing $f$.** Then $x \le a$ exactly when $z \ge f^{-1}(a)$, so $P(x \le a) = 1 - F_z(f^{-1}(a))$. Differentiating gives the same formula with $-f' = \lvert f'\rvert$.
:::

Where $f$ is steep, an interval of $z$ is stretched over a long interval of $x$, so the same probability is spread thinner, and the density falls. Where $f$ is flat, it piles up. In more dimensions, "steepness" becomes the factor by which $f$ changes volume, which is the absolute determinant of its Jacobian.

:::theorem[Change of variables]
Let $f : \mathbb{R}^d \to \mathbb{R}^d$ be invertible and differentiable with a differentiable inverse, and $\boldsymbol{x} = f(\boldsymbol{z})$. Then

$$
p_{\boldsymbol{x}}(\boldsymbol{x}) = p_{\boldsymbol{z}}(\boldsymbol{z})\,\big\lvert\det\mathbf{J}_f(\boldsymbol{z})\big\rvert^{-1}, \qquad \boldsymbol{z} = f^{-1}(\boldsymbol{x}),
$$

or, in logs, $\log p_{\boldsymbol{x}}(\boldsymbol{x}) = \log p_{\boldsymbol{z}}(\boldsymbol{z}) - \log\big\lvert\det\mathbf{J}_f(\boldsymbol{z})\big\rvert$.
:::

:::proof
**Step 1. Probabilities of sets.** For any region $A$, $\boldsymbol{x} \in A$ exactly when $\boldsymbol{z} \in f^{-1}(A)$, so $P(\boldsymbol{x} \in A) = \int_{f^{-1}(A)}p_{\boldsymbol{z}}(\boldsymbol{z})\,d\boldsymbol{z}$.

**Step 2. Substitute $\boldsymbol{z} = f^{-1}(\boldsymbol{x})$.** The multivariate substitution rule of calculus says $\int_{f^{-1}(A)}h(\boldsymbol{z})\,d\boldsymbol{z} = \int_A h\big(f^{-1}(\boldsymbol{x})\big)\,\big\lvert\det\mathbf{J}_{f^{-1}}(\boldsymbol{x})\big\rvert\,d\boldsymbol{x}$. So $P(\boldsymbol{x} \in A) = \int_A p_{\boldsymbol{z}}(f^{-1}(\boldsymbol{x}))\,\lvert\det\mathbf{J}_{f^{-1}}(\boldsymbol{x})\rvert\,d\boldsymbol{x}$.

**Step 3. Read off the density.** This holds for every region $A$, so the integrand is the density of $\boldsymbol{x}$.

**Step 4. Jacobian of the inverse.** Differentiating $f(f^{-1}(\boldsymbol{x})) = \boldsymbol{x}$ with the chain rule gives $\mathbf{J}_f(\boldsymbol{z})\,\mathbf{J}_{f^{-1}}(\boldsymbol{x}) = \mathbf{I}$, so $\mathbf{J}_{f^{-1}}(\boldsymbol{x}) = \mathbf{J}_f(\boldsymbol{z})^{-1}$ and $\det\mathbf{J}_{f^{-1}} = 1/\det\mathbf{J}_f$.
:::

:::corollary[Compositions add their log-determinants]
For $f = f_K \circ \dots \circ f_1$ with $\boldsymbol{z}_0 = \boldsymbol{z}$ and $\boldsymbol{z}_k = f_k(\boldsymbol{z}_{k-1})$,

$$
\log p_{\boldsymbol{x}}(\boldsymbol{x}) = \log p_{\boldsymbol{z}}(\boldsymbol{z}_0) - \sum_{k=1}^{K}\log\big\lvert\det\mathbf{J}_{f_k}(\boldsymbol{z}_{k-1})\big\rvert.
$$
:::

:::proof
**Step 1. Chain rule.** $\mathbf{J}_f(\boldsymbol{z}) = \mathbf{J}_{f_K}(\boldsymbol{z}_{K-1})\cdots\mathbf{J}_{f_1}(\boldsymbol{z}_0)$.

**Step 2. Determinants multiply.** $\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\det\mathbf{B}$, so the log of the absolute determinant of the product is the sum of the logs. Substitute into the theorem.
:::

The figure at the top is this formula, live. The map there is a radial stretch, a swirl and a shear. The swirl and the shear preserve area (determinant 1), so the density changes only through the stretch. The formula's readout is exact, and the heat map on the right is computed from it; I checked that it integrates to 1 and that the log-determinant matches a numerical Jacobian to ten decimal places.

## 3. Normalising flows: exact likelihood, invertible maps

A normalising flow makes the invertible map learnable: $\boldsymbol{x} = f_\theta(\boldsymbol{z})$ with $f_\theta = f_K \circ \dots \circ f_1$, each $f_k$ a small invertible network. The corollary gives $\log p_\theta(\boldsymbol{x})$ exactly, so the flow trains by maximum likelihood and samples in one pass (Rezende and Mohamed, 2015; Dinh, Sohl-Dickstein and Bengio, 2017).

The difficulty is cost. The determinant of a general $d \times d$ Jacobian costs $O(d^3)$, and for an image $d$ is in the hundreds of thousands. So flows are built from layers whose Jacobians are triangular, whose determinant is the product of the diagonal.

:::proposition[The affine coupling layer]
Split $\boldsymbol{x} = (\boldsymbol{x}_a, \boldsymbol{x}_b)$ and let $\boldsymbol{s}(\cdot)$ and $\boldsymbol{t}(\cdot)$ be any networks. The map

$$
\boldsymbol{y}_a = \boldsymbol{x}_a, \qquad \boldsymbol{y}_b = \boldsymbol{x}_b \odot e^{\boldsymbol{s}(\boldsymbol{x}_a)} + \boldsymbol{t}(\boldsymbol{x}_a)
$$

is invertible in closed form, and $\log\lvert\det\mathbf{J}\rvert = \sum_i s_i(\boldsymbol{x}_a)$, at cost $O(d)$.
:::

:::proof
**Step 1. The inverse.** $\boldsymbol{x}_a = \boldsymbol{y}_a$, and then $\boldsymbol{x}_b = \big(\boldsymbol{y}_b - \boldsymbol{t}(\boldsymbol{y}_a)\big)\odot e^{-\boldsymbol{s}(\boldsymbol{y}_a)}$. Nothing needs inverting except an exponential, because $\boldsymbol{s}$ and $\boldsymbol{t}$ are only ever evaluated at $\boldsymbol{x}_a$, which is known.

**Step 2. The Jacobian's blocks.** $\partial\boldsymbol{y}_a/\partial\boldsymbol{x}_a = \mathbf{I}$ and $\partial\boldsymbol{y}_a/\partial\boldsymbol{x}_b = \mathbf{0}$. $\partial\boldsymbol{y}_b/\partial\boldsymbol{x}_b = \operatorname{diag}(e^{\boldsymbol{s}(\boldsymbol{x}_a)})$. The block $\partial\boldsymbol{y}_b/\partial\boldsymbol{x}_a$ is complicated, but it sits below the diagonal.

**Step 3. The determinant.** The Jacobian is block lower-triangular, so its determinant is the product of the diagonal blocks' determinants: $1 \cdot \prod_ie^{s_i} = e^{\sum_is_i}$.

**Step 4. Stacking.** One layer leaves $\boldsymbol{x}_a$ unchanged, so consecutive layers swap the roles of the two halves.
:::

Flows have one limitation that no amount of training removes.

:::proposition[A flow cannot separate modes]
If $f$ is continuous and $\boldsymbol{z}$ has a positive density on all of $\mathbb{R}^d$, then the flow's density is positive on the connected set $f(\mathbb{R}^d)$. If the data's modes are separated by regions of zero density, the flow must still put positive density on paths between them.
:::

:::proof
**Step 1. The support.** $p_{\boldsymbol{x}}(\boldsymbol{x}) = p_{\boldsymbol{z}}(f^{-1}(\boldsymbol{x}))\lvert\det\mathbf{J}_f\rvert^{-1} > 0$ for every $\boldsymbol{x}$ in the image $f(\mathbb{R}^d)$, since both factors are positive.

**Step 2. The image is connected.** $\mathbb{R}^d$ is connected, and the continuous image of a connected set is connected.

**Step 3. Conclusion.** Any two modes lie in one connected set on which the density is positive, so there is a path between them along which the density never vanishes.
:::

On the ring of eight Gaussians (Section 8) the trained flow does exactly this: it draws thin bridges from mode to mode, and a quarter of its samples land on them. Continuous-time flows, which define $f$ by an ordinary differential equation $\dot{\boldsymbol{x}} = \boldsymbol{v}(\boldsymbol{x}, t)$, share the limitation, and they are the starting point of the flow-matching post.

## 4. Latent variables and VAEs

A *latent-variable model* does not need $f$ to be invertible. It draws a latent $\boldsymbol{z} \sim p(\boldsymbol{z})$ and then $\boldsymbol{x}$ from a decoder $p_\theta(\boldsymbol{x} \mid \boldsymbol{z})$, for example $\mathcal{N}(\boldsymbol{x} \mid \mathrm{NN}_\theta(\boldsymbol{z}), \sigma_x^2\mathbf{I})$. Sampling is easy, but the likelihood

$$
p_\theta(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x} \mid \boldsymbol{z})\,p(\boldsymbol{z})\,d\boldsymbol{z}
$$

has no closed form once the decoder is a neural network. Variational inference supplies a bound. The [sparse GP post](/aboutme/blog/sparse-variational-gaussian-processes/), Section 2, derives it in general; here it is for one data point.

:::proposition[The VAE bound (Kingma and Welling, 2014; Rezende, Mohamed and Wierstra, 2014)]
For any distribution $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$,

$$
\log p_\theta(\boldsymbol{x}) = \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\big[\log p_\theta(\boldsymbol{x} \mid \boldsymbol{z})\big] - \mathrm{KL}\big[q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) \,\Vert\, p(\boldsymbol{z})\big]}_{\text{ELBO}} + \mathrm{KL}\big[q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) \,\Vert\, p_\theta(\boldsymbol{z} \mid \boldsymbol{x})\big].
$$
:::

:::proof
**Step 1. Bayes' rule, pointwise in $\boldsymbol{z}$.** $\log p_\theta(\boldsymbol{x}) = \log p_\theta(\boldsymbol{x} \mid \boldsymbol{z}) + \log p(\boldsymbol{z}) - \log p_\theta(\boldsymbol{z} \mid \boldsymbol{x})$.

**Step 2. Add and subtract $\log q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$** and group:

$$
\log p_\theta(\boldsymbol{x}) = \log p_\theta(\boldsymbol{x} \mid \boldsymbol{z}) - \log\frac{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}{p(\boldsymbol{z})} + \log\frac{q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}{p_\theta(\boldsymbol{z} \mid \boldsymbol{x})}.
$$

**Step 3. Average over $q_\phi$.** The left side does not depend on $\boldsymbol{z}$. The three averages on the right are the reconstruction term and the two KL divergences.
:::

The last KL is non-negative, so the ELBO is a lower bound, and maximising it over $\phi$ makes the encoder approximate the true posterior. Two choices make it trainable at scale. The encoder is *amortised*: one network outputs $q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) = \mathcal{N}\big(\boldsymbol{\mu}_\phi(\boldsymbol{x}), \operatorname{diag}\boldsymbol{\sigma}^2_\phi(\boldsymbol{x})\big)$ for every $\boldsymbol{x}$, instead of separate parameters per data point. And the expectation is estimated by reparameterisation, $\boldsymbol{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon}$, proved in the [deep GP post](/aboutme/blog/deep-gaussian-processes/), Section 5. The KL term is in closed form:

:::proposition[KL to a standard Gaussian]
$$
\mathrm{KL}\big[\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}\boldsymbol{\sigma}^2) \,\Vert\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big] = \tfrac12\sum_{j=1}^{k}\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big).
$$
:::

:::proof
**Step 1. The general formula** for $\mathrm{KL}[\mathcal{N}(\boldsymbol{m}, \mathbf{S}) \,\Vert\, \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)]$ is $\tfrac12\big[\operatorname{tr}(\boldsymbol{\Sigma}_0^{-1}\mathbf{S}) + (\boldsymbol{\mu}_0 - \boldsymbol{m})^\top\boldsymbol{\Sigma}_0^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{m}) - k + \log\lvert\boldsymbol{\Sigma}_0\rvert - \log\lvert\mathbf{S}\rvert\big]$ (proved in [the Gaussian toolkit](/aboutme/blog/gaussian-identities/)).

**Step 2. Substitute** $\boldsymbol{\Sigma}_0 = \mathbf{I}$, $\boldsymbol{\mu}_0 = \mathbf{0}$, $\boldsymbol{m} = \boldsymbol{\mu}$, $\mathbf{S} = \operatorname{diag}\boldsymbol{\sigma}^2$: the trace is $\sum_j\sigma_j^2$, the quadratic form is $\sum_j\mu_j^2$, $\log\lvert\mathbf{I}\rvert = 0$ and $\log\lvert\mathbf{S}\rvert = \sum_j\log\sigma_j^2$.
:::

Why VAE samples look the way they do follows from the model, not from bad training:

- **The decoder adds isotropic noise.** A sample is the decoder mean plus $\sigma_x\boldsymbol{\varepsilon}$, which blurs every output by the same amount in every direction.
- **The prior has holes.** The encoder maps the data to a set of codes, but the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ also covers codes that no data point maps to. Those decode to points between the modes.
- **The bound has a gap.** Training maximises the ELBO, not the likelihood, and a decoder that ignores $\boldsymbol{z}$ can make the KL term zero. When this *posterior collapse* happens, the model explains the data as noise.

## 5. GANs: learning a sampler with a classifier

A generative adversarial network (Goodfellow et al., 2014) keeps only the sampler, $\boldsymbol{x} = G_\theta(\boldsymbol{z})$, with no density at all. To compare its samples with the data, it trains a classifier $D(\boldsymbol{x}) \in (0, 1)$ to tell real from generated, and trains $G$ to fool it:

$$
\min_G\max_D\ V(G, D) = \mathbb{E}_{p_{\mathrm{data}}}\big[\log D(\boldsymbol{x})\big] + \mathbb{E}_{\boldsymbol{z}}\big[\log\big(1 - D(G(\boldsymbol{z}))\big)\big].
$$

What is the generator really minimising? Write $p_g$ for the distribution of $G(\boldsymbol{z})$.

:::proposition[The optimal discriminator]
For fixed $G$, $V$ is maximised by $D^*(\boldsymbol{x}) = \dfrac{p_{\mathrm{data}}(\boldsymbol{x})}{p_{\mathrm{data}}(\boldsymbol{x}) + p_g(\boldsymbol{x})}$.
:::

:::proof
**Step 1. One integral.** $V = \int\big[p_{\mathrm{data}}(\boldsymbol{x})\log D(\boldsymbol{x}) + p_g(\boldsymbol{x})\log(1 - D(\boldsymbol{x}))\big]\,d\boldsymbol{x}$, writing the second expectation over $\boldsymbol{x} = G(\boldsymbol{z})$.

**Step 2. Maximise pointwise.** $D$ can be chosen separately at each $\boldsymbol{x}$, so maximise $a\log y + b\log(1 - y)$ over $y \in (0, 1)$, with $a = p_{\mathrm{data}}(\boldsymbol{x})$ and $b = p_g(\boldsymbol{x})$.

**Step 3. Set the derivative to zero.** $a/y - b/(1 - y) = 0$ gives $y = a/(a + b)$. The second derivative, $-a/y^2 - b/(1 - y)^2$, is negative, so it is a maximum.
:::

:::proposition[GANs minimise the Jensen–Shannon divergence]
At the optimal discriminator, $V(G, D^*) = 2\,\mathrm{JS}(p_{\mathrm{data}}, p_g) - \log 4$, where $\mathrm{JS}(p, q) = \tfrac12\mathrm{KL}[p \,\Vert\, m] + \tfrac12\mathrm{KL}[q \,\Vert\, m]$ with $m = \tfrac12(p + q)$.
:::

:::proof
**Step 1. Substitute $D^*$.**

$$
V(G, D^*) = \mathbb{E}_{p_{\mathrm{data}}}\Big[\log\frac{p_{\mathrm{data}}}{p_{\mathrm{data}} + p_g}\Big] + \mathbb{E}_{p_g}\Big[\log\frac{p_g}{p_{\mathrm{data}} + p_g}\Big].
$$

**Step 2. Bring in the mixture $m$.** $p_{\mathrm{data}} + p_g = 2m$, so each logarithm is $\log(p/m) - \log 2$.

**Step 3. Recognise the KLs.** $V = \mathrm{KL}[p_{\mathrm{data}} \,\Vert\, m] - \log 2 + \mathrm{KL}[p_g \,\Vert\, m] - \log 2 = 2\,\mathrm{JS} - \log 4$.
:::

So the ideal generator minimises the Jensen–Shannon divergence, which is zero only when $p_g = p_{\mathrm{data}}$. In practice the ideal is far away, for two reasons:

:::proposition[No gradient when the supports are disjoint]
If $p_{\mathrm{data}}$ and $p_g$ are positive on disjoint sets, then $\mathrm{JS}(p_{\mathrm{data}}, p_g) = \log 2$, however close or far apart the sets are.
:::

:::proof
**Step 1. On the support of $p_{\mathrm{data}}$,** $p_g = 0$, so $m = p_{\mathrm{data}}/2$ and $\log(p_{\mathrm{data}}/m) = \log 2$. Hence $\mathrm{KL}[p_{\mathrm{data}} \,\Vert\, m] = \log 2$.

**Step 2. Symmetrically,** $\mathrm{KL}[p_g \,\Vert\, m] = \log 2$, so $\mathrm{JS} = \tfrac12(\log 2 + \log 2) = \log 2$.

**Step 3. Consequence.** Moving $p_g$ a little closer to the data, without overlapping it, does not change the divergence: its gradient is zero. Early in training, and for data on thin sets such as images, this is the usual situation.
:::

The second reason is *mode collapse*: the generator can fool the current discriminator by producing a few convincing kinds of sample, then move on when the discriminator catches up. Its behaviour resembles the reverse KL of Section 1, sharp and mode-seeking.

### Divergences that see distance: Wasserstein and MMD

The fixes replace the Jensen–Shannon divergence with one that changes smoothly as the generated distribution moves towards the data, even without overlap. The *Wasserstein-1* distance is the least average distance mass must travel to turn $p_g$ into $p_{\mathrm{data}}$; it is estimated through its dual form with a constrained critic network (Arjovsky, Chintala and Bottou, 2017). The other fix comes from kernel methods, and returns in the post on one-step generators.

:::definition[Kernel mean embedding and MMD (Gretton et al., 2012)]
For a positive-definite kernel $k$ with reproducing kernel Hilbert space $\mathcal{H}$, the mean embedding of $p$ is $\mu_p = \mathbb{E}_{\boldsymbol{x} \sim p}[k(\boldsymbol{x}, \cdot)] \in \mathcal{H}$, and the maximum mean discrepancy is $\mathrm{MMD}_k(p, q) = \lVert\mu_p - \mu_q\rVert_{\mathcal{H}}$.
:::

:::proposition[MMD from samples]
$$
\mathrm{MMD}_k^2(p, q) = \mathbb{E}\big[k(\boldsymbol{x}, \boldsymbol{x}')\big] + \mathbb{E}\big[k(\boldsymbol{y}, \boldsymbol{y}')\big] - 2\,\mathbb{E}\big[k(\boldsymbol{x}, \boldsymbol{y})\big],
$$

with $\boldsymbol{x}, \boldsymbol{x}' \sim p$ and $\boldsymbol{y}, \boldsymbol{y}' \sim q$, all independent. Each term can be estimated by averaging the kernel over pairs of samples.
:::

:::proof
**Step 1. Expand the squared norm.** $\lVert\mu_p - \mu_q\rVert^2 = \langle\mu_p, \mu_p\rangle + \langle\mu_q, \mu_q\rangle - 2\langle\mu_p, \mu_q\rangle$.

**Step 2. The reproducing property.** $\langle k(\boldsymbol{x}, \cdot), k(\boldsymbol{y}, \cdot)\rangle_{\mathcal{H}} = k(\boldsymbol{x}, \boldsymbol{y})$.

**Step 3. Move the expectations out of the inner products** (the inner product is linear in each argument, and the expectations are finite for a bounded kernel): $\langle\mu_p, \mu_q\rangle = \mathbb{E}\langle k(\boldsymbol{x}, \cdot), k(\boldsymbol{y}, \cdot)\rangle = \mathbb{E}[k(\boldsymbol{x}, \boldsymbol{y})]$, and likewise for the other two terms.
:::

For a *characteristic* kernel such as the squared-exponential, $\mathrm{MMD}_k(p, q) = 0$ only when $p = q$ (Sriperumbudur et al., 2010), so the MMD is a genuine discrepancy that needs only samples. Generators trained to minimise it directly, without an adversary, were among the first alternatives to GANs (Li, Swersky and Zemel, 2015; Dziugaite, Roy and Ghahramani, 2015).

## 6. Energy-based models and the score

An *energy-based model* writes down an unnormalised density, $p_\theta(\boldsymbol{x}) = e^{-E_\theta(\boldsymbol{x})}/Z_\theta$, where $E_\theta$ is any network and $Z_\theta = \int e^{-E_\theta(\boldsymbol{x})}\,d\boldsymbol{x}$. Any shape is allowed, but $Z_\theta$ is an integral over all of $\mathbb{R}^d$ and is intractable. It enters the likelihood gradient:

:::proposition[The likelihood gradient of an energy-based model]
$$
\nabla_\theta\log p_\theta(\boldsymbol{x}) = -\nabla_\theta E_\theta(\boldsymbol{x}) + \mathbb{E}_{\boldsymbol{x}' \sim p_\theta}\big[\nabla_\theta E_\theta(\boldsymbol{x}')\big].
$$
:::

:::proof
**Step 1.** $\log p_\theta(\boldsymbol{x}) = -E_\theta(\boldsymbol{x}) - \log Z_\theta$.

**Step 2. Differentiate $\log Z_\theta$.** $\nabla_\theta\log Z_\theta = \frac{1}{Z_\theta}\int\nabla_\theta e^{-E_\theta(\boldsymbol{x}')}\,d\boldsymbol{x}' = -\int\frac{e^{-E_\theta(\boldsymbol{x}')}}{Z_\theta}\nabla_\theta E_\theta(\boldsymbol{x}')\,d\boldsymbol{x}' = -\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta]$.

**Step 3. Combine.** Subtracting gives the stated form: push the energy down at the data, and up at samples from the model.
:::

So training needs samples from the model at every step, which is expensive. The way out is a quantity that does not involve $Z_\theta$ at all.

:::proposition[The score does not need the normalising constant]
$\boldsymbol{s}_\theta(\boldsymbol{x}) = \nabla_{\boldsymbol{x}}\log p_\theta(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}}E_\theta(\boldsymbol{x})$.
:::

:::proof
**Step 1.** $\log p_\theta(\boldsymbol{x}) = -E_\theta(\boldsymbol{x}) - \log Z_\theta$.

**Step 2.** $Z_\theta$ depends on $\theta$ but not on $\boldsymbol{x}$, so its gradient in $\boldsymbol{x}$ is zero.
:::

The score points uphill in density. Remarkably, it is enough to *sample*.

:::proposition[Langevin dynamics leaves $p$ invariant]
The stochastic differential equation $d\boldsymbol{x} = \nabla\log p(\boldsymbol{x})\,dt + \sqrt{2}\,d\boldsymbol{W}$ has $p$ as a stationary distribution. Its discretisation $\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + \eta\,\boldsymbol{s}(\boldsymbol{x}_k) + \sqrt{2\eta}\,\boldsymbol{\xi}_k$, $\boldsymbol{\xi}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, samples approximately from $p$ for small $\eta$ and many steps.
:::

:::proof
We check the one-dimensional case with the Fokker–Planck equation, which the diffusion post derives. For $dx = a(x)\,dt + \sqrt{2}\,dW$, the density $\rho_t$ of $x$ evolves by $\partial_t\rho = -\partial_x(a\rho) + \partial_x^2\rho$.

**Step 1. Put in $a = (\log p)' = p'/p$ and $\rho = p$.** Then $a\rho = p'$.

**Step 2. The right-hand side.** $-\partial_x(p') + \partial_x^2p = -p'' + p'' = 0$.

**Step 3. Conclusion.** $\partial_t\rho = 0$: if $x$ is distributed as $p$, it stays distributed as $p$. The drift pushes mass uphill exactly as fast as the noise spreads it out.
:::

What remains is to learn the score from data, without knowing $p_{\mathrm{data}}$. Hyvärinen (2005) showed that the obvious objective can be rewritten without the unknown score.

:::proposition[Score matching (Hyvärinen, 2005)]
In one dimension, for a model score $s_\theta$ and data density $p$ that vanish fast enough at infinity,

$$
\mathbb{E}_p\big[(s_\theta(x) - (\log p)'(x))^2\big] = \mathbb{E}_p\big[s_\theta(x)^2 + 2\,s_\theta'(x)\big] + \text{const}.
$$

In $d$ dimensions the derivative becomes the divergence, $\nabla\cdot\boldsymbol{s}_\theta = \sum_i\partial s_{\theta,i}/\partial x_i$.
:::

:::proof
**Step 1. Expand the square.** $\mathbb{E}_p[s_\theta^2] - 2\,\mathbb{E}_p[s_\theta(\log p)'] + \mathbb{E}_p[((\log p)')^2]$. The last term does not depend on $\theta$.

**Step 2. Simplify the cross term.** $p\,(\log p)' = p'$, so $\mathbb{E}_p[s_\theta(\log p)'] = \int s_\theta(x)\,p'(x)\,dx$.

**Step 3. Integrate by parts.** $\int s_\theta p'\,dx = \big[s_\theta p\big]_{-\infty}^{\infty} - \int s_\theta'\,p\,dx = -\mathbb{E}_p[s_\theta']$, because the boundary term vanishes.

**Step 4. Substitute.** The cross term becomes $+2\,\mathbb{E}_p[s_\theta']$, which depends only on the model and the data samples.
:::

In high dimensions the divergence costs one backward pass per dimension, and the score is learned badly in regions with no data, where Langevin dynamics starts. Both problems disappear if the data are first *blurred with Gaussian noise of several sizes*, and the score of each blurred distribution is learned instead (Vincent, 2011; Song and Ermon, 2019). That step is the beginning of diffusion models, and of the next post.

## 7. Autoregressive models

The last family evaluates exactly and samples slowly. Order the coordinates of $\boldsymbol{x}$, and use the chain rule of probability:

$$
p(\boldsymbol{x}) = \prod_{i=1}^{d}p(x_i \mid x_1, \dots, x_{i-1}).
$$

It holds for every distribution, by repeated use of $p(a, b) = p(a)\,p(b \mid a)$, so nothing is approximated. A network outputs each conditional, for example a softmax over pixel values, so the likelihood is exact and training is maximum likelihood (van den Oord et al., 2016). Sampling must generate $x_1$, then $x_2$ given $x_1$, and so on: $d$ sequential network passes. For text that is natural, and large language models are exactly this. For an image with hundreds of thousands of values it is slow, and the ordering of pixels is arbitrary.

## 8. Same data, three families

Here is the ring of eight Gaussians (radius 2, standard deviation 0.1), and three models trained on it: a normalising flow (eight affine coupling layers), a VAE (two-dimensional latent, learned decoder noise) and a GAN. All three use the same small networks, one training run each with a fixed seed. Below each panel: how many of the eight modes receive at least 2% of the samples, the share of samples within three standard deviations of a mode (99% for the data), and the likelihood where the model has one.

<figure class="fig" data-families>
  <canvas data-figure="families" data-src="/aboutme/blog/generative/families.json" aria-label="Samples from the data, a normalising flow, a VAE and a GAN trained on a ring of eight Gaussians, with mode coverage and likelihood statistics."></canvas>
</figure>

Each family fails in the way its mathematics predicts:

- **The flow covers all eight modes and draws bridges between them.** A quarter of its samples lie off the modes, on the connecting paths that Section 3 proved it cannot avoid. Its likelihood is exact: 0.90 nats per point, against 0.32 for the true distribution.
- **The VAE covers all eight modes and fills the space between them.** Only 53% of its samples are on a mode. The decoder noise blurs, and codes from the prior's holes decode to points between the modes. Its $-\text{ELBO}$ of 1.71 is an upper bound on its negative log-likelihood, not the value itself.
- **The GAN's samples are sharp but smear along the ring, and it drops one mode entirely.** It covers 7 of 8. It has no likelihood to report at all.

With more tuning each model improves, and these small networks are far from the state of the art. But the characters are the ones reported throughout the literature: likelihood training covers the data and leaks between modes, adversarial training is sharp and drops modes.

## 9. The map, and where this series goes

<figure class="sketch-fig"><svg class="sketch" viewBox="0 0 960 560" role="img" aria-label="A map of generative models. Three ways to learn a distribution from samples: write down the density and maximise likelihood (autoregressive models, normalising flows, energy-based models); learn a sampler that pushes noise through a network (VAEs, GANs); or learn the score and sample by a dynamics (score matching with Langevin, diffusion models, flow matching). One-step generators combine the last route's quality with a one-pass sampler."><path d="M329.76139947256524 16.456233797897816 C390.5754986772223 15.108631398461355, 449.3175467212056 16.33415379550701, 631.4734966086137 18.863287989370473 M329.22475854242794 18.072327138660874 C417.41863005853776 18.799320293232455, 504.6451646126093 17.86962035222711, 630.5493106247219 17.42214955243988 M629.2458272562362 17.72748577883467 C631.0114289695999 30.08840428972617, 630.902553262808 39.497238354664304, 629.9799880997277 76.00556195965036 M630.9735754221213 17.55920121301897 C628.94432964441 39.88982759104111, 628.4167369392198 62.00229224306532, 629.2618925824296 76.11684645791539 M630.1808322130922 75.23334697662035 C521.6068010149278 77.72461563976482, 414.7970001041731 77.30207121274546, 328.5145495048257 74.91971424081983 M630.2539890985859 76.58538003918828 C556.1853943819721 77.81752161400827, 482.66459159734455 77.6120285915082, 329.65917964753504 76.8115971641472 M329.340021030698 77.75517282178626 C330.63963561799466 53.97069692807273, 330.81677275147376 30.15363732082769, 331.8531188325025 18.299163728859277 M328.8649603902828 77.1029993407894 C330.07106522894304 55.83912278288044, 329.6562049341838 32.65363296088762, 330.15758806313386 18.833395535545424" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="480" y="45" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:700;fill:var(--ink)">learn a distribution</text><text x="480" y="66" text-anchor="middle" style="font-family:Kalam,cursive;font-size:15px;font-weight:400;fill:var(--muted)">from samples x₁, …, xₙ</text><path d="M479.7741879868719 78.53897659974278 C416.8663036473545 90.46690278635326, 351.6133309302242 101.23170274560378, 161.39452011739385 128.81701746800374 M479.2663099445888 80.06845054770164 C387.25160253438287 94.34503296401287, 294.00560054675014 107.43765169456478, 160.51986866640544 127.45312118856806" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M159.80477091526612 126.73685081118718 C162.35084048158782 125.70299699610366, 163.27950423352289 125.35210915832118, 171.11719546930084 121.94165290441562 M159.36567751196213 128.05917992400936 C162.895165858686 126.5330907862802, 165.69728600521893 123.80322275939363, 170.36100274123402 120.76247639109579" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M159.80477091526612 126.73685081118718 C162.7797468680743 127.75584958258177, 164.01828106276085 129.47076469647587, 172.6657470934394 132.26533039867252 M159.36567751196213 128.05917992400936 C163.4438111663163 129.53962810345024, 166.69670464445585 129.814915620608, 171.90955436537257 131.0861538853527" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M19.672288024370175 132.37698950343255 C19.672288024370175 132.37698950343255, 19.672288024370175 132.37698950343255, 19.672288024370175 132.37698950343255 M19.672288024370175 132.37698950343255 C19.672288024370175 132.37698950343255, 19.672288024370175 132.37698950343255, 19.672288024370175 132.37698950343255 M19.891978109087134 145.20725754574005 C24.80660272796538 143.17832196468692, 28.728748232647355 137.0987444187073, 30.199764050522948 134.20067293605263 M20.51171347703253 144.8132036379219 C23.86255594246158 141.94074558526842, 25.914997722349142 137.24267781743498, 32.70051089483606 131.5615947740614 M18.956674280108878 158.24631760626738 C30.61093041582158 149.37592830138053, 37.55031139157769 138.97827800840912, 42.87743557462068 130.03123766940513 M20.118536091081793 159.27779507980387 C25.37057101383737 152.9400449537926, 31.462632242683682 145.9519564555121, 44.383784920056996 133.12049122599382 M20.41611702273297 172.61723346527822 C31.648083684641847 160.40536850133947, 39.74395009619958 148.25239552793465, 56.15918055882978 132.13179543988278 M20.66319165894459 173.65699803632842 C26.391815687206662 163.55745081281933, 34.28308018788572 154.63700122038236, 55.20094352563213 132.3719659558168 M19.21150837793522 185.18352910334977 C31.39992237007149 175.26650004655002, 43.047831143510855 158.45722626680578, 67.2829580709625 133.98450860968296 M20.02516156776384 186.23665532063492 C30.242213884757337 176.5443402575468, 39.4489703014472 164.3419392327033, 68.55260726006775 130.82051590956252 M23.289030469547622 192.64181814739908 C44.44495007454286 171.98249004169733, 62.82627910544619 153.57976737320743, 81.43759562566721 130.88233677353216 M25.557694286586383 193.41293042536321 C36.76887923583562 182.48255968562268, 47.86231179129944 168.364317113949, 80.58598834260212 130.84717593830516 M35.32228512019961 193.46548478170732 C53.771780180356465 172.81766457554576, 69.80215450449766 153.65256562355933, 92.3229515523519 131.46679676262983 M36.77754196763149 193.568767166142 C54.81186815881361 173.28400765120924, 72.50133440776362 152.68445926168445, 90.68135129160802 131.68061039521677 M47.90425564838456 194.55327580947866 C61.01999345131443 180.2884840929732, 76.27143749432985 164.23580748665873, 105.25852611114598 130.77474932375685 M49.505359743357396 194.54904424314824 C67.88145789554156 173.6321403316842, 86.56848554490765 151.77426465434647, 102.45906154055476 132.02046231467597 M60.06498050625089 193.6913148879397 C73.0437435287346 180.37314746640368, 84.78244619287614 164.89529928478836, 114.63148231895738 129.93228160061267 M61.78181686425235 193.4991519762061 C74.03623637261572 179.49735455793564, 88.65602776225245 163.71217429348215, 114.77604376406507 132.1577983936048 M74.43927896070657 193.82053291078978 C94.11271231146789 169.38545617619397, 114.62237742648799 148.51468187046765, 127.00525706543458 132.20149381292381 M73.08375987993917 194.25898518723756 C92.28597115341826 171.0532666166244, 114.25933871361593 148.53934266792473, 127.33283049387728 131.99845817057746 M83.57536220938104 194.4834546873099 C97.47007306771087 182.0756540550238, 109.27557641952124 166.65512787398316, 137.24787708397596 131.30083383666684 M85.33126125513712 193.88629237937133 C104.80568241914096 171.672662746979, 124.48568435174468 148.28762372732638, 139.55031558985462 131.70995110604315 M97.98285420668218 193.1180814236994 C115.23100553200617 175.8233257044414, 127.44251346298321 155.73508183378073, 153.61633395173 133.4707222040683 M96.18407769571897 193.9771207182794 C108.84269889516607 178.97742018988853, 123.04068660996423 165.42050570219928, 151.25256673998544 132.76364408232607 M109.4414027797585 194.90467740218486 C118.65105490206767 179.2568514195434, 133.83081801410054 167.3108395286411, 164.68490006060463 131.99388668382755 M109.24869876467302 194.08916521128597 C120.84979620113849 179.140660424044, 134.0793202983089 163.0297071816154, 163.84357893139247 131.68165848384825 M122.32899559950839 191.8861045612428 C131.2157366567167 182.11699723782456, 144.15114902914394 167.97338411702276, 176.25373973322925 130.84214896022561 M121.45510035311254 194.68866578759025 C135.98298514822199 176.1196575980665, 151.1408726940786 160.1047796781873, 174.414323066295 131.8602193762384 M132.7063351805803 194.23344749559988 C155.67676956071418 171.59323943851956, 173.7489546348593 146.88808243705654, 188.79693004901188 132.17030711446182 M132.9395571802758 193.59259479826656 C152.22939346017662 171.65347246302684, 172.14526736753123 148.48002143977462, 187.43920428806584 131.89010048766843 M145.84515485238217 195.56634803645528 C159.03836457434988 178.00666001323026, 173.81000964263797 159.28078973349093, 197.5710719601009 132.36292230376267 M143.62139626453563 193.5155540964867 C158.65729485329973 179.38447532789613, 174.36464324288497 162.54157548171344, 199.68311059719323 132.7799733879626 M157.93682987504346 196.5997919305902 C176.56678568580475 173.16761027186183, 197.6766028906603 145.40731253678166, 210.1436062623947 133.35465334160486 M156.51004553936127 195.49312421662955 C171.79646790400537 177.31326810416283, 187.5437878543964 157.9614177612178, 210.57816632229046 131.70330456882016 M170.34039826547203 192.6379913217038 C188.58476286696802 169.89861888966948, 210.16688590236055 148.30181965571012, 220.87718859554184 132.4958816977148 M168.7014921793755 194.10230935561697 C189.6956544575267 171.27729775288088, 209.64061232889864 146.80262550585195, 222.90654462009584 132.24054294381523 M180.49117191331644 192.62117142952275 C194.24613819861398 177.86297497545308, 211.65767295611985 159.4032776391541, 233.05800541395016 133.8730460782919 M181.58533121631663 193.88488655202318 C201.59572253068674 170.57470576191517, 222.9736897674844 146.08484400522002, 234.77722094648212 131.22601692861988 M194.829052551975 193.64590513002358 C205.37535282606336 178.27863523201955, 216.71738971031127 166.30832655189494, 245.20438510274911 130.55542910511724 M192.82151901004787 193.89430276629744 C204.2731383601763 182.00966240624288, 214.2810298318721 168.64518853005276, 247.53302167241858 132.87128362011282 M203.28290675859623 193.83073795543288 C218.23606766028325 179.01736713101025, 233.86432168073628 159.0086444531329, 260.71481880384664 133.48903979689 M203.92318574855761 193.337862380764 C223.97180056081274 172.82906703761816, 240.8418779293124 151.979907300187, 259.04708241002805 132.48557377771593 M215.19888192508805 194.43652267444995 C235.57095997848302 171.97315445783562, 251.31467401613378 152.11220505227388, 269.5765715318912 130.7434002483369 M216.83338891896855 194.5236242798448 C238.45382927212293 170.4061297418623, 258.4851337864641 144.77163282769735, 271.6298966285487 132.8991410025035 M227.24901737407984 195.91320597135578 C248.83692454175628 173.8949371437268, 265.75377760951693 150.42187232029585, 284.29399814528125 131.03188021043442 M228.03216192349993 193.98375102916847 C243.78976717535323 176.70158774056486, 258.34274635983286 161.1693107466651, 281.70606912534396 131.3101588174248 M242.27926518497964 194.5937109273498 C256.82259189225556 173.8736248741533, 275.3352761740347 153.59691760498134, 295.3168593171747 131.33203167063192 M240.2329724055247 193.79495385690714 C257.86166456841585 175.6011870380109, 274.23390599392735 157.1505128084101, 293.64106949691154 132.58161774752668 M251.01810130993584 192.54072369965849 C263.3167009941618 180.59166105265444, 276.3203896970067 168.1322856275869, 300.40046041952684 139.67005217374256 M251.88977086553098 195.0018418328849 C270.4773962282669 174.4959360681918, 289.179897859523 153.3676015015196, 300.5498085229216 138.3523511408451 M263.4048415448469 195.06702810175847 C274.1647302107618 182.80698576398635, 280.45295395449875 174.48983353570176, 302.08243972602423 150.50344276038402 M265.2737441321321 193.66381894880104 C272.7270727590771 183.74347929864146, 280.4560566437534 174.29713344623354, 301.07186386915043 151.99625560684532 M277.81795750037065 196.75584018359902 C279.94271439599544 189.50826504984678, 288.7382706281046 182.43189650662507, 302.85196981795133 167.72069884385957 M275.00006883307594 194.1519811984525 C283.7641600257464 186.3174990955944, 291.7020413693072 177.19204066585573, 301.1891189933475 165.64854439330333 M288.0836877178867 193.60759491071119 C292.17489119566875 188.52552046847964, 295.6807199120873 183.741333008152, 302.00842074346616 179.59351303661865 M288.5849600112066 194.522465756843 C292.86197224632787 189.67840405666473, 298.68186068634327 182.22751003513068, 300.4828638045106 178.53451271276612" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M19.750552544617452 130.38605322896737 C76.59818306582196 129.10541281916872, 131.27956569558415 130.3866482244047, 301.54048267824567 132.9025335967444 M19.18951558499255 132.07561517523882 C101.60087746264053 132.8219766054248, 183.00092664005172 131.85001192853815, 300.57428262638257 131.39588011273375 M299.2458272562362 131.72748577883468 C300.9800771655452 144.89861647011713, 300.8712014587533 155.1176627154462, 299.9799880997278 194.00556195965038 M300.9735754221212 131.55920121301898 C298.90588661032535 155.3822370315436, 298.37829390513514 178.98711112407034, 299.2618925824296 194.1168464579154 M300.18905295764444 193.19849445472957 C198.8046336130867 195.65697568012988, 99.26464755891041 195.21522212141284, 18.44702000409729 192.8706037802569 M300.2655356005216 194.61199177879936 C231.10437411429183 195.77745759744838, 162.5159075209954 195.5626227347068, 19.643685742948655 194.84849287454986 M19.340021030697972 195.75517282178626 C20.604625629973693 170.38423970201984, 20.78176276345281 144.98072286872193, 21.853118832502513 132.29916372885927 M18.864960390282796 195.10299934078938 C20.06347115888605 172.372626127908, 19.648610864126777 147.72063965094276, 20.15758806313388 132.83339553554543" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="160" y="160" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:700;fill:var(--ink)">write down p(x)</text><text x="160" y="182" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">and maximise likelihood</text><path d="M479.6746181921102 77.89475135197863 C481.2959696898445 88.1708429469727, 479.53818698918667 99.44706262042746, 482.0094213352539 129.17727403929456 M478.94279585327024 80.09863320668228 C480.578576095796 94.79483918626794, 480.44013339022433 107.49948713858612, 480.7491001218092 127.21197985042818" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M479.80477091526615 126.73685081118718 C479.2764598497165 125.02098840823746, 477.1773332500203 123.86155203108561, 475.9860663918176 117.90099919534462 M479.36567751196213 128.05917992400936 C478.4522855746692 125.34338119370963, 476.8498322094719 121.43730508572668, 475.22987366375077 116.7218226820248" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M479.80477091526615 126.73685081118718 C481.36537380831953 124.93368676568039, 481.35516116722636 123.77425038852854, 486.42523921048706 117.90099919534462 M479.36567751196213 128.05917992400936 C481.4910610583777 125.35269607106416, 482.9273831768889 121.4466199630812, 485.66904648242024 116.7218226820248" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M339.71238026598724 132.33086865530169 C339.71238026598724 132.33086865530169, 339.71238026598724 132.33086865530169, 339.71238026598724 132.33086865530169 M339.71238026598724 132.33086865530169 C339.71238026598724 132.33086865530169, 339.71238026598724 132.33086865530169, 339.71238026598724 132.33086865530169 M339.9320703507041 145.1611366976092 C344.8466949695824 143.13220111655608, 348.7688404742644 137.05262357057646, 350.23985629214 134.1545520879218 M340.5518057186495 144.76708278979106 C343.90264818407854 141.89462473713758, 345.95508996396615 137.19655696930414, 352.7406031364531 131.51547392593056 M338.9967665217259 158.20019675813654 C350.6510226574386 149.32980745324969, 357.5904036331947 138.93215716027828, 362.9175278162377 129.9851168212743 M340.1586283326988 159.23167423167303 C345.4106632554544 152.89392410566177, 351.5027244843007 145.90583560738125, 364.423877161674 133.07437037786298 M340.45620926435 172.57111261714738 C351.68817592625885 160.35924765320866, 359.78404233781663 148.2062746798038, 376.1992728004468 132.08567459175197 M340.70328390056164 173.61087718819758 C346.4319079288237 163.51132996468849, 354.3231724295028 154.59088037225155, 375.2410357672492 132.32584510768598 M339.2516006195523 185.13740825521893 C351.4400146116885 175.22037919841918, 363.0879233851279 158.41110541867494, 387.32305031257954 133.93838776155212 M340.06525380938086 186.19053447250408 C350.2823061263744 176.49821940941595, 359.4890625430643 164.29581838457247, 388.5926995016848 130.77439506143168 M343.32912271116464 192.59569729926824 C364.4850423161599 171.9363691935665, 382.8663713470632 153.5336465250766, 401.47768786728426 130.83621592540132 M345.5977865282034 193.36680957723237 C356.80897147745264 182.43643883749183, 367.9024040329165 168.31819626581816, 400.62608058421915 130.80105509017432 M355.36237736181664 193.41936393357648 C373.8118724219735 172.77154372741492, 389.8422467461147 153.60644477542846, 412.363043793969 131.42067591449896 M356.8176342092485 193.52264631801117 C374.85196040043064 173.2378868030784, 392.5414266493807 152.63833841355358, 410.7214435332251 131.6344895470859 M367.9443478900016 194.50715496134782 C381.0600856929315 180.24236324484235, 396.3115297359469 164.1896866385279, 425.298618352763 130.728628475626 M369.5454519849745 194.5029233950174 C387.92155013715865 173.58601948355337, 406.6085777865247 151.72814380621563, 422.4991537821718 131.97434146654513 M380.10507274786795 193.64519403980884 C393.08383577035164 180.3270266182728, 404.82253843449314 164.84917843665752, 434.6715745605744 129.8861607524818 M381.8219091058694 193.45303112807522 C394.07632861423275 179.45123370980477, 408.6961200038695 163.66605344535128, 434.8161360056821 132.11167754547392 M394.47937120232365 193.77441206265894 C414.15280455308493 169.33933532806313, 434.662469668105 148.46856102233681, 447.0453493070516 132.15537296479297 M393.12385212155624 194.21286433910672 C412.3260633950353 171.00714576849356, 434.299430955233 148.4932218197939, 447.3729227354943 131.95233732244662 M403.6154544509981 194.43733383917905 C417.51016530932793 182.02953320689295, 429.3156686611384 166.60900702585232, 457.287969325593 131.254712988536 M405.3713534967542 193.8401715312405 C424.84577466075797 171.62654189884816, 444.52577659336174 148.24150287919554, 459.5904078314717 131.6638302579123 M418.02294644829925 193.07196057556854 C435.07424218697406 175.99741521342446, 447.09405701263137 156.12968903500567, 473.00036716435653 134.17931093616022 M416.224169937336 193.93099987014853 C428.73502103577135 179.10633917584704, 442.78033172250423 165.72505951775258, 470.636599952612 133.472232814418 M429.4814950213755 194.858556554054 C438.69114714368476 179.21073057141254, 453.8709102557176 167.26471868051024, 484.7249923022217 131.94776583569669 M429.28879100629007 194.0430443631551 C440.88988844275553 179.09453957591313, 454.119412539926 162.98358633348454, 483.8836711730096 131.63553763571738 M442.36908784112546 191.83998371311202 C451.2558288983338 182.07087638969378, 464.191241270761 167.92726326889195, 496.29383197484634 130.7960281120948 M441.49519259472964 194.64254493945947 C456.02307738983905 176.07353674993573, 471.18096493569567 160.05865883005652, 494.4544153079121 131.8140985281076 M452.74642742219737 194.18732664746904 C475.7168618023313 171.54711859038872, 493.7890468764764 146.8419615889257, 508.83702229062897 132.12418626633098 M452.9796494218929 193.54647395013572 C472.2694857017937 171.607351614896, 492.1853596091483 148.43390059164378, 507.47929652968287 131.8439796395376 M465.8852470939993 195.52022718832444 C479.07845681596694 177.96053916509942, 493.850101884255 159.2346688853601, 517.6111642017179 132.31680145563183 M463.66148850615275 193.46943324835587 C478.69738709491685 179.3383544797653, 494.40473548450205 162.4954546335826, 519.7232028388103 132.73385253983176 M477.9769221166605 196.5536710824593 C496.6068779274218 173.12148942373094, 517.7166951322773 145.3611916886508, 530.1836985040118 133.308532493474 M476.55013778097833 195.44700336849866 C491.83656014562246 177.26714725603193, 507.58388009601344 157.91529691308693, 530.6182585639075 131.6571837206893 M490.38049050708906 192.59187047357292 C508.37961341541865 170.12642661871087, 529.7219551349865 148.805464235118, 540.2612218081683 133.2044704298067 M488.7415844209925 194.0561885074861 C509.4830944449065 171.5124335887191, 529.1769136627934 147.32666331449147, 542.2905778327223 132.94913167590713 M500.5312641549334 192.57505058139193 C514.286230440231 177.81685412732227, 531.6977651977369 159.3571567910233, 553.0980976555671 133.8269252301611 M501.6254234579336 193.83876570389236 C521.6358147723038 170.52858491378436, 543.0137820091014 146.0387231570892, 554.8173131880991 131.17989608048907 M514.869144793592 193.59978428189274 C525.4154450676804 178.2325143838887, 536.7574819519282 166.2622057037641, 565.244477344366 130.5093082569864 M512.861611251665 193.8481819181666 C524.3132306017932 181.96354155811204, 534.321122073489 168.59906768192192, 567.5731139140355 132.82516277198198 M522.6669399712227 194.5393266875248 C537.7960642100697 179.51996908218584, 553.6077741744052 159.3002044823492, 580.7549110454637 133.44291894875917 M523.307218961184 194.04645111285592 C543.5785361074471 173.2886743486896, 560.669245325693 152.18570670168367, 579.0871746516451 132.4394529295851 M535.238974166705 194.39040182631908 C555.6110522201 171.92703360970475, 571.3547662577507 152.066084204143, 589.6166637735082 130.69727940020604 M536.8734811605855 194.47750343171393 C558.4939215137399 170.36000889373142, 578.5252260280811 144.72551197956648, 591.6699888701656 132.85302015437264 M547.2891096156968 195.86708512322497 C568.8770167833734 173.84881629559595, 585.793869851134 150.37575147216504, 604.3340903868983 130.98575936230358 M548.072254165117 193.93763018103766 C563.8298594169703 176.65546689243402, 578.38283860145 161.12318989853426, 601.746161366961 131.26403796929395 M562.3193574265966 194.54759007921896 C576.8626841338727 173.82750402602247, 595.3753684156518 153.5507967568505, 615.3569515587917 131.28591082250108 M560.2730646471416 193.7488330087763 C577.9017568100328 175.55506618988005, 594.2739982355444 157.10439196027926, 613.6811617385285 132.53549689939584 M571.0581935515529 192.49460285152765 C583.3567932357789 180.5455402045236, 596.3604819386236 168.08616477945606, 620.4405526611438 139.62393132561172 M571.9298631071481 194.95572098475407 C590.517488469884 174.44981522006097, 609.2199901011402 153.32148065338876, 620.5899007645386 138.30623029271425 M583.4449337864639 195.0209072536276 C594.204822452379 182.76086491585548, 600.4930461961159 174.4437126875709, 622.1225319676412 150.45732191225315 M585.3138363737492 193.61769810067017 C592.7671650006941 183.6973584505106, 600.4961488853704 174.25101259810268, 621.1119561107674 151.95013475871446 M597.8580497419877 196.70971933546815 C599.9828066376125 189.46214420171592, 608.7783628697215 182.38577565849417, 622.8920620595684 167.6745779957287 M595.040161074693 194.10586035032162 C603.8042522673633 186.27137824746353, 611.7421336109243 177.14591981772486, 621.2292112349645 165.60242354517246 M608.1045121115233 193.6031842020142 C611.9911554154162 188.7752134818942, 615.3216926960139 184.23023539458296, 621.3330084858237 180.28980642162628 M608.5807207901772 194.47231150583943 C612.6438824135424 189.87045289067004, 618.172776431557 182.7921035702127, 619.883729393816 179.28375611396638" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M339.75055254461745 130.38605322896737 C396.59818306582196 129.10541281916872, 451.27956569558415 130.3866482244047, 621.5404826782457 132.9025335967444 M339.18951558499253 132.07561517523882 C421.60087746264054 132.8219766054248, 503.0009266400517 131.85001192853815, 620.5742826263826 131.39588011273375 M619.2458272562362 131.72748577883468 C620.9800771655453 144.89861647011713, 620.8712014587534 155.1176627154462, 619.9799880997277 194.00556195965038 M620.9735754221213 131.55920121301898 C618.9058866103254 155.3822370315436, 618.3782939051353 178.98711112407034, 619.2618925824296 194.1168464579154 M620.1890529576444 193.19849445472957 C518.8046336130867 195.65697568012988, 419.2646475589104 195.21522212141284, 338.4470200040973 192.8706037802569 M620.2655356005216 194.61199177879936 C551.1043741142918 195.77745759744838, 482.51590752099537 195.5626227347068, 339.64368574294866 194.84849287454986 M339.340021030698 195.75517282178626 C340.6046256299737 170.38423970201984, 340.78176276345283 144.98072286872193, 341.8531188325025 132.29916372885927 M338.8649603902828 195.10299934078938 C340.06347115888605 172.372626127908, 339.6486108641268 147.72063965094276, 340.15758806313386 132.83339553554543" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="480" y="160" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:700;fill:var(--ink)">learn a sampler x = G(z)</text><text x="480" y="182" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">push noise through a network</text><path d="M479.7741879868719 78.53897659974278 C544.9324786909374 86.75249076458302, 607.74568101739 97.51729072383354, 801.3945201173939 128.81701746800374 M479.2663099445888 80.06845054770164 C573.5514505544595 94.74135188097192, 666.6052965869036 107.83397061152382, 800.5198686664054 127.45312118856806" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M799.8047709152661 126.73685081118718 C798.1933392379118 127.94752857342793, 794.8454658024358 129.66244368732202, 789.7455585088652 132.26533039867252 M799.3656775119621 128.05917992400936 C796.7718929633512 129.5191764116049, 793.3528682385256 129.79446392876264, 788.9893657807984 131.0861538853527" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M799.8047709152661 126.73685081118718 C798.3841737369283 125.86877525655288, 795.3461707442038 125.5178874187704, 791.2941101330038 121.94165290441562 M799.3656775119621 128.05917992400936 C797.1247943189343 126.51540264068635, 794.1565429257154 123.78553461379978, 790.5379174049369 120.76247639109579" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M659.7524725076042 132.2847478071708 C659.7524725076042 132.2847478071708, 659.7524725076042 132.2847478071708, 659.7524725076042 132.2847478071708 M659.7524725076042 132.2847478071708 C659.7524725076042 132.2847478071708, 659.7524725076042 132.2847478071708, 659.7524725076042 132.2847478071708 M659.9721625923211 145.11501584947834 C664.8867872111994 143.08608026842518, 668.8089327158814 137.0065027224456, 670.279948533757 134.1084312397909 M660.5918979602665 144.7209619416602 C663.9427404256955 141.84850388900668, 665.9951822055831 137.15043612117324, 672.7806953780702 131.46935307779967 M659.0368587633428 158.15407591000567 C670.6911148990556 149.28368660511882, 677.6304958748117 138.88603631214744, 682.9576200578547 129.93899597314345 M660.1987205743158 159.18555338354216 C665.4507554970713 152.84780325753093, 671.5428167259176 145.85971475925038, 684.463969403291 133.02824952973214 M660.4963015059669 172.52499176901648 C671.7282681678759 160.31312680507776, 679.8241345794336 148.16015383167291, 696.2393650420638 132.03955374362107 M660.7433761421786 173.56475634006668 C666.4720001704407 163.4652091165576, 674.3632646711196 154.54475952412065, 695.2811280088662 132.27972425955508 M659.2916928611692 185.091287407088 C671.4801068533055 175.17425835028828, 683.1280156267449 158.364984570544, 707.3631425541965 133.89226691342122 M660.1053460509978 186.14441362437316 C670.3223983679912 176.45209856128503, 679.5291547846812 164.24969753644154, 708.6327917433017 130.72827421330078 M663.3692149527817 192.54957645113734 C684.3079707933218 172.149916277126, 702.4693278104669 154.00024246373638, 720.8617210799107 131.5448046574932 M665.6378787698204 193.32068872910148 C676.7142715262439 182.5440775686214, 687.6758797369596 168.5774815584485, 720.0101137968456 131.50964382226618 M675.4024696034336 193.3732430854456 C693.8519646635905 172.72542287928405, 709.8823389877317 153.5603239272976, 732.4031360355859 131.3745550663681 M676.8577264508655 193.4765254698803 C694.8920526420476 173.19176595494753, 712.5815188909976 152.59221756542271, 730.7615357748421 131.58836869895504 M687.9844401316186 194.46103411321693 C701.1001779345484 180.19624239671145, 716.3516219775639 164.143565790397, 745.33871059438 130.6825076274951 M689.5855442265914 194.4568025468865 C707.9616423787757 173.53989863542247, 726.6486700281417 151.68202295808473, 742.5392460237888 131.92822061841423 M700.145164989485 193.59907319167797 C713.1239280119687 180.28090577014194, 724.8626306761101 164.80305758852663, 754.7116668021914 129.8400399043509 M701.8620013474865 193.40691027994436 C714.1164208558499 179.40511286167384, 728.7362122454866 163.61993259722038, 754.8562282472991 132.06555669734303 M714.5194634439407 193.728291214528 C734.1928967947019 169.2932144799322, 754.702561909722 148.4224401742059, 767.0854415486687 132.10925211666208 M713.1639443631733 194.1667434909758 C732.3661556366523 170.96102492036263, 754.33952319685 148.44710097166296, 767.4130149771113 131.90621647431573 M723.6555466926151 194.39121299104818 C737.5502575509449 181.98341235876208, 749.3557609027553 166.56288617772145, 777.3280615672099 131.20859214040513 M725.4114457383712 193.79405068310962 C744.885866902375 171.5804210507173, 764.5658688349787 148.19538203106464, 779.6305000730887 131.61770940978144 M738.0630386899161 193.0258397274377 C755.114334428591 175.95129436529362, 767.1341492542482 156.0835681868748, 793.0404594059735 134.13319008802935 M736.2642621789529 193.88487902201769 C748.7751132773883 179.0602183277162, 762.8204239641211 165.67893866962174, 790.676692194229 133.42611196628712 M749.5215872629925 194.81243570592312 C758.7312393853017 179.16460972328167, 773.9110024973346 167.21859783237937, 804.7650845438386 131.90164498756582 M749.328883247907 193.99692351502424 C760.9299806843725 179.04841872778226, 774.159504781543 162.93746548535367, 803.9237634146265 131.58941678758652 M762.4091800827424 191.79386286498107 C771.2959211399507 182.02475554156283, 784.2313335123779 167.881142420761, 816.3339242164634 130.74990726396385 M761.5352848363466 194.59642409132852 C776.0631696314559 176.02741590180477, 791.2210571773127 160.01253798192556, 814.4945075495291 131.76797767997664 M772.1304606348237 194.8959153795609 C795.3599743572818 171.9590531905385, 813.6863528982785 146.96148005548744, 828.877114532246 132.07806541820008 M772.3636826345194 194.2550626822276 C791.8937345601125 172.03822491122293, 812.0552491241808 148.58219663695832, 827.5193887712999 131.7978587914067 M785.9253393356162 195.47410634019354 C799.1185490575839 177.91441831696852, 813.8901941258721 159.18854803722922, 837.6512564433349 132.270680607501 M783.7015807477696 193.42331240022497 C798.7374793365339 179.2922336316344, 814.444827726119 162.44933378545173, 839.7632950804273 132.68773169170092 M798.0170143582775 196.50755023432845 C816.6469701690388 173.0753685756001, 837.7567873738943 145.31507084051992, 850.2237907456289 133.26241164534312 M796.5902300225954 195.40088252036782 C811.8766523872396 177.2210264079011, 827.6239723376306 157.86917606495606, 850.6583508055246 131.61106287255842 M810.4205827487061 192.54574962544206 C828.4197056570356 170.08030577058, 849.7620473766035 148.75934338698715, 860.3013140497853 133.15834958167582 M808.7816766626096 194.01006765935523 C829.5231866865236 171.46631274058822, 849.2170059044105 147.2805424663606, 862.3306700743393 132.90301082777626 M820.5713563965504 192.528929733261 C834.326322681848 177.77073327919135, 851.737857439354 159.31103594289237, 873.1381898971842 133.78080438203017 M821.6655156995506 193.79264485576144 C841.6759070139208 170.48246406565343, 863.0538742507185 145.99260230895828, 874.8574054297162 131.13377523235815 M834.909237035209 193.55366343376187 C845.4555373092974 178.18639353575784, 856.7975741935452 166.21608485563323, 885.284569585983 130.4631874088555 M832.901703493282 193.80206107003573 C844.3533228434102 181.91742070998117, 854.361214315106 168.55294683379105, 887.6132061556525 132.77904192385108 M842.7070322128399 194.4932058393939 C857.8361564516869 179.47384823405494, 873.6478664160223 159.2540836342183, 900.7950032870807 133.39679810062827 M843.3473112028012 194.00033026472502 C863.6186283490641 173.2425535005587, 880.7093375673101 152.13958585355277, 899.1272668932621 132.3933320814542 M855.279066408322 194.34428097818818 C875.651144461717 171.88091276157385, 891.3948584993677 152.01996335601214, 909.6567560151252 130.6511585520752 M856.9135734022025 194.43138258358303 C878.5340137553569 170.31388804560052, 898.5653182696981 144.67939113143564, 911.7100811117826 132.8068993062418 M867.3292018573139 195.82096427509413 C888.6827068172008 174.0711347943884, 905.3633302616579 150.86982106645544, 923.7181235995248 131.6943480943955 M868.1123464067341 193.89150933290682 C883.6900993669847 176.80932116666386, 898.0632090205693 161.48396039852744, 921.1301945795875 131.97262670138588 M882.3594496682136 194.50146923108804 C896.9027763754897 173.78138317789154, 915.4154606572688 153.50467590871955, 935.3970438004087 131.23978997437015 M880.3131568887586 193.70271216064538 C897.9418490516498 175.50894534174913, 914.3140904771614 157.0582711121483, 933.7212539801455 132.4893760512649 M891.0982857931699 192.44848200339678 C903.2371041237373 180.6728106774335, 916.0861329163523 168.39135112695786, 939.8245858737704 140.33252005770362 M891.9699553487651 194.9096001366232 C910.3149981921384 174.67681830310195, 928.7735967016131 153.8290621821497, 939.9739339771652 139.01481902480614 M903.4850260280809 194.9747864054968 C914.244914693996 182.71474406772467, 920.5331384377329 174.39759183944008, 942.1626242092582 150.41120106412234 M905.3539286153662 193.57157725253936 C912.8072572423112 183.65123760237978, 920.5362411269874 174.20489174997186, 941.1520483523844 151.90401391058364 M917.8981419836047 196.66359848733725 C920.0228988792295 189.41602335358502, 928.8184551113386 182.3396548103633, 942.9321543011855 167.62845714759783 M915.08025331631 194.05973950219072 C923.8443445089804 186.22525739933263, 931.7822258525414 177.09979896959396, 941.2693034765816 165.5563026970416 M928.1446043531403 193.5570633538833 C932.0312476570332 188.72909263376332, 935.3617849376309 184.18411454645207, 941.3731007274407 180.24368557349538 M928.6208130317942 194.42619065770853 C932.6839746551594 189.82433204253914, 938.2128686731741 182.7459827220818, 939.923821635433 179.23763526583548" style="stroke:var(--dot);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M659.7505525446174 130.38605322896737 C716.5981830658219 129.10541281916872, 771.2795656955842 130.3866482244047, 941.5404826782457 132.9025335967444 M659.1895155849926 132.07561517523882 C741.6008774626405 132.8219766054248, 823.0009266400517 131.85001192853815, 940.5742826263826 131.39588011273375 M939.2458272562362 131.72748577883468 C940.9800771655453 144.89861647011713, 940.8712014587534 155.1176627154462, 939.9799880997277 194.00556195965038 M940.9735754221213 131.55920121301898 C938.9058866103254 155.3822370315436, 938.3782939051353 178.98711112407034, 939.2618925824296 194.1168464579154 M940.1890529576444 193.19849445472957 C838.8046336130867 195.65697568012988, 739.2646475589104 195.21522212141284, 658.4470200040973 192.8706037802569 M940.2655356005216 194.61199177879936 C871.1043741142918 195.77745759744838, 802.5159075209954 195.5626227347068, 659.6436857429486 194.84849287454986 M659.3400210306979 195.75517282178626 C660.6046256299737 170.38423970201984, 660.7817627634528 144.98072286872193, 661.8531188325026 132.29916372885927 M658.8649603902828 195.10299934078938 C660.063471158886 172.372626127908, 659.6486108641268 147.72063965094276, 660.1575880631339 132.83339553554543" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="800" y="160" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:700;fill:var(--ink)">learn the score ∇ log p</text><text x="800" y="182" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">then sample by a dynamics</text><text x="34" y="236" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--ink)">• autoregressive</text><text x="52" y="257" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">exact p, slow to sample</text><text x="34" y="294" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--ink)">• normalising flows</text><text x="52" y="315" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">exact p, invertible maps</text><text x="34" y="352" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--ink)">• energy-based</text><text x="52" y="373" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">p up to a constant</text><text x="354" y="236" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--ink)">• VAEs</text><text x="372" y="257" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">a lower bound on log p</text><text x="354" y="294" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--ink)">• GANs</text><text x="372" y="315" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">sharp samples, no p</text><text x="674" y="236" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--ink)">• score matching + Langevin</text><text x="692" y="257" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">the score needs no constant</text><text x="674" y="294" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--dot)">• diffusion models</text><text x="692" y="315" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">post 2 of this series</text><text x="674" y="352" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--dot)">• flow matching</text><text x="692" y="373" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">post 3</text><path d="M399.74474518479985 432.29363707519735 C399.74474518479985 432.29363707519735, 399.74474518479985 432.29363707519735, 399.74474518479985 432.29363707519735 M399.74474518479985 432.29363707519735 C399.74474518479985 432.29363707519735, 399.74474518479985 432.29363707519735, 399.74474518479985 432.29363707519735 M399.76982435535956 442.44044768148603 C403.59231017004265 440.8623866740002, 406.6428677847953 436.13382636046055, 407.78699119869856 433.87977076284017 M400.2518407526504 442.13396130873855 C402.85805155909526 439.89982726778584, 404.45439516567444 436.24577455947093, 409.7320165220532 431.82715441462483 M398.6345292356095 452.5198907411791 C408.51581594658563 445.6890170343364, 413.7040017750296 437.3058863137116, 417.9628773271877 429.58777786587626 M399.7963910465824 453.5513682147156 C403.85674519401135 448.5633718855961, 408.78381514469595 442.9154525484008, 419.469226672624 432.67703142246495 M400.55272990475953 463.31455980154055 C409.52079262531913 453.6445460331908, 415.39654879589057 444.0455177415644, 428.42308509297027 431.8856367388184 M400.7998045409712 464.3543243725907 C404.88238873568827 456.2899754747071, 411.0897843027847 449.30659550536456, 427.46484805977263 432.12580725475243 M399.15082015749726 473.0593182211855 C408.7683873020799 466.1445423455586, 417.9149448393289 452.2127440032909, 436.7253253866764 433.93565101108305 M399.96447334732585 474.1124444384707 C408.0519594869708 466.90920091802417, 415.02355256920356 457.27806117811895, 437.99497457578167 430.7716583109626 M397.7825689147211 483.7337466885905 C416.3325233462566 466.19043376317404, 432.0741882120607 450.8242973558878, 448.0584257229546 431.03078027739684 M400.05123273175985 484.50485896655465 C409.6449113671005 475.4196031779395, 419.1564517855874 463.1211193442754, 447.2068184398895 430.99561944216987 M397.8094599410794 495.32064854848204 C416.90423961890707 473.9350029660644, 433.5546405623379 454.05664497961106, 456.77830346020323 431.0578317887362 M399.2647167885113 495.42393093291673 C417.92955109481545 474.38867693571007, 436.26630294724293 453.0445116374957, 455.13670319945936 431.27164542132317 M398.38506684497065 507.17167480183673 C414.03499733384706 489.7421153343514, 431.9166101074041 470.66377571476124, 466.8923408005707 430.5630854523278 M399.9861709399435 507.1674432355063 C422.2281428421861 481.8147218580705, 444.6988991612004 455.6041642576731, 464.0928762299795 431.8087984432469 M406.41213642642947 508.0164341432078 C422.1856623536949 491.24755942780644, 436.81267851657725 472.4470866470897, 473.44375978995555 429.9179188316482 M408.1289727844309 507.8242712314742 C423.40717590007154 490.2805925613657, 441.22677208423033 470.81445795201984, 473.58832123506323 432.1434356246403 M418.62095669144907 507.58824368829966 C442.5424038698617 478.0111195330556, 467.44219523942775 452.09008268038815, 482.99599731800623 432.3844321464238 M417.26543761068166 508.02669596474743 C440.83351335945144 479.81651219522314, 467.27763552707444 452.1595733897036, 483.3235707464489 432.18139650407744 M424.935502721697 508.4484665672843 C441.50783541614294 493.1686354648493, 455.905174930096 474.76654284618684, 490.4170801181211 431.68107327263135 M426.6914017674531 507.85130425934574 C450.37396001295383 480.811199773925, 474.3722081377406 452.4585867601826, 492.71951862399976 432.09019054200763 M437.17751652956207 506.52568482591556 C457.772212827922 485.4873530357206, 473.2425035493337 461.6503083969483, 503.9639997674485 434.0482627424974 M435.3787400185988 507.38472412049555 C450.5494529352454 489.40934641258485, 467.3429501271547 472.8666398217843, 501.600232555704 433.34118462075514 M445.8145278842118 508.5095819068656 C457.54484892080654 489.7468003798677, 475.327236275588 474.80681178423987, 512.8670876868871 432.0140187444984 M445.6218238691264 507.69406971596675 C459.960404210373 489.3900386020068, 476.01365963754256 470.0307540470671, 512.025766557675 431.7017905445191 M455.88058348553517 505.68831016838806 C467.19859679799214 493.0422330644265, 482.5802493919162 476.0845424763414, 521.6143901410852 431.05958212336094 M455.00668823913935 508.4908713947355 C472.7855678080045 486.1963536001273, 491.2441823784191 466.3844235903359, 519.7749734741509 432.07765253937373 M463.4363858481805 508.2329542052096 C491.3293277324718 479.9563176412294, 514.2311886767965 449.6952541015942, 531.9921022674317 431.8303317998389 M463.669607847876 507.59210150787624 C487.5235413959316 480.37638589310245, 512.1065877808453 451.833967100615, 530.6343765064856 431.55012517304556 M474.4097273305464 509.00844626830684 C490.92204186919616 487.61819336058244, 509.0901214241492 464.9851721501878, 537.9447069600942 432.22024809160433 M472.18596874269986 506.95765232833827 C490.3660048834857 489.3788601370472, 509.1540341281367 468.9920423624662, 540.0567455971866 432.63729917580423 M483.67986513478104 510.2391912649062 C506.9857937409116 481.7244629157375, 532.6374731127685 448.7393504337866, 547.6957040439615 433.40928023191094 M482.2530807990989 509.13252355094556 C500.716250062898 487.093456338534, 519.8059024894287 463.89669230860915, 548.1302641038573 431.75793145912627 M493.26189630678306 506.4746917584844 C515.9206113852742 478.8046049373497, 541.8187981055097 452.2427423967931, 555.6077491586819 432.74780969048555 M491.6229902206865 507.9390097923976 C517.1648930751068 480.05137787821815, 541.6303467092107 450.37647012076604, 557.6371051832359 432.49247093658596 M500.59113273620096 506.6551729687679 C517.9208593557516 487.83137214446407, 538.9065070050419 465.26012825361533, 565.6230877876543 433.5675655933044 M501.68529203920116 507.9188880912683 C526.3958243726827 478.95392596959925, 552.640999779176 448.86498170311813, 567.3423033201863 430.92053644363233 M512.1074761564329 507.8772077717332 C525.5582583416976 489.1620258250004, 539.8124054921824 473.84171749625284, 574.9479302580266 430.44724972259417 M510.0999426145058 508.12560540800706 C524.084266000191 493.2359590936149, 536.7762176111166 476.7838272301734, 577.2765668276961 432.76310423758974 M517.7397931446277 508.2593416996071 C536.0362574523548 489.5322220377736, 555.1501744065747 465.5137028426694, 587.6368267406978 433.5781615168316 M518.380072134589 507.76646612493823 C542.6600312929932 482.5270237824025, 563.7221138066701 456.8555137630515, 585.9690903468792 432.5746954976575 M527.4902901216834 508.30771794086587 C551.867993174254 481.15211257247785, 571.6952403079554 456.5935957033779, 593.6770422503157 431.0298230707429 M529.1247971155639 508.3948195462607 C555.358367949997 479.0230269443336, 580.0254501377362 448.05567785179113, 595.7303673469731 433.18556382490954 M536.7188883522485 509.98170234023627 C562.5260352601396 483.1315252679876, 583.6950215473616 454.76694072559326, 605.5729316452791 431.515604135305 M537.5020329016687 508.05224739804896 C556.4969794023714 487.1705319069526, 574.2876101429617 467.9137628493131, 602.9850026253418 431.79388274229535 M549.5836579737124 508.10479881847203 C567.7016928143266 483.13584257854603, 589.9547403816573 458.5563395541461, 614.4303146277366 431.2583471177442 M547.5373651942574 507.3060417480294 C568.8816089081679 484.9929391640602, 588.9462371158845 462.29465983297024, 612.7545248074734 432.507933194639 M555.500956880242 506.24911269324525 C571.9538717595957 489.79187569918184, 588.9787181284133 472.7066875347245, 621.9408507435862 433.7559920815372 M556.3726264358372 508.7102308264716 C581.2673973020025 481.1031028513105, 606.3113800995766 452.67972869591915, 622.090198846981 432.43829104863977 M565.0661598967265 508.97271819780974 C582.960663036726 488.98052899250484, 595.9515873029834 472.95280184004895, 633.9224734114671 429.6924921661878 M566.9350624840117 507.5695090448523 C581.6663290029156 489.12845066157905, 596.57470839144 471.4231550385999, 632.9118975545933 431.1853050126491 M577.3137976628142 510.10412180189206 C589.0562781553654 491.79166016867384, 607.162323081225 474.00479957637305, 643.679528806797 433.522276908118 M574.4959089955195 507.50026281674553 C596.1910556566867 485.205670512702, 616.9777974143735 461.29928899376023, 642.0166779821931 431.45012245756175 M584.7579906619036 507.15317763146874 C604.5606534453678 483.5743884058452, 623.9281532266785 460.543435666734, 651.8235050357142 432.00761975933165 M585.2592629552236 508.0680484776006 C610.3364275540926 479.47200111667945, 637.1214681510638 447.90345840120176, 650.2979480967587 430.94861943547915 M593.8768242211434 509.57671246876635 C616.3096274735349 478.9013117870726, 640.114949327533 454.276656719039, 658.8331710191176 430.00310140830527 M594.0448951761125 508.1297797847416 C618.6311996008307 478.5644114635394, 643.715277560567 449.5247107710734, 659.6540431644034 432.2224265006005 M601.8967235114313 507.9441507522616 C619.2310361789079 487.42822160264797, 638.1022858316262 468.1955004399166, 667.7950105216555 430.3665251997656 M602.2210627992991 508.0259154173679 C621.3267129992347 485.9143658406857, 640.3388974887291 465.2246154912725, 669.7270972793657 431.4264066290001 M612.6768285474166 509.8031395766807 C632.2132810942834 485.9233050802836, 648.6172932214571 465.8645655497493, 679.5875971517264 430.7245421346889 M611.4873217940221 508.3226464765535 C625.655370175858 491.263498025454, 639.7555798954991 474.69498689352463, 677.4358558640088 432.020842718627 M622.6933691874973 508.84517369300056 C643.5166999804916 482.66735279031735, 663.7183707099017 458.4219877764815, 685.7659693614788 431.73827587030195 M621.1073394576449 507.4040839852479 C644.4524166809631 483.5530613929113, 665.910340475734 459.0570835154562, 686.858488168821 432.01362508930566 M632.3240401338219 506.9712205084477 C649.642135047911 482.81795483335077, 673.6290394152248 457.2049032281085, 695.9460054468108 433.86894772008134 M630.8917069156553 509.4144463882752 C643.4146920075631 492.02214839635263, 657.086522621146 477.33727701611497, 697.2653997724154 433.0524389827352 M641.0323759322886 506.45734508649855 C655.2378355718083 488.4763188898952, 673.4267565802897 473.03550094892427, 707.6887985473712 431.84350165375355 M640.7307516216666 508.0502470523346 C656.8700599661237 487.1946383317768, 674.9707686011844 464.902680930719, 706.1020450815306 432.2669520624898 M649.4521259725464 509.02033514687963 C666.5156337282443 489.74374465495174, 687.1719839763535 468.38389805945286, 715.4850316465584 434.420229297295 M648.9773498574176 508.30248176600026 C668.4027069659054 486.2738015105459, 688.7141690592424 461.95887188083947, 715.4217398310298 431.3610019140186 M659.8276885169837 507.2286164853449 C674.5803791735814 485.2994262323075, 695.079338676705 464.51111563494453, 726.6393525715206 431.25439493293254 M659.5357982617617 507.1941066594987 C672.3985903048763 493.1785799477065, 685.2828182567949 476.22354610662615, 723.3684735839009 433.05478354464594 M666.2742495847683 509.5867162945664 C691.3104224622434 477.8848673291258, 718.0695415941425 448.41753075647284, 733.4547489931564 431.06386049914255 M668.4203902147062 507.0410640255997 C684.5141228259934 487.8412793839415, 703.1198000390501 466.7621414082143, 732.9097222312577 431.77703688357104 M676.8985121257888 509.7726306827978 C704.1934468453148 478.0794498081153, 729.9063186172747 449.77548792895897, 743.5729824551048 432.34000101042875 M676.4245825113549 508.5162123382796 C693.7749432093844 486.81030531981594, 711.1800476128278 469.37636920212697, 743.9753195213286 432.3525247016452 M685.6588349046939 507.1222879947659 C711.9926071481641 478.8730469220317, 735.4973645167911 451.6186145454961, 754.1347272577831 432.3321678782135 M687.179571745946 507.8103520736128 C709.7987855980175 479.16836290303706, 733.6990838350529 451.0807130290978, 752.6660334434961 432.2130584322635 M695.8056665298822 507.24046423144705 C714.1367819387722 484.9624453269512, 732.5169448163348 466.4188125554032, 760.4321225453589 433.04918517016876 M695.8431658408431 507.2697036295484 C716.3818582894423 482.9505845623959, 738.5182173525535 457.8678824819541, 760.8045558152675 433.97555693069546 M703.5659204535016 507.2664080500679 C716.2131453561253 494.94772601902093, 728.5545313037475 480.1446353046232, 760.9567931318062 442.5532320915531 M705.8825965465342 507.2380948259106 C719.4886398994913 490.4700793314121, 735.611310864644 473.08776075196965, 760.0319606083317 445.2624914554223 M716.3273499714508 506.4394657641051 C727.7758291946993 493.90587390168196, 741.2206551273614 476.1215178206521, 762.0857002256694 455.1577284796429 M714.8073319416325 508.2491484043987 C724.5639989157373 495.34661093678636, 736.7587918366542 484.0310242314179, 760.3919921733228 456.27633366256816 M722.5229232233785 509.76470634328507 C733.1435380962857 497.53626044271397, 741.7612931935407 482.5021397441829, 758.1837455603469 466.23424721041465 M724.0518889328766 507.46846952858687 C736.9386635726186 493.1469212273724, 750.2776066534615 477.42547625504443, 760.9821068701044 466.9330305125821 M732.8151981926827 506.37854641358444 C744.2355316394663 497.37291600980575, 750.6023283246851 486.4023664375372, 759.3047957129869 476.7763338160948 M731.6086457551295 507.7858332839244 C740.2830650100636 499.72848192015454, 748.17575944736 490.04277991324506, 760.1529348866641 477.30238638778826 M740.7806977065122 510.0143752572297 C750.2624104754468 501.2639066942401, 753.0575802247795 495.4191610831831, 758.7760571044881 485.5651314851599 M741.9106544713798 507.3573721777352 C747.4822356514605 501.71337185996475, 750.8599403330484 497.30422406626826, 759.6410927391156 487.5002505923858 M750.4989415961348 509.3539680174885 C752.7463832998523 506.2639168386134, 755.716115191809 503.6524335928628, 758.8461061125086 497.5509661553003 M751.3880599220174 508.6046766960535 C753.6616736412175 506.1011283165774, 756.5283669378721 502.6370775556355, 760.5047874530475 498.45795530837904" style="stroke:var(--dot);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M399.7939402564087 430.66677550468916 C472.50744551142327 429.1851955998392, 543.4314897980698 430.2435789723139, 761.2725383997176 432.7455511672487 M399.3304874147341 432.06246302892697 C504.8718878462295 432.7242123829405, 609.5778785302821 431.92130664957915, 760.47439461974 431.5009578715583 M759.2458272562362 431.72748577883465 C760.8703458513537 447.73435910148544, 760.7614701445618 460.7891479781829, 759.9799880997277 508.0055619596504 M760.9735754221213 431.559201213019 C758.7713359910289 460.60567007330246, 758.2437432858387 489.43397720758804, 759.2618925824296 508.11684645791536 M760.1561699794357 507.33790454229273 C630.013303220451 509.85132992299947, 501.3940577399611 509.48641289107314, 398.7171380070109 507.0670456225086 M760.2193495927787 508.505544820355 C671.4284551850128 509.8736934758727, 583.1106438263921 509.69622597409716, 399.70566136129423 508.7009100329392 M399.340021030698 509.75517282178623 C400.4820906719005 478.8316394108347, 400.6592278053796 447.8755222863518, 401.8531188325025 432.2991637288593 M398.8649603902828 509.1029993407894 C400.03689191368653 481.2398878355045, 399.6220316189273 451.45516306613575, 400.15758806313386 432.83339553554543" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="580" y="464" text-anchor="middle" style="font-family:Kalam,cursive;font-size:21px;font-weight:700;fill:var(--ink)">one-step generators</text><text x="580" y="488" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">post 4: one network pass, diffusion quality</text><path d="M799.6746181921102 397.89475135197864 C781.055022202033 405.00506386749913, 759.2868996508154 412.2792155708419, 702.0094213352538 429.17727403929456 M798.9427958532702 400.0986332066823 C771.2796180362122 408.883738902607, 742.0318240775036 415.7665166042978, 700.7491001218092 427.2119798504282" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M699.8047709152661 426.7368508111872 C702.1544533789972 425.4755144098886, 702.9002783678452 424.88950263509497, 710.2034740410165 420.76664069189394 M699.3656775119621 428.05917992400936 C702.618047962994 426.1902370401377, 705.1541897294924 423.11833058104094, 709.4472813129497 419.58746417857407" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M699.8047709152661 426.7368508111872 C702.833596809129 427.46352425692163, 704.1426555130527 428.88906146454144, 713.0181874306045 430.81918851185094 M699.3656775119621 428.05917992400936 C703.532694344778 429.1189800646259, 706.8881809186206 428.97330506032995, 712.2619947025377 429.64001199853107" style="stroke:var(--dot);stroke-width:1;fill:none"/><text x="846" y="420" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--dot)">distil or shortcut</text><text x="846" y="438" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--dot)">the dynamics</text><path d="M479.6746181921102 349.89475135197864 C489.6229563222209 365.43922254887406, 495.86930956178696 382.3183373804856, 522.0094213352538 429.17727403929456 M478.94279585327024 352.0986332066823 C492.48776612607634 374.98114955035965, 503.9930639217595 395.8364158535563, 520.7491001218092 427.2119798504282" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M519.8047709152661 426.7368508111872 C518.4319938329883 425.7933325530415, 515.4460327777996 425.3691769920096, 511.5541853637319 421.5755035846308 M519.3656775119621 428.05917992400936 C517.197029311209 426.4090510142935, 514.3044841304251 423.57259929579976, 510.7979926356651 420.3963270713109" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M519.8047709152661 426.7368508111872 C520.2244510568271 424.74317351398525, 519.0870080285944 423.34611372823974, 520.7919996175672 416.7134960826122 M519.3656775119621 428.05917992400936 C519.8400036247014 425.0019950772873, 519.6365262431788 420.75024451707714, 520.0358068895003 415.53431956929234" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="424" y="398" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">keep a one-pass</text><text x="424" y="416" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">sampler</text></svg></figure>

Side by side:

| family | evaluate $p(\boldsymbol{x})$ | network passes per sample | typical failure | training |
|---|---|---|---|---|
| autoregressive | exact | $d$ (one per coordinate) | slow sampling | stable, maximum likelihood |
| normalising flow | exact | 1 | bridges between modes; restricted layers | stable, maximum likelihood |
| VAE | lower bound | 1 | blur, holes, posterior collapse | stable, ELBO |
| GAN | none | 1 | missing modes | unstable minimax game |
| energy-based | up to $Z_\theta$ | many (MCMC) | slow, hard to train | needs model samples |
| diffusion and flow matching | through an ODE, costly | tens to hundreds | slow sampling | stable, simple regression |

Xiao, Kreis and Vahdat (2022) summarised the tension as a *trilemma*: high sample quality, coverage of all modes, and fast sampling. GANs had quality and speed but not coverage. VAEs and flows had speed and coverage but lower quality. Diffusion models, which learn the score of the data blurred at every noise level and then follow a dynamics like Langevin's, achieve quality and coverage and pay in speed.

The rest of the series follows that route and then removes its cost:

- **Diffusion models** (next): add noise until the data are Gaussian, learn to undo it, and see why denoising is score matching in disguise.
- **Flow matching**: replace the noisy dynamics by a deterministic flow, learned by plain regression on a velocity field, and make its paths straighter.
- **One-step generators**: learn to jump along the whole flow in a single network pass. This brings back the GAN's speed with the diffusion model's coverage: consistency models, flow maps, MeanFlow, and drifting models, where the kernel discrepancies of Section 5 return.

## Questions

:::question
A model can produce samples but not evaluate its density. Which of the divergences in this post can you still use to train it, and why not maximum likelihood?
:::

:::answer
Maximum likelihood needs $\log p_\theta(\boldsymbol{x})$ at the data points, which a sampler cannot provide. Divergences that need only samples remain: the Jensen–Shannon divergence through a trained classifier (a GAN), the Wasserstein distance through a constrained critic, and the MMD, whose three terms are averages of a kernel over pairs of samples.
:::

:::question
Why is the density of a flow's output low where the map stretches space, and high where it compresses it?
:::

:::answer
By change of variables, $p_{\boldsymbol{x}}(\boldsymbol{x}) = p_{\boldsymbol{z}}(\boldsymbol{z})/\lvert\det\mathbf{J}_f(\boldsymbol{z})\rvert$, and $\lvert\det\mathbf{J}_f\rvert$ is the factor by which $f$ multiplies small volumes. The probability in a small region is preserved, so if the region's volume is multiplied by 3, its density is divided by 3.
:::

:::question
In the GAN proof, what does the discriminator output at the optimum when the generator is perfect, and what does that tell you about training dynamics?
:::

:::answer
With $p_g = p_{\mathrm{data}}$, $D^*(\boldsymbol{x}) = \tfrac12$ everywhere: the best classifier can only guess. The value is $V = -\log 4$, the minimum. During training the discriminator is never optimal and the generator never perfect, and the two chase each other; nothing in the game guarantees that this chase converges.
:::

:::question
Why can score matching be computed from data, although its definition contains the unknown $\nabla\log p_{\mathrm{data}}$?
:::

:::answer
Integration by parts moves the derivative from the data density onto the model's score (Step 3 of the proof), turning $\mathbb{E}_p[s_\theta(\log p)']$ into $-\mathbb{E}_p[\nabla\cdot\boldsymbol{s}_\theta]$. What remains are expectations under $p_{\mathrm{data}}$ of quantities that depend only on the model, which sample averages estimate.
:::

:::question
The VAE in Section 8 reports $-\text{ELBO} = 1.71$ and the flow an exact negative log-likelihood of $0.90$. Can you conclude that the flow fits the data better?
:::

:::answer
Not quite. The $-\text{ELBO}$ is only an upper bound on the VAE's negative log-likelihood; its true value lies somewhere below 1.71. An importance-weighted estimate would tighten it. The sample statistics (53% against 75% on a mode) point the same way as the numbers, so the conclusion is probably right, but the comparison of a bound with an exact value is not by itself a proof.
:::

## References

- I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio. *Generative adversarial nets.* NeurIPS, 2014.
- D. P. Kingma, M. Welling. *Auto-encoding variational Bayes.* ICLR, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- D. J. Rezende, S. Mohamed, D. Wierstra. *Stochastic backpropagation and approximate inference in deep generative models.* ICML, 2014.
- D. J. Rezende, S. Mohamed. *Variational inference with normalizing flows.* ICML, 2015.
- L. Dinh, J. Sohl-Dickstein, S. Bengio. *Density estimation using Real NVP.* ICLR, 2017. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, B. Lakshminarayanan. *Normalizing flows for probabilistic modeling and inference.* JMLR, 2021.
- A. Hyvärinen. *Estimation of non-normalized statistical models by score matching.* JMLR, 2005.
- P. Vincent. *A connection between score matching and denoising autoencoders.* Neural Computation, 2011.
- Y. Song, S. Ermon. *Generative modeling by estimating gradients of the data distribution.* NeurIPS, 2019.
- A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola. *A kernel two-sample test.* JMLR, 2012.
- B. K. Sriperumbudur, A. Gretton, K. Fukumizu, B. Schölkopf, G. R. G. Lanckriet. *Hilbert space embeddings and metrics on probability measures.* JMLR, 2010.
- Y. Li, K. Swersky, R. Zemel. *Generative moment matching networks.* ICML, 2015.
- G. K. Dziugaite, D. M. Roy, Z. Ghahramani. *Training generative neural networks via maximum mean discrepancy optimization.* UAI, 2015.
- M. Arjovsky, S. Chintala, L. Bottou. *Wasserstein generative adversarial networks.* ICML, 2017.
- A. van den Oord, N. Kalchbrenner, K. Kavukcuoglu. *Pixel recurrent neural networks.* ICML, 2016.
- Z. Xiao, K. Kreis, A. Vahdat. *Tackling the generative learning trilemma with denoising diffusion GANs.* ICLR, 2022.
- S. Bond-Taylor, A. Leach, Y. Long, C. G. Willcocks. *Deep generative modelling: a comparative review of VAEs, GANs, normalizing flows, energy-based and autoregressive models.* IEEE TPAMI, 2022.
