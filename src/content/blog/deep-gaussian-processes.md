---
title: "Deep Gaussian processes: stacking uncertainty"
description: "What composing Gaussian processes buys and what it costs. If a composition of GPs is not a GP, how is anything computed? What does all the mathematics buy, what is new, and where are deep GPs still the right tool? Fundamentals first, every step derived, with a deep GP fitted and taken apart."
date: 2026-02-07
updated: 2026-10-29
cover: /blog/thumbs/dgp.png
tags: [gaussian-processes, deep-gp, variational-inference, bayesian]
---

A Gaussian process is shallow in one precise sense. Its kernel, chosen before any data arrive, fixes the *texture* of every function it believes in: how often it wiggles, how smooth it is, and, for the usual stationary kernels, that this texture is the same everywhere. Real functions are calm in some places and sharp in others. A deep GP feeds one GP into another, so that the first layer can stretch and squeeze the input before the second one sees it.

Below are draws from deep GPs with one to five layers, each layer a GP with the same squared-exponential kernel. Raise the number of layers:

<figure class="fig" data-dgp>
  <canvas data-figure="dgp" aria-label="Three random draws from a deep GP with a chosen number of layers, length-scale and architecture."></canvas>
  <div class="controls">
    <label class="slider">layers L <input type="range" min="1" max="5" step="1" value="1" data-depth /></label>
    <label class="slider">length-scale ℓ <input type="range" min="0.3" max="1.5" step="0.05" value="1" data-ell /></label>
    <div class="seg" role="group" aria-label="Architecture">
      <button type="button" data-arch="plain" class="on">plain</button>
      <button type="button" data-arch="input">input-connected</button>
      <button type="button" data-arch="identity">identity mean</button>
    </div>
    <button type="button" data-resample>Resample</button>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

*What to try.* With one layer the draws look like any GP: about 6% of the axis is nearly flat. At three layers more than half of it is (55% for the default draws), with steep cliffs in between, and at five layers the steepest slope has grown from about 2 to about 16. That is the pathology of Section 3. Switch to **input-connected**: at five layers only about 5% of the axis is flat. **Identity mean** sits in between, at about 19%. Shorten the length-scale and the collapse slows down, because the ratio $\sigma/\ell$ approaches the critical value $1.89$ that Section 3 derives.

This is the third post in the Gaussian-process series, after [Gaussian processes, from first principles](/aboutme/blog/understanding-gaussian-processes/) and [sparse variational GPs](/aboutme/blog/sparse-variational-gaussian-processes/). It is written to answer four questions that beginners ask, and that many experts never quite settle:

1. **If a composition of GPs is not a GP, how is anything computed?** (Section 2.)
2. **What does all this mathematics buy**, compared with a GP or a neural network? (Section 6.)
3. **Is it still relevant, and for which tasks is it the right tool?** (Section 7.)
4. **What is new?** (Section 8.)

Every result is derived in numbered steps, with nothing left to the reader. The Gaussian identities it uses are proved in the primer and in [the Gaussian toolkit](/aboutme/blog/gaussian-identities/), and the one formula it needs from the sparse GP post is re-derived here.

The plan:

1. what one GP cannot do;
2. composing GPs: what the model is, and how anything is computed although it is not a GP;
3. what depth does: the pathology of deep compositions, and two fixes;
4. why exact inference is impossible;
5. doubly stochastic variational inference: one layer recalled, the bound derived, a forward pass by hand, and a deep GP fitted;
6. is the mathematics worth it?
7. where deep GPs are the right tool today;
8. what is new, 2018 to 2026;
9. deep GPs, neural networks and deep kernels;
10. a checklist.

## Notation

Plain italic letters are scalars, bold italic lower case are vectors, bold upright capitals are matrices. To keep the derivations readable, every hidden layer has one output; Section 5 explains what changes with more.

| Symbol | Meaning |
|---|---|
| $L$ | number of layers; $l = 1, \dots, L$ indexes them |
| $f_l$ | layer $l$, a GP: $f_l \sim \mathcal{GP}(0, k_l)$, independent across layers |
| $h_0(\boldsymbol{x}) = \boldsymbol{x}$, $\ h_l(\boldsymbol{x}) = f_l\big(h_{l-1}(\boldsymbol{x})\big)$ | hidden values; the model output is $h_L$ |
| $\sigma_l^2$, $\ell_l$ | amplitude and length-scale of the squared-exponential kernel $k_l(a, b) = \sigma_l^2e^{-(a - b)^2/(2\ell_l^2)}$ |
| $\boldsymbol{f}_l \in \mathbb{R}^N$, $f_{l,n}$ | layer-$l$ values at the $N$ training inputs, and the $n$-th of them |
| $\mathbf{Z}_l$, $\boldsymbol{u}_l = f_l(\mathbf{Z}_l)$ | inducing inputs and inducing variables of layer $l$ ($M$ of each) |
| $q(\boldsymbol{u}_l) = \mathcal{N}(\boldsymbol{m}_l, \mathbf{S}_l)$ | variational distribution of layer $l$ |
| $\mu_l(h)$, $s_l^2(h)$ | mean and variance of layer $l$'s variational marginal at a single input $h$ |
| $\sigma_n^2$ | observation-noise variance |
| $\varepsilon$, $Z$ | standard normal variables; $S$ is a number of Monte Carlo samples |

## 1. What one GP cannot do

### One texture everywhere

A kernel is *stationary* if $k(\boldsymbol{x}, \boldsymbol{x}') = k(\boldsymbol{x} - \boldsymbol{x}')$. Then shifting a zero-mean GP does not change its distribution: $f(\cdot + \boldsymbol{a})$ has the same covariance as $f$, for every shift $\boldsymbol{a}$. The prior therefore expects the same behaviour in every region of the input space. One concrete measure: for a one-dimensional stationary GP, the expected number of times a draw crosses zero upwards on an interval of unit length is $\frac{1}{2\pi}\sqrt{-k''(0)/k(0)}$ (Rice's formula; Rasmussen and Williams, 2006, Section 4.1). For the squared-exponential kernel that is $1/(2\pi\ell)$: the same in every region.

Data rarely cooperate. Take a step: flat, a jump, flat again. A stationary GP has one length-scale to spend. A long one cannot follow the jump. A short one follows the jump, but it also expects wiggles on both plateaus, rings near the jump, and forgets everything a short distance outside the data. Type-II maximum likelihood picks a compromise, and it is not a good one. The fitted figure in Section 5 shows it: the GP takes $\ell = 0.075$ and inflates the noise estimate to more than twice its true value, to explain the wiggles it cannot avoid.

### Two classic remedies

- **Design a non-stationary kernel.** Let the length-scale depend on the input (Gibbs, 1997), or put non-stationarity in the basis, as wavelet features do ([From Fourier to wavelets](/aboutme/blog/fourier-to-wavelets/)). This works when you know what form the non-stationarity takes.
- **Warp the input.** Use a stationary kernel on transformed inputs, $k\big(w(\boldsymbol{x}), w(\boldsymbol{x}')\big)$ (Sampson and Guttorp, 1992; Snoek et al., 2014). Where $w$ is steep, a small change of $\boldsymbol{x}$ is a large change of $w(\boldsymbol{x})$, so the function changes quickly there.

:::proposition[A warped kernel is a kernel]
If $k$ is a positive semi-definite kernel and $w$ is any function, then $k_w(\boldsymbol{x}, \boldsymbol{x}') = k\big(w(\boldsymbol{x}), w(\boldsymbol{x}')\big)$ is a positive semi-definite kernel.
:::

:::proof
**Step 1. What must be shown.** For any inputs $\boldsymbol{x}_1, \dots, \boldsymbol{x}_N$ and any real coefficients $c_1, \dots, c_N$, $\sum_{i,j}c_ic_j\,k_w(\boldsymbol{x}_i, \boldsymbol{x}_j) \geq 0$.

**Step 2. Rename the points.** Let $\boldsymbol{v}_i = w(\boldsymbol{x}_i)$. By the definition of $k_w$, the sum equals $\sum_{i,j}c_ic_j\,k(\boldsymbol{v}_i, \boldsymbol{v}_j)$.

**Step 3. Use the property of $k$.** That is the quadratic form $\boldsymbol{c}^\top\mathbf{K}_v\boldsymbol{c}$ of the Gram matrix of $k$ at the points $\boldsymbol{v}_1, \dots, \boldsymbol{v}_N$. It is non-negative because $k$ is positive semi-definite. Repeated points are allowed in that definition, so it does not matter whether $w$ is one-to-one.
:::

A deep GP takes the second remedy and does not choose the warp by hand: it puts a GP prior on it and learns it, with uncertainty, from the data.

## 2. Composing Gaussian processes

### The model

:::definition[Deep Gaussian process (Damianou and Lawrence, 2013)]
Let $f_1, \dots, f_L$ be independent GPs. The deep GP with $L$ layers is the random function

$$
h_L = f_L \circ f_{L-1} \circ \dots \circ f_1, \qquad\text{that is}\qquad h_l(\boldsymbol{x}) = f_l\big(h_{l-1}(\boldsymbol{x})\big),\ \ h_0(\boldsymbol{x}) = \boldsymbol{x},
$$

and the observations are $y = h_L(\boldsymbol{x}) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$. A layer with several outputs is a vector of independent GPs, one per output.
:::

### Sampling a deep GP

To draw the function at $N$ inputs $\mathbf{X}$:

1. Draw the first layer at the inputs: $\boldsymbol{f}_1 \sim \mathcal{N}\big(\mathbf{0},\ k_1(\mathbf{X}, \mathbf{X})\big)$. This is an ordinary GP draw.
2. Draw the second layer *at the points $\boldsymbol{f}_1$*: $\boldsymbol{f}_2 \sim \mathcal{N}\big(\mathbf{0},\ k_2(\boldsymbol{f}_1, \boldsymbol{f}_1)\big)$, where $k_2(\boldsymbol{f}_1, \boldsymbol{f}_1)$ is the $N \times N$ matrix with entries $k_2(f_{1,i}, f_{1,j})$. Another ordinary GP draw, at random inputs.
3. Continue to layer $L$.

Each step is a Cholesky factorisation and a matrix-vector product; the figure at the top does exactly this. Nothing in the recipe is exotic. But the result is not a GP. The next subsection shows what that means, and why it does not stop us computing anything.

### If it is not a GP, how does anything work?

Start with the smallest possible question: the output at one input.

:::proposition[At a single input, the output is Gaussian]
In a deep GP whose last layer is zero-mean and stationary, $h_L(\boldsymbol{x}) \sim \mathcal{N}(0, k_L(0))$ at every single input $\boldsymbol{x}$, exactly as for a GP. (For the squared-exponential kernel, $k_L(0) = \sigma_L^2$.)
:::

:::proof
**Step 1. Condition on the layer below.** Fix the lower layers, and with them the value $h = h_{L-1}(\boldsymbol{x})$. Then $h_L(\boldsymbol{x}) = f_L(h)$ is the GP $f_L$ at one fixed input, so $h_L(\boldsymbol{x}) \mid h \sim \mathcal{N}(0, k_L(h, h))$.

**Step 2. Stationarity removes the dependence.** $k_L(h, h) = k_L(h - h) = k_L(0)$, whatever $h$ is. The conditional distribution is the same for every value of the lower layers.

**Step 3. Average.** The marginal is the average of the conditional over the lower layers: $p(h_L(\boldsymbol{x})) = \int\mathcal{N}\big(h_L(\boldsymbol{x}) \mid 0, k_L(0)\big)\,p(h)\,dh = \mathcal{N}\big(h_L(\boldsymbol{x}) \mid 0, k_L(0)\big)$, because the integrand does not depend on $h$ and $p(h)$ integrates to 1.
:::

So one point at a time, you cannot tell a deep GP from a GP. The difference is entirely in how *different* points relate to each other. Take two inputs, $x$ and $x'$, and two one-dimensional layers.

:::proposition[At two inputs, a mixture of Gaussians]
Write $d = f_1(x) - f_1(x')$ and $\rho(d) = e^{-d^2/(2\ell_2^2)}$. With zero-mean layers,

$$
p\big(y(x), y(x')\big) = \int\mathcal{N}\!\left(\begin{bmatrix}y(x)\\ y(x')\end{bmatrix}\,\middle|\,\mathbf{0},\ \sigma_2^2\begin{bmatrix}1 & \rho(d)\\ \rho(d) & 1\end{bmatrix}\right)p(d)\,dd, \qquad d \sim \mathcal{N}(0, s^2),
$$

with $s^2 = k_1(x, x) + k_1(x', x') - 2k_1(x, x')$. It is a continuous mixture of bivariate Gaussians whose *correlation* is random, and it is not a Gaussian.
:::

:::proof
**Step 1. Condition on layer 1.** Given $f_1$, the pair $\big(f_2(f_1(x)), f_2(f_1(x'))\big)$ is the GP $f_2$ at two fixed inputs, so it is bivariate Gaussian with variances $k_2(0) = \sigma_2^2$ and covariance $k_2(f_1(x), f_1(x')) = \sigma_2^2\rho(d)$.

**Step 2. Only $d$ matters.** That conditional depends on $f_1$ only through $d = f_1(x) - f_1(x')$.

**Step 3. The law of $d$.** $f_1(x)$ and $f_1(x')$ are jointly Gaussian with mean zero. A linear combination of jointly Gaussian variables is Gaussian, so $d$ is Gaussian with mean $0$ and variance $\operatorname{Var}f_1(x) + \operatorname{Var}f_1(x') - 2\operatorname{Cov}(f_1(x), f_1(x')) = s^2$.

**Step 4. Average.** The joint density is the conditional averaged over $d$, which is the integral in the statement.

**Step 5. Not Gaussian.** If it were Gaussian, the difference $y(x) - y(x')$ would be Gaussian too, with kurtosis 3. The proposition on heavy tails below shows its kurtosis is strictly larger.
:::

Here is that joint density, next to the Gaussian with exactly the same covariance matrix. Move the distance between the two inputs:

<figure class="fig" data-dgpjoint>
  <canvas data-figure="dgpjoint" style="height:330px" aria-label="Joint density of the outputs at two inputs: a GP with the same covariance, and a two-layer deep GP."></canvas>
  <div class="controls">
    <label class="slider">distance |x − x′| <input type="range" min="0.03" max="1.2" step="0.01" value="0.3" data-delta /></label>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

Same covariance, very different shapes. The GP's density is a single ellipse. The deep GP's is a sharp ridge along the diagonal on top of a round cloud. When the two hidden values happen to be close, $f_2$ sees almost the same input twice, so $y(x) \approx y(x')$: that is the ridge. When they are far apart, the two outputs are almost unrelated: that is the cloud. Averaging the two situations gives the same correlation as the ellipse ($0.49$ at $|x - x'| = 0.3$) but not the same shape, and the difference $y(x) - y(x')$ has kurtosis $4.5$ instead of $3$. The shape carries information that a GP throws away: the deep GP says "either the same or unrelated", where the GP can only say "moderately related".

That answers the question in the heading. Three things still work exactly as for a GP, and one does not:

- **Each layer, given the layer below, is an ordinary GP.** Conditioning, sampling and Gaussian marginals all work within a layer.
- **Joint densities are averages of Gaussians**, so they can be computed by averaging: exactly in small cases like the one above, by Monte Carlo in general.
- **Predictions are averages too**, by the next proposition.
- **What is lost** is a closed form for the posterior and the evidence, because they need an average over *all* the hidden values at once. Section 5 replaces the evidence by a lower bound that can be computed.

:::proposition[Prediction by Monte Carlo]
Let $p(h_*)$ be the distribution of the hidden value at a test input (under the approximate posterior), and $p(y_* \mid h_*)$ the Gaussian of the last layer given it. Then

$$
p(y_*) = \mathbb{E}_{h_*}\big[p(y_* \mid h_*)\big] \approx \hat p_S(y_*) = \frac1S\sum_{s=1}^{S}p\big(y_* \mid \hat h^{(s)}\big), \qquad \hat h^{(s)} \sim p(h_*) \text{ independently},
$$

and $\hat p_S(y_*)$ is unbiased with variance $\operatorname{Var}_{h_*}\big[p(y_* \mid h_*)\big]/S$.
:::

:::proof
**Step 1. Sum rule.** $p(y_*) = \int p(y_* \mid h_*)\,p(h_*)\,dh_*$, which is the expectation of the function $h_* \mapsto p(y_* \mid h_*)$ under $p(h_*)$.

**Step 2. Unbiased.** Each term $p(y_* \mid \hat h^{(s)})$ has expectation $p(y_*)$ by Step 1, and the expectation of an average is the average of the expectations.

**Step 3. Variance.** The terms are independent and identically distributed, so the variance of their average is the variance of one term divided by $S$. The error therefore shrinks like $1/\sqrt{S}$.
:::

Each term is a Gaussian in $y_*$, so the prediction is a *mixture* of Gaussians. The animation does this for the deep GP fitted in Section 5, at $x^* = 1.2$, just beyond the data, where the hidden value is uncertain. Most samples of the hidden value land where the second layer is at $+1$; a few land where it falls away. The average is sharp at $+1$ with a long tail downwards. The Gaussian with the same mean and variance, in grey, spreads its mass where the model does not believe the answer lies.

<figure class="fig">
<video src="/aboutme/blog/deep-gp/predict.mp4" autoplay muted loop playsinline controls poster="/aboutme/blog/deep-gp/predict.jpg"></video>
<figcaption>Predicting at x* = 1.2 with the fitted two-layer deep GP: the Gaussian of the hidden value (top), samples pushed through layer 2 (middle), one Gaussian per sample and their average (right).</figcaption>
</figure>

### The covariance of a two-layer deep GP

For two one-dimensional layers with squared-exponential kernels, the covariance has a closed form. It shows what composition does to correlations, and it uses one tool worth proving first.

:::lemma[Law of total covariance]
For random variables $A$, $B$ and $C$ with finite second moments,

$$
\operatorname{Cov}(A, B) = \mathbb{E}\big[\operatorname{Cov}(A, B \mid C)\big] + \operatorname{Cov}\big(\mathbb{E}[A \mid C],\ \mathbb{E}[B \mid C]\big).
$$
:::

:::proof
**Step 1. Definitions.** $\operatorname{Cov}(A, B) = \mathbb{E}[AB] - \mathbb{E}[A]\,\mathbb{E}[B]$, and likewise conditionally: $\operatorname{Cov}(A, B \mid C) = \mathbb{E}[AB \mid C] - \mathbb{E}[A \mid C]\,\mathbb{E}[B \mid C]$.

**Step 2. Tower property.** $\mathbb{E}[AB] = \mathbb{E}\big[\mathbb{E}[AB \mid C]\big]$, $\mathbb{E}[A] = \mathbb{E}\big[\mathbb{E}[A \mid C]\big]$ and $\mathbb{E}[B] = \mathbb{E}\big[\mathbb{E}[B \mid C]\big]$.

**Step 3. Substitute the conditional identity of Step 1.**

$$
\mathbb{E}[AB] = \mathbb{E}\big[\operatorname{Cov}(A, B \mid C)\big] + \mathbb{E}\big[\mathbb{E}[A \mid C]\,\mathbb{E}[B \mid C]\big].
$$

**Step 4. Subtract $\mathbb{E}[A]\,\mathbb{E}[B] = \mathbb{E}\big[\mathbb{E}[A \mid C]\big]\,\mathbb{E}\big[\mathbb{E}[B \mid C]\big]$.** The last term of Step 3 minus this product is $\operatorname{Cov}\big(\mathbb{E}[A \mid C], \mathbb{E}[B \mid C]\big)$, by the definition of covariance.
:::

:::proposition[Two-layer covariance]
Let $f_1 \sim \mathcal{GP}(0, k_1)$ and $f_2 \sim \mathcal{GP}(0, k_2)$ be independent with squared-exponential kernels, and $y(x) = f_2(f_1(x))$. Then

$$
\operatorname{Cov}\big(y(x), y(x')\big) = \frac{\sigma_2^2}{\sqrt{1 + s^2/\ell_2^2}}, \qquad s^2 = 2\sigma_1^2\Big(1 - e^{-(x - x')^2/(2\ell_1^2)}\Big).
$$
:::

:::proof
**Step 1. Law of total covariance, conditioning on $f_1$.**

$$
\operatorname{Cov}\big(y(x), y(x')\big) = \mathbb{E}\big[\operatorname{Cov}(y(x), y(x') \mid f_1)\big] + \operatorname{Cov}\big(\mathbb{E}[y(x) \mid f_1],\ \mathbb{E}[y(x') \mid f_1]\big).
$$

**Step 2. The second term vanishes.** Given $f_1$, $y(x) = f_2(f_1(x))$ is the zero-mean GP $f_2$ at a fixed point, so $\mathbb{E}[y(x) \mid f_1] = 0$, and the covariance of two constants is zero.

**Step 3. The first term is an expected kernel.** Given $f_1$, the covariance is $k_2(f_1(x), f_1(x')) = \sigma_2^2e^{-d^2/(2\ell_2^2)}$ with $d = f_1(x) - f_1(x')$.

**Step 4. The law of $d$.** As in the previous proposition, $d \sim \mathcal{N}(0, s^2)$ with $s^2 = 2k_1(0) - 2k_1(x, x') = 2\sigma_1^2\big(1 - e^{-(x - x')^2/(2\ell_1^2)}\big)$.

**Step 5. Write the expectation as an integral and collect the exponents.**

$$
\mathbb{E}\big[e^{-d^2/(2\ell_2^2)}\big] = \int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi s^2}}\,e^{-d^2/(2s^2)}\,e^{-d^2/(2\ell_2^2)}\,dd = \frac{1}{\sqrt{2\pi s^2}}\int_{-\infty}^{\infty}e^{-\frac{a}{2}d^2}\,dd, \qquad a = \frac{1}{s^2} + \frac{1}{\ell_2^2}.
$$

**Step 6. The Gaussian integral.** $\int e^{-a d^2/2}\,dd = \sqrt{2\pi/a}$. So the expectation is $\sqrt{2\pi/a}\,/\sqrt{2\pi s^2} = 1/\sqrt{a s^2} = 1/\sqrt{1 + s^2/\ell_2^2}$.

**Step 7. Multiply by $\sigma_2^2$.**
:::

A Monte Carlo check with $\sigma_1 = \sigma_2 = 1$, $\ell_1 = 0.3$, $\ell_2 = 0.5$ and $|x - x'| = 0.2$ gives $0.6212$ against the formula's $0.6209$. Three readings:

- **Still stationary.** The covariance depends only on $x - x'$, because the first layer is stationary. The non-stationarity of a deep GP is not in its covariance. It is in its individual draws, and in its posterior once the data have chosen a warp.
- **Long-range memory.** As $|x - x'| \to \infty$, $s^2 \to 2\sigma_1^2$ and the covariance tends to $\sigma_2^2/\sqrt{1 + 2\sigma_1^2/\ell_2^2} > 0$. Far-apart inputs stay correlated: their hidden values land in the same range, and $f_2$ is evaluated there twice.
- **The covariance is not the whole story.** A GP with this kernel has the same covariance and Gaussian increments. The deep GP's increments are not Gaussian.

:::proposition[Heavy-tailed increments]
The increment $\Delta = y(x) - y(x')$ of the two-layer model has kurtosis

$$
\frac{\mathbb{E}[\Delta^4]}{\mathbb{E}[\Delta^2]^2} = 3\,\frac{1 - 2\bar\rho_1 + \bar\rho_2}{(1 - \bar\rho_1)^2} > 3, \qquad \bar\rho_1 = \Big(1 + \frac{s^2}{\ell_2^2}\Big)^{-1/2},\quad \bar\rho_2 = \Big(1 + \frac{2s^2}{\ell_2^2}\Big)^{-1/2},
$$

whenever $x \neq x'$. A Gaussian has kurtosis 3.
:::

:::proof
**Step 1. Conditionally Gaussian.** Given $f_1$, $\Delta$ is a difference of two jointly Gaussian values of $f_2$, so $\Delta \mid f_1 \sim \mathcal{N}(0, v)$ with $v = \operatorname{Var}(\Delta \mid f_1) = \sigma_2^2 + \sigma_2^2 - 2\sigma_2^2\rho(d) = 2\sigma_2^2\big(1 - \rho(d)\big)$.

**Step 2. Moments of a Gaussian.** For $\mathcal{N}(0, v)$ the second moment is $v$ and the fourth is $3v^2$. Averaging over $f_1$ (tower property): $\mathbb{E}[\Delta^2] = \mathbb{E}[v]$ and $\mathbb{E}[\Delta^4] = 3\,\mathbb{E}[v^2]$.

**Step 3. Compute $\mathbb{E}[v]$.** $\mathbb{E}[\rho(d)] = \bar\rho_1$ by Steps 5 and 6 of the covariance proof, so $\mathbb{E}[v] = 2\sigma_2^2(1 - \bar\rho_1)$.

**Step 4. Compute $\mathbb{E}[v^2]$.** $v^2 = 4\sigma_2^4(1 - 2\rho + \rho^2)$, and $\rho(d)^2 = e^{-d^2/\ell_2^2} = e^{-d^2/(2(\ell_2/\sqrt2)^2)}$. So $\mathbb{E}[\rho^2]$ is the same integral with $\ell_2^2$ replaced by $\ell_2^2/2$, which is $\bar\rho_2$. Hence $\mathbb{E}[v^2] = 4\sigma_2^4(1 - 2\bar\rho_1 + \bar\rho_2)$.

**Step 5. Divide.** $3\,\mathbb{E}[v^2]/\mathbb{E}[v]^2$ is the stated ratio; the factors $4\sigma_2^4$ cancel.

**Step 6. Why it exceeds 3.** $\mathbb{E}[v^2] - \mathbb{E}[v]^2 = \operatorname{Var}(v) \geq 0$, with equality only if $v$ is constant. For $x \neq x'$, $d$ is a non-degenerate Gaussian, so $v$ is not constant and the inequality is strict.
:::

With the numbers above the kurtosis is $5.15$ (Monte Carlo: $5.14$). A random variance makes a *scale mixture* of Gaussians: most increments are small, where the warp is flat, and a few are large, where it is steep. That is the prior of a function that is calm in most places and sharp in a few, which is what the step needed.

The animation shows the mechanism with the warp the deep GP of Section 5 actually learned. Evenly spaced inputs pass through the first layer; they bunch where it is flat and spread where it is steep. A stationary second layer then sees them, and the composition changes fast only where the warp is steep.

<figure class="fig">
<video src="/aboutme/blog/deep-gp/warp.mp4" autoplay muted loop playsinline controls poster="/aboutme/blog/deep-gp/warp.jpg"></video>
<figcaption>Layer 1 (left) is the warp learned on the step data; layer 2 (top right) is one draw from a stationary GP; their composition (bottom right) is calm where the warp is flat and busy where it is steep.</figcaption>
</figure>

## 3. What depth does

If two layers help, do ten help more? The figure at the top says no. With plain layers the draws become flat almost everywhere, with a few cliffs. Duvenaud, Rippel, Adams and Ghahramani (2014) explained why. The one-dimensional argument needs one fact about GPs, proved first.

:::lemma[The derivative of a stationary GP at a point]
If $f \sim \mathcal{GP}(0, k)$ with $k$ stationary and twice differentiable at 0, then at every input $t$ the derivative $f'(t)$ is Gaussian with mean 0 and variance $-k''(0)$, the same for every $t$. For the squared-exponential kernel, $-k''(0) = \sigma^2/\ell^2$.
:::

:::proof
**Step 1. A difference quotient is Gaussian.** For $\delta > 0$, $Q_\delta = \big(f(t + \delta) - f(t)\big)/\delta$ is a linear combination of two jointly Gaussian values, so it is Gaussian with mean 0.

**Step 2. Its variance.** Using $\operatorname{Var}f(t) = k(0)$ and $\operatorname{Cov}(f(t + \delta), f(t)) = k(\delta)$,

$$
\operatorname{Var}Q_\delta = \frac{k(0) + k(0) - 2k(\delta)}{\delta^2} = \frac{2\big(k(0) - k(\delta)\big)}{\delta^2}.
$$

**Step 3. Taylor expansion.** A stationary kernel is even, $k(\delta) = k(-\delta)$, so $k'(0) = 0$ and $k(\delta) = k(0) + \tfrac12k''(0)\delta^2 + o(\delta^2)$.

**Step 4. Take the limit.** Substituting, $\operatorname{Var}Q_\delta = -k''(0) + o(1) \to -k''(0)$ as $\delta \to 0$. Mean-square limits of Gaussian variables are Gaussian with the limiting mean and variance, so $f'(t) \sim \mathcal{N}(0, -k''(0))$. Nothing depends on $t$.

**Step 5. The squared-exponential case.** $k(\delta) = \sigma^2e^{-\delta^2/(2\ell^2)}$ has $k'(\delta) = -\sigma^2(\delta/\ell^2)e^{-\delta^2/(2\ell^2)}$ and $k''(0) = -\sigma^2/\ell^2$.
:::

:::proposition[The derivative of a deep composition]
Let every layer be one-dimensional, zero-mean and stationary, with $r_l^2 = -k_l''(0)$. At any fixed input $x$, the derivative of the composition is a product of independent Gaussians:

$$
D_L = \frac{d}{dx}\,h_L(x) = \prod_{l=1}^{L} g_l, \qquad g_l \sim \mathcal{N}(0, r_l^2)\ \text{independent}.
$$
:::

:::proof
**Step 1. Chain rule.** $h_l(x) = f_l(h_{l-1}(x))$, so $h_l'(x) = f_l'(h_{l-1}(x))\,h_{l-1}'(x)$. Unrolling from $h_0'(x) = 1$ gives $D_L = \prod_{l=1}^{L}f_l'\big(h_{l-1}(x)\big)$. Call the factors $g_l$.

**Step 2. Each factor, given the layers below.** $h_{l-1}(x)$ is determined by layers $1, \dots, l-1$, which are independent of $f_l$. Given them, $g_l$ is $f_l'$ evaluated at a fixed point, which is $\mathcal{N}(0, r_l^2)$ by the lemma.

**Step 3. The conditional law does not depend on the condition.** It is $\mathcal{N}(0, r_l^2)$ for every value of the lower layers, so $g_l$ is independent of those layers.

**Step 4. Hence of the earlier factors.** $g_1, \dots, g_{l-1}$ are functions of the lower layers, so $g_l$ is independent of them. By induction all factors are independent.
:::

:::corollary[The log-derivative is a random walk]
With $g_l = r_lZ_l$, $Z_l \sim \mathcal{N}(0, 1)$,

$$
\log\lvert D_L\rvert = \sum_{l=1}^{L}\big(\log r_l + \log\lvert Z_l\rvert\big), \qquad \mathbb{E}\log\lvert Z\rvert = -\tfrac12(\gamma + \log 2) \approx -0.635, \qquad \operatorname{Var}\log\lvert Z\rvert = \tfrac{\pi^2}{8} \approx 1.234,
$$

where $\gamma \approx 0.577$ is the Euler–Mascheroni constant. For equal layers the mean of $\log\lvert D_L\rvert$ is $L(\log r - 0.635)$ and its standard deviation is $1.11\sqrt{L}$.
:::

:::proof
**Step 1. Logs turn the product into a sum** of independent terms, because the $g_l$ are independent; $\log\lvert r_lZ_l\rvert = \log r_l + \log\lvert Z_l\rvert$.

**Step 2. Reduce to a Gamma variable.** $X = Z^2$ is chi-squared with one degree of freedom, which is the Gamma distribution with shape $k = \tfrac12$ and scale $\theta = 2$, with density $x^{k-1}e^{-x/\theta}/(\Gamma(k)\theta^k)$. And $\log\lvert Z\rvert = \tfrac12\log X$.

**Step 3. The mean of $\log X$.** Start from the normalisation $\int_0^\infty x^{k-1}e^{-x/\theta}\,dx = \Gamma(k)\theta^k$ and differentiate both sides with respect to $k$. On the left, $\partial_kx^{k-1} = x^{k-1}\log x$. On the right, $\Gamma'(k)\theta^k + \Gamma(k)\theta^k\log\theta$. Dividing by $\Gamma(k)\theta^k$ gives $\mathbb{E}\log X = \psi(k) + \log\theta$, where $\psi = \Gamma'/\Gamma$ is the digamma function.

**Step 4. The variance of $\log X$.** Differentiate once more. The left side becomes $\int x^{k-1}(\log x)^2e^{-x/\theta}\,dx$, and after dividing, $\mathbb{E}(\log X)^2 = \psi'(k) + \big(\psi(k) + \log\theta\big)^2$. Subtracting the square of the mean gives $\operatorname{Var}\log X = \psi'(k)$.

**Step 5. Plug in.** The standard values $\psi(\tfrac12) = -\gamma - 2\log 2$ and $\psi'(\tfrac12) = \pi^2/2$ give $\mathbb{E}\log X = -\gamma - \log 2$ and $\operatorname{Var}\log X = \pi^2/2$. Since $\log\lvert Z\rvert = \tfrac12\log X$, halve the mean and divide the variance by 4.
:::

So with $r = 1$ the typical slope shrinks by a factor $e^{-0.635} \approx 0.53$ per layer, while the spread of the log-slope grows like $\sqrt{L}$. A simulation of $D_L$ for $r = 1$:

| layers $L$ | median $\lvert D_L\rvert$ | share of inputs with $\lvert D_L\rvert < 0.1$ | 99.9% quantile of $\lvert D_L\rvert$ |
|---|---|---|---|
| 1 | 0.67 | 8% | 3.3 |
| 2 | 0.37 | 22% | 5.8 |
| 4 | 0.10 | 49% | 9.9 |
| 8 | 0.008 | 81% | 11.1 |

Flat almost everywhere, with rare steep places: the draws look piecewise constant. Unless $r$ is tuned to about $e^{0.635} \approx 1.89$, the median slope drifts to zero or to infinity, and even then the spread grows. With multi-dimensional hidden layers the same happens to the Jacobian, a product of random matrices: its singular values spread apart exponentially, so a deep representation keeps effectively one direction of variation (Duvenaud et al., 2014).

:::remark[Two fixes, both familiar from neural networks]
- **Input connections** (Duvenaud et al., 2014). Feed the original input to every layer: $h_l = f_l(h_{l-1}, \boldsymbol{x})$. The derivative gains a term $\partial f_l/\partial\boldsymbol{x}$ that is not multiplied through the earlier layers, so it cannot collapse.
- **Identity mean functions** (Salimbeni and Deisenroth, 2017). Make each hidden layer a correction to the identity: $h_l = h_{l-1} + f_l(h_{l-1})$. The chain rule now gives factors $1 + g_l$ instead of $g_l$. For small $r$, a second-order Taylor expansion gives $\log\lvert 1 + g_l\rvert \approx g_l - g_l^2/2$, whose mean is about $-r^2/2$: the drift per layer is $O(r^2)$ instead of $-0.635$. This is a residual connection, and it is the default in practice. For a hidden layer that changes dimension, a fixed linear map (for example a PCA projection of the inputs) plays the role of the identity.

Switch the figure to **input-connected** or **identity mean** and raise $L$: the draws stay alive.
:::

## 4. Why exact inference is impossible

For a GP, the evidence and the posterior are Gaussian integrals. For a deep GP, already with two layers,

$$
p(\boldsymbol{y}) = \int\mathcal{N}\big(\boldsymbol{y} \mid \mathbf{0},\ k_2(\boldsymbol{f}_1, \boldsymbol{f}_1) + \sigma_n^2\mathbf{I}\big)\,\mathcal{N}\big(\boldsymbol{f}_1 \mid \mathbf{0},\ k_1(\mathbf{X}, \mathbf{X})\big)\,d\boldsymbol{f}_1 .
$$

Look at where $\boldsymbol{f}_1$ appears in the first factor: inside $e^{-(f_{1,i} - f_{1,j})^2/(2\ell_2^2)}$, inside a matrix, which is then inverted and whose determinant is taken. As a function of $\boldsymbol{f}_1$ that is nothing like the exponential of a quadratic, so the integral over $\mathbb{R}^N$ has no closed form. The posterior over the hidden layer is not Gaussian, and typically multimodal. For example, with even kernels the pairs $(f_1, f_2)$ and $(-f_1,\ f_2(-\cdot))$ produce the same output and have the same prior probability, so every posterior mode has a mirror image.

So inference is approximate, by one of two routes:

- **Variational inference**, fast and scalable: Damianou and Lawrence (2013), and the doubly stochastic method of Salimbeni and Deisenroth (2017) that is now standard.
- **Monte Carlo**, slower but able to represent several modes: stochastic-gradient Hamiltonian Monte Carlo (Havasi et al., 2018), and elliptical slice sampling for full Bayesian surrogates (Sauer, Gramacy and Higdon, 2023).

## 5. Doubly stochastic variational inference

### One layer, recalled

Each layer is a sparse variational GP. Everything below needs one formula from the [sparse GP post](/aboutme/blog/sparse-variational-gaussian-processes/): the variational distribution of a layer's output at a single input. Here it is again, derived.

:::proposition[One layer's marginal at one input]
Let a layer have inducing inputs $\mathbf{Z}$, inducing variables $\boldsymbol{u} = f(\mathbf{Z})$ with $q(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$, and let $\boldsymbol{k}_u(h) = k(\mathbf{Z}, h)$ and $\boldsymbol{a}(h) = \mathbf{K}_{uu}^{-1}\boldsymbol{k}_u(h)$. Then $q(f(h)) = \int p(f(h) \mid \boldsymbol{u})\,q(\boldsymbol{u})\,d\boldsymbol{u} = \mathcal{N}\big(\mu(h), s^2(h)\big)$ with

$$
\mu(h) = \boldsymbol{a}(h)^\top\boldsymbol{m}, \qquad s^2(h) = \underbrace{k(h, h) - \boldsymbol{k}_u(h)^\top\boldsymbol{a}(h)}_{c(h)} + \boldsymbol{a}(h)^\top\mathbf{S}\,\boldsymbol{a}(h).
$$
:::

:::proof
**Step 1. The joint of $f(h)$ and $\boldsymbol{u}$.** Both are values of the same GP, so they are jointly Gaussian with mean zero, $\operatorname{Var}f(h) = k(h, h)$, $\operatorname{Cov}(f(h), \boldsymbol{u}) = \boldsymbol{k}_u(h)^\top$ and $\operatorname{Cov}(\boldsymbol{u}) = \mathbf{K}_{uu}$.

**Step 2. Condition on $\boldsymbol{u}$** (Gaussian conditioning, the primer's Lemma 5): $p(f(h) \mid \boldsymbol{u}) = \mathcal{N}\big(\boldsymbol{a}(h)^\top\boldsymbol{u},\ c(h)\big)$ with $c(h) = k(h, h) - \boldsymbol{k}_u(h)^\top\mathbf{K}_{uu}^{-1}\boldsymbol{k}_u(h)$.

**Step 3. Write it as a linear-Gaussian model.** $f(h) = \boldsymbol{a}(h)^\top\boldsymbol{u} + e$ with $e \sim \mathcal{N}(0, c(h))$ independent of $\boldsymbol{u}$.

**Step 4. Integrate out $\boldsymbol{u} \sim \mathcal{N}(\boldsymbol{m}, \mathbf{S})$.** A linear function of a Gaussian plus independent Gaussian noise is Gaussian. Its mean is $\boldsymbol{a}(h)^\top\boldsymbol{m}$, and its variance is $\operatorname{Var}(\boldsymbol{a}^\top\boldsymbol{u}) + \operatorname{Var}(e) = \boldsymbol{a}(h)^\top\mathbf{S}\boldsymbol{a}(h) + c(h)$, because variances of independent terms add.
:::

Read the variance as two parts. $c(h)$ is what the inducing points cannot pin down at $h$: zero at an inducing input, the full prior variance far from all of them. $\boldsymbol{a}(h)^\top\mathbf{S}\boldsymbol{a}(h)$ is our remaining uncertainty about the inducing values themselves. For a layer with an identity mean, add $h$ to $\mu(h)$; the variance is unchanged. Each evaluation costs $O(M^2)$ once $\mathbf{K}_{uu}$ is factorised.

### The model, augmented

Give every layer $M$ inducing inputs $\mathbf{Z}_l$ and inducing variables $\boldsymbol{u}_l = f_l(\mathbf{Z}_l)$. With $\boldsymbol{f}_0 = \mathbf{X}$, the augmented joint distribution is

$$
p\big(\boldsymbol{y}, \{\boldsymbol{f}_l, \boldsymbol{u}_l\}_{l=1}^{L}\big) = \prod_{n=1}^{N}p(y_n \mid f_{L,n})\ \prod_{l=1}^{L}p(\boldsymbol{f}_l \mid \boldsymbol{u}_l;\ \boldsymbol{f}_{l-1})\,p(\boldsymbol{u}_l),
$$

where $p(\boldsymbol{f}_l \mid \boldsymbol{u}_l; \boldsymbol{f}_{l-1})$ is the GP conditional of layer $l$ at the inputs $\boldsymbol{f}_{l-1}$. Integrating out every $\boldsymbol{u}_l$ gives back the deep GP, so nothing has been approximated yet.

### The variational family

Salimbeni and Deisenroth (2017) keep the layers' dependence on each other and approximate only the inducing variables:

$$
q\big(\{\boldsymbol{f}_l, \boldsymbol{u}_l\}\big) = \prod_{l=1}^{L}p(\boldsymbol{f}_l \mid \boldsymbol{u}_l;\ \boldsymbol{f}_{l-1})\,q(\boldsymbol{u}_l), \qquad q(\boldsymbol{u}_l) = \mathcal{N}(\boldsymbol{m}_l, \mathbf{S}_l).
$$

Each layer is a sparse variational GP whose inputs are the previous layer's outputs. The earlier method of Damianou and Lawrence (2013) instead gave the hidden values of each layer their own independent Gaussian, which breaks exactly the dependence between layers that makes the model deep.

<figure class="sketch-fig"><svg class="sketch" viewBox="0 0 960 520" role="img" aria-label="A two-layer deep GP: input x goes through GP layer 1 with its inducing points to a hidden value h, then through GP layer 2 to f and the noisy output y. Below, one training point's path under doubly stochastic variational inference: closed-form mean and variance at layer 1, a reparameterised sample of the hidden value, closed-form mean and variance at layer 2, and the expected log-likelihood in closed form."><text x="480" y="34" text-anchor="middle" style="font-family:Kalam,cursive;font-size:23px;font-weight:700;fill:var(--a1)">a two-layer deep GP</text><path d="M48.772895308283736 102.63005969025096 C53.23874752873494 99.89708491126233, 59.637000823818106 99.3999938010554, 64.55081988311261 100.87707606376037 C69.4646389424071 102.35415832646534, 74.99074485392951 106.30559729418451, 78.25580966405076 111.4925532664808 C81.520874474172 116.67950923877709, 84.39564560888599 125.67562585541906, 84.14120874384005 131.9988118975381 C83.88677187879412 138.32199793965717, 80.59956700849932 145.18461650007728, 76.7291884737752 149.4316695191951 C72.85880993905107 153.67872253831294, 66.3889886218361 157.29828934058162, 60.91893753549533 157.48113001224507 C55.44888644915456 157.66397068390853, 48.4433485322082 154.11027742914814, 43.908881955730564 150.52871354917585 C39.37441537925293 146.94714966920355, 34.98198375685864 141.71915081640012, 33.71213807662953 135.99174673241134 C32.44229239640042 130.26434264842257, 32.49735467062227 122.13985955837921, 36.289807874355915 116.16428904524315 C40.08226107808956 110.18871853210709, 51.79217105000477 102.50222134054863, 56.466857299031396 100.13832365359492 C61.141543548058024 97.77442596664122, 64.42422550887744 101.38947236130342, 64.33792536851568 101.98090292352093 M51.71244128817846 99.3207081081772 C56.71274629963572 97.3591170091384, 63.40432164619067 98.30599563332036, 68.57028236898334 101.46599969146456 C73.73624309177602 104.62600374960876, 80.46817778145765 112.3129394928788, 82.70820562493454 118.28073245704239 C84.94823346841142 124.24852542120598, 83.73787603304896 131.67929345635724, 82.01044942984467 137.2727574764461 C80.28302282664038 142.86622149653493, 76.70957290961711 148.43508921653807, 72.34364600570876 151.84151657757545 C67.97771910180042 155.24794393861282, 61.07957602002878 158.57322626044905, 55.814888006394625 157.7113216426703 C50.55019999276047 156.84941702489155, 44.37080020343453 151.54822184387479, 40.75551792390381 146.6700888709029 C37.14023564437309 141.79195589793102, 34.305561885706425 134.2408034818007, 34.12319432921032 128.44252380483906 C33.94082677271422 122.64424412787741, 36.58390371677377 116.55389818386679, 39.661312584927174 111.88041080913304 C42.73872145308058 107.20692343439929, 50.4660242818301 102.14717517926942, 52.58764753813077 100.40159955643657 C54.70927079443143 98.6560239336037, 52.46941473509631 101.19640665770721, 52.39105212273116 101.40695707213585" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="58" y="136" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">x</text><path d="M89.71587750986218 126.32327855750918 C96.58122758418321 128.83283207048476, 106.2103235386312 129.0407937489897, 127.12455053403974 128.3788286320865 M89.03761456571519 129.09270163811743 C98.39560598023236 128.74542768594623, 107.93004354648292 128.42574917671084, 127.53905941806734 128.33716952838003" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M129.02952650591732 126.99396713450551 C126.24593324369383 129.6491857188843, 125.06059659461769 130.8181381240186, 116.66936509219173 133.44688358858667 M128.6225687394291 128.65562098287046 C126.02738243532865 130.0172405915934, 123.57328063970651 131.05136197565798, 116.91807042260828 133.4218881263628" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M129.02952650591732 126.99396713450551 C126.1259514138362 127.56083492282168, 124.94061476476007 126.64143653189333, 116.66936509219173 123.00771076991714 M128.6225687394291 128.65562098287046 C126.09781607020525 127.56538361238184, 123.64371427458312 126.1476480172348, 116.91807042260828 122.98271530769325" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M132.17553215601978 77.79807335326348 C132.17553215601978 77.79807335326348, 132.17553215601978 77.79807335326348, 132.17553215601978 77.79807335326348 M132.17553215601978 77.79807335326348 C132.17553215601978 77.79807335326348, 132.17553215601978 77.79807335326348, 132.17553215601978 77.79807335326348 M132.1363508093779 88.89546819618575 C135.74941554450132 88.02259851862715, 137.44669275170074 80.999928477273, 141.46645298425506 77.29598399602592 M132.91213763926672 89.22153375944725 C135.5337472754928 86.85184137240724, 138.70681106767776 82.15678092502102, 143.3270993474538 76.87658196323336 M129.73640608038315 101.28373146669259 C138.03022928854188 93.33939049137787, 144.68757663240098 88.27474397916001, 152.5808643777605 78.98697079591123 M131.15102142650196 103.78110396766354 C136.76309525424267 94.75994535970081, 143.4547706688597 87.79227823499816, 154.27174571461342 77.9199820226398 M134.14875437548875 116.32846310196467 C140.35978432767294 107.41953535658193, 146.72866877050674 99.01000454356628, 162.99544201830602 77.56286846524593 M131.45466685934483 114.0299713188971 C145.27301084154013 100.40590978075005, 156.06154809771132 86.52886533380708, 165.12592485996163 78.1001180232924 M133.35635947601426 125.21444916228559 C145.785280522494 106.80094706165085, 163.76550719487292 90.95864220136426, 172.75046689395893 77.26725200494194 M131.67546050080585 125.74001687752809 C147.27761644277177 109.63678109040107, 163.3048963326213 89.39900149522595, 175.26235231158424 78.08824619618619 M133.4155446241472 138.32091328046783 C150.9733803498078 115.78881594286472, 173.4046098993013 91.28113692798924, 185.27023099690678 79.29887629781179 M132.5651235422913 139.6448710146503 C151.34460436256603 116.18276160711183, 168.36272852381168 96.63234862735501, 185.68001488588752 76.77698128741494 M130.83185181207506 151.46536033932392 C149.1670211420813 131.551021286136, 167.4051041669903 113.0272200918106, 196.24718604034655 79.37448705689073 M133.3320178002074 150.5777734763575 C149.20823870215767 131.72996179736754, 165.26763081408487 112.44205884463065, 195.74098099528723 78.75075699829519 M130.5789827529948 162.78328605188906 C148.54454742137645 144.0097073560357, 163.6260170767387 123.720778221882, 207.74117957444673 78.43498250454573 M130.6475866869699 164.76378296396612 C147.1753673873317 143.70817119878683, 163.2036960090848 125.08408824493107, 206.65713676417056 77.80448766447512 M131.398034997075 173.8343386405303 C161.49147722471028 140.88302101790552, 188.48838179215696 112.11786093154593, 218.91810081240433 79.29867085381204 M131.7330913269661 176.13843881312448 C160.23075763569497 142.67356113211287, 187.95707378091174 110.97505961994152, 217.21633185495335 78.00932573727125 M135.52856096044619 180.97260925672344 C161.8886112983958 155.39961011351954, 181.81743707835307 126.52710763392332, 225.74911674640688 78.16743198750221 M135.47816958554523 182.13055141358964 C162.1193455495595 153.40209428600977, 183.7710290248292 127.20161458639726, 227.2691155840418 77.23132346277737 M145.33509569327683 180.70930862283547 C171.87466868075535 157.37382214777378, 192.02340759376682 131.53386373634933, 239.0296131186418 76.70642703842867 M147.2548638370946 183.4373788510098 C179.2619521506646 147.50616364500533, 207.27239934389337 112.75094978796074, 236.73517789284213 77.22364246208717 M158.1897436768399 181.5207340936838 C187.26332577742537 150.68731229475188, 213.23589857441002 120.74785935455147, 247.85419442129668 76.569098389404 M157.1877077621059 181.84509396848367 C185.56439982673217 151.80372492338742, 214.13988839172708 120.37952531525802, 248.86584934776064 77.36405373129445 M167.06835940332292 183.27907805208102 C193.14347412056742 154.66489584449025, 218.6295175242459 124.95808903713814, 257.491683896559 76.32216967324062 M168.53740692698537 182.872092762105 C190.50540089541946 156.16430929849437, 209.76780572397675 131.64981626507713, 257.9066011402036 79.19243764149292 M178.12101611491332 180.96981413083952 C202.80763959044708 156.74545878558908, 227.21196297423586 125.42864943537664, 272.06012353661623 80.3706457727511 M179.45880752313923 183.09888060296754 C203.95951910790907 153.04777345203834, 230.75846947928784 123.32588485648937, 269.1595450865772 77.74828393877638 M188.5470058037987 183.2166972127719 C226.57504112367081 139.49735134683723, 259.1935248841685 97.47333381720627, 279.60177408365587 76.6828128207479 M188.66339636697376 183.1055722371651 C222.5067724699543 144.69539907727489, 255.29048720637806 109.28773610453912, 279.4459336282877 78.10819174423855 M200.46983921216648 180.40676343069055 C231.6754901912225 147.2738352872689, 260.05658201408176 115.31525695090563, 292.3581150298655 79.31523820004354 M199.88180089989524 182.4093917857913 C229.68243569060846 150.91311378544714, 256.6953093191397 118.96283385783717, 289.96865983174143 77.12336955397318 M213.21224583220416 182.4153301674081 C242.57614830899288 148.2382780299123, 270.97940552935137 108.86135931219033, 303.04484461743243 77.3377705434505 M210.72628918830986 182.80217274494 C230.80656237435446 160.96279162492868, 250.26711689521107 135.850704241778, 301.17475738834446 77.55713523907401 M222.11511487710874 182.68570563595966 C250.26054687673033 142.39779193239713, 286.61356923677937 105.7604386869214, 310.72687983854433 77.35853973378126 M221.29721089463334 184.04256756592767 C252.85780221379724 143.4703721170058, 287.80177741037744 105.82678450684621, 311.2179006145032 77.02241604563478 M232.6367313970488 183.33535055871232 C262.7195081098508 144.18544883950375, 296.41612110727647 109.8991962603088, 325.0933564353886 76.44895690384838 M230.46973339325305 181.7747093344025 C259.95544358404226 150.20447786632684, 287.39588251547235 117.70762032819187, 322.82090445162385 77.3192485947493 M241.49142210522123 182.2091356487013 C273.0078461931769 143.13582279326053, 307.29271733528 105.02714423734763, 334.7326289001529 75.8687144364643 M241.21998759658464 183.79396925168106 C266.34198747314093 156.05722434095597, 290.7854229120992 127.87560899861, 334.53822446611275 78.17227888251125 M250.7487914938136 183.5855032887391 C280.92366519527485 151.41568197660703, 306.5996551505505 118.28743370397592, 345.0072704750246 77.99078596444157 M252.11180407103438 183.9656183138571 C284.3705779579149 146.7626522414169, 314.9762290124888 111.13241648015817, 345.21845684937597 76.96826228001443 M261.8526628050134 181.45617711163823 C284.0378033307213 162.779638094621, 300.7511218143277 140.1755315721228, 355.92907315219117 76.02642637059269 M263.5031540080086 181.91703689202433 C284.0915606281116 158.9280519900448, 303.50467241333206 136.41915690883224, 354.86683841781985 76.89356353335259 M272.6960780310081 182.8511620002112 C299.4233344258138 154.61442518985703, 326.2223563245169 125.35301583003732, 366.4340476738402 79.75017178773008 M273.6086492239134 181.57521473227652 C303.58154898262666 145.96102310690304, 335.3217693455627 111.8622196415932, 364.30774661285074 77.0716567636194 M286.6962979139853 180.76374408727048 C304.14096779067773 159.21603247551255, 326.03572500056055 133.1083932963138, 376.2574235321591 76.42483959866635 M285.3620421601552 183.70425095570744 C312.3556955821297 150.60127984154065, 341.50309429446924 119.04727434703386, 376.0471995008747 78.30062546690874 M295.46980194613235 182.27549075562865 C310.6430721192951 162.59136091016825, 328.4557778974885 143.46220488257552, 375.56776353447793 87.0471721395555 M295.2787258831401 182.04285405841082 C325.9271862119236 146.929316247413, 356.88383534838715 112.6427515328542, 376.8944703384405 89.31130621765583 M307.1072883696369 181.48125063809806 C329.56115309034067 155.54435335963933, 353.71621231106684 127.73897917692167, 375.70856107885106 98.86561517009125 M305.1182340842745 182.24184191148038 C322.2297941677207 164.5324073709437, 337.65558364806833 146.0714393335127, 377.2911848478214 101.69756713883564 M314.4486741766862 182.81415361749137 C338.5296487162064 158.30885712740113, 355.8721681416567 136.7446039206576, 375.6627019074661 113.95782982308612 M315.5467410799644 183.60164176100994 C332.9089793666798 165.67819552138246, 346.5408494916694 147.7031108196212, 375.56328297042376 114.90631985154197 M328.15809510946804 183.57061082662094 C335.34298862583114 171.92035982296605, 349.7485059882333 157.54246933124244, 374.4244085704916 126.88033932090701 M328.1995618116163 183.60979022809315 C342.6683082367242 163.5683950000744, 359.0044513595018 146.42826792803254, 376.4804633306348 125.09966619142295 M340.1989056601698 182.8353146482994 C346.26373586822933 172.57559185366475, 355.49465037646456 162.53074669813603, 376.7939452811246 139.76707687658927 M336.94365184563645 183.67843330882388 C347.89944481774864 173.3223759186282, 356.56753704337814 161.91833410254495, 375.63838809087025 138.2659461097952 M347.30867907097894 183.2062387153392 C357.91677886303245 171.32521626087194, 367.24110814370704 162.0322688101249, 378.67685403155247 149.03554737294533 M349.1401073734309 181.68182637429584 C358.0194279126343 171.5149660461741, 367.5673492851773 160.28058635835723, 377.34541906753634 148.93839935275236 M357.6742073340831 180.51963791055277 C364.3706263891852 176.73857049719265, 368.0124204657324 171.6643811100099, 375.5028955639627 162.26653694278775 M360.10666521140615 182.49383417060272 C364.35649307969055 177.01481211356608, 368.71922075960856 171.59797307876894, 376.82905870203933 162.28737504138826 M370.533277237603 183.58156161625428 C372.58042610318785 180.80246173549855, 371.6613124074742 179.9227315275457, 377.1328169574969 174.56240080633995 M369.72012804225227 182.85675284229708 C371.7189288048055 180.16350456666547, 373.18172853680886 177.36937716253624, 376.83151730261824 174.52676015777377" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M133.41840406218944 76.6139637057309 C181.290893971967 78.17821124368658, 231.44799240461413 78.35011960420518, 375.2763230057958 78.31315293051911 M132.85772830900706 78.90326520008487 C190.13575198028835 76.9492327411544, 247.55963218091853 76.68497536060832, 375.6189704742422 78.27871606565853 M377.0844960220158 78.90747872516513 C377.36997423603833 120.66204580068587, 376.8982479853898 161.98683916404843, 375.1552303560078 180.12451485320926 M375.1293457683176 77.04958245791495 C376.0572634681344 115.29119176827372, 375.1678788288712 151.6500133741647, 376.5707743700594 181.8496171358973 M375.5841872076925 183.6388778840947 C284.47513269910183 186.20946044942852, 195.7387184760175 184.24659478197287, 132.19940671806341 181.5647685993352 M375.62534376275534 182.41558028209116 C293.291789898217 180.92203295915405, 213.2503359974905 179.94385137910388, 132.32580676453495 182.45658782696506 M132.08802561834455 183.08462310954928 C134.2653472202629 144.11551345437766, 130.59722662146987 110.5562296964228, 130.21176440790296 76.4797338835895 M132.47703127227723 183.17654409445822 C130.3064559995234 162.01940002553164, 129.78637449136374 140.2163090724498, 133.01777551732957 79.04199701584875" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="254" y="108" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">layer 1:  f₁ ~ GP(x, k₁)</text><text x="254" y="136" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">inducing inputs Z₁, q(u₁)</text><path d="M176 156L182 162L176 168L170 162Z" style="fill:var(--a1)"/><path d="M207 156L213 162L207 168L201 162Z" style="fill:var(--a1)"/><path d="M238 156L244 162L238 168L232 162Z" style="fill:var(--a1)"/><path d="M269 156L275 162L269 168L263 162Z" style="fill:var(--a1)"/><path d="M300 156L306 162L300 168L294 162Z" style="fill:var(--a1)"/><path d="M331 156L337 162L331 168L325 162Z" style="fill:var(--a1)"/><path d="M381.7158775098622 126.32327855750918 C388.58122758418324 128.83283207048476, 398.2103235386312 129.0407937489897, 419.1245505340397 128.3788286320865 M381.0376145657152 129.09270163811743 C390.39560598023235 128.74542768594623, 399.9300435464829 128.42574917671084, 419.5390594180673 128.33716952838003" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M421.0295265059173 126.99396713450551 C418.2459332436938 129.6491857188843, 417.06059659461766 130.8181381240186, 408.6693650921917 133.44688358858667 M420.62256873942914 128.65562098287046 C418.02738243532866 130.0172405915934, 415.57328063970647 131.05136197565798, 408.9180704226083 133.4218881263628" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M421.0295265059173 126.99396713450551 C418.1259514138362 127.56083492282168, 416.9406147647601 126.64143653189333, 408.6693650921917 123.00771076991714 M420.62256873942914 128.65562098287046 C418.0978160702053 127.56538361238184, 415.6437142745831 126.1476480172348, 408.9180704226083 122.98271530769325" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M442.77289530828375 102.63005969025096 C447.23874752873496 99.89708491126233, 453.6370008238181 99.3999938010554, 458.5508198831126 100.87707606376037 C463.4646389424071 102.35415832646534, 468.9907448539295 106.30559729418451, 472.2558096640508 111.4925532664808 C475.52087447417205 116.67950923877709, 478.39564560888596 125.67562585541906, 478.14120874384 131.9988118975381 C477.8867718787941 138.32199793965717, 474.5995670084993 145.18461650007728, 470.7291884737752 149.4316695191951 C466.85880993905107 153.67872253831294, 460.3889886218361 157.29828934058162, 454.91893753549533 157.48113001224507 C449.44888644915454 157.66397068390853, 442.44334853220823 154.11027742914814, 437.9088819557306 150.52871354917585 C433.3744153792529 146.94714966920355, 428.98198375685865 141.71915081640012, 427.71213807662957 135.99174673241134 C426.4422923964005 130.26434264842257, 426.49735467062226 122.13985955837921, 430.2898078743559 116.16428904524315 C434.0822610780896 110.18871853210709, 445.79217105000475 102.50222134054863, 450.4668572990314 100.13832365359492 C455.14154354805805 97.77442596664122, 458.42422550887744 101.38947236130342, 458.3379253685157 101.98090292352093 M445.71244128817847 99.3207081081772 C450.71274629963574 97.3591170091384, 457.4043216461906 98.30599563332036, 462.5702823689833 101.46599969146456 C467.73624309177603 104.62600374960876, 474.4681777814576 112.3129394928788, 476.7082056249345 118.28073245704239 C478.94823346841144 124.24852542120598, 477.737876033049 131.67929345635724, 476.0104494298447 137.2727574764461 C474.2830228266404 142.86622149653493, 470.7095729096171 148.43508921653807, 466.34364600570876 151.84151657757545 C461.9777191018004 155.24794393861282, 455.07957602002875 158.57322626044905, 449.8148880063946 157.7113216426703 C444.55019999276044 156.84941702489155, 438.37080020343456 151.54822184387479, 434.7555179239038 146.6700888709029 C431.1402356443731 141.79195589793102, 428.3055618857064 134.2408034818007, 428.1231943292103 128.44252380483906 C427.9408267727142 122.64424412787741, 430.58390371677376 116.55389818386679, 433.66131258492715 111.88041080913304 C436.73872145308053 107.20692343439929, 444.46602428183013 102.14717517926942, 446.5876475381308 100.40159955643657 C448.70927079443146 98.6560239336037, 446.4694147350963 101.19640665770721, 446.39105212273114 101.40695707213585" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="452" y="136" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">h</text><path d="M483.7158775098622 126.32327855750918 C490.58122758418324 128.83283207048476, 500.2103235386312 129.0407937489897, 521.1245505340397 128.3788286320865 M483.0376145657152 129.09270163811743 C492.39560598023235 128.74542768594623, 501.9300435464829 128.42574917671084, 521.5390594180674 128.33716952838003" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M523.0295265059173 126.99396713450551 C520.2459332436938 129.6491857188843, 519.0605965946177 130.8181381240186, 510.6693650921917 133.44688358858667 M522.6225687394291 128.65562098287046 C520.0273824353287 130.0172405915934, 517.5732806397066 131.05136197565798, 510.9180704226083 133.4218881263628" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M523.0295265059173 126.99396713450551 C520.1259514138362 127.56083492282168, 518.9406147647601 126.64143653189333, 510.6693650921917 123.00771076991714 M522.6225687394291 128.65562098287046 C520.0978160702052 127.56538361238184, 517.6437142745831 126.1476480172348, 510.9180704226083 122.98271530769325" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M525.8558625247035 78.16581119787767 C525.8558625247035 78.16581119787767, 525.8558625247035 78.16581119787767, 525.8558625247035 78.16581119787767 M525.8558625247035 78.16581119787767 C525.8558625247035 78.16581119787767, 525.8558625247035 78.16581119787767, 525.8558625247035 78.16581119787767 M525.8166811780617 89.2632060407999 C529.429745913185 88.39033636324132, 531.1270231203844 81.36766632188716, 535.1467833529388 77.66372184064011 M526.5924680079505 89.58927160406141 C529.2140776441765 87.2195792170214, 532.3871414363615 82.5245187696352, 537.0074297161375 77.24431980784753 M524.0727954780575 100.89675973108396 C532.1852648010477 93.16704057296883, 538.6602117991496 88.31222165597609, 546.2611947464443 79.3547086405254 M525.4874108241763 103.39413223205491 C530.9161649607937 94.57958454707908, 537.4255571978309 87.82161023103204, 547.9520760832971 78.28771986725397 M528.485143773163 115.94149136635608 C534.5610971690803 107.1916211563111, 540.795860068409 98.9363795296717, 556.6757723869898 77.93060630986011 M525.7910562570191 113.6429995832885 C539.3557504794942 100.29764018088778, 549.9003263225267 86.70124123620064, 558.8062552286455 78.46785586790658 M527.0366898446981 125.58218700689977 C539.4656108911779 107.16868490626503, 557.4458375635568 91.32638004597842, 566.4307972626427 77.63498984955612 M525.3557908694897 126.10775472214227 C540.9579468114556 110.00451893501527, 556.9852267013051 89.76673933984013, 568.942682680268 78.45598404080037 M527.0958749928309 138.68865112508198 C544.6537107184915 116.15655378747886, 567.0849402679851 91.64887477260339, 578.9505613655906 79.66661414242594 M526.245453910975 140.01260885926445 C545.0249347312498 116.55049945172597, 562.0430588924954 97.00008647196915, 579.3603452545714 77.1447191320291 M524.5121821807588 151.8330981839381 C543.0254275126713 131.72468602229532, 561.4365185555268 113.00186186992826, 590.5835754380208 78.98751532128216 M527.0123481688912 150.94551132097166 C543.0660896541233 131.89474430072596, 559.2971305580356 112.40938200055156, 590.0773703929615 78.36378526268662 M524.9153721506691 162.39631431628047 C542.8809368190507 143.6227356204271, 557.962406474413 123.3338064862734, 602.077568972121 78.04801076893713 M524.9839760846442 164.37681122835752 C541.511756785006 143.32119946317823, 557.5400854067592 124.69711650932247, 600.9935261618449 77.41751592886652 M525.7344243947493 173.44736690492167 C555.8278666223846 140.4960492822969, 582.8247711898313 111.73088919593731, 613.2544902100786 78.91169911820342 M526.0694807246404 175.75146707751585 C554.5671470333692 142.2865893965042, 582.2934631785861 110.5880878843329, 611.5527212526276 77.62235400166263 M529.20889132913 181.34034710133764 C555.7369634891809 155.5634844747284, 575.8394837470214 126.49116935526655, 620.0855061440813 77.78046025189366 M529.1584999542291 182.49828925820384 C555.9764287560198 153.56741649242244, 577.798244147282 127.17122241159196, 621.6055049817162 76.84435172716881 M539.0154260619606 181.07704646744963 C565.7175204702077 157.56187955344046, 586.0264023732034 131.53769770570037, 633.3660025163161 76.3194553028201 M540.9351942057784 183.80511669562398 C573.1679148043697 147.61564769736572, 601.3957855077255 112.61031670328389, 631.0715672905164 76.8366707264786 M551.8700740455238 181.88847193829798 C581.141972803132 150.82655346849634, 607.314617272668 120.656944397013, 642.190583818971 76.1821266537954 M550.8680381307897 182.21283181309784 C579.4512030668292 151.94413425467417, 608.2252180247058 120.2915561561742, 643.202238745435 76.97708199568584 M560.7486897720067 183.64681589669522 C587.0013162287385 154.82299382266592, 612.6706087067106 124.90538306959398, 651.8280732942333 75.93519793763205 M562.2177372956692 183.2398306067192 C584.3382347369701 156.34787695803578, 603.7555342990809 131.65519791669377, 652.2429905378779 78.80546590588435 M571.8013464835972 181.33755197545372 C596.6626766011192 156.9150037683595, 621.2403890670805 125.39873309585876, 666.3965129342905 79.98367403714255 M573.139137891823 183.46661844758174 C597.8274851289084 153.20579425137726, 624.8111186158532 123.2714520343339, 663.4959344842515 77.36131220316783 M582.2273361724824 183.58443505738603 C620.5050356879998 139.56335927056853, 653.3820069039115 97.24198593856636, 673.9381634813301 76.2958410851393 M582.3437267356576 183.47331008177923 C616.4255020260375 144.79539854405664, 649.4407942989748 109.12133608949362, 673.782323025962 77.72122000862996 M594.1501695808502 180.77450127530474 C625.5684125290509 147.39959297148835, 654.153845002733 115.20594760610759, 686.6945044275396 78.92826646443496 M593.5621312685789 182.7771296304055 C623.5661813359081 151.0560585731518, 650.7807108871384 118.87380004293985, 684.3050492294157 76.7363978183646 M606.8925762008879 182.78306801202223 C636.2564786776765 148.60601587452643, 664.6597358980349 109.22909715680447, 696.7251749861161 77.70550838806464 M604.4066195569935 183.16991058955415 C624.4868927430382 161.33052946954282, 643.9474472638949 136.21844208639217, 694.8550877570282 77.92487308368815 M616.451504274783 182.29873390035104 C644.3718589974238 142.28713678448793, 680.4904012533063 105.91952204296734, 704.407210207228 77.72627757839544 M615.6336002923076 183.65559583031904 C646.9582148103773 143.36288773759372, 681.6565732714839 106.00184986020781, 704.898230983187 77.39015389024897 M626.9731207947231 182.94837882310367 C656.8184541595571 144.064617308401, 690.2767745394224 110.05248902812154, 718.7736868040724 76.81669474846252 M624.8061227909274 181.38773759879385 C654.0907315464634 150.04931100206718, 681.3230820236164 117.79183184764008, 716.5012348203077 77.68698643936344 M635.8278115028954 181.8221639130927 C667.1148624007777 143.02365390967677, 701.1666948847271 105.18305566376515, 728.4129592688366 76.23645228107848 M635.5563769942588 183.40699751607247 C660.5077713653242 155.86393148761698, 684.7809571495583 127.87816596940604, 728.2185548347964 78.54001672712543 M645.0851808914879 183.19853155313046 C675.0598651757936 151.26160123132559, 700.5397229705095 118.35897719984224, 738.6876008437082 78.35852380905571 M646.4481934687087 183.57864657824848 C678.4858100586351 146.6362399553463, 708.8661394493408 111.265207117664, 738.8987872180596 77.33600012462857 M656.1890522026877 181.0692053760296 C678.2424348090917 162.5420778538047, 694.8244341765771 140.0890366937561, 749.6094035208749 76.39416421520683 M657.8395434056829 181.5300651564157 C678.2825856440329 158.71287156193407, 697.5497051480339 136.37192138873445, 748.5471687865037 77.26130137796673 M667.0324674286824 182.4641902646026 C693.5765992594584 154.44054526276412, 720.1941834396433 125.38785612220583, 760.114378042524 80.11790963234428 M667.9450386215877 181.18824299666792 C697.7045347872659 145.83006408896182, 729.2218139250278 111.9877251657592, 757.9880769815344 77.4393946082336 M681.0326873116595 180.37677235166188 C698.3320909138681 159.00020231650632, 720.0761227404649 133.06595285640594, 769.937753900843 76.79257744328049 M679.6984315578294 183.31727922009884 C706.4906483350159 150.4455045413493, 735.4372526519862 119.12248657562225, 769.7275298695586 78.66836331152288 M689.8061913438066 181.88851902002 C704.8388108960467 162.35994934587023, 722.5090905576269 143.39463582319254, 769.2480939031616 87.41490998416964 M689.6151152808145 181.65588232280217 C720.0189723637226 146.82601503286529, 750.7277603752751 112.82458192578233, 770.5748007071243 89.67904406226997 M701.4436777673111 181.0942789024895 C723.688343783292 155.4048617831515, 747.6310936633607 127.84372155848402, 769.3888914475348 99.23335301470544 M699.4546234819488 181.8548701758718 C716.4183989010418 164.31547877395641, 731.6896763270639 146.03225652235622, 770.9715152165052 102.06530498344983 M708.7850635743604 182.42718188188275 C732.8660381138806 157.9218853917925, 750.208557539331 136.357632185049, 769.9990913051404 113.5708580874775 M709.8831304776387 183.21467002540132 C727.2453687643541 165.29122378577384, 740.8772388893436 147.31613908401258, 769.8996723680981 114.51934811593335 M722.4944845071424 183.18363909101234 C729.6793780235054 171.53338808735745, 744.0848953859075 157.15549759563382, 768.7607979681659 126.4933675852984 M722.5359512092907 183.22281849248455 C737.0046976343984 163.1814232644658, 753.3408407571761 146.0412961924239, 770.816852728309 124.71269445581434 M733.8792360288536 183.20305249291357 C739.9440662369132 172.94332969827892, 749.1749807451483 162.8984845427502, 770.4742756498084 140.13481472120344 M730.6239822143202 184.04617115343805 C741.5797751864326 173.69011376324238, 750.247867412062 162.28607194715912, 769.318718459554 138.63368395440938 M740.9890094396627 183.57397655995337 C751.5971092317163 171.6929541054861, 760.9214385123909 162.40000665473906, 772.3571844002362 149.4032852175595 M742.8204377421147 182.04956421891 C751.699758281318 171.88270389078826, 761.2476796538612 160.6483242029714, 771.02574943622 149.30613719736652 M751.3545377027668 180.88737575516691 C758.050956757869 177.1063083418068, 761.6927508344162 172.03211895462405, 769.1832259326463 162.6342747874019 M753.7869955800899 182.86157201521686 C758.0368234483743 177.38254995818022, 762.3995511282922 171.96571092338309, 770.509389070723 162.6551128860024 M764.2809300016934 184.0686481359183 C766.5327937538368 181.01163826708702, 765.5217686885517 180.04393503833887, 771.5404236935767 174.14757124501253 M763.3864658868075 183.27135848456538 C765.5851467256161 180.30878538137063, 767.1942264308199 177.23524523682846, 771.2089940732102 174.10836653158972" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M527.4184040621894 76.6139637057309 C575.290893971967 78.17821124368658, 625.4479924046141 78.35011960420518, 769.2763230057958 78.31315293051911 M526.857728309007 78.90326520008487 C584.1357519802883 76.9492327411544, 641.5596321809186 76.68497536060832, 769.6189704742422 78.27871606565853 M771.0844960220159 78.90747872516513 C771.3699742360384 120.66204580068587, 770.8982479853898 161.98683916404843, 769.1552303560078 180.12451485320926 M769.1293457683175 77.04958245791495 C770.0572634681345 115.29119176827372, 769.1678788288713 151.6500133741647, 770.5707743700593 181.8496171358973 M769.5841872076925 183.6388778840947 C678.4751326991019 186.20946044942852, 589.7387184760174 184.24659478197287, 526.1994067180634 181.5647685993352 M769.6253437627553 182.41558028209116 C687.2917898982171 180.92203295915405, 607.2503359974905 179.94385137910388, 526.325806764535 182.45658782696506 M526.0880256183445 183.08462310954928 C528.2653472202628 144.11551345437766, 524.5972266214698 110.5562296964228, 524.2117644079029 76.4797338835895 M526.4770312722773 183.17654409445822 C524.3064559995233 162.01940002553164, 523.7863744913637 140.2163090724498, 527.0177755173296 79.04199701584875" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="648" y="108" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">layer 2:  f₂ ~ GP(0, k₂)</text><text x="648" y="136" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">inducing inputs Z₂, q(u₂)</text><path d="M570 156L576 162L570 168L564 162Z" style="fill:var(--a1)"/><path d="M601 156L607 162L601 168L595 162Z" style="fill:var(--a1)"/><path d="M632 156L638 162L632 168L626 162Z" style="fill:var(--a1)"/><path d="M663 156L669 162L663 168L657 162Z" style="fill:var(--a1)"/><path d="M694 156L700 162L694 168L688 162Z" style="fill:var(--a1)"/><path d="M725 156L731 162L725 168L719 162Z" style="fill:var(--a1)"/><path d="M775.7158775098621 126.32327855750918 C782.5812275841832 128.83283207048476, 792.2103235386312 129.0407937489897, 813.1245505340397 128.3788286320865 M775.0376145657152 129.09270163811743 C784.3956059802324 128.74542768594623, 793.9300435464829 128.42574917671084, 813.5390594180674 128.33716952838003" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M815.0295265059173 126.99396713450551 C812.2459332436937 129.6491857188843, 811.0605965946177 130.8181381240186, 802.6693650921917 133.44688358858667 M814.6225687394291 128.65562098287046 C812.0273824353287 130.01724059159343, 809.5732806397065 131.05136197565798, 802.9180704226083 133.4218881263628" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M815.0295265059173 126.99396713450551 C812.1259514138361 127.56083492282168, 810.9406147647601 126.64143653189333, 802.6693650921917 123.00771076991714 M814.6225687394291 128.65562098287046 C812.0978160702052 127.56538361238184, 809.643714274583 126.1476480172348, 802.9180704226083 122.98271530769325" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M836.7728953082838 102.63005969025096 C841.238747528735 99.89708491126233, 847.6370008238182 99.3999938010554, 852.5508198831127 100.87707606376037 C857.4646389424072 102.35415832646534, 862.9907448539295 106.30559729418451, 866.2558096640507 111.4925532664808 C869.5208744741719 116.67950923877709, 872.3956456088861 125.67562585541906, 872.1412087438401 131.9988118975381 C871.8867718787942 138.32199793965717, 868.5995670084994 145.18461650007728, 864.7291884737753 149.4316695191951 C860.8588099390511 153.67872253831294, 854.3889886218361 157.29828934058162, 848.9189375354954 157.48113001224507 C843.4488864491547 157.66397068390853, 836.4433485322082 154.11027742914814, 831.9088819557306 150.52871354917585 C827.374415379253 146.94714966920355, 822.9819837568587 141.71915081640012, 821.7121380766296 135.99174673241134 C820.4422923964005 130.26434264842257, 820.4973546706223 122.13985955837921, 824.2898078743559 116.16428904524315 C828.0822610780896 110.18871853210709, 839.7921710500048 102.50222134054863, 844.4668572990314 100.13832365359492 C849.141543548058 97.77442596664122, 852.4242255088775 101.38947236130342, 852.3379253685157 101.98090292352093 M839.7124412881784 99.3207081081772 C844.7127462996357 97.3591170091384, 851.4043216461907 98.30599563332036, 856.5702823689834 101.46599969146456 C861.736243091776 104.62600374960876, 868.4681777814576 112.3129394928788, 870.7082056249345 118.28073245704239 C872.9482334684114 124.24852542120598, 871.737876033049 131.67929345635724, 870.0104494298447 137.2727574764461 C868.2830228266404 142.86622149653493, 864.7095729096171 148.43508921653807, 860.3436460057088 151.84151657757545 C855.9777191018005 155.24794393861282, 849.0795760200289 158.57322626044905, 843.8148880063947 157.7113216426703 C838.5501999927604 156.84941702489155, 832.3708002034346 151.54822184387479, 828.7555179239038 146.6700888709029 C825.140235644373 141.79195589793102, 822.3055618857063 134.2408034818007, 822.1231943292103 128.44252380483906 C821.9408267727142 122.64424412787741, 824.5839037167738 116.55389818386679, 827.6613125849271 111.88041080913304 C830.7387214530805 107.20692343439929, 838.4660242818301 102.14717517926942, 840.5876475381308 100.40159955643657 C842.7092707944315 98.6560239336037, 840.4694147350964 101.19640665770721, 840.3910521227313 101.40695707213585" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="846" y="136" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">f</text><path d="M877.7158775098621 126.32327855750918 C882.9808319725097 128.85703778195978, 891.009532315284 129.06499946046472, 907.1245505340397 128.3788286320865 M877.0376145657152 129.09270163811743 C884.5166394975037 128.8245087099582, 892.1721105810254 128.5048302007228, 907.5390594180674 128.33716952838003" style="stroke:var(--ink);stroke-width:2;fill:none"/><path d="M909.0295265059173 126.99396713450551 C906.2459332436937 129.6491857188843, 905.0605965946177 130.8181381240186, 896.6693650921917 133.44688358858667 M908.6225687394291 128.65562098287046 C906.0273824353287 130.01724059159343, 903.5732806397065 131.05136197565798, 896.9180704226083 133.4218881263628" style="stroke:var(--ink);stroke-width:1;fill:none"/><path d="M909.0295265059173 126.99396713450551 C906.1259514138361 127.56083492282168, 904.9406147647601 126.64143653189333, 896.6693650921917 123.00771076991714 M908.6225687394291 128.65562098287046 C906.0978160702052 127.56538361238184, 903.643714274583 126.1476480172348, 896.9180704226083 122.98271530769325" style="stroke:var(--ink);stroke-width:1;fill:none"/><text x="932" y="136" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">y</text><text x="906" y="172" text-anchor="middle" style="font-family:Kalam,cursive;font-size:15px;font-weight:400;fill:var(--muted)">+ noise</text><text x="452" y="196" text-anchor="middle" style="font-family:Kalam,cursive;font-size:15px;font-weight:400;fill:var(--muted)">hidden values are random,</text><text x="452" y="216" text-anchor="middle" style="font-family:Kalam,cursive;font-size:15px;font-weight:400;fill:var(--muted)">so the kernel of layer 2 is random</text><text x="480" y="278" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:700;fill:var(--dot)">one training point, through the layers (doubly stochastic VI)</text><text x="46" y="352" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--ink)">xₙ</text><path d="M67.71587750986218 344.3232785575092 C74.18112868126481 346.83888349835354, 83.4101257327944 347.0468451768585, 103.12455053403974 346.37882863208654 M67.03761456571519 347.09270163811743 C75.92586435955018 346.7651979419492, 84.99056030511856 346.44551943271387, 103.53905941806734 346.33716952838006" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M105.0295265059173 344.9939671345055 C102.24593324369381 347.64918571888427, 101.06059659461768 348.81813812401856, 92.66936509219173 351.4468835885867 M104.62256873942911 346.65562098287046 C102.02738243532865 348.0172405915934, 99.57328063970651 349.051361975658, 92.91807042260828 351.4218881263628" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M105.0295265059173 344.9939671345055 C102.1259514138362 345.56083492282164, 100.94061476476007 344.6414365318933, 92.66936509219173 341.0077107699172 M104.62256873942911 346.65562098287046 C102.09781607020524 345.5653836123818, 99.6437142745831 344.1476480172348, 92.91807042260828 340.9827153076933" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M109.71587750986218 310.3232785575092 C147.78894201181828 312.360820696722, 188.62575239390134 312.56878237522693, 303.1245505340397 312.37882863208654 M109.03761456571519 313.09270163811743 C155.03545239344237 311.2033477177128, 201.209736372903 310.88366920847744, 303.5390594180673 312.33716952838006 M305.0844960220158 312.9074787251651 C304.9566023046561 341.3161832412705, 304.48487605400754 369.29511404521764, 303.1552303560078 380.12451485320923 M303.1293457683176 311.04958245791494 C304.2983896845542 337.3111794775352, 303.409005045291 361.68998879268764, 304.5707743700594 381.8496171358973 M303.49698126390575 383.982589995116 C230.21179491765798 386.53354527383, 159.79684859216212 384.159019464542, 108.24122710302473 381.47349010631444 M303.54676933847367 382.50273746289315 C237.17254243642094 381.16542300380024, 173.57112399078906 379.98209321893006, 108.39413627944887 382.55234527625146 M108.08802561834455 383.0846231095493 C110.25775405035542 356.1821982646361, 106.58963345156238 334.68959931693973, 106.21176440790296 310.4797338835895 M108.47703127227723 383.1765440944582 C106.65087848195807 368.8679842533544, 106.13079697379843 353.91347752809526, 109.01777551732957 313.0419970158488" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="206" y="342" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:700;fill:var(--ink)">μ₁(xₙ),  σ₁²(xₙ)</text><text x="206" y="366" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">closed form, O(M²)</text><path d="M309.7158775098622 344.3232785575092 C316.18112868126485 346.83888349835354, 325.4101257327944 347.0468451768585, 345.1245505340397 346.37882863208654 M309.0376145657152 347.09270163811743 C317.9258643595502 346.7651979419492, 326.99056030511855 346.44551943271387, 345.5390594180673 346.33716952838006" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M347.0295265059173 344.9939671345055 C344.2459332436938 347.64918571888427, 343.06059659461766 348.81813812401856, 334.6693650921917 351.4468835885867 M346.62256873942914 346.65562098287046 C344.02738243532866 348.0172405915934, 341.57328063970647 349.051361975658, 334.9180704226083 351.4218881263628" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M347.0295265059173 344.9939671345055 C344.1259514138362 345.56083492282164, 342.9406147647601 344.6414365318933, 334.6693650921917 341.0077107699172 M346.62256873942914 346.65562098287046 C344.0978160702053 345.5653836123818, 341.6437142745831 344.1476480172348, 334.9180704226083 340.9827153076933" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M349.90075721775344 312.11416576134127 C349.90075721775344 312.11416576134127, 349.90075721775344 312.11416576134127, 349.90075721775344 312.11416576134127 M349.90075721775344 312.11416576134127 C349.90075721775344 312.11416576134127, 349.90075721775344 312.11416576134127, 349.90075721775344 312.11416576134127 M349.9386855814051 324.50439626210095 C354.0033834084189 323.5224178748476, 355.9128202665183 315.6219140783241, 360.4350505281419 311.45497653692115 M350.81144576503 324.8712200207702 C353.7607566057843 322.20531608535015, 357.3304533719924 316.9233730820407, 362.5282776867405 310.9831492500295 M347.6589322445813 338.4213610931969 C356.67817099341426 329.61853284908386, 364.0651197203016 323.71457595596553, 373.1276266579207 313.1057621015245 M349.0735475907001 340.91873359416786 C355.41890018293407 331.0711312942575, 362.83970830787007 323.26469293493244, 374.8185079947736 312.038773328253 M352.1699310909192 354.87686133768227 C359.1914203807047 345.0135883802729, 366.3650340845704 335.6783224489999, 384.95297290767945 311.5830092196269 M349.47584357477524 352.57836955461465 C364.8160861152923 337.28209520198976, 377.0683918502961 321.72117774151167, 387.0834557493351 312.1202587776733 M351.476186742677 365.1736160072165 C365.6234908274815 344.82115138575165, 385.3251188929807 326.9986007466732, 395.4627073635552 311.94345178831344 M349.79528776746855 365.699183722459 C367.0957048115316 347.6374692209237, 384.86855722511376 325.3916381419336, 397.9745927811805 312.76444597955765 M350.97796341305167 380.4455583148348 C371.00003978140944 354.9993998344074, 395.90952467711753 327.6408141642201, 409.39324007571634 313.876425529951 M350.12754233119574 381.7695160490173 C371.2240338300209 355.64582343019, 390.52703197403775 333.46696360617193, 409.80302396469705 311.3545305195542 M358.9898656160599 382.92542069933984 C376.79080694034747 363.5933009715166, 394.5098659114168 345.6665686513161, 422.43702275735984 313.09867615757497 M361.4900316041922 382.0378338363734 C376.8336907562972 363.7988881811507, 392.3781364922686 345.1033632707264, 421.9308177123005 312.47494609897944 M371.3007686590316 381.31463299688556 C386.3869985632343 365.91525193678126, 398.61941121507886 348.903787969846, 435.34178490067336 312.06052105399766 M371.3693725930067 383.2951299089627 C385.07722081348265 365.7314893495681, 398.17791668556293 350.4752626188816, 434.2577420903972 311.4300262139271 M384.0275339761732 380.1916817507302 C406.083502023378 356.3058994922461, 425.32870401725734 336.4580531486922, 447.9294747478442 312.82555885203175 M384.36259030606425 382.4957819233244 C405.29166175058873 358.25759088633316, 425.1893836106514 335.56485691275407, 446.22770579039326 311.53621373549095 M395.47335980967216 380.4389155936861 C414.2724281530657 364.0397732037229, 426.38500242827695 344.1588395180802, 456.17125929106004 311.5956694344896 M395.42296843477124 381.59685775055226 C414.11026669884785 361.97710434204186, 428.1060139544467 344.5837717972376, 457.6912581286949 310.65956090976476 M406.69066315171614 380.0769644085659 C425.7542507838435 365.00677812508917, 438.5364121575858 347.64109778418384, 470.20646524351775 310.79072351440664 M408.6104312955339 382.80503463674023 C430.23843449812483 358.7534938744103, 448.2474002255094 335.5036683210793, 467.9120300177181 311.3079389380651 M420.30002071550194 381.5444489084047 C440.44935325005997 360.9933772986103, 457.41870078223485 341.410950266141, 480.44181515538594 310.5547443141496 M419.29798480076795 381.8688087832046 C438.3833976518027 362.05722283784183, 458.0251985371249 340.91005529638744, 481.4534700818499 311.3496996560401 M430.58940505119824 383.2041423155697 C448.6764914915174 364.0237540977119, 465.91632655198373 343.8031248477534, 491.4900732398615 310.209165046754 M432.05845257486067 382.79715702559366 C447.16379026428706 364.3770318902588, 459.4559320829408 347.8809092134558, 491.9049904835061 313.0794330150063 M443.05283037200195 380.7962278430959 C459.87765495806 365.49055128081693, 476.47946964408624 343.14950143357777, 507.469281489132 314.1589905950322 M444.39062178022783 382.92529431522394 C460.44772901079915 362.31145420168417, 478.93593918170427 342.1499785733816, 504.568703039093 311.5366287610575 M454.8895886701006 382.944460373796 C481.6827351859401 352.80296094759143, 502.6692834528086 324.15995452466416, 516.4217006453848 310.3725070917968 M455.00597923327564 382.8333353981892 C478.12139190328315 356.4713892407554, 500.4841174965964 333.05170295024703, 516.2658601900167 311.7978860152874 M468.2231906876816 380.03587604048244 C489.8622030552298 357.7920551148235, 509.047965591064 336.4114930842481, 530.5888102008078 312.90628191986013 M467.63515237541037 382.0385043955832 C488.28209971840766 360.6579139561671, 506.22045682548173 339.14667114564656, 528.1993550026838 310.7144132737898 M482.3763659169326 381.94579222596764 C501.81247152809874 359.8574093558487, 519.9409388689461 332.3002843063504, 542.0302493685975 311.58487329225756 M479.8904092730383 382.33263480349956 C493.5921009196651 367.9525460030394, 506.34334629816675 350.55863589153284, 540.1601621395096 311.8042379878811 M492.69000357105045 382.11751714328693 C510.4818808295603 354.5401664738975, 536.0488183979477 330.3107844103564, 551.1230531989227 311.5069919313561 M491.87209958857505 383.4743790732549 C512.5777580574099 355.7586020093739, 536.2233634222015 331.1123021068031, 551.6140739748816 311.1708682432096 M504.62238870020377 382.6685115148073 C522.1206679707028 357.62404063441284, 543.1877722374282 337.8663758977375, 562.3078852020468 315.7817256117503 M502.455390696408 381.1078702904975 C521.2827248187774 361.823297003918, 537.6944756735189 342.0134938023007, 560.035433218282 316.65201730265125 M514.8878480175895 381.4436460535641 C530.1187756103386 361.8813356918868, 547.8579020235243 342.8063591460641, 561.5488637541952 328.6876050371439 M514.6164135089529 383.0284796565439 C527.6254224956563 369.042935386961, 539.9811324491955 354.76665755820125, 561.3544593201549 330.99116948319084 M525.5559860153952 382.7213631423696 C537.9140015900094 371.2788399693564, 546.1342292554793 358.23114915887754, 561.4252114164509 344.2957984578988 M526.918998592616 383.10147816748764 C539.4947730505886 369.08830310393415, 550.0467960208947 356.52712754097684, 561.6363977908023 343.27327477347166 M538.0706259358083 380.4933864140365 C546.0259111766981 377.95328883459126, 548.5567651192551 371.66424145664814, 561.2926611520112 356.57227033705027 M539.7211171388035 380.9542461944226 C544.6101705295843 376.5187225022166, 548.256115943107 372.1478774864026, 560.2304264176399 357.43940749981016 M550.1167007370517 382.4986402312788 C553.5715040297355 380.48347211338745, 557.2170574062754 377.3868798259887, 561.0835623721249 373.1952551150989 M550.7555005720855 381.6054771437245 C553.9600111033548 378.0014023876545, 557.6071613004234 375.49573631946816, 559.5951516294323 371.32029459822144" style="stroke:var(--dot);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M351.51564489972634 310.5189418895192 C392.52203938808157 312.28129824852346, 435.9696671948283 312.46499204691696, 559.2267102340205 312.3346216036938 M350.91653117158364 312.96518991315895 C400.20685946763814 311.17406995889917, 449.6530437081957 310.8916960136205, 559.5928484183354 312.2978238674353 M561.0844960220159 312.9074787251651 C560.956602304656 341.3161832412705, 560.4848760540075 369.29511404521764, 559.1552303560078 380.12451485320923 M559.1293457683175 311.04958245791494 C560.2983896845542 337.3111794775352, 559.409005045291 361.68998879268764, 560.5707743700593 381.8496171358973 M559.5556805322956 383.7512336382259 C481.0913956968153 386.14845651187875, 405.1624110931054 384.05102361737977, 350.2130773474644 381.5349306518482 M559.5996586372898 382.4440710173983 C488.6173613774973 381.141864263295, 420.0843024641121 380.0966219643405, 350.34814294045486 382.48788989658453 M350.08802561834455 383.0846231095493 C352.2577540503554 356.1821982646361, 348.5896334515624 334.68959931693973, 348.21176440790293 310.4797338835895 M350.47703127227726 383.1765440944582 C348.65087848195805 368.8679842533544, 348.13079697379845 353.91347752809526, 351.0177755173296 313.0419970158488" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="455" y="342" text-anchor="middle" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">ĥₙ = μ₁ + σ₁ ε</text><text x="455" y="366" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">sample, ε ~ N(0, 1)</text><path d="M565.7158775098621 344.3232785575092 C572.1811286812648 346.83888349835354, 581.4101257327944 347.0468451768585, 601.1245505340397 346.37882863208654 M565.0376145657152 347.09270163811743 C573.9258643595502 346.7651979419492, 582.9905603051186 346.44551943271387, 601.5390594180674 346.33716952838006" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M603.0295265059173 344.9939671345055 C600.2459332436937 347.64918571888427, 599.0605965946177 348.81813812401856, 590.6693650921917 351.4468835885867 M602.6225687394291 346.65562098287046 C600.0273824353287 348.0172405915934, 597.5732806397065 349.051361975658, 590.9180704226083 351.4218881263628" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M603.0295265059173 344.9939671345055 C600.1259514138361 345.56083492282164, 598.9406147647601 344.6414365318933, 590.6693650921917 341.0077107699172 M602.6225687394291 346.65562098287046 C600.0978160702052 345.5653836123818, 597.643714274583 344.1476480172348, 590.9180704226083 340.9827153076933" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M607.7158775098621 310.3232785575092 C645.7889420118183 312.360820696722, 686.6257523939014 312.56878237522693, 801.1245505340397 312.37882863208654 M607.0376145657152 313.09270163811743 C653.0354523934425 311.2033477177128, 699.209736372903 310.88366920847744, 801.5390594180674 312.33716952838006 M803.0844960220159 312.9074787251651 C802.956602304656 341.3161832412705, 802.4848760540075 369.29511404521764, 801.1552303560078 380.12451485320923 M801.1293457683175 311.04958245791494 C802.2983896845542 337.3111794775352, 801.409005045291 361.68998879268764, 802.5707743700593 381.8496171358973 M801.4969812639058 383.982589995116 C728.2117949176579 386.53354527383, 657.7968485921622 384.159019464542, 606.2412271030247 381.47349010631444 M801.5467693384737 382.50273746289315 C735.1725424364208 381.16542300380024, 671.571123990789 379.98209321893006, 606.3941362794488 382.55234527625146 M606.0880256183445 383.0846231095493 C608.2577540503554 356.1821982646361, 604.5896334515623 334.68959931693973, 604.2117644079029 310.4797338835895 M606.4770312722773 383.1765440944582 C604.650878481958 368.8679842533544, 604.1307969737984 353.91347752809526, 607.0177755173296 313.0419970158488" style="stroke:var(--ink);stroke-width:2;fill:none"/><text x="704" y="342" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:700;fill:var(--ink)">μ₂(ĥₙ),  σ₂²(ĥₙ)</text><text x="704" y="366" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">closed form, O(M²)</text><path d="M807.7158775098621 344.3232785575092 C813.380930875428 346.85098635409105, 821.8097301211208 347.058948032596, 839.1245505340397 346.37882863208654 M807.0376145657152 347.09270163811743 C814.9863811181858 346.8047384539552, 823.1115938223899 346.48505994471986, 839.5390594180674 346.33716952838006" style="stroke:var(--dot);stroke-width:2;fill:none"/><path d="M841.0295265059173 344.9939671345055 C838.2459332436937 347.64918571888427, 837.0605965946177 348.81813812401856, 828.6693650921917 351.4468835885867 M840.6225687394291 346.65562098287046 C838.0273824353287 348.0172405915934, 835.5732806397065 349.051361975658, 828.9180704226083 351.4218881263628" style="stroke:var(--dot);stroke-width:1;fill:none"/><path d="M841.0295265059173 344.9939671345055 C838.1259514138361 345.56083492282164, 836.9406147647601 344.6414365318933, 828.6693650921917 341.0077107699172 M840.6225687394291 346.65562098287046 C838.0978160702052 345.5653836123818, 835.643714274583 344.1476480172348, 828.9180704226083 340.9827153076933" style="stroke:var(--dot);stroke-width:1;fill:none"/><text x="898" y="340" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:700;fill:var(--ink)">E log p(yₙ | f)</text><text x="898" y="362" text-anchor="middle" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">in closed form</text><text x="480" y="430" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">two sources of randomness: a minibatch of points n, and the noise ε</text><text x="480" y="458" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">both give unbiased gradients of the ELBO, and no N × N matrix appears anywhere</text></svg></figure>

:::theorem[The doubly stochastic ELBO (Salimbeni and Deisenroth, 2017)]
For this family,

$$
\mathcal{L} = \sum_{n=1}^{N}\mathbb{E}_{q(f_{L,n})}\big[\log p(y_n \mid f_{L,n})\big] - \sum_{l=1}^{L}\mathrm{KL}\big[q(\boldsymbol{u}_l) \,\Vert\, p(\boldsymbol{u}_l)\big],
$$

and the marginal $q(f_{L,n})$ depends only on the $n$-th input, through a chain of one-dimensional Gaussians:

$$
q(f_{L,n}) = \int\prod_{l=1}^{L}\mathcal{N}\big(f_{l,n} \mid \mu_l(f_{l-1,n}),\ s_l^2(f_{l-1,n})\big)\,df_{1,n}\cdots df_{L-1,n}.
$$
:::

:::proof
**Step 1. The ELBO.** By definition, $\mathcal{L} = \mathbb{E}_q\big[\log p(\boldsymbol{y}, \{\boldsymbol{f}_l, \boldsymbol{u}_l\}) - \log q(\{\boldsymbol{f}_l, \boldsymbol{u}_l\})\big]$.

**Step 2. Write the ratio out and cancel.**

$$
\frac{p(\boldsymbol{y}, \{\boldsymbol{f}_l, \boldsymbol{u}_l\})}{q(\{\boldsymbol{f}_l, \boldsymbol{u}_l\})} = \frac{\prod_n p(y_n \mid f_{L,n})\ \prod_l p(\boldsymbol{f}_l \mid \boldsymbol{u}_l; \boldsymbol{f}_{l-1})\,p(\boldsymbol{u}_l)}{\prod_l p(\boldsymbol{f}_l \mid \boldsymbol{u}_l; \boldsymbol{f}_{l-1})\,q(\boldsymbol{u}_l)} = \prod_n p(y_n \mid f_{L,n})\ \prod_l\frac{p(\boldsymbol{u}_l)}{q(\boldsymbol{u}_l)}.
$$

Every $N$-dimensional conditional cancels, exactly as in Titsias's derivation.

**Step 3. Take logs and expectations.**

$$
\mathcal{L} = \sum_n\mathbb{E}_q\big[\log p(y_n \mid f_{L,n})\big] + \sum_l\mathbb{E}_q\Big[\log\frac{p(\boldsymbol{u}_l)}{q(\boldsymbol{u}_l)}\Big].
$$

**Step 4. The KL terms.** Under $q$, integrate out $\boldsymbol{f}_L$ first: $p(\boldsymbol{f}_L \mid \boldsymbol{u}_L; \boldsymbol{f}_{L-1})$ is a normalised density in $\boldsymbol{f}_L$, so it integrates to 1. Then $\boldsymbol{f}_{L-1}$, and so on down to $\boldsymbol{f}_1$. What remains is $\prod_lq(\boldsymbol{u}_l)$: the inducing variables are independent under $q$ with marginals $q(\boldsymbol{u}_l)$. Each second term depends only on $\boldsymbol{u}_l$, so it equals $\mathbb{E}_{q(\boldsymbol{u}_l)}[\log p(\boldsymbol{u}_l) - \log q(\boldsymbol{u}_l)] = -\mathrm{KL}[q(\boldsymbol{u}_l) \,\Vert\, p(\boldsymbol{u}_l)]$.

**Step 5. Integrate out each $\boldsymbol{u}_l$.** Define $q(\boldsymbol{f}_l \mid \boldsymbol{f}_{l-1}) = \int p(\boldsymbol{f}_l \mid \boldsymbol{u}_l; \boldsymbol{f}_{l-1})\,q(\boldsymbol{u}_l)\,d\boldsymbol{u}_l$. By the one-layer proposition applied at each of the $N$ inputs, it is Gaussian with $n$-th mean $\mu_l(f_{l-1,n})$ and $n$-th variance $s_l^2(f_{l-1,n})$. Because the $\boldsymbol{u}_l$ are independent, the layers form a Markov chain under $q$:

$$
q(\boldsymbol{f}_1, \dots, \boldsymbol{f}_L) = \prod_{l=1}^{L}q(\boldsymbol{f}_l \mid \boldsymbol{f}_{l-1}).
$$

**Step 6. The $n$-th term needs one marginal.** $\log p(y_n \mid f_{L,n})$ depends only on $f_{L,n}$, so its expectation needs only the marginal $q(f_{L,n})$: integrate the chain over every other coordinate.

**Step 7. Integrate the top layer.** In $q(\boldsymbol{f}_L \mid \boldsymbol{f}_{L-1})$, integrate out the $N - 1$ coordinates of $\boldsymbol{f}_L$ other than $f_{L,n}$. The marginal of one coordinate of a Gaussian vector is Gaussian with that coordinate's mean and variance, $\mathcal{N}\big(f_{L,n} \mid \mu_L(f_{L-1,n}), s_L^2(f_{L-1,n})\big)$, and this depends on $\boldsymbol{f}_{L-1}$ only through $f_{L-1,n}$.

**Step 8. Induct downwards.** After Step 7, the coordinates of $\boldsymbol{f}_{L-1}$ other than $f_{L-1,n}$ appear only in $q(\boldsymbol{f}_{L-1} \mid \boldsymbol{f}_{L-2})$. Integrating them out leaves $\mathcal{N}\big(f_{L-1,n} \mid \mu_{L-1}(f_{L-2,n}), s_{L-1}^2(f_{L-2,n})\big)$. Repeating down to layer 1 leaves the chain in the statement.
:::

That is the whole method. Every point travels through the layers on its own, carrying one mean and one variance per layer, and no $N \times N$ matrix is ever formed.

### A forward pass by hand

Formulas are easier to trust after one pass with numbers. Take a tiny two-layer model:

- both layers: squared-exponential kernel with $\sigma = \ell = 1$, zero mean, two inducing inputs $\mathbf{Z} = (-1, 1)$;
- $q(\boldsymbol{u}_1) = \mathcal{N}\big((-0.8,\ 0.8),\ 0.01\,\mathbf{I}\big)$ and $q(\boldsymbol{u}_2) = \mathcal{N}\big((-1,\ 1),\ 0.02\,\mathbf{I}\big)$;
- one training point $x = 0.5$, $y = 0.9$, noise $\sigma_n = 0.1$, and one draw $\varepsilon = 0.5$.

Both layers share $\mathbf{K}_{uu} = \begin{bmatrix}1 & e^{-2}\\ e^{-2} & 1\end{bmatrix} = \begin{bmatrix}1 & 0.1353\\ 0.1353 & 1\end{bmatrix}$, since the two inducing inputs are 2 apart.

| step | formula | value |
|---|---|---|
| 1. kernel to the inducing inputs | $\boldsymbol{k}_u(0.5) = (e^{-1.5^2/2},\ e^{-0.5^2/2})$ | $(0.3247,\ 0.8825)$ |
| 2. interpolation weights | $\boldsymbol{a} = \mathbf{K}_{uu}^{-1}\boldsymbol{k}_u$ | $(0.2090,\ 0.8542)$ |
| 3. layer-1 mean | $\mu_1 = \boldsymbol{a}^\top\boldsymbol{m}_1 = -0.8(0.2090) + 0.8(0.8542)$ | $0.5161$ |
| 4. what $\boldsymbol{u}_1$ cannot pin down | $c_1 = 1 - \boldsymbol{k}_u^\top\boldsymbol{a}$ | $0.1783$ |
| 5. uncertainty about $\boldsymbol{u}_1$ | $\boldsymbol{a}^\top\mathbf{S}_1\boldsymbol{a} = 0.01(0.2090^2 + 0.8542^2)$ | $0.0077$ |
| 6. layer-1 variance | $s_1^2 = c_1 + \boldsymbol{a}^\top\mathbf{S}_1\boldsymbol{a}$ | $0.1860$, so $s_1 = 0.4313$ |
| 7. sample the hidden value | $\hat h = \mu_1 + s_1\varepsilon = 0.5161 + 0.4313(0.5)$ | $0.7318$ |
| 8. layer 2 at $\hat h$ | $\boldsymbol{k}_u(\hat h) = (e^{-1.7318^2/2},\ e^{-0.2682^2/2})$, then $\boldsymbol{a} = \mathbf{K}_{uu}^{-1}\boldsymbol{k}_u$ | $(0.2232,\ 0.9647)$, $(0.0944,\ 0.9519)$ |
| 9. layer-2 mean | $\mu_2 = -1(0.0944) + 1(0.9519)$ | $0.8575$ |
| 10. layer-2 variance | $s_2^2 = (1 - \boldsymbol{k}_u^\top\boldsymbol{a}) + \boldsymbol{a}^\top\mathbf{S}_2\boldsymbol{a} = 0.0607 + 0.0183$ | $0.0790$ |
| 11. expected log-likelihood | $-\tfrac12\log(2\pi\sigma_n^2) - \frac{(y - \mu_2)^2}{2\sigma_n^2} - \frac{s_2^2}{2\sigma_n^2} = 1.3836 - 0.0904 - 3.9482$ | $-2.6549$ |

Step 11 shows what training will do. The mean is close to $y$, so the data-fit term costs only $0.09$ nats. The variance term costs $3.95$, because the model is far less certain ($s_2^2 = 0.079$) than the noise ($\sigma_n^2 = 0.01$). To raise the ELBO, the optimiser will make the layers more certain where the data are, and move the hidden values to where layer 2 is certain. That is how a warp gets learned.

### Turning it into an algorithm

The chain has no closed form, because $\mu_l$ and $s_l^2$ are nonlinear functions of their input. But it is easy to *sample*, and samples can be differentiated.

:::proposition[The reparameterisation trick]
If $\varepsilon \sim \mathcal{N}(0, 1)$, then for any differentiable $g$,

$$
\nabla_\phi\,\mathbb{E}_{f \sim \mathcal{N}(\mu_\phi, s_\phi^2)}\big[g(f)\big] = \mathbb{E}_{\varepsilon}\big[\nabla_\phi\,g(\mu_\phi + s_\phi\varepsilon)\big].
$$
:::

:::proof
**Step 1. Same distribution.** If $\varepsilon \sim \mathcal{N}(0, 1)$, then $\mu_\phi + s_\phi\varepsilon \sim \mathcal{N}(\mu_\phi, s_\phi^2)$, an affine map of a Gaussian. So $\mathbb{E}_f[g(f)] = \mathbb{E}_\varepsilon[g(\mu_\phi + s_\phi\varepsilon)] = \int g(\mu_\phi + s_\phi\varepsilon)\,\mathcal{N}(\varepsilon \mid 0, 1)\,d\varepsilon$.

**Step 2. The density no longer depends on $\phi$.** Only the integrand does. Under the usual integrability conditions the gradient moves inside the integral, which is the statement.

**Step 3. Why this matters.** An average of $\nabla_\phi g(\mu_\phi + s_\phi\varepsilon)$ over a few draws of $\varepsilon$ is an unbiased estimate of the gradient. It uses the slope of $g$, which typically makes it far less noisy than estimators that only use the values of $g$.
:::

So one training step is:

1. pick a minibatch $\mathcal{B}$ of $B$ points and draw standard normals $\varepsilon_{l,n}$;
2. for each point, push a sample through the layers: $\hat f_{l,n} = \mu_l(\hat f_{l-1,n}) + s_l(\hat f_{l-1,n})\,\varepsilon_{l,n}$, starting from $\hat f_{0,n} = \boldsymbol{x}_n$, as in steps 1 to 7 of the table;
3. at the last layer, do the expectation in closed form instead of sampling; for a Gaussian likelihood that is step 11 of the table;
4. estimate the ELBO by $\frac{N}{B}\sum_{n \in \mathcal{B}}(\text{that term}) - \sum_l\mathrm{KL}_l$, and take a gradient step on everything: $\boldsymbol{m}_l$, $\mathbf{S}_l$, $\mathbf{Z}_l$, the kernel hyperparameters and $\sigma_n^2$.

The estimate is unbiased twice over, for the random minibatch and for the random $\varepsilon$: hence *doubly stochastic*. A step costs $O\big(SBLM^2 + LM^3\big)$ for $S$ samples per point, with no dependence on $N$; Salimbeni and Deisenroth (2017) report data sets "ranging in size from hundreds to a billion points". With several outputs per layer, each output dimension gets its own $q(\boldsymbol{u})$ and KL term, and the samples are vectors.

Three practical choices matter a lot:

- **Identity (or linear) mean functions** on hidden layers, the fix from Section 3.
- **Start inner layers almost deterministic.** Initialise $\mathbf{S}_l$ near zero for $l < L$, so training starts from a GP on almost unwarped inputs and the warp is learned only as far as the data justify. Sáez-Maldonado, Maroñas and Hernández-Lobato (2026) argue that this conditioning at initialisation, more than the identity mean itself, is what avoids *posterior collapse*, where the hidden layers fall back to their priors and the model explains the data as noise.
- **Predict with the mixture** of Section 2: push $S$ samples of a test input through the layers and average the final Gaussians.

### A deep GP, fitted

Here is an exact GP and a two-layer deep GP fitted to the same 150 noisy points of a step. I trained the deep GP with exactly the algorithm above: 20 whitened inducing points per layer, an identity mean on the hidden layer, 8 samples per step, 6000 Adam steps. The hyperparameters of both models were learned from the data. Move the pointer across the plot to see the predictive density $p(y \mid x^*)$ at that input, drawn sideways on the right:

<figure class="fig" data-dgpfit>
  <canvas data-figure="dgpfit" data-src="/aboutme/blog/deep-gp/fit.json" style="height:340px" aria-label="An exact GP and a two-layer deep GP fitted to a noisy step. Move the pointer to see the predictive density at that input."></canvas>
  <div class="controls">
    <div class="seg" role="group" aria-label="Model">
      <button type="button" data-model="dgp" class="on">Deep GP</button>
      <button type="button" data-model="gp">GP</button>
      <button type="button" data-model="hidden">Hidden layer h(x)</button>
    </div>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

- **The GP** has to choose a short length-scale to follow the jump. It rings on both plateaus, inflates the noise standard deviation to $0.111$ (the truth is $0.05$), and returns to zero within a short distance of the data.
- **The deep GP** learns a warp that is steep at $x = 0$ and flat elsewhere (**Hidden layer**), so its second layer can be smooth. The plateaus are flat, the noise estimate is right ($0.056$), and beyond the data it keeps predicting the plateau with growing uncertainty, through the skewed mixture of the animation in Section 2.
- **At the jump it does not hedge.** The two nearest data points are at $x \approx 0$ and $x \approx 0.026$. The deep GP switches from one plateau to the other inside that gap, so just outside it all its mass is on a plateau. Put the pointer at $x = 0.02$: the GP's predictive is centred at $0.15$, between the plateaus, where no data have ever been.
- **The numbers agree.** On 1000 held-out points the average log predictive density is $0.82$ for the deep GP and $0.60$ for the GP. The deep GP's ELBO is $162$ nats; it is a lower bound on its log evidence, against the GP's exact log evidence of $47$. The data prefer the deep model by a wide margin.

## 6. Is all this mathematics worth it?

Each piece of the machinery buys one specific thing:

| piece | what it buys | without it |
|---|---|---|
| composition | non-stationarity learned from data | one texture everywhere, or a hand-designed kernel |
| a GP prior on each layer | uncertainty about the warp itself, and a prior that prefers simple warps | a point-estimated warp that can overfit (deep kernel learning) |
| the ELBO with its KL terms | automatic complexity control on small data, and model comparison | early stopping and validation sets |
| the predictive mixture | skewed, heavy-tailed or multimodal uncertainty | a single Gaussian, which hedges between modes |
| inducing points and doubly stochastic VI | linear cost in $N$, minibatches, any likelihood | $O(N^3)$ per step, Gaussian likelihoods only |

Compared with the alternatives:

| | learns non-stationarity | uncertainty about the representation | predictive shape | data it suits | main risk |
|---|---|---|---|---|---|
| GP, stationary kernel | no | none needed | Gaussian | small | the wrong texture |
| GP, hand-designed non-stationary kernel | only the form you designed | none | Gaussian | small | the wrong design |
| deep kernel learning | yes | none (a point estimate) | Gaussian | medium | overfitting the marginal likelihood |
| neural-network ensembles | yes | from the spread of the members | any | large | poor uncertainty with little data |
| deep GP | yes | yes (variational or sampled) | a mixture | small to medium | harder optimisation; VI too narrow |

The costs are real. Training is slower than for a single sparse GP by roughly a factor $SL$. The optimisation has more local optima, and needs the initialisation above to avoid posterior collapse. And a variational posterior is zero-forcing: it tends to be too narrow (the [sparse GP post](/aboutme/blog/sparse-variational-gaussian-processes/), Section 2, proves this for the mean-field case). So the mathematics is worth it when three things matter at once: there is little data, the structure is non-stationary, and you will act on the uncertainty.

## 7. Where deep GPs are the right tool today

- **Surrogates of expensive simulators.** A finite-element, climate or chemistry code may be run only tens to thousands of times. Its response is often non-stationary (regimes, shocks, phase changes), and the surrogate's uncertainty drives design, calibration or the choice of the next run. This is where deep GPs are most used. Sauer, Gramacy and Higdon (2023) use them for active learning. Ming, Williamson and Guillas (2023) train them by stochastic imputation, which turns a deep GP into a chain of ordinary GPs. Booth (2025) adds gradient observations and predicts gradients, for molecular energies and forces.
- **Multi-fidelity modelling.** When a cheap, coarse simulator and an expensive, accurate one are both available, each layer can be one fidelity level, and the composition captures the non-linear relation between them (Cutajar et al., 2019).
- **Spatial statistics with non-stationary fields.** Deep compositions of injective warps model fields whose correlation length changes across space, for example satellite radiances (Zammit-Mangion et al., 2022).
- **Mappings between functions.** Stacking linear and non-linear GP layers in function space gives calibrated operator learning (Lowery et al., 2025), the setting of [the operator-kernel post](/aboutme/blog/vector-valued-kernel-ridge-regression/).
- **Sequential decisions.** Bayesian optimisation and active learning on non-stationary objectives, where better-shaped uncertainty chooses better next queries.

Where they are usually not the right tool: large data sets with plenty of compute (neural networks and ensembles), images and text (deep kernel machines, in Section 9, or neural networks), and problems where a single GP with a sensible kernel already fits.

## 8. What is new, 2018 to 2026

The doubly stochastic family has one weakness: each $q(\boldsymbol{u}_l)$ is Gaussian and independent of the other layers, so the approximate posterior is unimodal and ignores how the layers' inducing variables co-vary. Most work since attacks that, makes inference cheaper, or takes deep GPs to new settings.

| year | work | what it changes |
|---|---|---|
| 2018 | Havasi, Hernández-Lobato and Murillo-Fuentes | samples the inducing variables with stochastic-gradient HMC; the true posterior is multimodal |
| 2019 | Salimbeni, Dutordoir, Hensman and Deisenroth | latent input variables and an importance-weighted bound, for multimodal outputs |
| 2019 | Yu, Chen, Dai, Low and Jaillet | implicit (non-Gaussian) posteriors, trained adversarially |
| 2020 | Jankowiak, Pleiss and Gardner | deep sigma point processes: a learned quadrature replaces sampling |
| 2020 | Lindinger, Reeb, Lippert and Rakitsch | keeps covariances between layers, still marginalising the inducing variables in closed form |
| 2021 | Ober and Aitchison | "global" inducing points defined at the input and propagated through the layers |
| 2021 | Dutordoir et al. | neural networks as point estimates of deep GPs (Section 9) |
| 2022 | Zammit-Mangion, Ng, Vu and Filippone | deep compositional spatial models with injective warps |
| 2023 | Sauer, Gramacy and Higdon; Sauer, Cooper and Gramacy | fully Bayesian deep GP surrogates, made near-linear with the Vecchia approximation |
| 2023 | Ming, Williamson and Guillas | training by stochastic imputation, through ordinary GP fits |
| 2023–24 | Yang et al.; Milsom, Anson and Aitchison | deep kernel machines, including convolutional ones for images |
| 2024 | Meng and Zhang | amortised inference: a network maps each observation to its variational parameters |
| 2025 | Lowery et al. | deep GPs for mappings between functions |
| 2025 | Booth | gradient-enhanced deep GPs for computer experiments |
| 2026 | Xu, Zhao, Paisley and Zeng (ICLR) | diffusion-bridge variational inference: a diffusion started near the posterior |
| 2026 | Sáez-Maldonado, Maroñas and Hernández-Lobato | why posterior collapse happens, and an initialisation that prevents it |

## 9. Deep GPs, neural networks and deep kernels

### Width matters

A neural network with one hidden layer becomes a GP as its width goes to infinity (Neal, 1996), and so does a deep network, with a compositional kernel (Lee et al., 2018; Matthews et al., 2018). A Gaussian limit has a fixed kernel, so it cannot learn a representation. Deep GPs have the same limit.

:::proposition[Infinitely wide hidden layers make a deep GP shallow]
Let layer 1 have $D$ independent outputs $f_1^{(1)}, \dots, f_1^{(D)} \sim \mathcal{GP}(0, k_1)$, and let layer 2 use a squared-exponential kernel on the scaled distance, $k_2(\boldsymbol{h}, \boldsymbol{h}') = \sigma_2^2\exp\!\big(-\lVert\boldsymbol{h} - \boldsymbol{h}'\rVert^2/(2D\ell_2^2)\big)$. As $D \to \infty$, $k_2(\boldsymbol{h}(x), \boldsymbol{h}(x'))$ converges almost surely to the deterministic kernel $\sigma_2^2\exp\!\big(-s^2(x, x')/(2\ell_2^2)\big)$, with $s^2$ as in Section 2. So the output becomes an ordinary GP.
:::

:::proof
**Step 1. Write the distance as an average.** $\frac1D\lVert\boldsymbol{h}(x) - \boldsymbol{h}(x')\rVert^2 = \frac1D\sum_{d=1}^{D}\big(f_1^{(d)}(x) - f_1^{(d)}(x')\big)^2$.

**Step 2. Law of large numbers.** The $D$ terms are independent and identically distributed, each the square of a $\mathcal{N}(0, s^2)$ variable (Section 2), so each has mean $s^2$. The strong law of large numbers gives almost-sure convergence of the average to $s^2(x, x')$.

**Step 3. Continuity.** $t \mapsto \exp(-t/(2\ell_2^2))$ is continuous, so the kernel converges to its value at $t = s^2$, which is not random.

**Step 4. A fixed kernel means a GP.** Given layer 1, the output is Gaussian with this covariance. In the limit the covariance no longer depends on layer 1, so averaging over layer 1 changes nothing, and the output is a GP with that kernel.
:::

Finite width is what makes a deep GP deep. Deep kernel processes (Aitchison, Yang and Ober, 2021) and deep kernel machines (Yang et al., 2023; Milsom, Anson and Aitchison, 2024) change how the limit is taken, so that representation learning survives it, and obtain a deep generalisation of kernel methods that scales to images.

### Neural networks as deep GPs, and the other way round

- **Neural networks as point estimates.** Dutordoir et al. (2021) read activation functions as inter-domain inducing features (the inter-domain idea of the sparse post, Section 10). The posterior mean of a deep sparse GP then has the same structure as a network's forward pass, so a deep GP can be initialised and trained like a network, and a network gains calibrated uncertainty.
- **Deep kernel learning.** Wilson, Hu, Salakhutdinov and Xing (2016) use $k\big(g_\theta(\boldsymbol{x}), g_\theta(\boldsymbol{x}')\big)$ with a neural network $g_\theta$ fitted by maximising the marginal likelihood: a deep GP whose inner layers are point estimates. It is fast and often accurate. But with millions of warp parameters and no posterior over them, it can overfit the marginal likelihood badly (Ober, Rasmussen and van der Wilk, 2021).

## 10. A checklist

- **Start simple.** Fit a single GP, then one with a non-stationary kernel or warped inputs. Move to a deep GP only if the residuals or the held-out likelihood say the texture is wrong.
- **Two or three layers**, identity or linear mean functions on hidden layers, 100 to 500 inducing points per layer, whitening, and inner layers initialised nearly deterministic.
- **Monitor the ELBO and a held-out log density**, not only the error of the mean. If the hidden layers fall back to their priors, suspect posterior collapse and revisit the initialisation.
- **Look at predictive distributions**, not only means: the benefit often shows up as uncertainty with the right shape, sharp at a jump instead of smeared across it.
- **If the posterior itself matters**, as in active learning, consider sampling (Section 8): variational posteriors are unimodal and tend to be too narrow.

## Questions

:::question
A colleague says: "A deep GP is not a GP, so you cannot compute its predictive distribution." What is wrong with that?
:::

:::answer
Each layer is a GP given the layer below, so every conditional distribution is Gaussian and computable. The predictive is the average of the last layer's Gaussian over the hidden value, $p(y_*) = \mathbb{E}_{h_*}[p(y_* \mid h_*)]$, which Monte Carlo estimates without bias, with an error shrinking like $1/\sqrt{S}$. What is lost is a closed form, not computability.
:::

:::question
In the two-layer covariance, what happens as $\ell_2 \to \infty$, and why?
:::

:::answer
$s^2/\ell_2^2 \to 0$, so the covariance tends to $\sigma_2^2$ for every pair of inputs: $y(x)$ and $y(x')$ become perfectly correlated. Over the range of the hidden values, a GP with an enormous length-scale is essentially constant, so the composition is a single random constant.
:::

:::question
Every layer has a squared-exponential kernel with $\sigma_l/\ell_l = 2$. Does the typical slope of a deep composition grow or shrink with depth?
:::

:::answer
Here $r = 2$, and the mean of $\log\lvert D_L\rvert$ is $L(\log 2 - 0.635) \approx 0.058L$, so the median slope grows, very slowly. Its standard deviation still grows like $1.11\sqrt{L}$, so the slopes still spread out over orders of magnitude: flat in places, cliffs in others. Tuning $r$ moves the drift but cannot remove the spread.
:::

:::question
In the forward pass by hand, what would happen to the expected log-likelihood if $\mathbf{S}_2$ were ten times larger, and what does that tell you about training?
:::

:::answer
$\boldsymbol{a}^\top\mathbf{S}_2\boldsymbol{a}$ would grow from $0.0183$ to $0.183$, so $s_2^2 = 0.0607 + 0.183 = 0.244$ and the variance term would cost $0.244/0.02 = 12.2$ nats instead of $3.95$. The ELBO strongly rewards certainty about the inducing values where there are data; the KL term pulls $q(\boldsymbol{u}_2)$ back towards the prior, and training balances the two.
:::

:::question
The variational distribution $q(\boldsymbol{f}_l \mid \boldsymbol{f}_{l-1})$ is an $N$-dimensional Gaussian with a full covariance. Why does doubly stochastic VI never form an $N \times N$ matrix?
:::

:::answer
The likelihood factorises over points, and Steps 6 to 8 of the theorem show that the marginal of $f_{L,n}$ needs only the one-dimensional conditionals along point $n$'s own path. Those need $\mu_l$ and $s_l^2$ at one input, which cost $O(M^2)$ each. Covariances between different points never enter the objective.
:::

:::question
Show that with an identity mean the derivative of the composition is $\prod_l(1 + g_l)$, and explain why that stops the collapse.
:::

:::answer
With $h_l = h_{l-1} + f_l(h_{l-1})$, the chain rule gives $h_l'(x) = \big(1 + f_l'(h_{l-1}(x))\big)\,h_{l-1}'(x)$, so $D_L = \prod_l(1 + g_l)$. When the GP part is small, each factor is close to 1 and $\log\lvert 1 + g_l\rvert \approx g_l - g_l^2/2$ has mean about $-r^2/2$. The drift per layer is $O(r^2)$ instead of the $-0.635$ of a plain layer, which is why residual connections let networks and deep GPs grow deep.
:::

## References

- A. Damianou, N. D. Lawrence. *Deep Gaussian processes.* AISTATS, 2013. [arXiv:1211.0358](https://arxiv.org/abs/1211.0358)
- H. Salimbeni, M. P. Deisenroth. *Doubly stochastic variational inference for deep Gaussian processes.* NeurIPS, 2017. [arXiv:1705.08933](https://arxiv.org/abs/1705.08933)
- D. Duvenaud, O. Rippel, R. P. Adams, Z. Ghahramani. *Avoiding pathologies in very deep networks.* AISTATS, 2014. [arXiv:1402.5836](https://arxiv.org/abs/1402.5836)
- C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning.* MIT Press, 2006.
- M. N. Gibbs. *Bayesian Gaussian processes for regression and classification.* PhD thesis, University of Cambridge, 1997.
- P. D. Sampson, P. Guttorp. *Nonparametric estimation of nonstationary spatial covariance structure.* JASA, 1992.
- J. Snoek, K. Swersky, R. Zemel, R. P. Adams. *Input warping for Bayesian optimization of non-stationary functions.* ICML, 2014.
- M. Havasi, J. M. Hernández-Lobato, J. J. Murillo-Fuentes. *Inference in deep Gaussian processes using stochastic gradient Hamiltonian Monte Carlo.* NeurIPS, 2018.
- H. Salimbeni, V. Dutordoir, J. Hensman, M. P. Deisenroth. *Deep Gaussian processes with importance-weighted variational inference.* ICML, 2019.
- H. Yu, Y. Chen, Z. Dai, B. K. H. Low, P. Jaillet. *Implicit posterior variational inference for deep Gaussian processes.* NeurIPS, 2019.
- M. Jankowiak, G. Pleiss, J. R. Gardner. *Deep sigma point processes.* UAI, 2020.
- J. Lindinger, D. Reeb, C. Lippert, B. Rakitsch. *Beyond the mean-field: structured deep Gaussian processes improve the predictive uncertainties.* NeurIPS, 2020. [arXiv:2005.11110](https://arxiv.org/abs/2005.11110)
- S. W. Ober, L. Aitchison. *Global inducing point variational posteriors for Bayesian neural networks and deep Gaussian processes.* ICML, 2021. [PMLR](https://proceedings.mlr.press/v139/ober21a.html)
- V. Dutordoir, J. Hensman, M. van der Wilk, C. H. Ek, Z. Ghahramani, N. Durrande. *Deep neural networks as point estimates for deep Gaussian processes.* NeurIPS, 2021. [arXiv:2105.04504](https://arxiv.org/abs/2105.04504)
- A. Zammit-Mangion, T. L. J. Ng, Q. Vu, M. Filippone. *Deep compositional spatial models.* JASA, 2022.
- A. Sauer, R. B. Gramacy, D. Higdon. *Active learning for deep Gaussian process surrogates.* Technometrics, 2023.
- A. Sauer, A. Cooper, R. B. Gramacy. *Vecchia-approximated deep Gaussian processes for computer experiments.* JCGS, 2023. [arXiv:2204.02904](https://arxiv.org/abs/2204.02904)
- D. Ming, D. Williamson, S. Guillas. *Deep Gaussian process emulation using stochastic imputation.* Technometrics, 2023. [arXiv:2107.01590](https://arxiv.org/abs/2107.01590)
- K. Cutajar, M. Pullin, A. Damianou, N. D. Lawrence, J. González. *Deep Gaussian processes for multi-fidelity modeling.* 2019. [arXiv:1903.07320](https://arxiv.org/abs/1903.07320)
- Q. Meng, Y. Zhang. *Amortized variational inference for deep Gaussian processes.* 2024. [arXiv:2409.12301](https://arxiv.org/abs/2409.12301)
- M. Lowery, Z. Xu, D. Long, K. Chen, D. S. Johnson, Y. Bai, V. Shankar, S. Zhe. *Deep Gaussian processes for functional maps.* 2025. [arXiv:2510.22068](https://arxiv.org/abs/2510.22068)
- A. S. Booth. *Gradient-enhancement and gradient predictions for deep Gaussian process modeling of expensive computer experiments.* 2025. [arXiv:2512.18066](https://arxiv.org/abs/2512.18066)
- J. Xu, Q. Zhao, J. Paisley, D. Zeng. *Diffusion bridge variational inference for deep Gaussian processes.* ICLR, 2026. [arXiv:2509.19078](https://arxiv.org/abs/2509.19078)
- F. J. Sáez-Maldonado, J. Maroñas, D. Hernández-Lobato. *An analysis of posterior collapse, parameterization and initialization in variational deep Gaussian processes.* 2026. [arXiv:2606.25882](https://arxiv.org/abs/2606.25882)
- R. M. Neal. *Bayesian Learning for Neural Networks.* Springer, 1996.
- J. Lee, Y. Bahri, R. Novak, S. S. Schoenholz, J. Pennington, J. Sohl-Dickstein. *Deep neural networks as Gaussian processes.* ICLR, 2018.
- A. G. de G. Matthews, M. Rowland, J. Hron, R. E. Turner, Z. Ghahramani. *Gaussian process behaviour in wide deep neural networks.* ICLR, 2018.
- L. Aitchison, A. X. Yang, S. W. Ober. *Deep kernel processes.* ICML, 2021.
- A. X. Yang, M. Robeyns, E. Milsom, B. Anson, N. Schoots, L. Aitchison. *A theory of representation learning gives a deep generalisation of kernel methods.* ICML, 2023. [PMLR](https://proceedings.mlr.press/v202/yang23k.html)
- E. Milsom, B. Anson, L. Aitchison. *Convolutional deep kernel machines.* ICLR, 2024.
- A. G. Wilson, Z. Hu, R. Salakhutdinov, E. P. Xing. *Deep kernel learning.* AISTATS, 2016.
- S. W. Ober, C. E. Rasmussen, M. van der Wilk. *The promises and pitfalls of deep kernel learning.* UAI, 2021.
