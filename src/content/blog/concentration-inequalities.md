---
title: "Concentration inequalities: how far can an average stray?"
description: "Markov, Chebyshev, Chernoff and Hoeffding from one picture, with step-by-step proofs, interactive figures, and the places they decide real numbers: sample sizes, generalisation bounds, random features and high-dimensional geometry."
date: 2025-12-07
updated: 2026-10-26
cover: /blog/thumbs/concentration.svg
tags: [probability, inequalities, learning-theory]
---

Almost every number in machine learning is an average: a test accuracy, a Monte Carlo estimate, a minibatch gradient, a kernel approximated with random features. Each comes with the same two questions. **How far from the truth can it be, and how many samples make that unlikely?** Concentration inequalities answer both, without knowing the distribution.

Below, each sample is the mean of $n$ fair coin flips. The left panel shows its distribution, with the outcomes more than $\varepsilon$ from $\tfrac12$ highlighted. The right panel shows the probability of such a miss as $n$ grows, on a log scale: the exact value, and two bounds that hold for *any* variables bounded in $[0, 1]$.

<figure class="fig" data-conc>
  <canvas data-figure="conc" style="height:320px" aria-label="Distribution of a sample mean and its tail probability against Chebyshev and Hoeffding bounds."></canvas>
  <figcaption>
    <label class="slider">n <input type="range" min="0.5" max="3.17" step="0.01" value="2" data-n /></label>
    <label class="slider">ε <input type="range" min="0.02" max="0.3" step="0.01" value="0.1" data-eps /></label>
    <span class="mono" data-out></span>
  </figcaption>
</figure>

On a log scale the exact tail is a straight line falling at a constant rate: it decays **exponentially** in $n$. Hoeffding's bound is parallel to it, which is why it is useful. Chebyshev's bound bends: it decays only like $1/n$, and by $n = 1000$ it is wrong by many orders of magnitude. The difference between those two shapes is the subject of this post.

## 1. The one idea behind all of them

Each inequality below is the same move with a different function.

:::lemma[Bounding an indicator]
Let $A$ be an event defined by $X$, and let $g \ge 0$ with $g(x) \ge 1$ whenever $x \in A$. Then $\;\mathbb{P}(X \in A) \le \mathbb{E}[g(X)]$.
:::

:::proof
**Step 1. Write the probability as an expectation.** $\mathbb{P}(X \in A) = \mathbb{E}[\mathbb{1}\{X \in A\}]$.

**Step 2. Compare pointwise.** On $A$ the indicator is $1 \le g(x)$; off $A$ it is $0 \le g(x)$. So $\mathbb{1}\{x \in A\} \le g(x)$ for every $x$.

**Step 3. Take expectations.** Expectation preserves pointwise inequalities: $\mathbb{E}[\mathbb{1}\{X \in A\}] \le \mathbb{E}[g(X)]$.
:::

So a tail bound is a choice of a function lying above a step. The closer it hugs the step *where the distribution has mass*, the tighter the bound. Below, $X$ is the number of heads in 20 fair flips and the event is $X \ge a$. Markov uses a line, Chebyshev a parabola, Chernoff an exponential whose steepness $t$ you choose.

<figure class="fig" data-bounds>
  <canvas data-figure="bounds" style="height:330px" aria-label="Markov, Chebyshev and Chernoff functions lying above the indicator of a tail event."></canvas>
  <figcaption>
    <label class="slider">threshold a <input type="range" min="11" max="20" step="1" value="14" data-a /></label>
    <label class="slider">Chernoff t <input type="range" min="0.02" max="2" step="0.01" value="0.4" data-t /></label>
    <span class="mono" data-out></span>
  </figcaption>
</figure>

The line and the parabola stay well above zero across the bulk of the distribution, where the probability mass sits, and pay for it. The exponential can be made almost zero on the bulk and still reach $1$ at the threshold. Push $a$ towards 20: Markov barely moves, Chebyshev improves, Chernoff keeps pace with the truth.

## 2. Markov: knowing only the mean

:::theorem[Markov's inequality]
If $X \ge 0$ and $a > 0$, then $\;\mathbb{P}(X \ge a) \le \dfrac{\mathbb{E}[X]}{a}$.
:::

:::proof
**Step 1. Choose $g$.** Let $g(x) = x/a$. It is non-negative because $X \ge 0$, and $g(x) \ge 1$ whenever $x \ge a$.

**Step 2. Apply Lemma 1.** $\mathbb{P}(X \ge a) \le \mathbb{E}[X/a] = \mathbb{E}[X]/a$.
:::

:::example[Markov cannot be improved without more information]
Let $X = a$ with probability $p$ and $X = 0$ otherwise. Then $\mathbb{E}[X] = pa$ and $\mathbb{P}(X \ge a) = p = \mathbb{E}[X]/a$: equality. Any distribution with all its tail mass exactly at $a$ makes Markov tight, so a better bound needs more than the mean.
:::

:::corollary[Bounded above instead of below]
If $X \le U$ and $a < U$, then $\;\mathbb{P}(X \le a) \le \dfrac{U - \mathbb{E}[X]}{U - a}$.
:::

:::proof
**Step 1. Flip.** $Y = U - X \ge 0$, and $X \le a$ exactly when $Y \ge U - a$.

**Step 2. Apply Theorem 1 to $Y$.** $\mathbb{P}(Y \ge U - a) \le \mathbb{E}[Y]/(U - a) = (U - \mathbb{E}[X])/(U - a)$.
:::

For example, if a classifier's accuracy, averaged over a family of tasks, is $0.9$, then (since accuracy is at most $1$) at most $\tfrac{1 - 0.9}{1 - 0.5} = 20\%$ of those tasks can have accuracy below $0.5$.

## 3. Chebyshev: knowing the variance

:::theorem[Chebyshev's inequality]
If $\operatorname{Var}[X] = \sigma^2 < \infty$ and $\varepsilon > 0$, then $\;\mathbb{P}(\lvert X - \mathbb{E}[X]\rvert \ge \varepsilon) \le \dfrac{\sigma^2}{\varepsilon^2}$.
:::

:::proof
**Step 1. Choose $g$.** $g(x) = (x - \mathbb{E}[X])^2/\varepsilon^2 \ge 0$, and $g(x) \ge 1$ whenever $\lvert x - \mathbb{E}[X]\rvert \ge \varepsilon$.

**Step 2. Apply Lemma 1.** $\mathbb{P}(\lvert X - \mathbb{E}X\rvert \ge \varepsilon) \le \mathbb{E}[(X - \mathbb{E}X)^2]/\varepsilon^2 = \sigma^2/\varepsilon^2$.
:::

Applied to an average, Chebyshev already proves that averages work.

:::proposition[Weak law of large numbers, with a sample size]
Let $X_1, \dots, X_n$ be independent with mean $\mu$ and variance $\sigma^2$, and $\bar X = \tfrac1n\sum_iX_i$. Then $\mathbb{P}(\lvert\bar X - \mu\rvert \ge \varepsilon) \le \dfrac{\sigma^2}{n\varepsilon^2}$, so $n \ge \dfrac{\sigma^2}{\delta\varepsilon^2}$ samples give $\mathbb{P}(\lvert\bar X - \mu\rvert \ge \varepsilon) \le \delta$.
:::

:::proof
**Step 1. Mean of the average.** By linearity, $\mathbb{E}[\bar X] = \tfrac1n\sum_i\mu = \mu$.

**Step 2. Variance of a sum of independent variables.** Independence makes every covariance $\operatorname{Cov}[X_i, X_j]$ with $i \ne j$ vanish, so $\operatorname{Var}[\sum_iX_i] = \sum_i\operatorname{Var}[X_i] = n\sigma^2$.

**Step 3. Scale.** $\operatorname{Var}[\bar X] = \tfrac{1}{n^2}\cdot n\sigma^2 = \sigma^2/n$.

**Step 4. Apply Theorem 2** to $\bar X$, then solve $\sigma^2/(n\varepsilon^2) \le \delta$ for $n$.
:::

The weakness is the $1/\delta$. Asking for ten times more confidence costs ten times more samples. The figure at the top shows why this is pessimistic: the true tail falls exponentially. To see that, the bound must use more than two moments.

## 4. Chernoff: using every moment at once

:::theorem[Chernoff bound]
For any $X$ and $a$, $\;\mathbb{P}(X \ge a) \le \inf_{t > 0}\,e^{-ta}\,\mathbb{E}[e^{tX}]$.
:::

:::proof
**Step 1. Choose $g$.** For any $t > 0$, $g(x) = e^{t(x - a)} > 0$, and $g(x) \ge 1$ whenever $x \ge a$, because $t(x - a) \ge 0$ there.

**Step 2. Apply Lemma 1.** $\mathbb{P}(X \ge a) \le \mathbb{E}[e^{t(X - a)}] = e^{-ta}\,\mathbb{E}[e^{tX}]$.

**Step 3. Optimise.** Step 2 holds for every $t > 0$, so it holds for the infimum.
:::

$M(t) = \mathbb{E}[e^{tX}]$ is the moment generating function, $\sum_k t^k\mathbb{E}[X^k]/k!$: one function that packs every moment, weighted by $t$. Two properties make it the right tool for averages.

:::lemma[MGFs of independent sums multiply]
If $X_1, \dots, X_n$ are independent, $\;\mathbb{E}\big[e^{t\sum_iX_i}\big] = \prod_i\mathbb{E}[e^{tX_i}]$.
:::

:::proof
**Step 1.** $e^{t\sum_iX_i} = \prod_ie^{tX_i}$.

**Step 2.** The factors are functions of independent variables, so they are independent, and the expectation of a product of independent variables is the product of the expectations.
:::

:::example[The Gaussian tail, by Chernoff]
For $X \sim \mathcal{N}(0, \sigma^2)$, the MGF is $\mathbb{E}[e^{tX}] = e^{\sigma^2t^2/2}$ (complete the square in the integral; see [Gaussian identities](/aboutme/blog/gaussian-identities/), Proposition 3, Step 1).

**Step 1. The bound for fixed $t$.** $\mathbb{P}(X \ge a) \le e^{-ta + \sigma^2t^2/2}$.

**Step 2. Minimise the exponent.** $\tfrac{d}{dt}\big(-ta + \sigma^2t^2/2\big) = -a + \sigma^2t = 0$ at $t^* = a/\sigma^2$.

**Step 3. Substitute.** $-a^2/\sigma^2 + a^2/(2\sigma^2) = -a^2/(2\sigma^2)$, so $\mathbb{P}(X \ge a) \le e^{-a^2/(2\sigma^2)}$.

The exact tail is $e^{-a^2/(2\sigma^2)}$ times a factor of order $\sigma/a$, so Chernoff captures the exponential rate exactly and loses only a polynomial factor.
:::

:::definition[Sub-Gaussian variable]
$X$ with mean $\mu$ is sub-Gaussian with variance proxy $s^2$ if $\mathbb{E}[e^{t(X - \mu)}] \le e^{s^2t^2/2}$ for all $t \in \mathbb{R}$. By the example, such an $X$ has Gaussian-type tails: $\mathbb{P}(X - \mu \ge a) \le e^{-a^2/(2s^2)}$.
:::

## 5. Hoeffding: bounded variables are sub-Gaussian

Bounded variables are the common case: accuracies, losses clipped to $[0, 1]$, indicators, features with bounded range. The key fact is that boundedness alone gives a Gaussian-type MGF.

:::lemma[Hoeffding's lemma]
If $\mathbb{E}[X] = 0$ and $a \le X \le b$, then $\;\mathbb{E}[e^{tX}] \le e^{t^2(b - a)^2/8}$ for all $t$. That is, $X$ is sub-Gaussian with $s^2 = (b - a)^2/4$.
:::

:::proof
**Step 1. Convexity.** On $[a, b]$, the convex function $e^{tx}$ lies below its chord:

$$
e^{tx} \le \frac{b - x}{b - a}\,e^{ta} + \frac{x - a}{b - a}\,e^{tb}.
$$

**Step 2. Take expectations, using $\mathbb{E}[X] = 0$.**

$$
\mathbb{E}[e^{tX}] \le \frac{b}{b - a}\,e^{ta} - \frac{a}{b - a}\,e^{tb}.
$$

**Step 3. Rewrite as one exponential.** Let $\theta = -a/(b - a) \in [0, 1]$ (note $a \le 0 \le b$ because the mean is $0$) and $u = t(b - a)$. Then $ta = -\theta u$, and the right side of Step 2 equals $e^{\phi(u)}$ with

$$
\phi(u) = -\theta u + \log\big(1 - \theta + \theta e^u\big).
$$

**Step 4. Value and slope at zero.** $\phi(0) = \log 1 = 0$. Differentiating, $\phi'(u) = -\theta + \dfrac{\theta e^u}{1 - \theta + \theta e^u}$, so $\phi'(0) = -\theta + \theta = 0$.

**Step 5. Curvature is at most $\tfrac14$.** Write $q = \dfrac{\theta e^u}{1 - \theta + \theta e^u} \in [0, 1]$. Differentiating $\phi'$ once more gives $\phi''(u) = q(1 - q)$, and $q(1 - q) \le \tfrac14$ for every $q \in [0, 1]$ (its maximum is at $q = \tfrac12$).

**Step 6. Taylor's theorem with remainder.** For some $\xi$ between $0$ and $u$, $\phi(u) = \phi(0) + \phi'(0)\,u + \tfrac12\phi''(\xi)\,u^2 \le \tfrac18u^2 = \tfrac18t^2(b - a)^2$.
:::

:::theorem[Hoeffding's inequality]
Let $X_1, \dots, X_n$ be independent with $a_i \le X_i \le b_i$, and $S = \sum_iX_i$. For $s > 0$,

$$
\mathbb{P}\big(S - \mathbb{E}[S] \ge s\big) \le \exp\!\Big(-\frac{2s^2}{\sum_i(b_i - a_i)^2}\Big).
$$

For the mean of $n$ variables in $[0, 1]$: $\;\mathbb{P}(\lvert\bar X - \mu\rvert \ge \varepsilon) \le 2e^{-2n\varepsilon^2}$.
:::

:::proof
**Step 1. Chernoff** (Theorem 3) for the centred sum: $\mathbb{P}(S - \mathbb{E}S \ge s) \le e^{-ts}\,\mathbb{E}\big[e^{t(S - \mathbb{E}S)}\big]$.

**Step 2. Factorise** (Lemma 2), since $S - \mathbb{E}S = \sum_i(X_i - \mathbb{E}X_i)$ is a sum of independent terms.

**Step 3. Bound each factor** with Hoeffding's lemma: $X_i - \mathbb{E}X_i$ has mean $0$ and lies in an interval of length $b_i - a_i$. So

$$
\mathbb{P}(S - \mathbb{E}S \ge s) \le \exp\!\Big(-ts + \frac{t^2}{8}\sum_i(b_i - a_i)^2\Big).
$$

**Step 4. Optimise $t$.** Writing $c = \sum_i(b_i - a_i)^2$, the exponent $-ts + t^2c/8$ is minimised at $t = 4s/c$, where it equals $-4s^2/c + 2s^2/c = -2s^2/c$.

**Step 5. The mean.** With $b_i - a_i = 1$, $c = n$; the event $\bar X - \mu \ge \varepsilon$ is $S - \mathbb{E}S \ge n\varepsilon$, giving $e^{-2n\varepsilon^2}$. The lower tail is the same bound applied to $-X_i$; adding both tails gives the factor $2$.
:::

Solving $2e^{-2n\varepsilon^2} \le \delta$ gives the sample size that every estimate in machine learning quietly relies on:

$$
n \ge \frac{\log(2/\delta)}{2\varepsilon^2}.
$$

The dependence on the confidence is now **logarithmic**. For $\varepsilon = 0.05$ and variables in $[0, 1]$ (so $\sigma^2 \le \tfrac14$ in Chebyshev):

| confidence $1 - \delta$ | Chebyshev: $n \ge \tfrac{1}{4\delta\varepsilon^2}$ | Hoeffding: $n \ge \tfrac{\log(2/\delta)}{2\varepsilon^2}$ |
|---|---|---|
| 95% | 2 000 | 738 |
| 99% | 10 000 | 1 060 |
| 99.9% | 100 000 | 1 521 |

Each extra "9" of confidence multiplies Chebyshev's requirement by ten, but adds only a few hundred samples to Hoeffding's.

## 6. Where these numbers are used

### How big must a test set be?

A classifier's test accuracy is the mean of $n$ indicators. By Hoeffding, with $n = 1060$ test examples the measured accuracy is within $5$ points of the true accuracy with probability at least $99\%$, whatever the classifier. To claim $\pm 1$ point at the same confidence takes $25$ times more, $n \ge 26\,492$: halving the error bar always costs four times the data.

### Why a model chosen on training data still generalises

Pick the hypothesis with the lowest training error from a finite class $\mathcal{H}$. For each fixed $h$, Hoeffding controls the gap between its training error $\hat R(h)$ and true error $R(h)$; the danger is that we *chose* $h$ after seeing the data. One more tool handles that.

:::lemma[Union bound]
For any events $A_1, \dots, A_m$, $\;\mathbb{P}\big(\bigcup_jA_j\big) \le \sum_j\mathbb{P}(A_j)$.
:::

:::proof
**Step 1.** $\mathbb{1}\{\bigcup_jA_j\} \le \sum_j\mathbb{1}\{A_j\}$ pointwise: the left side is $1$ only if at least one term on the right is $1$.

**Step 2.** Take expectations.
:::

:::proposition[Generalisation bound for a finite class]
For a loss in $[0, 1]$ and $n$ independent training examples, with probability at least $1 - \delta$, simultaneously for every $h \in \mathcal{H}$,

$$
\lvert R(h) - \hat R(h)\rvert \le \sqrt{\frac{\log(2\lvert\mathcal{H}\rvert/\delta)}{2n}} .
$$
:::

:::proof
**Step 1. One hypothesis.** For a fixed $h$, $\hat R(h)$ is a mean of $n$ independent losses in $[0, 1]$ with mean $R(h)$. By Theorem 4, $\mathbb{P}(\lvert R(h) - \hat R(h)\rvert \ge \varepsilon) \le 2e^{-2n\varepsilon^2}$.

**Step 2. All hypotheses at once.** The event "some $h$ has a gap $\ge \varepsilon$" is the union over $h$. By Lemma 4 its probability is at most $2\lvert\mathcal{H}\rvert e^{-2n\varepsilon^2}$.

**Step 3. Solve for $\varepsilon$.** Setting $2\lvert\mathcal{H}\rvert e^{-2n\varepsilon^2} = \delta$ gives the stated $\varepsilon$. Because the bound holds for every $h$ simultaneously, it holds for whichever $h$ the learning algorithm picked.
:::

$\log\lvert\mathcal{H}\rvert$ is the number of nats needed to name a hypothesis. The bound says: **the gap grows only with the log of how many models you tried**, which is why trying a thousand hyperparameter settings costs little, and trying all functions costs everything. Replacing $\log\lvert\mathcal{H}\rvert$ by a capacity measure (VC dimension, Rademacher complexity) extends the argument to infinite classes.

### Random features approximate a kernel

Random Fourier features approximate a stationary kernel by an average: $\hat k(\boldsymbol{x}, \boldsymbol{x}') = \tfrac1D\sum_{j=1}^Dz_j(\boldsymbol{x})\,z_j(\boldsymbol{x}')$ with $z_j(\boldsymbol{x}) = \sqrt2\cos(\boldsymbol{\omega}_j^\top\boldsymbol{x} + b_j)$, $\boldsymbol{\omega}_j$ drawn from the kernel's spectral measure (Bochner's theorem, GP Theorem 1) and $b_j$ uniform on $[0, 2\pi]$.

:::proposition[How many random features]
For each fixed pair $\boldsymbol{x}, \boldsymbol{x}'$, $\;\mathbb{P}\big(\lvert\hat k(\boldsymbol{x}, \boldsymbol{x}') - k(\boldsymbol{x}, \boldsymbol{x}')\rvert \ge \varepsilon\big) \le 2e^{-D\varepsilon^2/8}$.
:::

:::proof
**Step 1. Each term is unbiased.** Using $2\cos\alpha\cos\beta = \cos(\alpha - \beta) + \cos(\alpha + \beta)$, the term is $\cos(\boldsymbol{\omega}^\top(\boldsymbol{x} - \boldsymbol{x}')) + \cos(\boldsymbol{\omega}^\top(\boldsymbol{x} + \boldsymbol{x}') + 2b)$. The second part averages to $0$ over uniform $b$; the first averages to $k(\boldsymbol{x}, \boldsymbol{x}')$ over $\boldsymbol{\omega}$ by Bochner's theorem.

**Step 2. Each term is bounded.** $z_j(\boldsymbol{x})z_j(\boldsymbol{x}') = 2\cos(\cdot)\cos(\cdot) \in [-2, 2]$, an interval of length $4$.

**Step 3. Hoeffding** for the mean of $D$ independent terms in an interval of length $4$: the bound of Theorem 4 with $\sum_i(b_i - a_i)^2 = 16D$ and $s = D\varepsilon$ gives $e^{-2D^2\varepsilon^2/(16D)} = e^{-D\varepsilon^2/8}$ per tail.
:::

So $D = 8\log(2/\delta)/\varepsilon^2$ features reach accuracy $\varepsilon$ at a given pair: about $4200$ for $\varepsilon = 0.1$ at $99\%$ confidence. The [wavelet post](/aboutme/blog/fourier-to-wavelets/) builds on these features.

### In high dimensions, Gaussians live on a shell

Draw $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$. The density is largest at the origin, yet as $d$ grows almost no samples are near it. Move the slider.

<figure class="fig" data-shell>
  <canvas data-figure="shell" style="height:260px" aria-label="Density of the norm of a standard Gaussian vector divided by the square root of the dimension."></canvas>
  <figcaption><label class="slider">dimension d <input type="range" min="0" max="3" step="0.01" value="1" data-d /></label> <span class="mono" data-out></span></figcaption>
</figure>

:::proposition[Norms concentrate]
For $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ and $0 < \varepsilon \le 1$, $\;\mathbb{P}\Big(\big\lvert\tfrac1d\lVert\boldsymbol{z}\rVert^2 - 1\big\rvert \ge \varepsilon\Big) \le 2e^{-d\varepsilon^2/8}$.
:::

:::proof
**Step 1. One coordinate's MGF.** For $z \sim \mathcal{N}(0, 1)$ and $t < \tfrac12$, completing the square in the Gaussian integral gives $\mathbb{E}[e^{tz^2}] = (1 - 2t)^{-1/2}$. So the centred variable $w = z^2 - 1$ has $\log\mathbb{E}[e^{tw}] = -t - \tfrac12\log(1 - 2t)$.

**Step 2. Bound it for $0 \le t \le \tfrac14$.** Expand $-\tfrac12\log(1 - 2t) = \sum_{k\ge1}(2t)^k/(2k)$. The $k = 1$ term cancels $-t$, and $1/(2k) \le \tfrac14$ for $k \ge 2$, so

$$
-t - \tfrac12\log(1 - 2t) \le \tfrac14\sum_{k\ge2}(2t)^k = \frac{t^2}{1 - 2t} \le 2t^2 .
$$

**Step 3. Upper tail by Chernoff.** The $d$ coordinates are independent, so by Lemma 2 and Step 2,

$$
\mathbb{P}\Big(\sum_i w_i \ge d\varepsilon\Big) \le e^{-td\varepsilon + 2dt^2}.
$$

Choose $t = \varepsilon/4 \le \tfrac14$: the exponent is $-d\varepsilon^2/4 + d\varepsilon^2/8 = -d\varepsilon^2/8$.

**Step 4. Lower tail.** For $t \ge 0$, $\log\mathbb{E}[e^{-tw}] = t - \tfrac12\log(1 + 2t) \le t^2$, using $\log(1 + 2t) \ge 2t - 2t^2$. Chernoff with $t = \varepsilon/2$ gives $e^{-d\varepsilon^2/4} \le e^{-d\varepsilon^2/8}$.

**Step 5. Add the two tails.**
:::

So $\lVert\boldsymbol{z}\rVert \approx \sqrt d$ with fluctuations of order one: in $d = 1000$ dimensions, essentially every sample lies within $10\%$ of the sphere of radius $\sqrt{1000}$. This is why random vectors in high dimensions are nearly orthogonal, why the Johnson–Lindenstrauss lemma can compress dimension while preserving distances, and why the noise in a diffusion model is a thin shell, not a ball.

## 7. Beyond Hoeffding

Hoeffding uses only the range. Two refinements are worth knowing by name; their proofs follow the same Chernoff pattern.

- **Bernstein's inequality** uses the variance too. If the $X_i$ are independent with mean $0$, $\lvert X_i\rvert \le c$ and $\sum_i\operatorname{Var}[X_i] = v$, then $\mathbb{P}(\sum_iX_i \ge s) \le \exp\!\big(-\tfrac{s^2}{2(v + cs/3)}\big)$. When the variance is much smaller than the range suggests (rare events), it is far tighter.
- **McDiarmid's inequality** replaces the sum by any function $f(X_1, \dots, X_n)$ that changes by at most $c_i$ when only $X_i$ changes: $\mathbb{P}(f - \mathbb{E}f \ge s) \le \exp\!\big(-2s^2/\sum_ic_i^2\big)$. With $f$ a sum it is Hoeffding; with $f$ a supremum over hypotheses it is the start of Rademacher complexity.
- **Matrix concentration** (Tropp, 2015) extends these bounds to sums of random matrices, which is what controls random-feature and sketching approximations of whole Gram matrices rather than single entries.

## Questions

:::question
In the indicator figure, why does Markov's bound barely improve as $a$ moves towards 20, while Chernoff's tracks the truth?
:::

:::answer
Markov's function $x/a$ is a line through the origin; at $x = \mu = 10$ it still equals $10/a \ge \tfrac12$, so most of the probability mass contributes heavily. Chernoff's $e^{t(x - a)}$ is exponentially small on the bulk for large $t$, so only mass near the tail contributes.
:::

:::question
Chebyshev needs $1/\delta$ samples, Hoeffding only $\log(1/\delta)$. Is Chebyshev ever the better choice?
:::

:::answer
Yes, when the variables are unbounded and only a variance is known (heavy-tailed data, where the MGF may not exist and Chernoff-type bounds fail), or when the variance is far below the square of the range, at moderate $\delta$.
:::

:::question
You tune 50 hyperparameter settings and report the best validation accuracy, measured on $n = 2000$ examples. How much optimism should you expect at 95% confidence?
:::

:::answer
By Proposition 2 with $\lvert\mathcal{H}\rvert = 50$: $\sqrt{\log(2\cdot 50/0.05)/(2\cdot 2000)} = \sqrt{7.6/4000} \approx 0.044$, about 4.4 points. A fresh test set removes it.
:::

## References

- S. Boucheron, G. Lugosi, P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
- R. Vershynin. *High-Dimensional Probability*. Cambridge University Press, 2018.
- M. J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019.
- W. Hoeffding. *Probability inequalities for sums of bounded random variables*. Journal of the American Statistical Association, 1963.
- H. Chernoff. *A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations*. Annals of Mathematical Statistics, 1952.
- A. Rahimi, B. Recht. *Random features for large-scale kernel machines*. NeurIPS 2007.
- J. A. Tropp. *An Introduction to Matrix Concentration Inequalities*. Foundations and Trends in Machine Learning, 2015.
