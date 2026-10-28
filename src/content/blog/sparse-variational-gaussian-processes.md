---
title: "Sparse variational Gaussian processes: Titsias, Hensman and what came after"
description: "How a Gaussian process scales from thousands to millions of points without changing the model. Variational inference from scratch, inducing points, Titsias's collapsed bound, Hensman's stochastic version, how many inducing points you need, and the 2025 bounds that go beyond p(f | u). Every step derived."
date: 2026-10-29
updated: 2026-10-29
cover: /blog/thumbs/svgp.png
tags: [gaussian-processes, variational-inference, sparse-gp, tutorial]
---

An exact Gaussian process has to factorise an $N \times N$ matrix. That is fine for a thousand points, slow for ten thousand, and impossible for a million. Sparse variational GPs keep the model exactly as it is and approximate only the *posterior*, through the values of the function at $M \ll N$ chosen inputs, the inducing points. The cost drops from $O(N^3)$ to $O(NM^2)$, or to $O(M^3)$ per minibatch step. The approximation also comes with a number, a lower bound on the log evidence, that says how much was lost.

Drag the diamonds, change $M$, or press **Greedy** and then **Optimise Z**:

<figure class="fig" data-svgp>
  <canvas data-figure="svgp" style="height:390px" aria-label="Exact GP and sparse variational GP on the same data. Drag the inducing points; a panel on the right compares the exact log evidence with three bounds."></canvas>
  <div class="controls">
    <label class="slider">inducing points M <input type="range" min="1" max="24" step="1" value="6" data-m /></label>
    <span class="mono" data-mval>6</span>
    <button type="button" data-greedy>Greedy</button>
    <button type="button" data-opt>Optimise Z</button>
    <button type="button" data-spread>Spread evenly</button>
  </div>
  <figcaption class="mono" data-out></figcaption>
</figure>

*What to try.* With six evenly spaced inducing points the sparse fit (teal) is far too smooth, and the bound $\mathcal{F}$ sits about a hundred nats below the exact log evidence. The grey bars under the plot show, at every training input, how much prior variance the inducing points fail to explain; the bound pays for each bar. Raise $M$: by $M = 14$ the sparse fit is indistinguishable from the exact one and $\mathcal{F}$ is within two nats of the exact log evidence. The upper bound $\mathcal{U}$ closes in more slowly, because it replaces the unexplained covariance by its trace (Section 6). **Greedy** places points where the bars are tallest, and **Optimise Z** then climbs $\mathcal{F}$ by moving them; at $M = 14$ the pair brings the gap below half a nat. Finally, drag one point out of the data into the empty stretch near $x = 0.5$: the bars grow where it left, and the gap jumps from about 1.6 to 7.6 nats.

This is the second post in the Gaussian-process series, after [Gaussian processes, from first principles](/aboutme/blog/understanding-gaussian-processes/). It re-derives everything it uses, apart from three Gaussian facts proved in that post (its Lemmas 5, 6 and 8) and in [the Gaussian toolkit](/aboutme/blog/gaussian-identities/): conditioning a joint Gaussian, the Woodbury identity with the matrix determinant lemma, and the marginal and posterior of a linear-Gaussian model. Proofs are folded and written as numbered steps.

The plan:

1. what an exact GP costs, and why that bill is paid again at every training step;
2. variational inference from scratch: the KL divergence, the ELBO and why it underestimates variance;
3. inducing points, and what the Nyström matrix $\mathbf{Q}_{ff}$ keeps and loses;
4. two ways to approximate: change the prior (DTC, FITC) or change the posterior (variational);
5. Titsias's collapsed bound, derived and read three ways;
6. an upper bound, so you know how far from the truth you are;
7. predictions and a numerically stable implementation;
8. Hensman's stochastic version: minibatches, non-Gaussian likelihoods, whitening and natural gradients;
9. how many inducing points you need, and where to put them;
10. a map of the field after 2013;
11. the 2025 bounds that relax $q(\boldsymbol{f} \mid \boldsymbol{u}) = p(\boldsymbol{f} \mid \boldsymbol{u})$;
12. a practical recipe.

## Notation

Plain italic letters are scalars, bold italic lower case are column vectors, bold upright capitals are matrices. In this post $\boldsymbol{f}$ is shorthand for $\boldsymbol{f}_{\mathbf{X}}$, the latent function values at the training inputs (the primer's notation).

| Symbol | Meaning |
|---|---|
| $\mathbf{X} \in \mathbb{R}^{N\times D}$, $\boldsymbol{y} \in \mathbb{R}^N$ | training inputs (rows $\boldsymbol{x}_i^\top$), observations |
| $\boldsymbol{f} = f(\mathbf{X}) \in \mathbb{R}^N$ | latent values at the training inputs |
| $\sigma_n^2$ | noise variance; $\boldsymbol{y} = \boldsymbol{f} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2\mathbf{I})$ |
| $M$, $\mathbf{Z} \in \mathbb{R}^{M\times D}$ | number of inducing points, inducing inputs (rows $\boldsymbol{z}_j^\top$) |
| $\boldsymbol{u} = f(\mathbf{Z}) \in \mathbb{R}^M$ | inducing variables |
| $\mathbf{K}_{ff}$, $\mathbf{K}_{fu}$, $\mathbf{K}_{uu}$ | $k(\mathbf{X}, \mathbf{X})$, $k(\mathbf{X}, \mathbf{Z})$, $k(\mathbf{Z}, \mathbf{Z})$; $\mathbf{K}_{uf} = \mathbf{K}_{fu}^\top$ |
| $\mathbf{A} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}$ | interpolation weights, $N \times M$, rows $\boldsymbol{a}_i^\top$ |
| $\mathbf{Q}_{ff} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}$ | Nyström approximation of $\mathbf{K}_{ff}$, entries $q_{ij}$ |
| $\mathbf{C} = \mathbf{K}_{ff} - \mathbf{Q}_{ff}$, $c_i = k_{ii} - q_{ii}$ | covariance of $\boldsymbol{f}$ given $\boldsymbol{u}$, its diagonal |
| $t = \operatorname{tr}\mathbf{C} = \sum_i c_i$ | total prior variance the inducing variables leave unexplained |
| $q(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$ | variational distribution over $\boldsymbol{u}$ |
| $\boldsymbol{\Sigma} = (\mathbf{K}_{uu} + \sigma_n^{-2}\mathbf{K}_{uf}\mathbf{K}_{fu})^{-1}$ | appears in the optimal $q^*(\boldsymbol{u})$ |
| $\mathcal{L}(q)$, $\mathcal{F}$, $\mathcal{U}$ | ELBO of a given $q$; collapsed (Titsias) bound; upper bound |
| $\mathrm{KL}[q \,\Vert\, p]$ | Kullback–Leibler divergence |

$\mathcal{N}(\boldsymbol{a} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ is the Gaussian density in $\boldsymbol{a}$. For symmetric matrices, $\mathbf{P} \succeq \mathbf{R}$ means $\mathbf{P} - \mathbf{R}$ is positive semi-definite.

## 1. The exact GP and its bill

With a zero-mean prior $f \sim \mathcal{GP}(0, k)$ and Gaussian noise, the [primer](/aboutme/blog/understanding-gaussian-processes/) derives two formulas. The log marginal likelihood (the *evidence*) is

$$
\log p(\boldsymbol{y}) = -\tfrac12\,\boldsymbol{y}^\top(\mathbf{K}_{ff} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y} - \tfrac12\log\lvert\mathbf{K}_{ff} + \sigma_n^2\mathbf{I}\rvert - \tfrac N2\log 2\pi ,
$$

and the predictive distribution at a test input $\boldsymbol{x}_*$ has mean $\boldsymbol{k}_*^\top(\mathbf{K}_{ff} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y}$ and variance $k_{**} - \boldsymbol{k}_*^\top(\mathbf{K}_{ff} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{k}_*$.

Both need the Cholesky factor of the $N \times N$ matrix $\mathbf{K}_{ff} + \sigma_n^2\mathbf{I}$: about $N^3/3$ floating-point operations and $N^2$ numbers of memory. In double precision:

| $N$ | memory for $\mathbf{K}_{ff}$ | Cholesky flops | in practice |
|---|---|---|---|
| $10^3$ | 8 MB | $3\times10^{8}$ | instant |
| $10^4$ | 0.8 GB | $3\times10^{11}$ | seconds |
| $10^5$ | 80 GB | $3\times10^{14}$ | needs a large GPU cluster |
| $10^6$ | 8 TB | $3\times10^{17}$ | out of reach |

Learning the kernel hyperparameters $\boldsymbol{\theta}$ (length-scales, amplitude, noise) means maximising $\log p(\boldsymbol{y})$ by gradient ascent, and every step repeats the factorisation. So the target is a method that

- costs $O(N)$ in the number of data points, or less per step;
- keeps the *same model*, so that the hyperparameters we learn mean what they meant before;
- reports how much accuracy it gave up.

Variational inference with inducing points does all three.

## 2. Variational inference from scratch

### The problem it solves

Write $\boldsymbol{z}$ for whatever is unknown and $\boldsymbol{y}$ for the data. Bayes' rule gives the posterior

$$
p(\boldsymbol{z} \mid \boldsymbol{y}) = \frac{p(\boldsymbol{y} \mid \boldsymbol{z})\,p(\boldsymbol{z})}{p(\boldsymbol{y})}, \qquad p(\boldsymbol{y}) = \int p(\boldsymbol{y} \mid \boldsymbol{z})\,p(\boldsymbol{z})\,d\boldsymbol{z} .
$$

Often the posterior is intractable because the integral has no closed form. For GP regression it has one, but it costs $O(N^3)$. Variational inference (VI) turns both cases into optimisation: pick a family $\mathcal{Q}$ of distributions we can handle, and find the member closest to the posterior.

"Closest" needs a notion of distance between distributions.

:::definition[Kullback–Leibler divergence]
For densities $q$ and $p$ on the same space, with $p > 0$ wherever $q > 0$,

$$
\mathrm{KL}[q \,\Vert\, p] = \int q(\boldsymbol{z})\log\frac{q(\boldsymbol{z})}{p(\boldsymbol{z})}\,d\boldsymbol{z} = \mathbb{E}_{q}\big[\log q(\boldsymbol{z}) - \log p(\boldsymbol{z})\big].
$$

The integral runs over the set where $q > 0$. If $p = 0$ somewhere $q > 0$, the divergence is $+\infty$.
:::

It measures, on average under $q$, how much more probable $q$ thinks a point is than $p$ does. It is not symmetric: $\mathrm{KL}[q \,\Vert\, p] \neq \mathrm{KL}[p \,\Vert\, q]$ in general. But it is never negative.

:::proposition[Gibbs' inequality]
$\mathrm{KL}[q \,\Vert\, p] \geq 0$, with equality if and only if $q = p$ almost everywhere.
:::

:::proof
**Step 1. Flip the ratio.** On the set $\{q > 0\}$,

$$
-\mathrm{KL}[q \,\Vert\, p] = \mathbb{E}_q\Big[\log\frac{p(\boldsymbol{z})}{q(\boldsymbol{z})}\Big].
$$

**Step 2. Jensen's inequality.** The logarithm is concave, so $\mathbb{E}[\log W] \leq \log \mathbb{E}[W]$ for any positive random variable $W$. With $W = p(\boldsymbol{z})/q(\boldsymbol{z})$ and $\boldsymbol{z} \sim q$:

$$
\mathbb{E}_q\Big[\log\frac{p}{q}\Big] \leq \log\mathbb{E}_q\Big[\frac{p}{q}\Big] = \log\int_{q>0} q\,\frac{p}{q}\,d\boldsymbol{z} = \log\int_{q>0} p\,d\boldsymbol{z}.
$$

**Step 3. Bound the last integral.** $p$ is a density, so its integral over any set is at most 1, and $\log$ of a number at most 1 is at most 0. Hence $-\mathrm{KL} \leq 0$.

**Step 4. Equality.** Jensen's inequality for a strictly concave function is an equality only if $W$ is constant $q$-almost everywhere, say $p = \lambda q$. Step 3 is an equality only if $\int_{q>0} p = 1$. Together these force $\lambda = 1$, so $p = q$ almost everywhere.
:::

Now the central identity of variational inference.

:::theorem[The evidence decomposition]
For any density $q(\boldsymbol{z})$ that is positive wherever the posterior is,

$$
\log p(\boldsymbol{y}) = \underbrace{\mathbb{E}_{q}\big[\log p(\boldsymbol{y}, \boldsymbol{z}) - \log q(\boldsymbol{z})\big]}_{\mathcal{L}(q),\ \text{the ELBO}} + \mathrm{KL}\big[q(\boldsymbol{z}) \,\Vert\, p(\boldsymbol{z} \mid \boldsymbol{y})\big].
$$
:::

:::proof
**Step 1. The product rule, pointwise.** For every $\boldsymbol{z}$, $p(\boldsymbol{y}, \boldsymbol{z}) = p(\boldsymbol{z} \mid \boldsymbol{y})\,p(\boldsymbol{y})$, so

$$
\log p(\boldsymbol{y}) = \log p(\boldsymbol{y}, \boldsymbol{z}) - \log p(\boldsymbol{z} \mid \boldsymbol{y}).
$$

**Step 2. Add and subtract $\log q(\boldsymbol{z})$.**

$$
\log p(\boldsymbol{y}) = \big[\log p(\boldsymbol{y}, \boldsymbol{z}) - \log q(\boldsymbol{z})\big] + \big[\log q(\boldsymbol{z}) - \log p(\boldsymbol{z} \mid \boldsymbol{y})\big].
$$

**Step 3. Average over $q$.** The left side does not depend on $\boldsymbol{z}$, so its expectation under $q$ is itself. The expectation of the first bracket is $\mathcal{L}(q)$; the expectation of the second is the KL divergence by definition.
:::

Three consequences follow at once, and the rest of the post uses all of them.

- **It is a lower bound.** By Gibbs' inequality the KL term is non-negative, so $\mathcal{L}(q) \leq \log p(\boldsymbol{y})$: the name ELBO stands for *evidence lower bound*.
- **Maximising it is the right goal.** $\log p(\boldsymbol{y})$ does not depend on $q$. So raising $\mathcal{L}(q)$ by some amount lowers $\mathrm{KL}[q \,\Vert\, p(\cdot \mid \boldsymbol{y})]$ by exactly that amount.
- **It can replace the evidence when learning hyperparameters.** When the model has hyperparameters $\boldsymbol{\theta}$, maximising $\mathcal{L}(q, \boldsymbol{\theta})$ jointly over $q$ and $\boldsymbol{\theta}$ approximates type-II maximum likelihood. The gap is the KL term, so the approximation is good when the KL term is small.

:::corollary[The ELBO as fit minus complexity]
$$
\mathcal{L}(q) = \mathbb{E}_{q}\big[\log p(\boldsymbol{y} \mid \boldsymbol{z})\big] - \mathrm{KL}\big[q(\boldsymbol{z}) \,\Vert\, p(\boldsymbol{z})\big].
$$
:::

:::proof
**Step 1.** Split the joint: $\log p(\boldsymbol{y}, \boldsymbol{z}) = \log p(\boldsymbol{y} \mid \boldsymbol{z}) + \log p(\boldsymbol{z})$.

**Step 2.** Substitute into $\mathcal{L}(q)$ and group the last two terms:

$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(\boldsymbol{y} \mid \boldsymbol{z})] + \mathbb{E}_q[\log p(\boldsymbol{z}) - \log q(\boldsymbol{z})] = \mathbb{E}_q[\log p(\boldsymbol{y} \mid \boldsymbol{z})] - \mathrm{KL}[q \,\Vert\, p(\boldsymbol{z})].
$$
:::

The first term rewards explaining the data; the second penalises moving away from the prior. That reading is the same one the GP evidence has (data fit against complexity), and it will reappear in the inducing-point bound.

:::remark[Why KL(q ‖ p) and not KL(p ‖ q)]
We minimise $\mathrm{KL}[q \,\Vert\, p(\cdot \mid \boldsymbol{y})]$, an expectation under $q$, because we can sample from $q$ and evaluate it, but not the posterior. The choice has a consequence. Where the posterior is tiny and $q$ is not, $\log(q/p)$ is huge, so the optimum keeps $q$ small wherever $p$ is small. The resulting $q$ is *zero-forcing*: it sits inside the posterior and tends to be too narrow. The reverse divergence $\mathrm{KL}[p \,\Vert\, q]$ (used by expectation propagation) behaves the opposite way and covers all of $p$.
:::

The next example makes "too narrow" exact. It is also the clearest preview of the 2025 bound in Section 11.

:::example[Mean-field VI underestimates variance]
Let the target be $p = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ in $D$ dimensions, with precision $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$. Approximate it with a fully factorised Gaussian $q = \mathcal{N}(\boldsymbol{m}, \operatorname{diag}(s_1, \dots, s_D))$ by minimising $\mathrm{KL}[q \,\Vert\, p]$. The optimum is $\boldsymbol{m} = \boldsymbol{\mu}$ and $s_d = 1/\Lambda_{dd}$, and $1/\Lambda_{dd} \leq \Sigma_{dd}$.

For two variables with unit variances and correlation $0.9$, $\Lambda_{11} = 1/(1 - 0.81) \approx 5.26$. So $q$ has standard deviation $0.44$ where the truth has $1$. The approximation matches the *conditional* variance of each coordinate given the others, not its marginal variance.
:::

:::proof
**Step 1. The Gaussian KL.** For $q = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$ and $p = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ (proved in [the toolkit](/aboutme/blog/gaussian-identities/)),

$$
\mathrm{KL}[q \,\Vert\, p] = \tfrac12\Big[\operatorname{tr}(\boldsymbol{\Lambda}\mathbf{S}) + (\boldsymbol{m} - \boldsymbol{\mu})^\top\boldsymbol{\Lambda}(\boldsymbol{m} - \boldsymbol{\mu}) - D + \log\lvert\boldsymbol{\Sigma}\rvert - \log\lvert\mathbf{S}\rvert\Big].
$$

**Step 2. The mean.** Only the quadratic term depends on $\boldsymbol{m}$. It is non-negative because $\boldsymbol{\Lambda}$ is positive definite, and zero at $\boldsymbol{m} = \boldsymbol{\mu}$.

**Step 3. The variances.** With $\mathbf{S} = \operatorname{diag}(s_d)$, $\operatorname{tr}(\boldsymbol{\Lambda}\mathbf{S}) = \sum_d \Lambda_{dd}s_d$ and $\log\lvert\mathbf{S}\rvert = \sum_d \log s_d$. So the KL separates into $\tfrac12\sum_d(\Lambda_{dd}s_d - \log s_d)$ plus constants.

**Step 4. Minimise each term.** $\frac{d}{ds}(\Lambda_{dd}s - \log s) = \Lambda_{dd} - 1/s$, which vanishes at $s = 1/\Lambda_{dd}$. The second derivative $1/s^2$ is positive, so this is the minimum.

**Step 5. Compare with the marginal variance.** By the Cauchy–Schwarz inequality applied to the vectors $\boldsymbol{\Sigma}^{1/2}\boldsymbol{e}_d$ and $\boldsymbol{\Sigma}^{-1/2}\boldsymbol{e}_d$:

$$
1 = (\boldsymbol{e}_d^\top\boldsymbol{e}_d)^2 = \big(\boldsymbol{e}_d^\top\boldsymbol{\Sigma}^{1/2}\,\boldsymbol{\Sigma}^{-1/2}\boldsymbol{e}_d\big)^2 \leq (\boldsymbol{e}_d^\top\boldsymbol{\Sigma}\boldsymbol{e}_d)(\boldsymbol{e}_d^\top\boldsymbol{\Lambda}\boldsymbol{e}_d) = \Sigma_{dd}\Lambda_{dd}.
$$

Hence $1/\Lambda_{dd} \leq \Sigma_{dd}$, with equality only when coordinate $d$ is uncorrelated with the rest.
:::

:::note[What VI means for a GP]
In GP regression with Gaussian noise the posterior is already Gaussian, so VI is not about approximating a strange shape. It is about *cost*. We will restrict the posterior to those that are fully determined by $M$ numbers, the function values at $M$ inputs, and let VI pick the best such posterior and tell us its distance from the exact one.
:::

## 3. Inducing points

:::definition[Inducing inputs and inducing variables]
Choose $M$ inputs $\mathbf{Z} = [\boldsymbol{z}_1, \dots, \boldsymbol{z}_M]^\top$, the *inducing inputs*. The *inducing variables* are the function values there, $\boldsymbol{u} = f(\mathbf{Z}) \in \mathbb{R}^M$.
:::

The inducing variables are not data and not new parameters of the model. They are values of the same random function $f$, at places of our choosing, and nothing is observed there. Because $f$ is a GP, $\boldsymbol{f}$ and $\boldsymbol{u}$ are jointly Gaussian:

$$
\begin{bmatrix}\boldsymbol{f}\\ \boldsymbol{u}\end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\ \begin{bmatrix}\mathbf{K}_{ff} & \mathbf{K}_{fu}\\ \mathbf{K}_{uf} & \mathbf{K}_{uu}\end{bmatrix}\right).
$$

:::lemma[The function given the inducing variables]
$$
p(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{u} \mid \mathbf{0}, \mathbf{K}_{uu}), \qquad p(\boldsymbol{f} \mid \boldsymbol{u}) = \mathcal{N}\big(\boldsymbol{f} \mid \mathbf{A}\boldsymbol{u},\ \mathbf{K}_{ff} - \mathbf{Q}_{ff}\big),
$$

with $\mathbf{A} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}$ and $\mathbf{Q}_{ff} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}$.
:::

:::proof
**Step 1. The marginal of $\boldsymbol{u}$.** Any sub-vector of a Gaussian vector is Gaussian with the matching blocks of the mean and covariance, so $\boldsymbol{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_{uu})$.

**Step 2. Conditioning.** For a joint Gaussian with zero mean, the conditional of the first block given the second has mean $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{u}$ and covariance $\boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$, the Schur complement (the primer's Lemma 5, Gaussian conditioning). Here $\boldsymbol{\Sigma}_{11} = \mathbf{K}_{ff}$, $\boldsymbol{\Sigma}_{12} = \mathbf{K}_{fu}$, $\boldsymbol{\Sigma}_{22} = \mathbf{K}_{uu}$.

**Step 3. Name the pieces.** The mean is $\mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\boldsymbol{u} = \mathbf{A}\boldsymbol{u}$, and the covariance is $\mathbf{K}_{ff} - \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf} = \mathbf{K}_{ff} - \mathbf{Q}_{ff}$.
:::

Three readings of this lemma carry the whole post.

- **The mean interpolates.** $\mathbf{A}\boldsymbol{u}$ is what noise-free GP regression would predict at $\mathbf{X}$ if we had observed the values $\boldsymbol{u}$ at $\mathbf{Z}$. Row $i$ of $\mathbf{A}$ holds the kernel weights that turn the inducing values into a guess for $f(\boldsymbol{x}_i)$.
- **$\mathbf{Q}_{ff}$ is a low-rank stand-in for $\mathbf{K}_{ff}$.** It is the Nyström approximation (Williams and Seeger, 2001), built from the $M$ columns of the kernel at $\mathbf{Z}$. Its rank is at most $M$.
- **$\mathbf{C} = \mathbf{K}_{ff} - \mathbf{Q}_{ff}$ is what the inducing variables cannot pin down.** Its diagonal entry $c_i$ is the prior variance of $f(\boldsymbol{x}_i)$ that remains after $\boldsymbol{u}$ is known. These are the grey bars in the figure.

:::proposition[What the Nyström matrix keeps and loses]
$\mathbf{Q}_{ff} \succeq \mathbf{0}$ and $\mathbf{C} = \mathbf{K}_{ff} - \mathbf{Q}_{ff} \succeq \mathbf{0}$. In particular $0 \leq c_i \leq k_{ii}$ for every $i$, and $c_i = 0$ whenever $\boldsymbol{x}_i$ is one of the inducing inputs.
:::

:::proof
**Step 1. $\mathbf{Q}_{ff}$ is positive semi-definite.** For any $\boldsymbol{v}$, $\boldsymbol{v}^\top\mathbf{Q}_{ff}\boldsymbol{v} = (\mathbf{K}_{uf}\boldsymbol{v})^\top\mathbf{K}_{uu}^{-1}(\mathbf{K}_{uf}\boldsymbol{v}) \geq 0$, because $\mathbf{K}_{uu}^{-1}$ is positive definite.

**Step 2. $\mathbf{C}$ is positive semi-definite.** By the lemma, $\mathbf{C}$ is the covariance of $\boldsymbol{f}$ given $\boldsymbol{u}$, and every covariance matrix is positive semi-definite.

**Step 3. The diagonal.** Diagonal entries of a positive semi-definite matrix are non-negative: $q_{ii} \geq 0$ and $c_i \geq 0$. Since $c_i = k_{ii} - q_{ii}$, also $c_i \leq k_{ii}$.

**Step 4. At an inducing input.** If $\boldsymbol{x}_i = \boldsymbol{z}_j$, then $f(\boldsymbol{x}_i) = u_j$ is known exactly once $\boldsymbol{u}$ is, so its conditional variance $c_i$ is zero.
:::

:::example[One inducing point]
With a single inducing input $z$ and the squared-exponential kernel $k(x, x') = \sigma_f^2\exp\!\big(-(x - x')^2/(2\ell^2)\big)$,

$$
q_{ii} = \frac{k(x_i, z)^2}{k(z, z)} = \sigma_f^2 e^{-(x_i - z)^2/\ell^2}, \qquad c_i = \sigma_f^2\big(1 - e^{-(x_i - z)^2/\ell^2}\big).
$$

One inducing point explains the variance of nearby inputs and almost none of the inputs more than about $2\ell$ away. With length-scale $\ell$ on a domain of width $W$, you need on the order of $W/\ell$ well-spread points before every $c_i$ is small. The figure uses $\ell = 0.07$ on $[0, 1]$, which is why about fourteen are needed.
:::

## 4. Two ways to approximate

The exact prior can be written as $p(\boldsymbol{f}) = \int p(\boldsymbol{f} \mid \boldsymbol{u})\,p(\boldsymbol{u})\,d\boldsymbol{u}$. Before 2009 most sparse GPs approximated the factor $p(\boldsymbol{f} \mid \boldsymbol{u})$, and so changed the prior. Quiñonero-Candela and Rasmussen (2005) unified them this way.

- **DTC (deterministic training conditional).** Replace $p(\boldsymbol{f} \mid \boldsymbol{u})$ by a point mass at $\mathbf{A}\boldsymbol{u}$. The prior on $\boldsymbol{f}$ becomes $\mathcal{N}(\mathbf{0}, \mathbf{Q}_{ff})$, a degenerate rank-$M$ Gaussian, and the evidence becomes $\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{Q}_{ff} + \sigma_n^2\mathbf{I})$.
- **FITC (fully independent training conditional; Snelson and Ghahramani, 2006).** Keep only the diagonal of the conditional covariance: $\mathcal{N}(\boldsymbol{f} \mid \mathbf{A}\boldsymbol{u}, \operatorname{diag}\mathbf{C})$. The prior becomes $\mathcal{N}(\mathbf{0}, \mathbf{Q}_{ff} + \operatorname{diag}\mathbf{C})$, which has the correct variances but the wrong correlations.

Each is a new GP with a new, $\mathbf{Z}$-dependent kernel. Fitting $\mathbf{Z}$ and $\boldsymbol{\theta}$ by maximising its evidence turns the inducing inputs into extra model parameters. Then nothing stops them from fitting the data better than the original model could: FITC is known to underestimate the noise and to overfit (Bauer, van der Wilk and Rasmussen, 2016).

The variational route keeps $p(\boldsymbol{f})$ exact and approximates the posterior instead. Then:

- the hyperparameters belong to the original model;
- $\mathbf{Z}$ are *variational* parameters, which can only move the approximation closer to the true posterior;
- the objective is a lower bound on the true evidence, so its value is a certificate.

## 5. Titsias's bound

### The augmented model

Add $\boldsymbol{u}$ to the model:

$$
p(\boldsymbol{y}, \boldsymbol{f}, \boldsymbol{u}) = p(\boldsymbol{y} \mid \boldsymbol{f})\,p(\boldsymbol{f} \mid \boldsymbol{u})\,p(\boldsymbol{u}).
$$

Integrating out $\boldsymbol{u}$ returns $p(\boldsymbol{y}, \boldsymbol{f}) = p(\boldsymbol{y} \mid \boldsymbol{f})\,p(\boldsymbol{f})$, so the augmentation changes nothing: the evidence $p(\boldsymbol{y})$ is the same. We now approximate the posterior over $(\boldsymbol{f}, \boldsymbol{u})$.

### The variational family

Titsias (2009) chose

$$
q(\boldsymbol{f}, \boldsymbol{u}) = p(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u}).
$$

Given $\boldsymbol{u}$, the function is distributed exactly as under the prior. All the information the data carry has to pass through the $M$ inducing variables. The exact posterior has this form only when $\boldsymbol{u}$ is a sufficient statistic for $\boldsymbol{f}$, which is why it is an approximation.

:::lemma[The ELBO for the inducing-point family]
For $q(\boldsymbol{f}, \boldsymbol{u}) = p(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u})$,

$$
\mathcal{L}(q) = \mathbb{E}_{q(\boldsymbol{f}, \boldsymbol{u})}\big[\log p(\boldsymbol{y} \mid \boldsymbol{f})\big] - \mathrm{KL}\big[q(\boldsymbol{u}) \,\Vert\, p(\boldsymbol{u})\big].
$$
:::

:::proof
**Step 1. Write the ELBO with $\boldsymbol{z} = (\boldsymbol{f}, \boldsymbol{u})$.**

$$
\mathcal{L}(q) = \mathbb{E}_{q(\boldsymbol{f}, \boldsymbol{u})}\left[\log\frac{p(\boldsymbol{y} \mid \boldsymbol{f})\,p(\boldsymbol{f} \mid \boldsymbol{u})\,p(\boldsymbol{u})}{p(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u})}\right].
$$

**Step 2. Cancel.** The factor $p(\boldsymbol{f} \mid \boldsymbol{u})$ appears above and below:

$$
\mathcal{L}(q) = \mathbb{E}_{q(\boldsymbol{f}, \boldsymbol{u})}\big[\log p(\boldsymbol{y} \mid \boldsymbol{f})\big] + \mathbb{E}_{q(\boldsymbol{f}, \boldsymbol{u})}\left[\log\frac{p(\boldsymbol{u})}{q(\boldsymbol{u})}\right].
$$

**Step 3. The second term depends only on $\boldsymbol{u}$.** Its expectation under $q(\boldsymbol{f}, \boldsymbol{u})$ equals its expectation under the marginal $q(\boldsymbol{u})$, which is $-\mathrm{KL}[q(\boldsymbol{u}) \,\Vert\, p(\boldsymbol{u})]$.
:::

:::note[Why the cancellation is the whole trick]
$p(\boldsymbol{f} \mid \boldsymbol{u})$ is an $N$-dimensional Gaussian with an $N \times N$ covariance. Evaluating its density would cost $O(N^3)$, which is exactly what we are avoiding. Because the same factor sits in the model and in $q$, its density never has to be evaluated. We only need its moments, and only on the diagonal, as the next two lemmas show.
:::

:::lemma[Expected Gaussian log-likelihood]
If $\boldsymbol{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{P})$, then

$$
\mathbb{E}\big[\log\mathcal{N}(\boldsymbol{y} \mid \boldsymbol{f}, \sigma_n^2\mathbf{I})\big] = \log\mathcal{N}(\boldsymbol{y} \mid \boldsymbol{\mu}, \sigma_n^2\mathbf{I}) - \frac{1}{2\sigma_n^2}\operatorname{tr}\mathbf{P}.
$$
:::

:::proof
**Step 1. Write the log-density.** $\log\mathcal{N}(\boldsymbol{y} \mid \boldsymbol{f}, \sigma_n^2\mathbf{I}) = -\tfrac N2\log(2\pi\sigma_n^2) - \frac{1}{2\sigma_n^2}\lVert\boldsymbol{y} - \boldsymbol{f}\rVert^2$. Only the norm is random.

**Step 2. Split the residual.** $\boldsymbol{y} - \boldsymbol{f} = (\boldsymbol{y} - \boldsymbol{\mu}) - (\boldsymbol{f} - \boldsymbol{\mu})$, so

$$
\lVert\boldsymbol{y} - \boldsymbol{f}\rVert^2 = \lVert\boldsymbol{y} - \boldsymbol{\mu}\rVert^2 - 2(\boldsymbol{y} - \boldsymbol{\mu})^\top(\boldsymbol{f} - \boldsymbol{\mu}) + \lVert\boldsymbol{f} - \boldsymbol{\mu}\rVert^2 .
$$

**Step 3. Take expectations.** The middle term has mean zero because $\mathbb{E}[\boldsymbol{f}] = \boldsymbol{\mu}$. The last has mean $\mathbb{E}\sum_i(f_i - \mu_i)^2 = \sum_i P_{ii} = \operatorname{tr}\mathbf{P}$.

**Step 4. Reassemble.** $\mathbb{E}[\log\mathcal{N}] = -\tfrac N2\log(2\pi\sigma_n^2) - \frac{1}{2\sigma_n^2}\lVert\boldsymbol{y} - \boldsymbol{\mu}\rVert^2 - \frac{1}{2\sigma_n^2}\operatorname{tr}\mathbf{P}$, and the first two terms are $\log\mathcal{N}(\boldsymbol{y} \mid \boldsymbol{\mu}, \sigma_n^2\mathbf{I})$.
:::

Only the trace of the covariance enters: the expected log-likelihood is the log-likelihood at the mean, minus a penalty for the spread of $\boldsymbol{f}$.

### The optimal q(u) and the collapsed bound

:::theorem[Titsias (2009)]
Over all densities $q(\boldsymbol{u})$, the ELBO of the family $p(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u})$ is maximised by

$$
q^*(\boldsymbol{u}) \propto \mathcal{N}(\boldsymbol{y} \mid \mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I})\,p(\boldsymbol{u}),
$$

and its maximum value is

$$
\boxed{\ \mathcal{F} = \log\mathcal{N}\big(\boldsymbol{y} \mid \mathbf{0},\ \mathbf{Q}_{ff} + \sigma_n^2\mathbf{I}\big) - \frac{1}{2\sigma_n^2}\operatorname{tr}\big(\mathbf{K}_{ff} - \mathbf{Q}_{ff}\big)\ }
$$
:::

:::proof
**Step 1. Do the inner expectation over $\boldsymbol{f}$.** For fixed $\boldsymbol{u}$, $\boldsymbol{f} \sim p(\boldsymbol{f} \mid \boldsymbol{u}) = \mathcal{N}(\mathbf{A}\boldsymbol{u}, \mathbf{C})$. The expected-log-likelihood lemma gives

$$
\mathbb{E}_{p(\boldsymbol{f} \mid \boldsymbol{u})}\big[\log p(\boldsymbol{y} \mid \boldsymbol{f})\big] = \log\mathcal{N}(\boldsymbol{y} \mid \mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I}) - \frac{t}{2\sigma_n^2}, \qquad t = \operatorname{tr}\mathbf{C}.
$$

**Step 2. Put it into the ELBO.** Write the KL as an expectation and collect everything under one $\mathbb{E}_{q(\boldsymbol{u})}$:

$$
\mathcal{L}(q) = \mathbb{E}_{q(\boldsymbol{u})}\left[\log\frac{\mathcal{N}(\boldsymbol{y} \mid \mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I})\,p(\boldsymbol{u})}{q(\boldsymbol{u})}\right] - \frac{t}{2\sigma_n^2}.
$$

**Step 3. Normalise the numerator.** Let $Z_0 = \int\mathcal{N}(\boldsymbol{y} \mid \mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I})\,p(\boldsymbol{u})\,d\boldsymbol{u}$ and $q^*(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{y} \mid \mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I})\,p(\boldsymbol{u})/Z_0$, a probability density. Then the numerator is $Z_0\,q^*(\boldsymbol{u})$, and

$$
\mathcal{L}(q) = \log Z_0 - \mathrm{KL}\big[q \,\Vert\, q^*\big] - \frac{t}{2\sigma_n^2}.
$$

**Step 4. Maximise.** By Gibbs' inequality the KL term is at least zero, with equality exactly at $q = q^*$. So $q^*$ is the maximiser and the maximum is $\log Z_0 - t/(2\sigma_n^2)$.

**Step 5. Compute $Z_0$.** It is the marginal of $\boldsymbol{y}$ in the linear-Gaussian model $\boldsymbol{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_{uu})$, $\boldsymbol{y} \mid \boldsymbol{u} \sim \mathcal{N}(\mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I})$. That marginal is $\mathcal{N}(\mathbf{0}, \mathbf{A}\mathbf{K}_{uu}\mathbf{A}^\top + \sigma_n^2\mathbf{I})$ (the primer's Lemma 8, linear-Gaussian model), and

$$
\mathbf{A}\mathbf{K}_{uu}\mathbf{A}^\top = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf} = \mathbf{Q}_{ff}.
$$

So $\log Z_0 = \log\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{Q}_{ff} + \sigma_n^2\mathbf{I})$.
:::

:::corollary[The optimal q(u) in closed form]
$q^*(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{m}^*, \mathbf{S}^*)$ with

$$
\mathbf{S}^* = \mathbf{K}_{uu}\boldsymbol{\Sigma}\mathbf{K}_{uu}, \qquad \boldsymbol{m}^* = \sigma_n^{-2}\,\mathbf{K}_{uu}\boldsymbol{\Sigma}\mathbf{K}_{uf}\boldsymbol{y}, \qquad \boldsymbol{\Sigma} = \big(\mathbf{K}_{uu} + \sigma_n^{-2}\mathbf{K}_{uf}\mathbf{K}_{fu}\big)^{-1}.
$$
:::

:::proof
**Step 1. It is a linear-Gaussian posterior.** $q^*$ is proportional to a Gaussian prior on $\boldsymbol{u}$ times a Gaussian likelihood with design matrix $\mathbf{A}$. So it is Gaussian, with precision $\mathbf{K}_{uu}^{-1} + \sigma_n^{-2}\mathbf{A}^\top\mathbf{A}$ and mean $\mathbf{S}^*\sigma_n^{-2}\mathbf{A}^\top\boldsymbol{y}$ (the primer's Lemma 8, in information form).

**Step 2. Factor the precision.** With $\mathbf{A}^\top\mathbf{A} = \mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}\mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}$,

$$
\mathbf{K}_{uu}^{-1} + \sigma_n^{-2}\mathbf{A}^\top\mathbf{A} = \mathbf{K}_{uu}^{-1}\big(\mathbf{K}_{uu} + \sigma_n^{-2}\mathbf{K}_{uf}\mathbf{K}_{fu}\big)\mathbf{K}_{uu}^{-1} = \mathbf{K}_{uu}^{-1}\boldsymbol{\Sigma}^{-1}\mathbf{K}_{uu}^{-1}.
$$

**Step 3. Invert.** $(\mathbf{P}\mathbf{R}\mathbf{P})^{-1} = \mathbf{P}^{-1}\mathbf{R}^{-1}\mathbf{P}^{-1}$, so $\mathbf{S}^* = \mathbf{K}_{uu}\boldsymbol{\Sigma}\mathbf{K}_{uu}$.

**Step 4. The mean.** $\boldsymbol{m}^* = \sigma_n^{-2}\mathbf{K}_{uu}\boldsymbol{\Sigma}\mathbf{K}_{uu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}\boldsymbol{y} = \sigma_n^{-2}\mathbf{K}_{uu}\boldsymbol{\Sigma}\mathbf{K}_{uf}\boldsymbol{y}$.
:::

Every matrix here is $M \times M$ or $M \times N$. Nothing $N \times N$ is ever formed.

### Reading the bound

- **The first term is the DTC evidence.** $\log\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{Q}_{ff} + \sigma_n^2\mathbf{I})$ is the log evidence of the low-rank model from Section 4, with its usual data-fit and complexity parts.
- **The second term is what makes it honest.** $-t/(2\sigma_n^2) = -\sum_i c_i/(2\sigma_n^2)$ charges, at every training input, the prior variance the inducing points leave unexplained, measured in units of noise variance. Inducing points that leave data uncovered are penalised. In the figure this is the sum of the grey bars, divided by $2\sigma_n^2$.
- **The optimal $q^*(\boldsymbol{u})$ is the DTC posterior over $\boldsymbol{u}$.** The variational solution predicts like DTC, but it learns hyperparameters and $\mathbf{Z}$ with the extra trace term.

:::proposition[The gap is a KL divergence]
$$
\log p(\boldsymbol{y}) - \mathcal{F} = \mathrm{KL}\big[p(\boldsymbol{f} \mid \boldsymbol{u})\,q^*(\boldsymbol{u}) \,\big\Vert\, p(\boldsymbol{f}, \boldsymbol{u} \mid \boldsymbol{y})\big] \geq 0.
$$
:::

:::proof
**Step 1.** Apply the evidence decomposition of Section 2 to the augmented model, with $\boldsymbol{z} = (\boldsymbol{f}, \boldsymbol{u})$ and $q(\boldsymbol{f}, \boldsymbol{u}) = p(\boldsymbol{f} \mid \boldsymbol{u})\,q^*(\boldsymbol{u})$.

**Step 2.** The augmented model has the same evidence $p(\boldsymbol{y})$, and $\mathcal{L}(q) = \mathcal{F}$ at this $q$ by the theorem. Rearranging the decomposition gives the identity, and Gibbs' inequality gives the sign.
:::

Matthews, Hensman, Turner and Ghahramani (2016) proved more: the same number is the KL divergence between the approximating *process* and the posterior *process*, over the whole function, not only at $\mathbf{X}$ and $\mathbf{Z}$. So moving $\mathbf{Z}$ to raise $\mathcal{F}$ always brings the approximation closer to the true posterior. It cannot overfit in the way FITC's evidence can.

:::proposition[More inducing points never hurt]
1. If $\mathbf{Z} = \mathbf{X}$, then $\mathcal{F} = \log p(\boldsymbol{y})$.
2. Adding an inducing input to $\mathbf{Z}$ cannot lower the optimal bound $\mathcal{F}$.
:::

:::proof
**Step 1. The case $\mathbf{Z} = \mathbf{X}$.** Then $\mathbf{K}_{fu} = \mathbf{K}_{uu} = \mathbf{K}_{ff}$, so $\mathbf{Q}_{ff} = \mathbf{K}_{ff}$ and $\mathbf{C} = \mathbf{0}$. The trace term vanishes and the first term is $\log\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{K}_{ff} + \sigma_n^2\mathbf{I}) = \log p(\boldsymbol{y})$.

**Step 2. Embed the old family in the new one.** Let $\boldsymbol{u}' = (\boldsymbol{u}, u_{\mathrm{new}})$. In the new family choose $q(\boldsymbol{u}') = p(u_{\mathrm{new}} \mid \boldsymbol{u})\,q(\boldsymbol{u})$. Then $q(\boldsymbol{f}, \boldsymbol{u}') = p(\boldsymbol{f}, u_{\mathrm{new}} \mid \boldsymbol{u})\,q(\boldsymbol{u})$.

**Step 3. Same ELBO.** The ratio inside the new ELBO is

$$
\frac{p(\boldsymbol{y} \mid \boldsymbol{f})\,p(\boldsymbol{f}, u_{\mathrm{new}} \mid \boldsymbol{u})\,p(\boldsymbol{u})}{p(\boldsymbol{f}, u_{\mathrm{new}} \mid \boldsymbol{u})\,q(\boldsymbol{u})} = \frac{p(\boldsymbol{y} \mid \boldsymbol{f})\,p(\boldsymbol{u})}{q(\boldsymbol{u})},
$$

the same as before. So every ELBO value reachable with $\mathbf{Z}$ is reachable with the larger set.

**Step 4. Conclude.** The maximum over a larger set is at least the maximum over a smaller one.
:::

## 6. A sandwich: the upper bound

A lower bound alone cannot say how close it is to the truth. Titsias (2014) gave an upper bound that costs the same $O(NM^2)$; Burt, Rasmussen and van der Wilk (2019) used it to prove their convergence results.

:::proposition[Upper bound on the evidence]
With $t = \operatorname{tr}(\mathbf{K}_{ff} - \mathbf{Q}_{ff})$,

$$
\log p(\boldsymbol{y}) \leq \mathcal{U} = -\tfrac12\,\boldsymbol{y}^\top\big(\mathbf{Q}_{ff} + (\sigma_n^2 + t)\mathbf{I}\big)^{-1}\boldsymbol{y} - \tfrac12\log\big\lvert\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I}\big\rvert - \tfrac N2\log 2\pi .
$$
:::

:::proof
**Step 1. Split the exact covariance.** $\mathbf{K}_{ff} + \sigma_n^2\mathbf{I} = (\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I}) + \mathbf{C}$ with $\mathbf{C} \succeq \mathbf{0}$ (Section 3).

**Step 2. The determinant can only grow.** If $\mathbf{P} \succeq \mathbf{R} \succ \mathbf{0}$, then $\mathbf{R}^{-1/2}\mathbf{P}\mathbf{R}^{-1/2} \succeq \mathbf{I}$ has all eigenvalues at least 1, so $\lvert\mathbf{P}\rvert \geq \lvert\mathbf{R}\rvert$. Hence

$$
-\tfrac12\log\lvert\mathbf{K}_{ff} + \sigma_n^2\mathbf{I}\rvert \leq -\tfrac12\log\lvert\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I}\rvert .
$$

**Step 3. Bound $\mathbf{C}$ by a multiple of the identity.** $\mathbf{C}$ is positive semi-definite, so its eigenvalues are non-negative and sum to $t$. Its largest eigenvalue is therefore at most $t$, and $\mathbf{C} \preceq t\,\mathbf{I}$.

**Step 4. So the exact covariance is dominated.** $\mathbf{K}_{ff} + \sigma_n^2\mathbf{I} \preceq \mathbf{Q}_{ff} + (\sigma_n^2 + t)\mathbf{I}$.

**Step 5. Inversion reverses the order.** If $\mathbf{P} \preceq \mathbf{R}$ (both positive definite), then $\mathbf{P}^{-1} \succeq \mathbf{R}^{-1}$. Hence

$$
-\tfrac12\boldsymbol{y}^\top(\mathbf{K}_{ff} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y} \leq -\tfrac12\boldsymbol{y}^\top\big(\mathbf{Q}_{ff} + (\sigma_n^2 + t)\mathbf{I}\big)^{-1}\boldsymbol{y}.
$$

**Step 6. Add the pieces.** The exact $\log p(\boldsymbol{y})$ is the sum of the two left-hand sides and $-\tfrac N2\log 2\pi$; replacing each by its upper bound gives $\mathcal{U}$.
:::

So $\mathcal{F} \leq \log p(\boldsymbol{y}) \leq \mathcal{U}$, and $\mathcal{U} - \mathcal{F}$ bounds the KL divergence of the approximation. Both depend on how small $t$ is. Burt et al. (2019, Lemma 1) turn this into an explicit bound:

$$
\mathrm{KL} \leq \frac{t}{2\sigma_n^2}\left(1 + \frac{\lVert\boldsymbol{y}\rVert^2}{\sigma_n^2 + t}\right).
$$

To make the approximation good, make $t$ small: place inducing points so that the training inputs' prior variance is explained. That is what **Greedy** does in the figure.

## 7. Predictions and an implementation you can trust

The variational posterior is a GP. At a test input, the inducing variables carry everything the data said.

:::proposition[The predictive distribution]
With $q(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$, $\boldsymbol{k}_u = k(\mathbf{Z}, \boldsymbol{x}_*)$ and $\boldsymbol{a}_* = \mathbf{K}_{uu}^{-1}\boldsymbol{k}_u$,

$$
q(f_*) = \mathcal{N}\big(f_* \mid \boldsymbol{a}_*^\top\boldsymbol{m},\ k_{**} - \boldsymbol{k}_u^\top\mathbf{K}_{uu}^{-1}\boldsymbol{k}_u + \boldsymbol{a}_*^\top\mathbf{S}\,\boldsymbol{a}_*\big).
$$

At the optimum $q^*$ this is mean $\sigma_n^{-2}\boldsymbol{k}_u^\top\boldsymbol{\Sigma}\mathbf{K}_{uf}\boldsymbol{y}$ and variance $k_{**} - \boldsymbol{k}_u^\top(\mathbf{K}_{uu}^{-1} - \boldsymbol{\Sigma})\boldsymbol{k}_u$.
:::

:::proof
**Step 1. The conditional at the test point.** Exactly as in the lemma of Section 3, $p(f_* \mid \boldsymbol{u}) = \mathcal{N}(\boldsymbol{a}_*^\top\boldsymbol{u},\ k_{**} - \boldsymbol{k}_u^\top\mathbf{K}_{uu}^{-1}\boldsymbol{k}_u)$.

**Step 2. Integrate out $\boldsymbol{u}$.** $q(f_*) = \int p(f_* \mid \boldsymbol{u})\,q(\boldsymbol{u})\,d\boldsymbol{u}$ is a linear-Gaussian marginal: the mean is $\boldsymbol{a}_*^\top\boldsymbol{m}$ and the variance is the conditional variance plus $\boldsymbol{a}_*^\top\mathbf{S}\boldsymbol{a}_*$ (the primer's Lemma 8).

**Step 3. Substitute the optimum.** From the corollary, $\mathbf{K}_{uu}^{-1}\boldsymbol{m}^* = \sigma_n^{-2}\boldsymbol{\Sigma}\mathbf{K}_{uf}\boldsymbol{y}$ and $\mathbf{K}_{uu}^{-1}\mathbf{S}^*\mathbf{K}_{uu}^{-1} = \boldsymbol{\Sigma}$. So $\boldsymbol{a}_*^\top\boldsymbol{m}^* = \sigma_n^{-2}\boldsymbol{k}_u^\top\boldsymbol{\Sigma}\mathbf{K}_{uf}\boldsymbol{y}$ and $\boldsymbol{a}_*^\top\mathbf{S}^*\boldsymbol{a}_* = \boldsymbol{k}_u^\top\boldsymbol{\Sigma}\boldsymbol{k}_u$.
:::

Far from every inducing point $\boldsymbol{k}_u \approx \mathbf{0}$, so the prediction returns to the prior: mean $0$, variance $k_{**}$. That is the right behaviour in a gap of the data, and the wrong one if the inducing points do not cover the data, which the bound punishes.

The formulas above invert $\mathbf{K}_{uu}$ and $\boldsymbol{\Sigma}^{-1}$ directly, which is numerically fragile: $\mathbf{K}_{uu}$ is often badly conditioned. Production code (for example GPflow's SGPR) works with two $M \times M$ Cholesky factors instead.

:::proposition[A stable form of the bound and the predictions]
Let $\mathbf{L}\mathbf{L}^\top = \mathbf{K}_{uu}$, $\hat{\mathbf{A}} = \mathbf{L}^{-1}\mathbf{K}_{uf}/\sigma_n$ ($M \times N$), $\mathbf{B} = \mathbf{I} + \hat{\mathbf{A}}\hat{\mathbf{A}}^\top$ with $\mathbf{L}_B\mathbf{L}_B^\top = \mathbf{B}$, and $\boldsymbol{b} = \mathbf{L}_B^{-1}\hat{\mathbf{A}}\boldsymbol{y}/\sigma_n$. Then

$$
\mathcal{F} = -\tfrac N2\log 2\pi - N\log\sigma_n - \sum_{j}\log(\mathbf{L}_B)_{jj} - \frac{\boldsymbol{y}^\top\boldsymbol{y}}{2\sigma_n^2} + \tfrac12\boldsymbol{b}^\top\boldsymbol{b} - \frac{\operatorname{tr}\mathbf{K}_{ff}}{2\sigma_n^2} + \tfrac12\operatorname{tr}\big(\hat{\mathbf{A}}\hat{\mathbf{A}}^\top\big),
$$

and with $\boldsymbol{g}_1 = \mathbf{L}^{-1}\boldsymbol{k}_u$, $\boldsymbol{g}_2 = \mathbf{L}_B^{-1}\boldsymbol{g}_1$, the optimal predictive has mean $\boldsymbol{g}_2^\top\boldsymbol{b}$ and variance $k_{**} - \boldsymbol{g}_1^\top\boldsymbol{g}_1 + \boldsymbol{g}_2^\top\boldsymbol{g}_2$.
:::

:::proof
**Step 1. Rewrite $\mathbf{Q}_{ff}$.** $\mathbf{Q}_{ff} = \mathbf{K}_{fu}\mathbf{L}^{-\top}\mathbf{L}^{-1}\mathbf{K}_{uf} = \sigma_n^2\hat{\mathbf{A}}^\top\hat{\mathbf{A}}$, so $\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I} = \sigma_n^2(\mathbf{I}_N + \hat{\mathbf{A}}^\top\hat{\mathbf{A}})$.

**Step 2. The determinant.** Sylvester's identity gives $\lvert\mathbf{I}_N + \hat{\mathbf{A}}^\top\hat{\mathbf{A}}\rvert = \lvert\mathbf{I}_M + \hat{\mathbf{A}}\hat{\mathbf{A}}^\top\rvert = \lvert\mathbf{B}\rvert$. To see it, take the determinant of the block matrix $\begin{bmatrix}\mathbf{I}_M & -\hat{\mathbf{A}}\\ \hat{\mathbf{A}}^\top & \mathbf{I}_N\end{bmatrix}$ in two ways, eliminating either block by its Schur complement: one way gives $\lvert\mathbf{I}_N + \hat{\mathbf{A}}^\top\hat{\mathbf{A}}\rvert$, the other $\lvert\mathbf{I}_M + \hat{\mathbf{A}}\hat{\mathbf{A}}^\top\rvert$. So $\log\lvert\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I}\rvert = 2N\log\sigma_n + 2\sum_j\log(\mathbf{L}_B)_{jj}$.

**Step 3. The quadratic form.** The Woodbury identity (the primer's Lemma 6 with $\boldsymbol{\Sigma}_w = \mathbf{I}$, $\boldsymbol{\Phi} = \hat{\mathbf{A}}$ and unit noise) gives $(\mathbf{I}_N + \hat{\mathbf{A}}^\top\hat{\mathbf{A}})^{-1} = \mathbf{I}_N - \hat{\mathbf{A}}^\top\mathbf{B}^{-1}\hat{\mathbf{A}}$. Hence

$$
\boldsymbol{y}^\top(\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y} = \sigma_n^{-2}\boldsymbol{y}^\top\boldsymbol{y} - \sigma_n^{-2}\lVert\mathbf{L}_B^{-1}\hat{\mathbf{A}}\boldsymbol{y}\rVert^2 = \sigma_n^{-2}\boldsymbol{y}^\top\boldsymbol{y} - \boldsymbol{b}^\top\boldsymbol{b}.
$$

**Step 4. The trace term.** $\operatorname{tr}\mathbf{Q}_{ff} = \sigma_n^2\operatorname{tr}(\hat{\mathbf{A}}^\top\hat{\mathbf{A}}) = \sigma_n^2\operatorname{tr}(\hat{\mathbf{A}}\hat{\mathbf{A}}^\top)$, so $-\operatorname{tr}(\mathbf{K}_{ff} - \mathbf{Q}_{ff})/(2\sigma_n^2) = -\operatorname{tr}\mathbf{K}_{ff}/(2\sigma_n^2) + \tfrac12\operatorname{tr}(\hat{\mathbf{A}}\hat{\mathbf{A}}^\top)$. Steps 2 to 4 assemble into the stated $\mathcal{F}$.

**Step 5. $\boldsymbol{\Sigma}$ in factored form.** $\boldsymbol{\Sigma}^{-1} = \mathbf{K}_{uu} + \sigma_n^{-2}\mathbf{K}_{uf}\mathbf{K}_{fu} = \mathbf{L}(\mathbf{I} + \hat{\mathbf{A}}\hat{\mathbf{A}}^\top)\mathbf{L}^\top = \mathbf{L}\mathbf{B}\mathbf{L}^\top$, so $\boldsymbol{\Sigma} = \mathbf{L}^{-\top}\mathbf{B}^{-1}\mathbf{L}^{-1}$.

**Step 6. The predictive mean.** $\sigma_n^{-2}\boldsymbol{k}_u^\top\boldsymbol{\Sigma}\mathbf{K}_{uf}\boldsymbol{y} = \sigma_n^{-2}\boldsymbol{g}_1^\top\mathbf{B}^{-1}(\sigma_n\hat{\mathbf{A}}\boldsymbol{y}) = \boldsymbol{g}_1^\top\mathbf{L}_B^{-\top}\mathbf{L}_B^{-1}\hat{\mathbf{A}}\boldsymbol{y}/\sigma_n = \boldsymbol{g}_2^\top\boldsymbol{b}$.

**Step 7. The predictive variance.** $\boldsymbol{k}_u^\top\mathbf{K}_{uu}^{-1}\boldsymbol{k}_u = \boldsymbol{g}_1^\top\boldsymbol{g}_1$ and $\boldsymbol{k}_u^\top\boldsymbol{\Sigma}\boldsymbol{k}_u = \boldsymbol{g}_1^\top\mathbf{B}^{-1}\boldsymbol{g}_1 = \boldsymbol{g}_2^\top\boldsymbol{g}_2$.
:::

In NumPy, with a kernel function `k(A, B)` that returns the matrix of kernel values (the figure above runs the same lines in JavaScript):

```python
import numpy as np
from scipy.linalg import cholesky, solve_triangular

def sgpr(X, y, Z, k, sn, jitter=1e-6):
    """Titsias's collapsed bound F and a predictor, in O(N M^2) time and O(N M) memory."""
    N, M = len(X), len(Z)
    L = cholesky(k(Z, Z) + jitter * np.eye(M), lower=True)     # K_uu = L L^T
    A = solve_triangular(L, k(Z, X), lower=True) / sn            # \hat{A}, M x N
    B = np.eye(M) + A @ A.T
    LB = cholesky(B, lower=True)
    b = solve_triangular(LB, A @ y, lower=True) / sn
    trace_Kff = sum(k(X[i:i+1], X[i:i+1])[0, 0] for i in range(N))   # only the diagonal is needed
    F = (-0.5 * N * np.log(2 * np.pi) - N * np.log(sn) - np.log(np.diag(LB)).sum()
         - 0.5 * y @ y / sn**2 + 0.5 * b @ b
         - 0.5 * trace_Kff / sn**2 + 0.5 * np.sum(A * A))

    def predict(Xs):
        g1 = solve_triangular(L, k(Z, Xs), lower=True)
        g2 = solve_triangular(LB, g1, lower=True)
        mean = g2.T @ b
        var = np.array([k(x[None], x[None])[0, 0] for x in Xs]) - np.sum(g1**2, 0) + np.sum(g2**2, 0)
        return mean, var
    return F, predict
```

The dominant costs are forming $\hat{\mathbf{A}}$ and $\hat{\mathbf{A}}\hat{\mathbf{A}}^\top$, both $O(NM^2)$, plus two $O(M^3)$ Cholesky factorisations. For $N = 10^6$ and $M = 10^3$ that is about $10^{12}$ flops, minutes on a CPU. But $\hat{\mathbf{A}}$ alone holds $10^9$ numbers, 8 GB. The collapsed bound still touches every data point at every step, which is what Hensman's method removes.

## 8. Hensman: stochastic variational GPs

### Why the collapsed bound does not minibatch

Stochastic gradient methods need an objective that is a *sum over data points*: then a random minibatch, rescaled, gives an unbiased estimate of it and of its gradient. $\mathcal{F}$ is not such a sum. Its quadratic term $\boldsymbol{y}^\top(\mathbf{Q}_{ff} + \sigma_n^2\mathbf{I})^{-1}\boldsymbol{y}$ couples every pair of points, because optimising $q(\boldsymbol{u})$ analytically has already mixed all the data into $q^*$.

Hensman, Fusi and Lawrence (2013) undo that step. They keep $q(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$ explicit and optimise $\boldsymbol{m}$ and $\mathbf{S}$ by stochastic gradients along with everything else.

:::proposition[The SVGP bound]
For a likelihood that factorises over data points, $p(\boldsymbol{y} \mid \boldsymbol{f}) = \prod_i p(y_i \mid f_i)$, and $q(\boldsymbol{u}) = \mathcal{N}(\boldsymbol{m}, \mathbf{S})$:

$$
\mathcal{L}(\boldsymbol{m}, \mathbf{S}) = \sum_{i=1}^{N}\mathbb{E}_{q(f_i)}\big[\log p(y_i \mid f_i)\big] - \mathrm{KL}\big[\mathcal{N}(\boldsymbol{m}, \mathbf{S}) \,\Vert\, \mathcal{N}(\mathbf{0}, \mathbf{K}_{uu})\big],
$$

where each $q(f_i)$ is the one-dimensional Gaussian

$$
q(f_i) = \mathcal{N}\big(f_i \mid \boldsymbol{a}_i^\top\boldsymbol{m},\ c_i + \boldsymbol{a}_i^\top\mathbf{S}\,\boldsymbol{a}_i\big).
$$
:::

:::proof
**Step 1. Start from the lemma of Section 5.** $\mathcal{L} = \mathbb{E}_{q(\boldsymbol{f}, \boldsymbol{u})}[\log p(\boldsymbol{y} \mid \boldsymbol{f})] - \mathrm{KL}[q(\boldsymbol{u}) \,\Vert\, p(\boldsymbol{u})]$.

**Step 2. Only the marginal of $\boldsymbol{f}$ matters.** The first term depends on $\boldsymbol{f}$ alone, so it is an expectation under $q(\boldsymbol{f}) = \int p(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u})\,d\boldsymbol{u}$. That is a linear-Gaussian marginal: $q(\boldsymbol{f}) = \mathcal{N}(\mathbf{A}\boldsymbol{m}, \mathbf{C} + \mathbf{A}\mathbf{S}\mathbf{A}^\top)$.

**Step 3. The log-likelihood is a sum.** $\log p(\boldsymbol{y} \mid \boldsymbol{f}) = \sum_i\log p(y_i \mid f_i)$, and the expectation of a sum is the sum of expectations.

**Step 4. Each term needs one marginal.** $\log p(y_i \mid f_i)$ depends only on $f_i$, whose marginal under $q(\boldsymbol{f})$ has mean $(\mathbf{A}\boldsymbol{m})_i = \boldsymbol{a}_i^\top\boldsymbol{m}$ and variance $C_{ii} + (\mathbf{A}\mathbf{S}\mathbf{A}^\top)_{ii} = c_i + \boldsymbol{a}_i^\top\mathbf{S}\boldsymbol{a}_i$.
:::

For a Gaussian likelihood each term has a closed form, by the expected-log-likelihood lemma in one dimension:

$$
\mathbb{E}_{q(f_i)}\big[\log\mathcal{N}(y_i \mid f_i, \sigma_n^2)\big] = \log\mathcal{N}(y_i \mid \boldsymbol{a}_i^\top\boldsymbol{m}, \sigma_n^2) - \frac{c_i + \boldsymbol{a}_i^\top\mathbf{S}\,\boldsymbol{a}_i}{2\sigma_n^2}.
$$

The KL term, between two $M$-dimensional Gaussians, is

$$
\mathrm{KL} = \tfrac12\Big[\operatorname{tr}(\mathbf{K}_{uu}^{-1}\mathbf{S}) + \boldsymbol{m}^\top\mathbf{K}_{uu}^{-1}\boldsymbol{m} - M + \log\lvert\mathbf{K}_{uu}\rvert - \log\lvert\mathbf{S}\rvert\Big].
$$

:::proposition[SVGP contains Titsias's bound]
For a Gaussian likelihood, $\max_{\boldsymbol{m}, \mathbf{S}}\mathcal{L}(\boldsymbol{m}, \mathbf{S}) = \mathcal{F}$, attained at $(\boldsymbol{m}^*, \mathbf{S}^*)$. For any other $(\boldsymbol{m}, \mathbf{S})$, $\mathcal{L}(\boldsymbol{m}, \mathbf{S}) \leq \mathcal{F}$.
:::

:::proof
**Step 1.** Titsias's theorem maximised $\mathcal{L}$ over *all* densities $q(\boldsymbol{u})$ and found the maximiser $q^*$, which is Gaussian (corollary).

**Step 2.** Gaussians are a subset of all densities, and $q^*$ belongs to that subset. So the maximum over Gaussians is the same value $\mathcal{F}$, at the same point, and every other Gaussian does no better.
:::

So SVGP gives up nothing at its optimum. Away from the optimum it is a looser bound, the price of being able to minibatch.

### Minibatches

Pick a random minibatch $\mathcal{B}$ of size $B$. Then

$$
\hat{\mathcal{L}} = \frac{N}{B}\sum_{i \in \mathcal{B}}\mathbb{E}_{q(f_i)}\big[\log p(y_i \mid f_i)\big] - \mathrm{KL}\big[q(\boldsymbol{u}) \,\Vert\, p(\boldsymbol{u})\big]
$$

has expectation $\mathcal{L}$ over the random choice of batch, and so does its gradient. One step costs $O(BM^2)$ for the marginals and $O(M^3)$ for the KL and the Cholesky factor of $\mathbf{K}_{uu}$, independent of $N$. The parameters are $\boldsymbol{m}$, a lower-triangular factor of $\mathbf{S}$, $\mathbf{Z}$ and $\boldsymbol{\theta}$.

### Any likelihood

Nothing in the bound assumed a Gaussian likelihood except the closed form above. For classification, counts or robust regression, only the one-dimensional expectations change (Hensman, Matthews and Ghahramani, 2015). Gauss–Hermite quadrature with $J$ nodes $\xi_j$ and weights $w_j$ computes them accurately:

$$
\mathbb{E}_{\mathcal{N}(f \mid \mu, v)}\big[g(f)\big] \approx \frac{1}{\sqrt{\pi}}\sum_{j=1}^{J} w_j\,g\big(\mu + \sqrt{2v}\,\xi_j\big).
$$

It comes from the substitution $f = \mu + \sqrt{2v}\,\xi$, which turns the Gaussian weight into $e^{-\xi^2}/\sqrt{\pi}$. Twenty nodes are typically plenty. This is how GP classification scales to millions of points.

### Whitening

The parameters $(\boldsymbol{m}, \mathbf{S})$ live on the scale of $\mathbf{K}_{uu}$, which changes whenever a hyperparameter does. Optimising both at once is like steering while someone moves the road. Whitening changes variables to remove that coupling.

:::proposition[The whitened parametrisation]
Write $\boldsymbol{u} = \mathbf{L}\boldsymbol{v}$ with $\mathbf{L}\mathbf{L}^\top = \mathbf{K}_{uu}$, so that $p(\boldsymbol{v}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, and let $q(\boldsymbol{v}) = \mathcal{N}(\tilde{\boldsymbol{m}}, \tilde{\mathbf{S}})$. Then, with $\tilde{\boldsymbol{a}}_i = \mathbf{L}^{-1}k(\mathbf{Z}, \boldsymbol{x}_i)$,

$$
q(f_i) = \mathcal{N}\big(\tilde{\boldsymbol{a}}_i^\top\tilde{\boldsymbol{m}},\ c_i + \tilde{\boldsymbol{a}}_i^\top\tilde{\mathbf{S}}\,\tilde{\boldsymbol{a}}_i\big), \qquad \mathrm{KL} = \tfrac12\big[\operatorname{tr}\tilde{\mathbf{S}} + \tilde{\boldsymbol{m}}^\top\tilde{\boldsymbol{m}} - M - \log\lvert\tilde{\mathbf{S}}\rvert\big].
$$
:::

:::proof
**Step 1. The prior on $\boldsymbol{v}$.** $\boldsymbol{v} = \mathbf{L}^{-1}\boldsymbol{u}$ has covariance $\mathbf{L}^{-1}\mathbf{K}_{uu}\mathbf{L}^{-\top} = \mathbf{I}$.

**Step 2. The KL is unchanged by an invertible linear map.** Both densities pick up the same Jacobian factor $\lvert\mathbf{L}\rvert^{-1}$, which cancels in the ratio. So $\mathrm{KL}[q(\boldsymbol{u}) \,\Vert\, p(\boldsymbol{u})] = \mathrm{KL}[q(\boldsymbol{v}) \,\Vert\, \mathcal{N}(\mathbf{0}, \mathbf{I})]$, which is the Gaussian KL with $\boldsymbol{\Lambda} = \mathbf{I}$.

**Step 3. The marginals.** $q(\boldsymbol{u}) = \mathcal{N}(\mathbf{L}\tilde{\boldsymbol{m}}, \mathbf{L}\tilde{\mathbf{S}}\mathbf{L}^\top)$. Substituting into Hensman's marginal, $\boldsymbol{a}_i^\top\mathbf{L} = k(\boldsymbol{x}_i, \mathbf{Z})\mathbf{K}_{uu}^{-1}\mathbf{L} = k(\boldsymbol{x}_i, \mathbf{Z})\mathbf{L}^{-\top} = \tilde{\boldsymbol{a}}_i^\top$, which gives both moments.
:::

No $\mathbf{K}_{uu}^{-1}$ remains. When a length-scale changes, the prior on $\boldsymbol{v}$ stays fixed, so $(\tilde{\boldsymbol{m}}, \tilde{\mathbf{S}})$ stay meaningful. In practice whitening makes the optimisation much better conditioned, and libraries use it by default.

### Natural gradients

Ordinary gradients on $(\boldsymbol{m}, \mathbf{S})$ ignore the geometry of distributions. A small change to $\mathbf{S}$ can be a large change to $q$, and the reverse. Natural gradients precondition by the Fisher information, and for this model they do something remarkable.

:::proposition[One natural-gradient step solves the Gaussian case]
Write $q(\boldsymbol{u})$ in exponential-family form with natural parameters $\boldsymbol{\eta} = (\mathbf{S}^{-1}\boldsymbol{m},\ -\tfrac12\mathbf{S}^{-1})$, and let $\boldsymbol{\eta}^*$ be those of $q^*$. For a Gaussian likelihood the natural gradient of the ELBO is $\boldsymbol{\eta}^* - \boldsymbol{\eta}$. So the update $\boldsymbol{\eta} \leftarrow \boldsymbol{\eta} + \rho\,(\boldsymbol{\eta}^* - \boldsymbol{\eta})$ with step size $\rho = 1$ lands exactly on the optimum.
:::

:::proof
**Step 1. Exponential-family facts.** $q(\boldsymbol{u}) = h(\boldsymbol{u})\exp(\langle\boldsymbol{\eta}, T(\boldsymbol{u})\rangle - A(\boldsymbol{\eta}))$ with $T(\boldsymbol{u}) = (\boldsymbol{u}, \boldsymbol{u}\boldsymbol{u}^\top)$. The mean parameters are $\boldsymbol{\mu} = \mathbb{E}_q[T(\boldsymbol{u})] = \nabla A(\boldsymbol{\eta})$, and the Fisher information is $\mathbf{F}(\boldsymbol{\eta}) = \nabla^2 A(\boldsymbol{\eta}) = \partial\boldsymbol{\mu}/\partial\boldsymbol{\eta}$, a symmetric matrix.

**Step 2. The natural gradient is the gradient in mean parameters.** By the chain rule, $\nabla_{\boldsymbol{\eta}}\mathcal{L} = \mathbf{F}\,\nabla_{\boldsymbol{\mu}}\mathcal{L}$. So $\mathbf{F}^{-1}\nabla_{\boldsymbol{\eta}}\mathcal{L} = \nabla_{\boldsymbol{\mu}}\mathcal{L}$.

**Step 3. The ELBO as a KL to $q^*$.** Step 3 of Titsias's proof gave $\mathcal{L}(q) = \text{const} - \mathrm{KL}[q \,\Vert\, q^*]$, and $q^*$ is in the same family. Writing both log-densities out,

$$
-\mathrm{KL}[q \,\Vert\, q^*] = \mathbb{E}_q\big[\langle\boldsymbol{\eta}^* - \boldsymbol{\eta}, T(\boldsymbol{u})\rangle\big] - A(\boldsymbol{\eta}^*) + A(\boldsymbol{\eta}) = \langle\boldsymbol{\eta}^* - \boldsymbol{\eta}, \boldsymbol{\mu}\rangle - A(\boldsymbol{\eta}^*) + A(\boldsymbol{\eta}).
$$

**Step 4. Differentiate in $\boldsymbol{\mu}$.** Here $\boldsymbol{\eta}$ is a function of $\boldsymbol{\mu}$. Write $\mathbf{J} = \partial\boldsymbol{\eta}/\partial\boldsymbol{\mu}$. Then

$$
\nabla_{\boldsymbol{\mu}}\big[\langle\boldsymbol{\eta}^* - \boldsymbol{\eta}, \boldsymbol{\mu}\rangle + A(\boldsymbol{\eta})\big] = (\boldsymbol{\eta}^* - \boldsymbol{\eta}) - \mathbf{J}^\top\boldsymbol{\mu} + \mathbf{J}^\top\nabla A(\boldsymbol{\eta}) = \boldsymbol{\eta}^* - \boldsymbol{\eta},
$$

because $\nabla A(\boldsymbol{\eta}) = \boldsymbol{\mu}$.

**Step 5. Take the step.** $\boldsymbol{\eta} + 1\cdot(\boldsymbol{\eta}^* - \boldsymbol{\eta}) = \boldsymbol{\eta}^*$.
:::

With minibatches the natural gradient is noisy, so $\rho < 1$. Salimbeni, Eleftheriadis and Hensman (2018) recommend natural-gradient steps for $q(\boldsymbol{u})$ and Adam for $\boldsymbol{\theta}$ and $\mathbf{Z}$. That combination is far more robust than Adam on everything.

## 9. How many inducing points, and where

### How many

The figure's answer was "about the domain width divided by the length-scale", and Burt, Rasmussen and van der Wilk (ICML 2019; JMLR 2020) made it precise. The KL divergence of the approximation is controlled by $t$, the total unexplained variance (Section 6). For well-placed inducing points, $t$ is controlled by how fast the eigenvalues of the kernel's integral operator decay under the input distribution.

- **Smooth kernels need very few.** For the squared-exponential kernel with Gaussian-distributed inputs in $D$ dimensions, $M = O(\log^D N)$ inducing points suffice to make the KL divergence small with high probability. As $N$ grows, $M$ barely has to.
- **Rough kernels need more.** For Matérn kernels the eigenvalues decay polynomially, and $M$ must grow polynomially in $N$; the smoother the kernel, the slower.
- **Concentrated inputs help.** Inputs packed into a small region need fewer inducing points than inputs spread thinly.

### Where

- **Greedy conditional variance.** Repeatedly add the training input with the largest current $c_i$. This is a pivoted Cholesky factorisation of $\mathbf{K}_{ff}$, it attacks $t$ directly, and it costs $O(NM^2)$. Burt et al. (2020) show that this kind of initialisation, close to sampling from a determinantal point process, achieves their bounds. It is the **Greedy** button above.
- **k-means.** Cluster centres of the inputs are a common, cheap alternative.
- **Then optimise.** $\mathbf{Z}$ are variational parameters, so gradient ascent on $\mathcal{F}$ or $\mathcal{L}$ can only improve the approximation.

:::remark[VFE and FITC behave differently]
Bauer, van der Wilk and Rasmussen (2016) compared the two in detail. FITC tends to underestimate the noise variance and can overfit badly, because its inducing inputs act as model parameters. The variational bound (VFE) behaves well as $M$ grows, but with too few inducing points it overestimates the noise: inflating $\sigma_n^2$ shrinks the trace penalty $t/(2\sigma_n^2)$. Section 11 shows a bound that reduces this bias.
:::

## 10. After 2013: a map of the field

The ideas above are the core. Most later work changes one ingredient: what $\boldsymbol{u}$ is, how $q$ is structured, or how the bound is computed.

### Inter-domain inducing variables

$\boldsymbol{u}$ need not be function values. Any linear functional works, for example $u_m = \int f(\boldsymbol{x})\,g_m(\boldsymbol{x})\,d\boldsymbol{x}$; only $\mathbf{K}_{uu}$ and $\mathbf{K}_{fu}$ change, by integrating the kernel against the $g_m$ (Lázaro-Gredilla and Figueiras-Vidal, 2009). With Fourier features on an interval, **variational Fourier features** make $\mathbf{K}_{uu}$ diagonal plus low rank, and much of the cost can be precomputed (Hensman, Durrande and Solin, 2017). On the sphere, spherical harmonics make $\mathbf{K}_{uu}$ exactly diagonal by the Funk–Hecke theorem (Dutordoir, Durrande and Hensman, 2020); the harmonics themselves are introduced in [Fourier and wavelets on the sphere](/aboutme/blog/fourier-and-wavelets-on-the-sphere/).

### Decoupled and orthogonal bases

The mean of $q$ usually needs many more basis functions than its covariance. Decoupled representations give each its own inducing set (Cheng and Boots, 2017), and orthogonally decoupled ones make the two sets complementary (Salimbeni et al., 2018). SOLVE-GP (Shi, Titsias and Mnih, 2020) splits $f$ into a part spanned by the inducing points and an independent orthogonal remainder with its own inducing points, which gives tighter bounds at similar cost.

### Nearest neighbours instead of low rank

Low-rank approximations struggle when the length-scale is short compared with the domain, because $M$ must then grow with the volume. Sparse-within-sparse GPs (Tran et al., 2021) let each data point use only its nearest inducing points. The variational nearest-neighbour GP (Wu, Pleiss and Cunningham, 2022) puts a Vecchia-style prior with a sparse precision on $\boldsymbol{u}$, with cost $O(K^3)$ per point for $K$ neighbours, and can use one inducing point per observation.

### Tighter bounds with conjugate gradients

The quadratic term of $\mathcal{F}$ is where the low-rank approximation loses most. Artemev, Burt and van der Wilk (2021) replace it with a bound tightened by a few conjugate-gradient iterations on the exact $\mathbf{K}_{ff} + \sigma_n^2\mathbf{I}$, which reduces hyperparameter bias.

### Sampling whole functions

Bayesian optimisation and Thompson sampling need posterior *functions*, not marginals. Pathwise conditioning (Wilson et al., 2020) draws them by Matheron's rule. A cheap random-feature prior sample is corrected by an inducing-point update, so the cost stays linear in the number of evaluation points. The rule itself is derived in the [primer](/aboutme/blog/understanding-gaussian-processes/), Section 6.

### Numerical stability

Inducing points that nearly coincide make $\mathbf{K}_{uu}$ singular. Jitter helps but biases the result. Terenin et al. (2024) give conditions on the minimum separation of inducing points under which the computations are provably stable, and build such sets with cover trees for low-dimensional problems.

### Or skip sparsity

On a GPU, exact GPs reach a million points with conjugate gradients and preconditioning, using only matrix-vector products (Gardner et al., 2018; Wang et al., 2019). When the hardware is there and the length-scale is short, exact inference can beat a sparse approximation.

## 11. 2025: beyond p(f | u)

For sixteen years the family $q(\boldsymbol{f}, \boldsymbol{u}) = p(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u})$ was taken as fundamental: it is what makes the $N \times N$ density cancel. In 2025 three papers relaxed it.

The weakness is visible in the bound. Given $\boldsymbol{u}$, each $f_i$ keeps its prior conditional variance $c_i$, *as if $y_i$ had never been observed*. But $y_i$ is a direct, noisy measurement of $f_i$, and it should shrink that variance.

:::theorem[A tighter collapsed bound (Titsias, ICML 2025)]
Let $q(\boldsymbol{f} \mid \boldsymbol{u}) = \mathcal{N}\big(\boldsymbol{f} \mid \mathbf{A}\boldsymbol{u},\ \mathbf{C}^{1/2}\mathbf{V}\mathbf{C}^{1/2}\big)$ with $\mathbf{V} = \operatorname{diag}(v_1, \dots, v_N)$, $v_i > 0$: the prior conditional mean, with a rescaled covariance. For a Gaussian likelihood, the optimal scales are

$$
v_i^* = \Big(1 + \frac{c_i}{\sigma_n^2}\Big)^{-1},
$$

the optimal $q(\boldsymbol{u})$ is unchanged, and the collapsed bound is

$$
\boxed{\ \mathcal{F}_{\mathrm{new}} = \log\mathcal{N}\big(\boldsymbol{y} \mid \mathbf{0},\ \mathbf{Q}_{ff} + \sigma_n^2\mathbf{I}\big) - \frac12\sum_{i=1}^{N}\log\Big(1 + \frac{c_i}{\sigma_n^2}\Big)\ }
$$

with $\mathcal{F} \leq \mathcal{F}_{\mathrm{new}} \leq \log p(\boldsymbol{y})$, and $\mathcal{F}_{\mathrm{new}} > \mathcal{F}$ whenever some $c_i > 0$.
:::

:::proof
We assume $\mathbf{C}$ is invertible to keep the algebra short; the paper treats the general case.

**Step 1. The ELBO of the new family.** With $q(\boldsymbol{f}, \boldsymbol{u}) = q(\boldsymbol{f} \mid \boldsymbol{u})\,q(\boldsymbol{u})$, the log-ratio inside the ELBO is $\log p(\boldsymbol{y} \mid \boldsymbol{f}) + \log\frac{p(\boldsymbol{f} \mid \boldsymbol{u})}{q(\boldsymbol{f} \mid \boldsymbol{u})} + \log\frac{p(\boldsymbol{u})}{q(\boldsymbol{u})}$. Taking expectations,

$$
\mathcal{L} = \mathbb{E}_q\big[\log p(\boldsymbol{y} \mid \boldsymbol{f})\big] - \mathbb{E}_{q(\boldsymbol{u})}\mathrm{KL}\big[q(\boldsymbol{f} \mid \boldsymbol{u}) \,\Vert\, p(\boldsymbol{f} \mid \boldsymbol{u})\big] - \mathrm{KL}\big[q(\boldsymbol{u}) \,\Vert\, p(\boldsymbol{u})\big].
$$

The middle term was zero before; now it is the price of departing from the prior conditional.

**Step 2. The conditional KL.** Both Gaussians have mean $\mathbf{A}\boldsymbol{u}$; the covariances are $\mathbf{C}^{1/2}\mathbf{V}\mathbf{C}^{1/2}$ and $\mathbf{C}$. The Gaussian KL formula gives

$$
\mathrm{KL} = \tfrac12\Big[\operatorname{tr}\big(\mathbf{C}^{-1}\mathbf{C}^{1/2}\mathbf{V}\mathbf{C}^{1/2}\big) - N + \log\lvert\mathbf{C}\rvert - \log\lvert\mathbf{C}^{1/2}\mathbf{V}\mathbf{C}^{1/2}\rvert\Big].
$$

The trace is $\operatorname{tr}(\mathbf{C}^{-1/2}\mathbf{V}\mathbf{C}^{1/2}) = \operatorname{tr}\mathbf{V}$ by the cyclic property, and $\lvert\mathbf{C}^{1/2}\mathbf{V}\mathbf{C}^{1/2}\rvert = \lvert\mathbf{C}\rvert\lvert\mathbf{V}\rvert$. So

$$
\mathrm{KL} = \tfrac12\sum_{i=1}^{N}\big(v_i - 1 - \log v_i\big),
$$

which does not depend on $\boldsymbol{u}$.

**Step 3. The expected log-likelihood given $\boldsymbol{u}$.** By the expected-log-likelihood lemma with covariance $\mathbf{C}^{1/2}\mathbf{V}\mathbf{C}^{1/2}$,

$$
\mathbb{E}_{q(\boldsymbol{f} \mid \boldsymbol{u})}\big[\log p(\boldsymbol{y} \mid \boldsymbol{f})\big] = \log\mathcal{N}(\boldsymbol{y} \mid \mathbf{A}\boldsymbol{u}, \sigma_n^2\mathbf{I}) - \frac{1}{2\sigma_n^2}\operatorname{tr}(\mathbf{V}\mathbf{C}),
$$

and $\operatorname{tr}(\mathbf{V}\mathbf{C}) = \sum_i v_i c_i$ because $\mathbf{V}$ is diagonal.

**Step 4. Separate the $\mathbf{V}$-terms.** Everything that involves $\mathbf{V}$ is a sum of independent one-dimensional terms,

$$
\sum_{i=1}^{N} g_i(v_i), \qquad g_i(v) = -\frac{c_i}{2\sigma_n^2}\,v - \tfrac12\big(v - 1 - \log v\big),
$$

and the rest is exactly the expression Titsias maximised over $q(\boldsymbol{u})$.

**Step 5. Maximise each $g_i$.** $g_i'(v) = -\frac{c_i}{2\sigma_n^2} - \frac12 + \frac{1}{2v}$, which is zero at $v = (1 + c_i/\sigma_n^2)^{-1}$. Since $g_i''(v) = -1/(2v^2) < 0$, this is the maximum.

**Step 6. Its value.** At $v_i^*$, $v_i^*(1 + c_i/\sigma_n^2) = 1$, so

$$
g_i(v_i^*) = -\tfrac12\big[v_i^*(1 + c_i/\sigma_n^2) - 1 - \log v_i^*\big] = -\tfrac12\log\Big(1 + \frac{c_i}{\sigma_n^2}\Big).
$$

**Step 7. Maximise over $q(\boldsymbol{u})$.** The remaining terms are those of Titsias's proof, Steps 2 to 5, so the optimal $q(\boldsymbol{u})$ is the same $q^*$ and contributes $\log\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \mathbf{Q}_{ff} + \sigma_n^2\mathbf{I})$.

**Step 8. Compare.** $\mathbf{V} = \mathbf{I}$ recovers the 2009 family, with $g_i(1) = -c_i/(2\sigma_n^2)$. And $\log(1 + a) \leq a$ for $a \geq 0$, with equality only at $a = 0$, so each new term is at least the old one. Finally $\mathcal{F}_{\mathrm{new}}$ is an ELBO, so it is at most $\log p(\boldsymbol{y})$.
:::

:::note[The optimal scale is one observation's worth of shrinkage]
$v_i^*c_i = \dfrac{c_i\,\sigma_n^2}{c_i + \sigma_n^2} = \Big(\dfrac{1}{c_i} + \dfrac{1}{\sigma_n^2}\Big)^{-1}$. That is the posterior variance of a Gaussian quantity with prior variance $c_i$ after one noisy observation with noise variance $\sigma_n^2$: precisions add. The new family lets each $f_i$ learn from its own $y_i$ what $\boldsymbol{u}$ could not tell it. Because $\mathbf{V}$ is diagonal it ignores the correlations among the residuals, just as the mean-field example of Section 2 ignored correlations. The same idea, one level down.
:::

What changes in practice:

- **Hyperparameters.** The penalty grows like $\log(1 + c_i/\sigma_n^2)$ instead of $c_i/\sigma_n^2$, so there is less pressure to inflate $\sigma_n^2$ to hide unexplained variance. Titsias reports less biased hyperparameters and often better predictions.
- **Predictions.** Test points get no extra parameters and $q^*(\boldsymbol{u})$ is unchanged, so Section 7's predictive formulas still apply.
- **Code.** In the stochastic SVGP bound the per-point term $-\frac{c_i}{2\sigma_n^2}$ becomes $-\frac12\log(1 + c_i/\sigma_n^2)$; the rest is untouched, minibatching included. The figure's readout shows how much of the 2009 gap this closes.

Two companion papers develop the idea. Bui, Ashman and Turner (TMLR 2025) independently relax the same assumption, choosing conditional posteriors with smaller variances than the prior at the training points, at unchanged cost. Bui and Titsias (2025) replace the diagonal $\mathbf{V}$ with a block-diagonal one, which provably tightens the bound further, and revisit Power-EP, which unifies VFE- and FITC-like methods, with these structured posteriors.

## 12. A practical recipe

- **$N$ up to about $10^4$:** use the exact GP. With a GPU and conjugate gradients, exact inference goes much further.
- **Gaussian likelihood, and $\hat{\mathbf{A}}$ fits in memory:** use the collapsed bound (SGPR), with the 2025 regulariser where your library has it. $M$ from a few hundred to a few thousand.
- **Larger $N$, streaming data, or a non-Gaussian likelihood:** use SVGP with minibatches, whitening, natural gradients for $q(\boldsymbol{u})$ and Adam for $\boldsymbol{\theta}$ and $\mathbf{Z}$.
- **Initialise $\mathbf{Z}$** with greedy conditional variance or k-means, then optimise.
- **Check the gap.** Compute $\mathcal{U} - \mathcal{F}$ on a subset. If it is large, increase $M$ before trusting the hyperparameters.
- **Short length-scales on a large domain:** consider nearest-neighbour variants (VNNGP) or inter-domain features rather than a huge $M$.
- **Numerics:** factorise $\mathbf{K}_{uu}$ in double precision with a small jitter, and keep inducing points from collapsing onto each other.

## Questions

:::question
With no inducing points ($M = 0$), what is $\mathcal{F}$, and what does it mean?
:::

:::answer
With no $\boldsymbol{u}$, $\mathbf{Q}_{ff} = \mathbf{0}$ and $t = \operatorname{tr}\mathbf{K}_{ff}$, so $\mathcal{F} = \log\mathcal{N}(\boldsymbol{y} \mid \mathbf{0}, \sigma_n^2\mathbf{I}) - \operatorname{tr}\mathbf{K}_{ff}/(2\sigma_n^2)$. The model explains the data as pure noise, and pays for all the prior variance of $f$ at the data, unexplained. It is the ELBO of the approximation $q(\boldsymbol{f}) = p(\boldsymbol{f})$, which ignores the data entirely.
:::

:::question
Show that at an inducing input $\boldsymbol{z}_j$ the predictive variance of $f(\boldsymbol{z}_j)$ is $S_{jj}$.
:::

:::answer
At $\boldsymbol{x}_* = \boldsymbol{z}_j$, $\boldsymbol{k}_u$ is the $j$-th column of $\mathbf{K}_{uu}$, so $\boldsymbol{a}_* = \mathbf{K}_{uu}^{-1}\boldsymbol{k}_u = \boldsymbol{e}_j$ and $k_{**} - \boldsymbol{k}_u^\top\mathbf{K}_{uu}^{-1}\boldsymbol{k}_u = k_{jj} - k_{jj} = 0$. The variance is $\boldsymbol{e}_j^\top\mathbf{S}\boldsymbol{e}_j = S_{jj}$: of course, since $f(\boldsymbol{z}_j) = u_j$.
:::

:::question
Why does the 2009 bound push the noise variance up when $M$ is too small?
:::

:::answer
$\sigma_n^2$ appears in the penalty $-t/(2\sigma_n^2)$. When the inducing points leave a lot of variance unexplained ($t$ large), raising $\sigma_n^2$ shrinks the penalty, at some cost in the DTC evidence. The optimiser trades data fit for a smaller penalty by calling the unexplained signal "noise". The 2025 penalty $\tfrac12\sum_i\log(1 + c_i/\sigma_n^2)$ grows only logarithmically in $c_i/\sigma_n^2$, so the incentive is weaker.
:::

:::question
For GP classification with a Bernoulli likelihood, which parts of the SVGP objective change, and which stay?
:::

:::answer
Only the $N$ one-dimensional expectations $\mathbb{E}_{q(f_i)}[\log p(y_i \mid f_i)]$ change, now computed by Gauss–Hermite quadrature. The marginals $q(f_i)$, the KL term, whitening, minibatching and the predictive distribution of $f_*$ are all unchanged. To predict the class probability, average the likelihood over $q(f_*)$, again by quadrature.
:::

:::question
What does one SVGP step cost with minibatch size $B$, and which term dominates when $B \approx M$?
:::

:::answer
The marginals cost $O(BM^2)$ (one triangular solve per point) and the Cholesky factor of $\mathbf{K}_{uu}$ plus the KL cost $O(M^3)$. With $B \approx M$ both are $O(M^3)$, so the inducing points, not the data, set the cost. That is why $M$ in the low thousands is the practical limit, and why structured $\mathbf{K}_{uu}$ (inter-domain features) or local methods are attractive.
:::

## References

- M. K. Titsias. *Variational learning of inducing variables in sparse Gaussian processes.* AISTATS, 2009. [PMLR](https://proceedings.mlr.press/v5/titsias09a.html)
- J. Hensman, N. Fusi, N. D. Lawrence. *Gaussian processes for big data.* UAI, 2013. [arXiv:1309.6835](https://arxiv.org/abs/1309.6835)
- J. Hensman, A. G. de G. Matthews, Z. Ghahramani. *Scalable variational Gaussian process classification.* AISTATS, 2015. [arXiv:1411.2005](https://arxiv.org/abs/1411.2005)
- A. G. de G. Matthews, J. Hensman, R. E. Turner, Z. Ghahramani. *On sparse variational methods and the Kullback–Leibler divergence between stochastic processes.* AISTATS, 2016. [arXiv:1504.07027](https://arxiv.org/abs/1504.07027)
- J. Quiñonero-Candela, C. E. Rasmussen. *A unifying view of sparse approximate Gaussian process regression.* JMLR, 2005. [JMLR](https://jmlr.org/papers/v6/quinonero-candela05a.html)
- E. Snelson, Z. Ghahramani. *Sparse Gaussian processes using pseudo-inputs.* NeurIPS, 2006.
- C. K. I. Williams, M. Seeger. *Using the Nyström method to speed up kernel machines.* NeurIPS, 2001.
- M. Bauer, M. van der Wilk, C. E. Rasmussen. *Understanding probabilistic sparse Gaussian process approximations.* NeurIPS, 2016. [arXiv:1606.04820](https://arxiv.org/abs/1606.04820)
- M. K. Titsias. *Variational inference for Gaussian and determinantal point processes.* NeurIPS Workshop on Advances in Variational Inference, 2014. (The upper bound of Section 6, as cited by Burt et al.)
- D. R. Burt, C. E. Rasmussen, M. van der Wilk. *Rates of convergence for sparse variational Gaussian process regression.* ICML, 2019. [arXiv:1903.03571](https://arxiv.org/abs/1903.03571)
- D. R. Burt, C. E. Rasmussen, M. van der Wilk. *Convergence of sparse variational inference in Gaussian processes regression.* JMLR, 2020.
- H. Salimbeni, S. Eleftheriadis, J. Hensman. *Natural gradients in practice: non-conjugate variational inference in Gaussian process models.* AISTATS, 2018. [arXiv:1803.09151](https://arxiv.org/abs/1803.09151)
- M. Lázaro-Gredilla, A. Figueiras-Vidal. *Inter-domain Gaussian processes for sparse inference using inducing features.* NeurIPS, 2009.
- J. Hensman, N. Durrande, A. Solin. *Variational Fourier features for Gaussian processes.* JMLR, 2017. [arXiv:1611.06740](https://arxiv.org/abs/1611.06740)
- V. Dutordoir, N. Durrande, J. Hensman. *Sparse Gaussian processes with spherical harmonic features.* ICML, 2020.
- C.-A. Cheng, B. Boots. *Variational inference for Gaussian process models with linear complexity.* NeurIPS, 2017.
- H. Salimbeni, C.-A. Cheng, B. Boots, M. Deisenroth. *Orthogonally decoupled variational Gaussian processes.* NeurIPS, 2018.
- J. Shi, M. K. Titsias, A. Mnih. *Sparse orthogonal variational inference for Gaussian processes.* AISTATS, 2020. [PMLR](https://proceedings.mlr.press/v108/shi20b.html)
- G.-L. Tran, D. Milios, P. Michiardi, M. Filippone. *Sparse within sparse Gaussian processes using neighbor information.* ICML, 2021. [PMLR](https://proceedings.mlr.press/v139/tran21a.html)
- L. Wu, G. Pleiss, J. P. Cunningham. *Variational nearest neighbor Gaussian process.* ICML, 2022. [arXiv:2202.01694](https://arxiv.org/abs/2202.01694)
- A. Artemev, D. R. Burt, M. van der Wilk. *Tighter bounds on the log marginal likelihood of Gaussian process regression using conjugate gradients.* ICML, 2021.
- J. T. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, M. P. Deisenroth. *Efficiently sampling functions from Gaussian process posteriors.* ICML, 2020. [arXiv:2002.09309](https://arxiv.org/abs/2002.09309)
- A. Terenin, D. R. Burt, A. Artemev, S. Flaxman, M. van der Wilk, C. E. Rasmussen, H. Ge. *Numerically stable sparse Gaussian processes via minimum separation using cover trees.* JMLR, 2024. [JMLR](https://jmlr.org/papers/v25/22-1170.html)
- J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, A. G. Wilson. *GPyTorch: blackbox matrix-matrix Gaussian process inference with GPU acceleration.* NeurIPS, 2018. [arXiv:1809.11165](https://arxiv.org/abs/1809.11165)
- K. A. Wang, G. Pleiss, J. R. Gardner, S. Tyree, K. Q. Weinberger, A. G. Wilson. *Exact Gaussian processes on a million data points.* NeurIPS, 2019. [arXiv:1903.08114](https://arxiv.org/abs/1903.08114)
- M. K. Titsias. *New bounds for sparse variational Gaussian processes.* ICML, 2025. [arXiv:2502.08730](https://arxiv.org/abs/2502.08730)
- T. D. Bui, M. Ashman, R. E. Turner. *Tighter sparse variational Gaussian processes.* TMLR, 2025. [arXiv:2502.04750](https://arxiv.org/abs/2502.04750)
- T. D. Bui, M. K. Titsias. *Sparse Gaussian processes: structured approximations and Power-EP revisited.* 2025. [arXiv:2507.02377](https://arxiv.org/abs/2507.02377)
- F. Leibfried, V. Dutordoir, S. T. John, N. Durrande. *A tutorial on sparse Gaussian processes and variational inference.* 2020. [arXiv:2012.13962](https://arxiv.org/abs/2012.13962)
- D. M. Blei, A. Kucukelbir, J. D. McAuliffe. *Variational inference: a review for statisticians.* JASA, 2017. [arXiv:1601.00670](https://arxiv.org/abs/1601.00670)
- C. M. Bishop. *Pattern Recognition and Machine Learning*, Chapter 10. Springer, 2006.
