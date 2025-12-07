---
title: "Concentration inequalities"
date: 2025-12-07
summary: "Introduction to concentration inequalities."
tags: ['math', 'theoretical-ml', 'inequalities']
---

## Introduction

Question: Why concentration inequalities are needed?

Concentration inequalities address the following problems:
- How can we estimate the confidence interval (range of values) which would be a good estimate?
- How can we determine the level of significance (confidence) of that estimate?

Some areas of common applications are as follows:
- Statistics
- Learning Theory
- Discrete mathematics
- Statistical mechanics
- Information theory
- High-dimensional geometry

## Inequalities

Some of the inequalities we are going to understand in this blog are as follows:
1. Markov
2. Chebyshev's
3. Chernoff bound
4. Hoeffding's Lemma and inequality

### Markov's inequality

> **Definition:**
> For a positive random variable (rv) $X \geq 0$ and $a > 0$, the probability that $X$ is no less than $a$ is less than or equal to the expectation of $X$ divided by $a$:
> $$ \mathbb{P}(X\geq a) \leq \frac{\mathbb{E}[X]}{a} $$

**Proof:**

$$
\begin{aligned}
\mathbb{E}[X] &= \int_{0}^{\infty} x p(x) \, dx \\
&= \int_{0}^{a} x p(x) \, dx + \int_{a}^{\infty} x p(x) \, dx \\
&\geq \int_{a}^{\infty} x p(x) \, dx \geq a \int_{a}^{\infty} p(x) \, dx \\
&= a \mathbb{P}(X \geq a)
\end{aligned}
$$

By rearranging we get:
$$ \mathbb{P}(X \geq a) \leq \frac{\mathbb{E}[X]}{a} $$

> **Corollary:**
> Given maximum value $U$ of a random variable $X$,
> $$ \mathbb{P}(X \leq a) \leq \frac{U - \mathbb{E}[X]}{U - a} $$

> **Note:**
> In this corollary there is no need for the random variable $X$ to be positive.

**Proof**:
$$
\begin{aligned}
\mathbb{P}(X \leq a) &= \mathbb{P}(U - X \geq U - a) \\
&\leq \frac{\mathbb{E}[U - X]}{U - a} \quad \text{(Applying Markov's inequality)} \\
&\leq \frac{U - \mathbb{E}[X]}{U - a}
\end{aligned}
$$

**Features:**
- This needs almost no assumption about the rv.
- Gives weaker bounds.

> **Note:**
> It is applied where rv is too complicated to be analyzed by more powerful inequalities.

## Chebyshev's Inequality

> **Definition:**
> For a rv $X$, if expectation and variance are finite, then $\forall a > 0$,
> $$ \mathbb{P}(|X - \mathbb{E}[X]| \geq a) \leq \frac{\operatorname{Var}(X)}{a^{2}} $$

**Proof:**
$$
\begin{aligned}
\mathbb{P}(|X - \mathbb{E}[X]| \geq a) &= \mathbb{P}((X - \mathbb{E}[X])^{2} \geq a^{2}) \\
&\leq \frac{\mathbb{E}[(X - \mathbb{E}[X])^{2}]}{a^{2}} \\
&=\frac{\operatorname{Var}(X)}{a^{2}}
\end{aligned}
$$
>[!note] 
>Usage of chebhyshev's inequality has great utility because it can be applied to any probability distribution in which the mean and variance are defined and bounded.


## Chernoff Bound

>[!info] Definition
>The generic Chernoff bound for a rv $X$ states,
> $$ \Pr\big(X\geq a\big) = \Pr\big(e^{ tX }\geq e^{ ta }\big) \quad \forall t>0$$

As $e^{ tX }\geq 0$ and is monotonically increasing function, we can use Markov's inequality,
$$
\Pr\big(X\geq a\big) \leq \frac{\mathbb{E}\big[e^{ tX }\big]}{e^{ ta }}
$$
When $X = X_{1}+X_{2}+\dots+X_{n}$ for any $t>0$,
$$
\Pr\big(X\geq a\big) \leq e^{ -ta }\mathbb{E}\big[\prod_{i}e^{ -tX_{i} }\big]
$$
For a better tighter bounds we can optimise over $t$.

