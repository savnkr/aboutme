---
title: "Convex Optimization"
date: 2025-12-07
summary: "Introduction to concentration inequalities."
tags: ['math', 'theoretical-ml', 'inequalities']
---

## Introduction

Question: Why concenteration inequalities are needed?

Concenteration inequalities address the following problems:
- How can be estimate the confidence internval (range of values) which would be a good estimate?
- How can we determine the level of significance (confidence) of that estimate?

Some areas of common applications are as follows:
- Statistics
- Leanring Theory
- Discrete mathematics
- Statistical mechanics
- Information theory
- High-dimensional geometry

## Inequalities

Some of the inequalities we are going to understand in this blog is as follows:
1. Markov
2. Chebyshev's
3. Chernoff boun
4. Hoeffding's Lemma and inequality

### Markov's inequality

>[! definition] Definition:
>  For  a positive random variable (rv) $X \geq 0$ and $a=0$, the probability that  $X$ is no less than $a$ is less than or equal to the expecation of $X$ divided by $a$:
>  $$ \mathbb{P}r[X\geq a] \leq \frac{\mathbb{E}[X]}{a}$$

proof:
$$
\begin{align}
\mathbb{E}[X] &= \int_{0}^{\infty} xp(x) \, dx \\ \\
&= \int_{0}^{a} xp(x) \, dx + \int_{a}^{\infty}xp(x)  \, dx \\ \\
&\geq \int_{a}^{\infty} xp(x)  \, dx \geq a\int_{a}^{\infty} p(x) \, dx \\ \\
&= a\mathbb{P}r(X\geq) \\
\text{By rearanging we get:} \\
 Pr(X\geq a) &= \frac{\mathbb{E}[X]}{a} 
\end{align}

$$
> [!note]- **Proof**
> This is a simple proof:
> $$
> \begin{align}
> \mathbb{E}[X] 
>     &= \int_{0}^{\infty} x p(x)\, dx \\[6pt]
> &= \int_{0}^{a} x p(x)\, dx 
>    + \int_{a}^{\infty} x p(x)\, dx \\[6pt]
> &\ge \int_{a}^{\infty} x p(x)\, dx 
>    \ge a \int_{a}^{\infty} p(x)\, dx \\[6pt]
> &= a\, \Pr(X \ge a).\\
> \text{Rearranging gives Markov's inequality:} \\
> \Pr(X \ge a) &= \frac{\mathbb{E}[X]}{a}.
>\end{align}
> $$

>[!note] Corollary
>Given maximum value $U$ of a random variable $X$,
> $$ \Pr\big(X\leq a\big) \leq \frac{U-\mathbb{E}\big[X\big]}{U-a}$$

>[!note]
>In this corollary there is no need for the random variable $X$ to be positive.

**Proof**:
$$
\begin{align}
\Pr\big(X\leq a\big) &= \Pr\big(U-X \geq U-a\big)   \\
 & \leq \frac{\mathbb{E}\big[U-X\big]}{U-a} \quad \text{(Applying Markov's inequality)}\\
 & \leq \frac{U-\mathbb{E}\big[X\big]}{U-a}\\
\end{align}
$$
Features:
- This needs alomost no assumption about the rv
- Gives weaker bounds
>[!Note]
>It is applied where rv is too complicated to be analyzed by more powerful inequalities

## Chebyshev's Inequality

>[!info]  Definition:
>For a rv $X$ expectation and variance should be finite, then $\forall  a >0$, 
> $$ \Pr\big(|X -\mathbb{E}\big[X\big]|\geq a\big) \leq \frac{\operatorname{Var}\big(X\big)}{a^{2}}$$

**Proof:** 
$$
\begin{align}
\Pr\big(|X-\mathbb{E}\big[X\big]|\geq a\big)  & = \Pr\big((X-\mathbb{E}\big[x\big])^{2}\geq a^{2}\big) \\
 & \leq \frac{\mathbb{E}\big[(X-\mathbb{E}\big[X\big])^{2}\big]}{a^{2}} \\
 & =\frac{\operatorname{Var}\big(X\big)}{a^{2}}
\end{align}
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

