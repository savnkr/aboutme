---
title: "From Fourier to wavelets: putting non-stationarity in the basis"
description: "Fourier features made Gaussian processes scale, but every sinusoid spans the whole domain. Wavelets are local in space and frequency. How that one change gives fast, non-stationary kernels, and what it does for global weather fields."
date: 2026-10-23
updated: 2026-10-23
cover: /blog/wavelets/atoms.jpg
tags: [fourier, wavelets, kernels, gaussian-processes, climate]
---

Every scalable kernel method makes a quiet choice: **which basis to write functions in**. Choose sinusoids and you get speed and elegant theory, along with an assumption you may not have meant to make. This post is about that assumption, and how wavelets remove it.

## The Fourier transform: global by design

The Fourier transform writes a function as a superposition of waves,

$$
\hat f(\omega) = \int f(x)\, e^{-i\,\omega\cdot x}\, dx, \qquad f(x) = \frac{1}{(2\pi)^d}\int \hat f(\omega)\, e^{i\,\omega\cdot x}\, d\omega .
$$

Each wave $e^{i\omega\cdot x}$ has a perfectly defined frequency and no location at all: it spans the entire domain. That is ideal when the statistics of a signal are the same everywhere, and awkward when they are not.

## Bochner's theorem and random Fourier features

For Gaussian processes the Fourier picture comes with a theorem. **Bochner**: a continuous stationary kernel $k(x - y)$ is positive definite if and only if it is the Fourier transform of a non-negative measure,

$$
k(x - y) = \int e^{i\,\omega\cdot(x-y)}\, \mu(d\omega).
$$

Rahimi and Recht (2007) turned this into an algorithm. Sample frequencies from the spectral measure and phases uniformly,

$$
z(x) = \sqrt{\tfrac{2}{D}}\,\big[\cos(\omega_1^\top x + b_1), \dots, \cos(\omega_D^\top x + b_D)\big], \qquad \omega_i \sim \mu,\; b_i \sim \mathcal U[0, 2\pi],
$$

and $z(x)^\top z(y)$ is an unbiased estimate of $k(x,y)$. A GP becomes Bayesian linear regression on $D$ features: cost $O(ND^2)$ instead of $O(N^3)$.

**The catch** is in the theorem itself. Bochner characterises *stationary* kernels, so random Fourier features can only approximate kernels that look the same everywhere. Terrain, weather fronts and speech do not: they are smooth in one place and sharp in another.

## Time and frequency: you cannot have both sharply

To localise a frequency you need a window, and windows have a price. The Gabor–Heisenberg uncertainty principle bounds the spread in space and frequency of any function,

$$
\Delta x \;\Delta\omega \;\ge\; \tfrac12 .
$$

The short-time Fourier transform picks one window width for every frequency. Wavelets make a better trade: **narrow windows for high frequencies, wide windows for low ones**, so each scale gets the resolution it needs.

## The wavelet transform

Take one mother wavelet $\psi$, a localised oscillation with zero mean, and generate a family by scaling and shifting it. The continuous wavelet transform is

$$
\mathcal W_f(a, b) = \frac{1}{\sqrt{a}} \int f(x)\, \overline{\psi\!\Big(\frac{x - b}{a}\Big)}\, dx, \qquad a > 0,
$$

where $a$ is the scale and $b$ the location. It can be inverted whenever $\psi$ satisfies the **admissibility condition**

$$
C_\psi = \int \frac{|\hat\psi(\omega)|^2}{|\omega|}\, d\omega < \infty,
$$

which in particular forces $\hat\psi(0) = 0$: a wavelet has no DC component, it only responds to change.

My favourite example is the **Mexican hat**, $\psi(x) \propto (d - \lVert x\rVert^2)\,e^{-\lVert x\rVert^2/2}$, which is the negative Laplacian of a Gaussian. (That connection gets [a post of its own](/aboutme/blog/one-operator-i-keep-coming-back-to/).)

## Random wavelet features

Here is the idea behind my ICLR 2026 paper: **put the non-stationarity in the basis, not in extra latent functions.**

Define atoms that are local in both space and frequency, $\psi_{s,t}(x) = s^{-d/2}\,\psi\big((x - t)/s\big)$, and integrate them against a density over scales and locations:

$$
k(x, y) = \int_0^\infty \!\! \int_{\mathbb R^d} \psi_{s,t}(x)\, \psi_{s,t}(y)\; p(s, t)\; dt\, ds .
$$

This kernel is positive definite by construction, since it is an average of rank-one kernels. It is **non-stationary** whenever $p$ puts different scales in different places. Sampling atoms gives random features,

$$
z(x) = \tfrac{1}{\sqrt D}\big[\psi_{s_1,t_1}(x), \dots, \psi_{s_D,t_D}(x)\big], \qquad (s_i, t_i) \sim p,
$$

with log-uniform scales and uniform shifts. The estimator is unbiased with variance at most $B^2/D$, and it converges **uniformly** on a compact set with $D = O\big(\varepsilon^{-2}\, d\, \log(\operatorname{diam}\mathcal M / \varepsilon)\big)$ features. Inference stays a single closed-form solve,

$$
S_w = \big(I_D + \sigma^{-2} Z^\top Z\big)^{-1}, \qquad m_w = \sigma^{-2}\, S_w Z^\top y .
$$

![Random Fourier features vs random wavelet features: global sinusoids give the same grain everywhere, wavelet atoms give coarse and fine detail where needed](/aboutme/blog/wavelets/atoms.jpg)

The top row shows why Fourier features are stationary: every feature spans the whole domain, so their random sum has the same texture everywhere. The bottom row shows wavelet atoms at four scales: their sum can be smooth on the left and detailed on the right.

## On the globe

Weather is the archetypal non-stationary field: calm over oceans, sharp over mountains and fronts. Interpolating ERA5 reanalysis fields from a quarter of the grid, with the same solver and feature count for both, swapping sinusoids for wavelets cut held-out error by **10–66% across six variables**. For surface pressure it fell from 10.3 hPa to 2.6 hPa.

![ERA5 surface pressure, and the interpolation error of stationary (RFF) and wavelet (RWF) features](/aboutme/blog/wavelets/era5.jpg)

On the sphere, the natural Fourier basis is the spherical harmonics, the eigenfunctions of the Laplacian on $\mathbb S^2$, and the natural wavelets are needlets. That story, with proofs, is in [Fourier and wavelets on the sphere](/aboutme/blog/fourier-and-wavelets-on-the-sphere/).

## References

- A. Rahimi, B. Recht. *Random features for large-scale kernel machines.* NeurIPS, 2007.
- S. Mallat. *A Wavelet Tour of Signal Processing.* Academic Press, 3rd ed., 2008.
- I. Daubechies. *Ten Lectures on Wavelets.* SIAM, 1992.
- S. Kumar, S. Chakraborty. *Scalable random wavelet features: efficient non-stationary kernel approximation with convergence guarantees.* ICLR, 2026. [arXiv:2602.00987](https://arxiv.org/abs/2602.00987)
