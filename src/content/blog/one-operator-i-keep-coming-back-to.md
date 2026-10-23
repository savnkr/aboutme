---
title: "One operator I keep coming back to"
description: "The Laplacian is everywhere: heat, diffusion models, Wasserstein gradient flows, Gaussian processes, spherical harmonics, wavelets, even music. A tour of the operator that asks how different you are from your neighbours."
date: 2026-10-23
updated: 2026-10-23
cover: /blog/laplacian/cover.png
tags: [laplacian, diffusion-models, gaussian-processes, spectral, geometry]
---

<!-- First pass. Rewrite the opening and the closing in your own voice; the mathematics in between is checked. -->

Some ideas you learn once. Others you keep meeting, in a different coat each time, until you realise it was the same idea all along. For me that idea is the **Laplacian**.

It shows up when heat spreads through a metal plate, when a diffusion model turns an image into noise, when a Gaussian process decides how smooth a function should be, when a weather model describes the globe, and when a drum makes a sound. It is like a good analogy: short, and straight to the point.

This post follows that thread.

## What it measures

In $d$ dimensions the Laplacian is the sum of second derivatives,

$$
\Delta f = \nabla \cdot \nabla f = \sum_{i=1}^{d} \frac{\partial^2 f}{\partial x_i^2}.
$$

There are two ways to read it, and both are worth keeping.

**It is the divergence of the gradient.** It tells you how the gradient itself is changing: whether slopes pile up into a valley or spread apart over a peak. Where $\Delta f > 0$ the function curves up; where $\Delta f < 0$ it curves down.

**It compares a point with its neighbourhood.** Average $f$ over a small sphere of radius $r$ around $x$, and

$$
\langle f \rangle_{\partial B_r(x)} - f(x) = \frac{r^2}{2d}\, \Delta f(x) + O(r^4).
$$

So $\Delta f(x)$ measures *how different $x$ is from the average of its surroundings*. On a graph this becomes literal: $(Lf)_i = \sum_j w_{ij}\,(f_i - f_j)$, the weighted sum of disagreements with your neighbours (with the sign convention $L = -\Delta$).

Hold on to that second reading. Everything below is a consequence of it.

## Heat: differences get smoothed away

If every point keeps moving towards the average of its neighbours, you get the heat equation

$$
\partial_t u = \Delta u.
$$

In Fourier space the Laplacian is just multiplication by $-\lVert\omega\rVert^2$, so each frequency decays on its own:

$$
\hat u(\omega, t) = e^{-\lVert\omega\rVert^2 t}\, \hat u(\omega, 0).
$$

High frequencies die first. Sharp edges round off long before the broad shapes fade.

![Heat diffusion smoothing a signal: sharp edges vanish first](/aboutme/blog/laplacian/heat.png)

The solution is a convolution with the **heat kernel**,

$$
G_t(x) = (4\pi t)^{-d/2} \exp\!\Big(-\frac{\lVert x\rVert^2}{4t}\Big).
$$

Look closely and you will recognise it. The RBF kernel every GP practitioner uses, $\exp(-\lVert x-x'\rVert^2 / 2\ell^2)$, *is* the heat kernel at time $t = \ell^2/2$, up to a constant. Choosing a length-scale is choosing how long heat has been allowed to flow.

## Diffusion models: heat on probability densities

Now let the thing that diffuses be a probability density. Add Brownian noise to data, $x_t = x_0 + \sqrt{t}\,\varepsilon$ with $\varepsilon \sim \mathcal N(0, I)$, and the density of $x_t$ obeys

$$
\partial_t p_t = \tfrac12\, \Delta p_t .
$$

The forward process of a diffusion model is the heat equation applied to the data distribution. Images blur into noise exactly the way the signal above flattens out.

Generation runs this backwards, and to do that you need the **score** $\nabla \log p_t$. The Laplacian is hiding there too, through Tweedie's formula:

$$
\mathbb E[x_0 \mid x_t] = x_t + t\, \nabla \log p_t(x_t).
$$

The best denoiser is one step along the score, and the score is what heat flow leaves behind.

## Gradient flows: the heat equation is a descent

A beautiful result of Jordan, Kinderlehrer and Otto (1998): the heat equation is the **gradient flow of entropy** in the Wasserstein-2 geometry of probability distributions. Add a potential and you get the Fokker–Planck equation,

$$
\partial_t p = \nabla\cdot\big(p\,\nabla V\big) + \Delta p,
$$

which is the Wasserstein gradient flow of $\mathrm{KL}(p \,\Vert\, e^{-V})$. Langevin sampling is just following it with particles.

This is where my current work lives. **Drifting models** (Deng et al., 2026) train a one-step generator by moving its samples along a kernel-defined drift field until the generated and data distributions agree. Turan, Dufour and Ovsjanikov (2026) show this drift is a Wasserstein gradient flow of a kernel-smoothed KL, and that with a Gaussian kernel the drift is a difference of scores of smoothed distributions. Their spectral analysis finds a **high-frequency bottleneck** for the Gaussian kernel.

That is the heat equation again: Gaussian smoothing damps frequency $\omega$ by $e^{-\lVert\omega\rVert^2 t}$, so fine detail is the last thing to be learned. Choosing the kernel in a drifting model is choosing which frequencies you are willing to wait for, which is exactly a question about multi-scale bases and wavelets.

## Gaussian processes: smoothness as an operator

Whittle (1954), and later Lindgren, Rue and Lindström (2011), showed that a Matérn Gaussian process is the solution of a stochastic PDE in the Laplacian,

$$
\Big(\tfrac{2\nu}{\ell^2} - \Delta\Big)^{\frac{\nu}{2} + \frac{d}{4}} f = \mathcal W,
$$

driven by white noise $\mathcal W$. Its spectral density is $S(\omega) \propto \big(\tfrac{2\nu}{\ell^2} + \lVert\omega\rVert^2\big)^{-\nu - d/2}$, and as $\nu \to \infty$ it becomes the RBF, the heat kernel again.

The payoff is that this definition does not care where you are. Replace $\lVert\omega\rVert^2$ by the eigenvalues $\lambda_n$ of the Laplacian on any space, a sphere, a mesh, a graph, and you get a valid Matérn kernel there (Borovitskiy et al., 2020):

$$
k(x, x') \propto \sum_{n} \Big(\tfrac{2\nu}{\ell^2} + \lambda_n\Big)^{-\nu - \frac d2}\, \phi_n(x)\, \phi_n(x').
$$

Smoothness is no longer a formula in coordinates. It is a statement about which eigenfunctions of the Laplacian you trust.

## The sphere: harmonics of the globe

On the sphere those eigenfunctions are the **spherical harmonics**,

$$
\Delta_{\mathbb S^2}\, Y_\ell^m = -\ell(\ell+1)\, Y_\ell^m,
$$

the Fourier basis of the globe. Weather and climate models are written in them, and by the addition theorem, $\sum_m Y_\ell^m(x)\,Y_\ell^m(x') = \frac{2\ell+1}{4\pi} P_\ell(x\cdot x')$, every isotropic kernel on the sphere is a series in Legendre polynomials with non-negative coefficients (Schoenberg's theorem, the sphere's version of Bochner; worked through in [Fourier and wavelets on the sphere](/aboutme/blog/fourier-and-wavelets-on-the-sphere/)). Drag the sliders:

<figure class="sphere-embed" data-sphere>
  <canvas data-figure="sphere" aria-label="A rotating sphere coloured by a spherical harmonic"></canvas>
  <figcaption>
    <span class="mono">Y<sub data-show-l>3</sub><sup data-show-m>2</sup> · eigenvalue −<span data-eig>12</span></span>
    <label>ℓ <input type="range" min="0" max="8" step="1" value="3" data-l /></label>
    <label>m <input type="range" min="-3" max="3" step="1" value="2" data-m /></label>
  </figcaption>
</figure>

Higher $\ell$ means finer structure and a larger eigenvalue, so under heat flow it fades faster. A weather pattern the size of a continent outlives a thunderstorm for the same reason a broad bump outlives a sharp edge.

## Wavelets: the Laplacian of a Gaussian

The Mexican-hat wavelet is, up to sign and scale, the Laplacian of a Gaussian:

$$
\psi(x) \;\propto\; -\Delta\, e^{-\lVert x\rVert^2/2} = \big(d - \lVert x\rVert^2\big)\, e^{-\lVert x\rVert^2/2}.
$$

Marr and Hildreth used this to find edges in images: blur with heat, then ask where the Laplacian changes sign. In my [random wavelet features](/aboutme/blog/fourier-to-wavelets/) it is the atom that makes a kernel local in space *and* frequency, which is what lets a Gaussian process be smooth in one region and sharp in another.

## Music: you can hear the eigenvalues

A vibrating string obeys $\partial_t^2 u = c^2\, \Delta u$. Its natural modes are the eigenfunctions of the Laplacian on the string, $\sin(n\pi x / L)$, and the notes it can play are set by the eigenvalues. The harmonics of a guitar *are* the spectrum of a Laplacian.

In 1966 Mark Kac asked the question that made this famous: *can one hear the shape of a drum?* If you know every eigenvalue, do you know the shape? Gordon, Webb and Wolpert answered in 1992: not always, there are differently shaped drums that sound exactly alike. The spectrum tells you a great deal, but not everything.

## And life

<!-- This section is yours. A first attempt to react to, not keep. -->

The Laplacian asks one question everywhere: *how different are you from what surrounds you?* Left alone, the answer shrinks. Heat smooths, noise blurs, sharp edges round off.

But some shapes survive diffusion without changing, they only grow quieter. Those are the eigenfunctions, the patterns that are already at peace with the operator. Maybe that is a decent definition of character: not refusing to be shaped by your surroundings, but staying recognisably yourself while they act on you.

And like the drum, you can learn a lot about someone from how they resonate, but never quite everything. I find that reassuring.

## Further reading

- R. Jordan, D. Kinderlehrer, F. Otto. *The variational formulation of the Fokker–Planck equation.* SIAM J. Math. Anal., 1998.
- F. Lindgren, H. Rue, J. Lindström. *An explicit link between Gaussian fields and Gaussian Markov random fields: the SPDE approach.* JRSS-B, 2011.
- V. Borovitskiy, A. Terenin, P. Mostowsky, M. P. Deisenroth. *Matérn Gaussian processes on Riemannian manifolds.* NeurIPS, 2020.
- M. Deng, H. Li, T. Li, Y. Du, K. He. *Generative modeling via drifting.* [arXiv:2602.04770](https://arxiv.org/abs/2602.04770), 2026.
- E. Turan, N. Dufour, M. Ovsjanikov. *Generative drifting is secretly score matching: a spectral and variational perspective.* [arXiv:2603.09936](https://arxiv.org/abs/2603.09936), 2026.
- M. Kac. *Can one hear the shape of a drum?* Amer. Math. Monthly, 1966.
