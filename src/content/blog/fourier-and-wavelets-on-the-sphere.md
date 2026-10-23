---
title: "Fourier and wavelets on the sphere"
description: "Why a flat FFT breaks at the poles, how spherical harmonics fix it, what Bochner's theorem becomes on a globe, and how needlets give a wavelet transform for the Earth. A from-scratch tutorial with proofs."
date: 2026-10-23
updated: 2026-10-23
cover: /blog/sphere/needlets.png
tags: [spherical-harmonics, wavelets, needlets, gaussian-processes, climate]
---

The [previous post](/aboutme/blog/fourier-to-wavelets/) went from Fourier features to wavelets on flat space. Weather and climate live on a sphere, and most of that story has to be rebuilt there. This post does it from scratch: Fourier analysis on $S^2$, then Gaussian processes, then wavelets. Each step comes with the short argument for why it is true.

## 1. Why a flat FFT fails on a globe

Use colatitude $\theta \in [0, \pi]$ (angle from the north pole) and longitude $\phi \in [0, 2\pi)$:

$$
p(\theta, \phi) = (\sin\theta\cos\phi,\; \sin\theta\sin\phi,\; \cos\theta), \qquad d\omega = \sin\theta\, d\theta\, d\phi, \qquad \int_{S^2} d\omega = 4\pi .
$$

It is tempting to treat the latitude–longitude grid as a rectangle and run a 2-D FFT. Three things go wrong.

1. **The chart is singular.** At $\theta = 0$ a whole row of the grid is one point. A smooth function must take the same value along it, and a longitudinal wave $e^{im\phi}$ must vanish at the pole like $\sin^{|m|}\theta$. A 2-D Fourier series knows neither fact, so truncating it leaves fields that jump at the pole.
2. **Latitude is not periodic.** $\theta$ lives on an interval, not a circle, so periodic extension puts a kink at each pole and Gibbs ringing follows.
3. **Resolution is not uniform.** Meridians converge. A stationary kernel in latitude–longitude coordinates is *not* isotropic on the sphere, and an unweighted grid average is biased. The cap north of $60^\circ$ is only $\tfrac{1-\cos 30^\circ}{2} \approx 6.7\%$ of the Earth's area, but on an equiangular grid it holds about 17% of the points. Every honest score on a global grid is area-weighted.

The fix is to use the sphere's own "pure frequencies".

## 2. Spherical harmonics, from the Laplacian

On the line, $e^{ikx}$ is an eigenfunction of $d^2/dx^2$. On the sphere the analogue is the **Laplace–Beltrami operator**,

$$
\Delta_{S^2} f = \frac{1}{\sin\theta}\,\partial_\theta\big(\sin\theta\,\partial_\theta f\big) + \frac{1}{\sin^2\theta}\,\partial_\phi^2 f .
$$

Two properties do all the work. It is **symmetric and non-positive** (integrate by parts on a surface without boundary: $\int f\,\Delta g = -\int \nabla f\cdot\nabla g$), so eigenvalues are real and eigenfunctions with different eigenvalues are orthogonal. And it is **rotation invariant**, because rotations are isometries.

**Why the eigenvalues are $\ell(\ell+1)$.** Separate variables, $f = \Theta(\theta)\,e^{im\phi}$ with $m \in \mathbb Z$, and set $t = \cos\theta$. The equation $\Delta_{S^2} f = -\Lambda f$ becomes the associated Legendre equation, which is singular at the poles $t = \pm 1$. For $m = 0$ try a power series $\Theta = \sum_k a_k t^k$. The recursion is

$$
a_{k+2} = \frac{k(k+1) - \Lambda}{(k+1)(k+2)}\, a_k ,
$$

so $a_{k+2}/a_k \to 1$. Unless the series stops, it behaves like $\sum t^{2k}$ and blows up logarithmically at a pole. It stops exactly when $\Lambda = \ell(\ell+1)$ for an integer $\ell$, and then the solution is the Legendre polynomial $P_\ell$. **Boundedness at the poles quantises the frequencies.**

The spherical harmonics are

$$
Y_{\ell m}(\theta,\phi) = N_{\ell m}\, P_\ell^{m}(\cos\theta)\, e^{im\phi}, \qquad \Delta_{S^2} Y_{\ell m} = -\ell(\ell+1)\, Y_{\ell m}, \qquad |m| \le \ell ,
$$

an orthonormal basis of $L^2(S^2)$. Degree $\ell$ has $2\ell+1$ harmonics, so the band-limited space $V_L = \bigoplus_{\ell \le L} \mathcal H_\ell$ has dimension $(L+1)^2$.

**Degree is scale, order is direction.** $\mathrm{Re}\,Y_{\ell m}$ has $m$ nodal meridians and $\ell - m$ nodal circles of latitude, so $\ell$ counts how fine the pattern is and $m$ how it is oriented: zonal bands ($m = 0$), "orange slices" ($m = \ell$), checkerboards in between. Try it:

<figure class="sphere-embed" data-sphere>
  <canvas data-figure="sphere" aria-label="A rotating sphere coloured by a spherical harmonic"></canvas>
  <figcaption>
    <span class="mono">Y<sub data-show-l>3</sub><sup data-show-m>2</sup> · eigenvalue −<span data-eig>12</span></span>
    <label>ℓ <input type="range" min="0" max="8" step="1" value="3" data-l /></label>
    <label>m <input type="range" min="-3" max="3" step="1" value="2" data-m /></label>
  </figcaption>
</figure>

**Wavelength.** Matching $\lvert k\rvert^2 a^2 = \ell(\ell+1)$ on a sphere of radius $a$ gives the Jeans relation $\lambda_\ell = 2\pi a / \sqrt{\ell(\ell+1)}$. On Earth ($a = 6371$ km):

| $\ell$ | 1 | 8 | 30 | 64 | 120 |
|---|---|---|---|---|---|
| $\lambda_\ell$ (km) | 28 300 | 4 700 | 1 310 | 620 | 330 |

## 3. Two theorems that make everything easy

**The addition theorem.** For all points $p, q$,

$$
\sum_{m=-\ell}^{\ell} Y_{\ell m}(p)\,\overline{Y_{\ell m}(q)} = \frac{2\ell+1}{4\pi}\, P_\ell(p\cdot q).
$$

*Why.* The left side is the reproducing kernel of $\mathcal H_\ell$, so it does not depend on the choice of basis. Rotations map $\mathcal H_\ell$ to itself, so the kernel is rotation invariant. Fix $q$ at the north pole: a function on $\mathcal H_\ell$ invariant under rotations about the axis is zonal, so it is $c\,P_\ell(\cos\theta)$. Integrating the diagonal gives $4\pi c = 2\ell+1$. A corollary: $\sum_m \lvert Y_{\ell m}(p)\rvert^2 = \frac{2\ell+1}{4\pi}$ at *every* point. The poles are not special.

**Funk–Hecke: convolution is multiplication per degree.** For a zonal kernel $k(p\cdot q)$,

$$
\int_{S^2} k(p\cdot q)\, Y_{\ell m}(q)\, d\omega(q) = \hat k_\ell\, Y_{\ell m}(p), \qquad \hat k_\ell = 2\pi \int_{-1}^{1} k(t)\, P_\ell(t)\, dt .
$$

The operator commutes with rotations, so by Schur's lemma it is a scalar on each degree. Averaging over a spherical cap of radius $\rho$, for example, multiplies degree $\ell$ by

$$
\hat k_\ell = \frac{P_{\ell-1}(\cos\rho) - P_{\ell+1}(\cos\rho)}{(2\ell+1)(1 - \cos\rho)}, \qquad \hat k_0 = 1,
$$

a multiplier that decays with $\ell$. This is the spherical version of "smoothing is a low-pass filter". Spectral neural operators on the sphere learn exactly such a $\hat k_\ell$.

**Rotations mix orders, never degrees.** Rotating a field acts on each $\mathcal H_\ell$ by a unitary Wigner matrix, so the angular power spectrum $\widehat C_\ell = \frac{1}{2\ell+1}\sum_m \lvert f_{\ell m}\rvert^2$ is rotation invariant.

## 4. Sampling, aliasing and the cost of a transform

When does a grid compute the coefficients exactly? Gauss–Legendre nodes in latitude integrate polynomials of degree $\le 2N - 1$ exactly, and $N_\phi$ equispaced longitudes integrate $e^{im\phi}$ for $\lvert m\rvert < N_\phi$. The analysis integrand $f\,\overline{Y_{\ell m}}$ has degree $\le 2L$, so:

> a Gauss grid with $N_{\text{lat}} \ge L+1$ and $N_{\text{lon}} \ge 2L+1$ reads every band-limited field in $V_L$ exactly.

A product of two fields in $V_L$ lives in $V_{2L}$, so projecting it back exactly needs exactness at degree $3L$. That is the "3/2 rule" of spectral weather models. Break the rule and you get **aliasing**: in longitude an exact fold ($m$ looks like $m \pm N_{\text{lon}}$), in latitude a quadrature error without a simple fold.

A spherical harmonic transform is an FFT per latitude ring plus a Legendre sum per $(\ell, m)$: $O(L^3)$ in its plain matrix form, which is what GPU libraries use at $L \le 10^3$.

## 5. Gaussian processes on the sphere

**Schoenberg is Bochner for the sphere.** On $\mathbb R^d$, Bochner says a stationary kernel is valid exactly when its Fourier transform is non-negative. On $S^2$, Schoenberg (1942) says: a continuous $\kappa(p\cdot q)$ is a valid covariance if and only if

$$
\kappa(t) = \sum_{\ell \ge 0} a_\ell\, P_\ell(t), \qquad a_\ell \ge 0, \qquad \sum_\ell a_\ell < \infty .
$$

The "if" direction is one line: by the addition theorem each $P_\ell(p\cdot q)$ is a sum of rank-one kernels. The "only if" direction is Funk–Hecke: the integral operator has eigenvalue $\hat\kappa_\ell$ on $\mathcal H_\ell$, and a valid covariance has non-negative eigenvalues. In words, **an isotropic kernel *is* a non-negative spectrum** $C_\ell$, with

$$
k(p,q) = \sum_\ell C_\ell\, \frac{2\ell+1}{4\pi}\, P_\ell(p\cdot q), \qquad \mathrm{Var} f(p) = \sum_\ell \frac{2\ell+1}{4\pi}\, C_\ell .
$$

*A trap.* Kernels of the **chordal** distance $\lVert p - q\rVert$ are always valid on $S^2$, since they are restrictions of kernels on $\mathbb R^3$. Kernels of the **geodesic** distance need not be: the squared exponential in great-circle distance is not positive definite for every length scale (Gneiting 2013). Defining kernels through $C_\ell$ avoids the problem entirely.

**Sampling is a filter on white noise.** Draw $\varepsilon_{\ell m} \sim \mathcal N(0,1)$ independently and set

$$
f = \sum_{\ell, m} \sqrt{C_\ell}\, \varepsilon_{\ell m}\, Y_{\ell m}
$$

(Karhunen–Loève). Samples are square-integrable iff $\sum_\ell (2\ell+1)\,C_\ell < \infty$. White noise ($C_\ell \equiv 1$) is not a random function on the sphere, only a random distribution; on a band-limited $V_L$ it is perfectly fine.

**Matérn on the sphere** comes from the Laplacian, exactly as on flat space. Replace $\Delta$ by $\Delta_{S^2}$ in the SPDE $\big(\tfrac{2\nu}{\kappa^2} - \Delta\big)^{\nu/2 + d/4} f = \mathcal W$. The operator is diagonal in $Y_{\ell m}$, so

$$
C_\ell \;\propto\; \Big(\frac{2\nu}{\kappa^2} + \ell(\ell+1)\Big)^{-\nu - 1} \qquad (d = 2),
$$

where $\kappa$ is the length scale. As $\nu \to \infty$, $\big(1 + \kappa^2\ell(\ell+1)/2\nu\big)^{-\nu} \to e^{-\kappa^2\ell(\ell+1)/2}$: the heat kernel at time $t = \kappa^2/2$, exactly as the RBF is on flat space. (The [Laplacian post](/aboutme/blog/one-operator-i-keep-coming-back-to/) tells this part as a story.)

**The limit of all of this** is stationarity. An isotropic GP must use the same variance and the same correlation length everywhere: storm tracks and calm tropical oceans get one compromise. That is the same problem random Fourier features have on flat space, and it has the same cure: wavelets.

## 6. Wavelets on the sphere: needlets

Spherical harmonics are perfectly localised in scale and not at all in space. We want both, approximately.

**Step 1: split the spectrum into octaves.** Take a smooth cutoff $\chi$ equal to 1 on $[0, \tfrac12]$ and 0 on $[1, \infty)$, and define windows by $b(t)^2 = \chi(t/2) - \chi(t)$, $\;b_j(\ell) = b(\ell / 2^j)$. Window $j$ covers $2^{j-1} < \ell < 2^{j+1}$, and the squares form a **partition of unity**:

$$
\sum_{j \ge 0} b_j(\ell)^2 = 1 \quad (\ell \ge 1).
$$

*Proof by telescoping:* $\sum_{j=0}^{J} b_j(\ell)^2 = \chi(\ell/2^{J+1}) - \chi(\ell) \to 1 - 0$.

![Littlewood–Paley windows: one per octave, their squares sum to one](/aboutme/blog/sphere/windows.png)

**Step 2: centre a filtered addition-theorem kernel at each point.** With a cubature rule $\{\xi_{jk}, \varpi_{jk}\}$ exact up to degree $2^{j+2}$, the needlets (Narcowich, Petrushev and Ward, 2006) are

$$
\psi_{jk}(p) = \sqrt{\varpi_{jk}}\, \sum_\ell b_j(\ell)\, \frac{2\ell+1}{4\pi}\, P_\ell(p\cdot\xi_{jk}) .
$$

![Needlet profiles at four scales: each is a bump that halves in width per octave](/aboutme/blog/sphere/needlets.png)

They have the two properties we wanted.

- **A tight frame.** For every zero-mean $f$, $\;f = \sum_{j,k} \langle f, \psi_{jk}\rangle\, \psi_{jk}$ and $\lVert f\rVert^2 = \sum_{j,k} \lvert\langle f,\psi_{jk}\rangle\rvert^2$. Cubature exactness turns $\sum_k \psi_{jk}(p)\psi_{jk}(q)$ into $\sum_\ell b_j(\ell)^2 \frac{2\ell+1}{4\pi} P_\ell(p\cdot q)$, and the partition of unity sums those to the identity.
- **Localised in space.** For every $M$, $\;\lvert\psi_{jk}(p)\rvert \le C_M\, 2^j \big/ \big(1 + 2^j d(p, \xi_{jk})\big)^M$. As on the line, a smooth, compactly supported window in frequency buys fast decay in space.

A tight frame is redundant: it has more atoms than dimensions. That redundancy is what lets it be both tight and localised, something no orthonormal basis on the sphere can do as cleanly.

**One field, two pictures.** A random needlet expansion $\sum_k w_{jk}\psi_{jk}$ with white $w$, and band-pass-filtered white noise $\sum_{\ell,m} b_j(\ell)\,\varepsilon_{\ell m} Y_{\ell m}$, have the *same* covariance $\sum_\ell b_j(\ell)^2 \frac{2\ell+1}{4\pi}P_\ell(p\cdot q)$, so they are the same Gaussian law. In practice the second picture is cheaper: one transform and no cubature points. For isotropic fields, needlet coefficients at the same scale and well-separated centres also become asymptotically uncorrelated (Baldi, Kerkyacharian, Marinucci and Picard, 2009). That is a big part of why "one variance per scale and place" is a useful way to think about fields on the globe.

## Self-check

1. *Why must $\Lambda = \ell(\ell+1)$ with integer $\ell$?* Otherwise the Legendre series does not terminate and blows up at a pole.
2. *A Gauss grid has 64 latitudes. What band limit does it read exactly? And for exact products?* $L = 63$; for products, $\lceil (3L+1)/2 \rceil \le 64$ gives $L \le 42$.
3. *Is $\kappa(t) = t^2$ a valid kernel on $S^2$?* $t^2 = \tfrac13 P_0 + \tfrac23 P_2$, non-negative coefficients: yes. *And $t^2 - \tfrac12$?* $-\tfrac16 P_0 + \tfrac23 P_2$: no.
4. *Is $\widehat C_\ell$ rotation invariant? Is $\lvert f_{\ell 0}\rvert^2$?* Yes; no, because $m = 0$ is tied to the chosen axis.
5. *What fraction of the Earth lies within $15^\circ$ of the equator?* $\sin 15^\circ \approx 25.9\%$.

## References

- I. J. Schoenberg. *Positive definite functions on spheres.* Duke Math. J., 1942.
- F. Narcowich, P. Petrushev, J. Ward. *Localized tight frames on spheres.* SIAM J. Math. Anal., 2006.
- P. Baldi, G. Kerkyacharian, D. Marinucci, D. Picard. *Asymptotics for spherical needlets.* Ann. Statist., 2009.
- D. Marinucci, G. Peccati. *Random Fields on the Sphere.* Cambridge University Press, 2011.
- T. Gneiting. *Strictly and non-strictly positive definite functions on spheres.* Bernoulli, 2013.
- A. Lang, C. Schwab. *Isotropic Gaussian random fields on the sphere: regularity, fast simulation and stochastic PDEs.* Ann. Appl. Probab., 2015.
- V. Borovitskiy, A. Terenin, P. Mostowsky, M. P. Deisenroth. *Matérn Gaussian processes on Riemannian manifolds.* NeurIPS, 2020.
- K. Atkinson, W. Han. *Spherical Harmonics and Approximations on the Unit Sphere.* Springer, 2012.
