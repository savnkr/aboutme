---
title: "Wavelets, from first principles"
description: "Windows that shrink as frequency grows; when such a transform can be inverted; Haar by hand; multiresolution analysis and the two-scale equation; the fast wavelet transform as a filter bank, and lifting; perfect reconstruction; vanishing moments and why they make signals sparse; Daubechies-4 solved by hand; denoising with a proof of the universal threshold; and what the discrete transform gets wrong. Part 3 of Seeing in frequencies."
date: 2026-11-18
updated: 2026-11-18
cover: /blog/thumbs/wavelets.png
tags: [wavelets, multiresolution, signal-processing, denoising, computer-vision]
---

*Seeing in frequencies*, part 3 of 4. [Fourier analysis](/aboutme/blog/fourier-analysis-from-first-principles/) → [the discrete Fourier world](/aboutme/blog/the-discrete-fourier-world/) → **wavelets** → [wavelets for images](/aboutme/blog/wavelets-for-images/).

This is the picture most people meet first: an image split into a half-size approximation and three images of detail, then the approximation split again, and again.

<figure class="fig">
  <canvas data-figure="pyramid" data-src="/aboutme/blog/spectral/zoo.json" data-base="/aboutme/blog/spectral/img/" aria-label="The Eye sketch and its two-dimensional wavelet transform. Each level splits the current approximation into a half-size approximation in the top-left corner and three detail images that show vertical edges, horizontal edges and diagonal detail."></canvas>
  <div class="controls">
    <button type="button" data-pplay>Pause</button>
    <label class="slider">levels <input type="range" min="0" max="4" step="1" value="0" data-plev /> <span class="mono" data-plevval>0</span></label>
    <span class="seg"><button type="button" data-pw="haar" class="on">Haar</button><button type="button" data-pw="db2">D4</button></span>
  </div>
  <div class="insight" data-insight></div>
</figure>

It is beautiful, and it hides almost everything: which numbers produce each quadrant, why there are three detail images, why the transform loses nothing, and why most of it is black. This post builds the 1-D machinery behind it and connects it, line by line, to the code that draws it. Part 4 then does the 2-D picture by hand.

Part 2 ended at a wall. A spectrogram looks at a signal through one window, and the uncertainty principle forbids any window from being sharp in both time and frequency, so the song's chord and its click could not both be seen clearly. Here is the same song through a different lens: a window that **shrinks as the frequency grows**.

<figure class="fig">
  <canvas data-figure="scalogram" aria-label="Top: the song, a chord, a chirp and a click, with one chosen wavelet drawn over it. Bottom: its scalogram, the size of the wavelet transform against time and log frequency. Click the scalogram to pick a wavelet."></canvas>
  <div class="controls">
    <label class="slider">cycles per wavelet <input type="range" min="4" max="12" step="1" value="8" data-nc /> <span class="mono" data-ncval>8</span></label>
  </div>
  <div class="insight" data-insight></div>
</figure>

Low frequencies get long windows, so the chord's two tones separate (the band ripples because the tones beat against each other 16 times a second, but a slice through it, or its average over the chord, has two peaks, at 40 and 56 Hz); high frequencies get short windows, so the click stays a click; and the chirp climbs as a bright diagonal. Each wavelet still obeys the uncertainty principle, but it is spent differently at every scale. This post builds that idea from the ground up, and then makes it fast, exact and compact: the **discrete wavelet transform**, which costs less than an FFT, reconstructs perfectly, and represents piecewise-smooth signals with very few numbers.

The route: Sections 1 to 3 define the continuous transform and prove when it can be inverted. Section 4 does the simplest wavelet, Haar, by hand. Sections 5 and 6 build the machinery: multiresolution analysis and the fast algorithm. **Section 7 connects that theory to the code you will actually run**: which way the levels count, why one filter is low-pass and the other high-pass, how libraries store the four filters, and where the samples enter. Sections 8 and 9 cover lifting and perfect reconstruction. Sections 10 to 12 explain what makes a wavelet good, construct Daubechies' famous one by hand, and show where the numbers in every wavelet library come from. Section 13 uses all of it to remove noise, with a proof of the standard threshold, Section 14 is honest about what goes wrong, and Section 17 collects the traps. Images get their own post, Part 4.

## Notation and conventions

| Symbol | Meaning |
|---|---|
| $\psi_{a,b}(t) = a^{-1/2}\psi\big(\frac{t-b}{a}\big)$ | continuous wavelet family |
| $W_f(a, b) = \langle f, \psi_{a,b}\rangle$ | continuous wavelet transform |
| $\varphi$, $\psi$ | scaling function, wavelet |
| $\psi_{j,k}(x) = 2^{j/2}\psi(2^jx - k)$ | discrete family; larger $j$ is finer |
| $V_j$, $W_j$ | approximation and detail spaces |
| $h_k$, $g_k$ | low-pass and high-pass filters |
| $H(\omega) = \sum_k h_ke^{-ik\omega}$ | frequency response, $\omega$ in radians per sample |
| $a_j[k]$, $d_j[k]$ | approximation and detail coefficients |

Filter responses use radians per sample, $\omega = 2\pi\xi$, because every wavelet book does; continuous transforms keep the $\xi$ of Parts 1 and 2. Daubechies and Strang count levels upward ($V_j \subset V_{j+1}$, larger $j$ finer), as here; Mallat's book counts the other way, and libraries count the number of splits. Section 7 translates between all three.

## 1. Let the window scale with the frequency

Pick one function $\psi$, a short oscillation with zero mean, and make a family from it by stretching and sliding:

$$
\psi_{a,b}(t) = \frac{1}{\sqrt a}\,\psi\!\left(\frac{t - b}{a}\right), \qquad a > 0,\ b \in \mathbb{R}.
$$

The scale $a$ stretches it (large $a$, long and slow; small $a$, short and fast) and $b$ slides it to time $b$. The factor $a^{-1/2}$ keeps $\lVert\psi_{a,b}\rVert = \lVert\psi\rVert$.

:::definition[Continuous wavelet transform]
$$
W_f(a, b) = \langle f, \psi_{a,b}\rangle = \int f(t)\,\frac{1}{\sqrt a}\,\overline{\psi\!\left(\frac{t - b}{a}\right)}\,dt .
$$
:::

Two wavelets are worth knowing by name. The **Morlet** wavelet is a Gaussian-windowed complex wave, $\psi(t) = e^{2\pi i f_0t}e^{-t^2/2\sigma^2}$, the one in the figure above. The **Mexican hat** is the negative second derivative of a Gaussian, $(1 - t^2)e^{-t^2/2}$ up to normalisation, the 1-D case of the Laplacian of a Gaussian that computer vision uses to find blobs (see [the Laplacian post](/aboutme/blog/one-operator-i-keep-coming-back-to/)).

What does $W_f(a, \cdot)$ do in frequency? By Plancherel, $\langle f, \psi_{a,b}\rangle = \langle\hat f, \hat\psi_{a,b}\rangle$, and by the shift and scaling rules of Part 1, $\hat\psi_{a,b}(\xi) = \sqrt a\,\hat\psi(a\xi)\,e^{-2\pi i\xi b}$. So

$$
W_f(a, b) = \int\hat f(\xi)\,\sqrt a\,\overline{\hat\psi(a\xi)}\,e^{2\pi i\xi b}\,d\xi .
$$

For each scale, $W_f(a, \cdot)$ is $f$ passed through a **band-pass filter** centred at $\xi_c/a$, where $\xi_c$ is the centre frequency of $\hat\psi$. Doubling $a$ halves the centre frequency *and* halves the bandwidth: every filter has the same ratio of bandwidth to centre frequency (engineers call this "constant-Q"). That is the difference from the spectrogram, whose filters all have the same bandwidth.

The figure uses this directly. It labels each row by its centre frequency $f = \xi_c/a$ and gives each wavelet a fixed number of cycles $n$: its envelope has standard deviation $\sigma_t = n/(2\pi f)$ in time and $\sigma_f = f/n$ in frequency, so the ratio of bandwidth to frequency is $1/n$ at every scale. These Gaussians meet the uncertainty bound of Part 2 *exactly*: the spreads of $\lvert\psi\rvert^2$ and $\lvert\hat\psi\rvert^2$ are $\sigma_t/\sqrt2$ and $\sigma_f/\sqrt2$, whose product is $\sigma_t\sigma_f/2 = 1/(4\pi)$. What changes from row to row is only how the fixed area is shared between time and frequency.

## 2. When can you invert it?

A transform that cannot be undone is a feature extractor, not a representation. When does $W_f$ determine $f$?

:::theorem[Admissibility and the resolution of the identity]
Let $\psi$ be real with $C_\psi = \displaystyle\int_0^\infty\frac{\lvert\hat\psi(\xi)\rvert^2}{\xi}\,d\xi < \infty$. Then for every $f \in L^2$,
$$
\int_0^\infty\!\!\int_{-\infty}^{\infty}\lvert W_f(a, b)\rvert^2\,db\,\frac{da}{a^2} = C_\psi\,\lVert f\rVert^2 ,
$$
and $f = \frac{1}{C_\psi}\displaystyle\int_0^\infty\!\!\int W_f(a, b)\,\psi_{a,b}\,db\,\frac{da}{a^2}$ (in the weak sense: inner products with any $g$ agree).
:::

:::proof
**Step 1. Fix the scale and integrate over $b$.** By Section 1, $b \mapsto W_f(a, b)$ is the inverse Fourier transform of $\xi \mapsto \hat f(\xi)\sqrt a\,\overline{\hat\psi(a\xi)}$. Plancherel gives $\int\lvert W_f(a, b)\rvert^2db = \int\lvert\hat f(\xi)\rvert^2\,a\,\lvert\hat\psi(a\xi)\rvert^2\,d\xi$.

**Step 2. Integrate over $a$ and swap** (everything is non-negative, so Tonelli allows it): $\int_0^\infty\frac{da}{a^2}\int\lvert\hat f(\xi)\rvert^2a\lvert\hat\psi(a\xi)\rvert^2d\xi = \int\lvert\hat f(\xi)\rvert^2\Big(\int_0^\infty\lvert\hat\psi(a\xi)\rvert^2\frac{da}{a}\Big)d\xi$.

**Step 3. The inner integral does not depend on $\xi$.** For $\xi > 0$ substitute $u = a\xi$, so $da/a = du/u$: it equals $\int_0^\infty\lvert\hat\psi(u)\rvert^2du/u = C_\psi$. For $\xi < 0$ substitute $u = a\lvert\xi\rvert$ and use $\lvert\hat\psi(-u)\rvert = \lvert\hat\psi(u)\rvert$, true because $\psi$ is real (Part 1): again $C_\psi$.

**Step 4. Plancherel once more.** What remains is $C_\psi\int\lvert\hat f\rvert^2 = C_\psi\lVert f\rVert^2$.

**Step 5. Inversion.** Applying Steps 1 to 4 to $f + g$ and $f + ig$ (polarisation) gives $\iint W_f\,\overline{W_g}\,db\,da/a^2 = C_\psi\langle f, g\rangle$ for all $f, g$. The left side is $\big\langle\iint W_f(a,b)\psi_{a,b}\,db\,da/a^2,\ g\big\rangle$, which is the claimed formula tested against $g$.
:::

:::question
Why must a wavelet have zero mean?
:::

:::answer
Because $C_\psi$ must be finite. If $\hat\psi(0) = \int\psi \ne 0$, then by continuity $\lvert\hat\psi(\xi)\rvert^2 \ge c > 0$ on some interval $(0, \delta)$, and $\int_0^\delta c\,d\xi/\xi = \infty$. So $\hat\psi(0) = 0$: the wavelet must oscillate, which is where the name ("small wave") comes from. A wavelet responds to *change*, never to a constant.
:::

The Morlet wavelet is not quite admissible: its Gaussian spectrum is centred at $f_0$ but leaks a little to zero frequency, so the classical definition subtracts a tiny correction term. In practice it is negligible. For the 8-cycle wavelets in the figure the spectrum at zero is $e^{-32} \approx 10^{-14}$ of its peak.

## 3. Too many coefficients

The continuous transform turns a function of one variable into a function of two. It is extremely redundant: nearby $(a, b)$ give almost the same number, which is why the scalogram looks smooth. For analysis by eye, redundancy is a gift. For storage, compression or fast computation, it is waste.

Keep only a grid: scales $a = 2^{-j}$ and, at each scale, positions $b = k\,2^{-j}$, spaced in proportion to the wavelet's width. Fine scales get many positions, coarse scales few. On this **dyadic grid** the family becomes

$$
\psi_{j,k}(x) = 2^{j/2}\,\psi(2^jx - k), \qquad j, k \in \mathbb{Z}.
$$

For a general $\psi$ this family is a **frame**: it is redundant but stable, and every $f$ can be rebuilt from $\langle f, \psi_{j,k}\rangle$ using a dual family. The remarkable fact is that for special $\psi$ it is an **orthonormal basis** of $L^2(\mathbb{R})$: no redundancy at all, the three facts of Part 1 (coefficients are inner products, the sum rebuilds, energy is preserved) hold exactly, and the coefficients can be computed faster than an FFT. The simplest such $\psi$ is a hundred years older than the word "wavelet".

## 4. Haar, by hand

In 1910 Alfréd Haar used the function that is $+1$ on $[0, \tfrac12)$ and $-1$ on $[\tfrac12, 1)$. Here is what it does to eight numbers.

<figure class="fig">
  <canvas data-figure="haar" aria-label="Eight bars of a signal and, below, the Haar transform after one, two and three steps of averaging and half-differencing. Averages are teal, details amber. A threshold sets small details to zero and the rebuilt signal is drawn over the original."></canvas>
  <div class="controls">
    <span class="seg"><button type="button" data-step="0">signal</button><button type="button" data-step="1">step 1</button><button type="button" data-step="2">step 2</button><button type="button" data-step="3" class="on">step 3</button></span>
    <label class="slider">drop details below <input type="range" min="0" max="4" step="0.5" value="0" data-t /> <span class="mono" data-tval>0</span></label>
  </div>
  <div class="insight" data-insight></div>
</figure>

:::example[Eight numbers, three steps]
Start with $(9, 7, 3, 5, 6, 10, 2, 6)$ (the first four are the example of Stollnitz, DeRose and Salesin's primer).

**Step 1.** Pair them up, $(9,7), (3,5), (6,10), (2,6)$, and replace each pair $(p, q)$ by its average $\frac{p+q}2$ and half-difference $\frac{p-q}2$: averages $(8, 4, 8, 4)$, details $(1, -1, -2, -2)$. Nothing is lost, since $p = \text{average} + \text{detail}$ and $q = \text{average} - \text{detail}$.

**Step 2.** Repeat on the averages only: $(8, 4) \to 6, 2$ and $(8, 4) \to 6, 2$. Averages $(6, 6)$, details $(2, 2)$.

**Step 3.** $(6, 6) \to 6, 0$.

Result: $(6 \mid 0 \mid 2, 2 \mid 1, -1, -2, -2)$. One overall mean, then details at three scales, coarsest first. Setting the two details of size 1 to zero changes the rebuilt signal by at most 1 anywhere, while keeping 5 of 8 numbers: compression, in miniature.
:::

Averages and half-differences are easy to compute but do not preserve energy. Dividing by $\sqrt2$ instead of $2$ does: $\big(\frac{p+q}{\sqrt2}\big)^2 + \big(\frac{p-q}{\sqrt2}\big)^2 = p^2 + q^2$. That is the **orthonormal Haar transform**, and its coefficients are the averaging ones multiplied by $2^{j/2}$ at level $j$: here $(16.97 \mid 0 \mid 4, 4 \mid 1.41, -1.41, -2.83, -2.83)$, exactly what PyWavelets returns (Section 17).

:::proposition[The Haar system is an orthonormal basis]
On $[0, 1)$, the constant $1$ and the functions $\psi_{j,k}(x) = 2^{j/2}\psi(2^jx - k)$, $j \ge 0$, $0 \le k < 2^j$, with $\psi = \mathbf{1}_{[0,1/2)} - \mathbf{1}_{[1/2,1)}$, form an orthonormal basis of $L^2[0, 1)$.
:::

:::proof
**Step 1. Unit length.** $\psi_{j,k}$ equals $\pm2^{j/2}$ on an interval of length $2^{-j}$, so $\int\psi_{j,k}^2 = 2^j\cdot2^{-j} = 1$.

**Step 2. Two different Haar functions are orthogonal.** Their supports are dyadic intervals, which are either disjoint (product zero) or nested. If nested and different, the coarser function is constant on the finer one's support, and the finer one has integral zero there. The constant $1$ is orthogonal to every $\psi_{j,k}$ for the same reason.

**Step 3. Completeness.** By induction on $J$, the span of $1$ and $\psi_{j,k}$ with $j < J$ is all functions that are constant on each interval $[m2^{-J}, (m+1)2^{-J})$: it has the right dimension, $2^J$, and Step 1 of the example shows how to reach any such function. Step functions on dyadic intervals are dense in $L^2[0, 1)$, so nothing is orthogonal to all of them.
:::

Haar is perfect for steps and poor for smooth curves: it approximates a sine wave by a staircase. Can we find wavelets that are orthonormal like Haar but smooth? The framework that answers yes is the next section.

## 5. Multiresolution analysis

The Haar example had a structure worth naming: a sequence of ever-finer approximations, each adding detail to the last.

:::definition[Multiresolution analysis (Mallat, Meyer)]
A sequence of closed subspaces $\cdots \subset V_{-1} \subset V_0 \subset V_1 \subset \cdots$ of $L^2(\mathbb{R})$ such that
1. **nested:** $V_j \subset V_{j+1}$;
2. **self-similar:** $f(x) \in V_j$ if and only if $f(2x) \in V_{j+1}$;
3. **complete:** the union of all $V_j$ is dense in $L^2$, and their intersection is $\{0\}$;
4. **generated by one function:** there is a **scaling function** $\varphi$ whose integer shifts $\varphi(x - k)$ form an orthonormal basis of $V_0$.
:::

For Haar, $V_j$ is the functions constant on intervals of length $2^{-j}$ and $\varphi = \mathbf{1}_{[0,1)}$. Read $V_j$ as "what a camera with pixels $2^{-j}$ wide can see". Each $V_j$ sits inside $V_{j+1}$, and the difference is a detail space $W_j$, the orthogonal complement: $V_{j+1} = V_j \oplus W_j$.

<figure class="sketch-fig">
<svg class="sketch" viewBox="0 0 900 420" role="img" aria-label="Four nested rectangles labelled V0 inside V1 inside V2 inside V3; the rings between them are labelled W0, W1 and W2. Text: V j is what a camera with pixels 2 to the minus j wide sees; V j plus 1 equals V j plus W j, where W j is the detail V j misses; phi spans V0 and psi spans W0; so V3 equals V0 plus W0 plus W1 plus W2."><text x="450" y="36" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--a1)">a multiresolution analysis: spaces inside spaces</text><path d="M86.7 56.1 C171.5 54.3, 256.8 53.3, 512.2 55.1 M87.4 55.1 C198.6 56.2, 310.5 56.4, 512.5 54.6 M512.2 56.3 C510.2 170.1, 511.3 288.8, 511.8 395.8 M511.9 54.4 C511.5 163.4, 511.3 273.4, 512.9 395.7 M512.9 395.8 C391 395.7, 269.4 397.5, 88 394.7 M512.9 394.7 C404.4 393.9, 297.6 393.6, 87 394.6 M88.7 394.5 C85.5 296.5, 87.1 198.2, 86.3 54 M88.1 395.7 C90 300.7, 89.8 206.9, 87.3 55.5" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="107.5" y="79" text-anchor="start" style="font-family:Kalam,cursive;font-size:20px;font-weight:700;fill:var(--ink)">V₃</text><path d="M138.9 98.4 C202.6 96.3, 267 95, 459.5 97.1 M139.9 97.2 C223.5 98.1, 308.1 98.3, 460 96.5 M459.7 98.6 C457.5 183, 458.8 273.4, 459.2 353.9 M459.3 96.3 C459 178.4, 458.7 261.7, 460.5 353.8 M460.5 354.1 C368.5 353.3, 276.9 355.7, 140.7 352.6 M460.6 352.6 C378.4 351.9, 298.4 351.6, 139.3 352.5 M141.5 352.4 C137.9 279, 140 205.4, 138.5 95.8 M140.7 353.8 C142.3 281.9, 142 211.4, 139.7 97.6" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="160" y="121" text-anchor="start" style="font-family:Kalam,cursive;font-size:20px;font-weight:700;fill:var(--ink)">V₂</text><path d="M188.6 138.8 C232.2 136.4, 276.8 134.8, 409.4 137.2 M189.9 137.2 C247.3 137.7, 305.9 138, 410 136.4 M409.6 138.9 C407.2 195.2, 408.8 258.8, 409 314.2 M409.2 136.1 C408.9 192.6, 408.6 250.7, 410.7 314 M410.6 314.3 C347.1 312.6, 284 315.7, 190.9 312.5 M410.7 312.6 C353.6 312.1, 299.2 311.7, 189.1 312.3 M191.8 312.2 C187.9 262.5, 190.5 212.3, 188.2 135.5 M190.8 314 C191.9 264, 191.6 215.7, 189.6 137.7" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="210" y="161" text-anchor="start" style="font-family:Kalam,cursive;font-size:20px;font-weight:700;fill:var(--ink)">V₁</text><path d="M237.7 174.8 C237.7 174.8, 237.7 174.8, 237.7 174.8 M237.7 174.8 C237.7 174.8, 237.7 174.8, 237.7 174.8 M236.8 182.9 C239.4 181, 243.7 175.9, 246.2 175.5 M237.1 183.7 C239 182.6, 240.5 179.9, 245.4 175.5 M235.4 195.4 C242.8 184.4, 248.1 180.6, 253.9 174 M237.9 194 C240.5 190, 244 185.5, 253 174.4 M237.2 201.5 C243.7 197, 248.2 189.5, 262.3 174.2 M237.1 202.4 C247.2 191.6, 254.8 182.8, 260.2 174.9 M238.4 209.8 C248.4 200.6, 258.7 184.7, 271.3 176.9 M237 211.2 C245 204, 253.3 194.8, 269.6 175 M236.1 221.4 C251.5 205.2, 268.1 187.1, 278.4 174 M237.3 221.8 C246.6 210.5, 255 198.4, 278.3 175.5 M236.2 231.4 C248.5 216.6, 257.8 207.8, 284.8 174.5 M236.3 230 C255.7 208.6, 273.1 188.6, 284.7 174.5 M236.9 237.6 C251.3 221.7, 266.9 206, 292.1 174.3 M236.5 239.4 C251.6 220.7, 267.7 202.3, 292.7 174.5 M237.2 250 C254.7 231.6, 265.8 215.3, 299.7 177.2 M238.3 247.5 C251.5 231.3, 265.2 217.6, 301.7 175.7 M236.4 258.2 C260.9 232.8, 283 207.6, 309.5 173.8 M238.7 255.9 C255.7 237.1, 275.2 214.3, 309.9 174.7 M236.6 264.4 C260.2 243.3, 282.6 219.2, 315.8 175 M236.8 265.4 C267.6 229.7, 299.3 195.9, 316.5 175.4 M237.5 277 C268.1 239.3, 299.7 206, 325.7 174.5 M236.7 275.2 C260.2 251.2, 282.5 224.5, 325.7 174.4 M244 276.2 C273.9 245.6, 300.6 211.3, 332.2 177.4 M245 275.1 C277.8 238, 309.5 201.2, 332.1 174.4 M253.6 275.3 C272.1 252, 292.3 228.5, 340.4 174.1 M252.8 274.8 C272.5 253.1, 292.5 231.3, 340.7 175.1 M263.3 276.6 C295.3 234.2, 327.6 199.6, 349.4 174.4 M261.4 274.9 C289.6 242.3, 316.8 209.9, 348.8 175.3 M271.6 275.2 C288.8 255.4, 305.1 234.9, 355.2 173.5 M269.9 274.3 C289.6 252, 310.4 230.3, 357.8 174 M275.3 277.4 C295.5 255.8, 311.3 234.1, 365.4 178.1 M277.4 275.4 C312.5 237.4, 345.5 198.2, 363.6 176.1 M285 276.9 C315.4 240.8, 343.1 210.8, 362.1 184.7 M285.1 274 C306.6 253, 326.6 230.7, 363.6 185.2 M295.7 276.8 C318.4 245.6, 344 215.5, 362.5 195.7 M292.9 275.3 C312 256.8, 327.2 236.6, 362.2 195.3 M300.9 276.5 C315.5 261.9, 325.9 245, 364.7 203.1 M300.8 275.2 C326 247.7, 349 220.8, 363.4 204.6 M311.2 276.3 C325.9 253.6, 345.7 235.6, 364.5 211.5 M309.9 275.3 C327.9 255.1, 345.7 235, 362.4 213.4 M316.6 276.3 C329.5 256.9, 344.2 244.4, 363.6 222.1 M317.5 274.7 C334.1 256, 350.3 237.1, 363.8 223.5 M326.4 273 C334 267.5, 339.7 258.4, 362 232.6 M326.4 275.1 C332.8 264.4, 342.6 254.7, 362.1 232.5 M332.5 275.4 C342.5 261.8, 356.8 250.7, 361.8 243.3 M332.3 276 C342.7 264.6, 352.1 253.5, 363.7 242 M339.2 276 C346.6 269.9, 349.9 261.2, 364.4 252.4 M340.7 276.4 C348 267.5, 354.7 261, 363.8 249.7 M348.4 275 C350.6 273.5, 356 267.9, 364.7 257.9 M349.6 276 C353.7 269.3, 359 262.9, 363.6 260 M356.9 275.7 C359.5 273.6, 360.9 270.2, 363.1 268.7 M357.6 275.3 C358.3 273.1, 359.5 271.8, 363.3 268.3" style="stroke:var(--a1);stroke-width:1.1;stroke-opacity:.45;fill:none"/><path d="M235.9 177.1 C260.4 174.6, 286 172.8, 361.8 175.2 M237.4 175.2 C269.8 175.1, 303.7 175.5, 362.5 174.3 M362.1 176.9 C360 207.2, 361.6 244.8, 361.5 276.2 M361.7 174.1 C361.7 206.3, 361.4 240, 363.2 276 M363.2 276.5 C326.7 273.8, 290.8 277.3, 238.5 274.4 M363.3 274.5 C330 274.5, 299.9 274.1, 236.5 274.2 M239.3 274.2 C235.9 246.6, 238.4 218.5, 235.7 173.5 M238.3 276 C238.6 247, 238.2 219.7, 237.1 175.7" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="257.5" y="199" text-anchor="start" style="font-family:Kalam,cursive;font-size:20px;font-weight:700;fill:var(--a1)">V₀</text><text x="376.25" y="231" text-anchor="end" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--dot)">W₀</text><text x="425" y="231" text-anchor="end" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--dot)">W₁</text><text x="476.25" y="231" text-anchor="end" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--dot)">W₂</text><text x="560" y="110" text-anchor="start" style="font-family:Kalam,cursive;font-size:21px;font-weight:400;fill:var(--ink)">V₀ ⊂ V₁ ⊂ V₂ ⊂ V₃ ⊂ …</text><text x="560" y="150" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--muted)">Vⱼ: what a camera with pixels 2⁻ʲ wide sees</text><text x="560" y="200" text-anchor="start" style="font-family:Kalam,cursive;font-size:21px;font-weight:400;fill:var(--ink)">Vⱼ₊₁ = Vⱼ ⊕ Wⱼ</text><text x="560" y="240" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--dot)">Wⱼ: the detail Vⱼ misses</text><text x="560" y="290" text-anchor="start" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--a1)">φ spans V₀ (by its shifts)</text><text x="560" y="324" text-anchor="start" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--dot)">ψ spans W₀ (by its shifts)</text><text x="560" y="372" text-anchor="start" style="font-family:Kalam,cursive;font-size:19px;font-weight:700;fill:var(--ink)">so  V₃ = V₀ ⊕ W₀ ⊕ W₁ ⊕ W₂</text></svg>
</figure>

<figure class="fig">
<video autoplay muted loop playsinline controls preload="metadata" poster="/aboutme/blog/spectral/projections.jpg"><source src="/aboutme/blog/spectral/projections.webm" type="video/webm" /><source src="/aboutme/blog/spectral/projections.mp4" type="video/mp4" /></video>
<figcaption>A function projected onto the Haar spaces V₀ ⊂ V₁ ⊂ … ⊂ V₆. Each step adds the detail the finer space can see.</figcaption>
</figure>

Everything else follows from axiom 4 combined with nesting.

:::proposition[The two-scale equation]
There are numbers $h_k$ with
$$
\varphi(x) = \sqrt2\sum_k h_k\,\varphi(2x - k), \qquad \sum_k h_k = \sqrt2, \qquad \sum_k h_k\,h_{k - 2m} = \begin{cases}1 & m = 0\\ 0 & m \ne 0.\end{cases}
$$
:::

:::proof
**Step 1. A basis of $V_1$.** By axiom 4, $\{\varphi(x - k)\}$ is orthonormal in $V_0$; by axiom 2 and the substitution $x \to 2x$, $\{\sqrt2\varphi(2x - k)\}$ is orthonormal and spans $V_1$ (the $\sqrt2$ restores unit length).

**Step 2. Expand $\varphi$ in it.** $\varphi \in V_0 \subset V_1$, so $\varphi = \sum_k h_k\sqrt2\varphi(2x - k)$ with $h_k = \langle\varphi, \sqrt2\varphi(2\cdot{} - k)\rangle$, by fact 1 of Part 1.

**Step 3. The sum of the $h_k$.** Normalise $\int\varphi = 1$ (one can show $\int\varphi \ne 0$ from axiom 3). Integrate both sides: $1 = \sum_kh_k\sqrt2\int\varphi(2x - k)dx = \sum_kh_k\sqrt2\cdot\tfrac12$, so $\sum_kh_k = \sqrt2$.

**Step 4. Orthogonality of shifts.** Shift Step 2 by $m$: $\varphi(x - m) = \sum_lh_l\sqrt2\varphi(2x - 2m - l) = \sum_kh_{k - 2m}\sqrt2\varphi(2x - k)$. The inner product of $\varphi$ with $\varphi(\cdot - m)$ is, by orthonormality of the $\sqrt2\varphi(2x - k)$, $\sum_kh_kh_{k-2m}$, and by axiom 4 it is $1$ if $m = 0$ and $0$ otherwise.
:::

For Haar, $\varphi(x) = \varphi(2x) + \varphi(2x - 1)$, so $h = (\tfrac1{\sqrt2}, \tfrac1{\sqrt2})$. The whole multiresolution analysis is encoded in this short list of numbers, and so is the wavelet.

:::proposition[The wavelet from the scaling function]
Let $g_k = (-1)^kh_{1-k}$ and $\psi(x) = \sqrt2\sum_kg_k\varphi(2x - k)$. Then the shifts $\psi(x - m)$ are orthonormal, and each is orthogonal to every $\varphi(x - n)$.
:::

:::proof
**Step 1. Reduce to sums of filter coefficients.** As in Step 4 above, both functions are combinations of the orthonormal $\sqrt2\varphi(2x - k)$, so $\langle\psi(\cdot - m), \varphi(\cdot - n)\rangle = \sum_kg_{k-2m}h_{k-2n}$ and $\langle\psi(\cdot - m), \psi(\cdot - n)\rangle = \sum_kg_{k-2m}g_{k-2n}$.

**Step 2. $\psi$ against $\varphi$.** Write the terms of $\sum_kg_{k-2m}h_{k-2n} = \sum_k(-1)^kh_{1-k+2m}h_{k-2n}$ as $T(k)$ (using $(-1)^{k - 2m} = (-1)^k$). The substitution $k' = 1 + 2m + 2n - k$ gives $T(k') = (-1)^{1 + 2m + 2n - k}h_{k - 2n}h_{1 + 2m - k} = -T(k)$. The map $k \mapsto k'$ is its own inverse and has no fixed point (that would need $2k = 1 + 2m + 2n$, impossible for integers), so the terms cancel in pairs and the sum is $0$.

**Step 3. $\psi$ against its shifts.** $g_{k - 2m}g_{k - 2n} = h_{1 - k + 2m}h_{1 - k + 2n}$, since the signs $(-1)^{k-2m}(-1)^{k-2n} = 1$. Substituting $l = 1 - k + 2m$, the sum becomes $\sum_lh_lh_{l - 2(m - n)}$, which is $1$ if $m = n$ and $0$ otherwise, by the two-scale proposition.
:::

One more fact, which we state without its longer proof: the shifts of $\psi$ span all of $W_0$ (see Daubechies, *Ten Lectures*, ch. 5). Then $\psi_{j,k}$ over all $j, k$ is an orthonormal basis of $L^2(\mathbb{R})$, and $V_J = V_0 \oplus W_0 \oplus \cdots \oplus W_{J-1}$.

:::question
What is the scaling function for, if the wavelet does the work?
:::

:::answer
The scaling function is the camera; the wavelet records what the camera misses. A finite transform never uses all scales: it stops at some coarsest level, and everything coarser is summarised by the coefficients $\langle f, \varphi_{J,k}\rangle$, the low-resolution image that the details refine. That is the $a_3 = 6$ in the Haar example. Without $\varphi$, a finite transform would throw away the signal's mean and slow trends.
:::

## 6. From functions to filters: the fast wavelet transform

We never need $\varphi$ or $\psi$ themselves to compute the coefficients. Write $\varphi_{j,k}(x) = 2^{j/2}\varphi(2^jx - k)$, and let $a_j[k] = \langle f, \varphi_{j,k}\rangle$ and $d_j[k] = \langle f, \psi_{j,k}\rangle$.

:::proposition[Mallat's algorithm]
$$
a_j[k] = \sum_nh_{n - 2k}\,a_{j+1}[n], \qquad d_j[k] = \sum_ng_{n-2k}\,a_{j+1}[n],
$$
$$
a_{j+1}[n] = \sum_kh_{n-2k}\,a_j[k] + \sum_kg_{n - 2k}\,d_j[k].
$$
:::

:::proof
**Step 1. The two-scale equation at level $j$.** Substitute $x \to 2^jx - k$ in it and multiply by $2^{j/2}$: $\varphi_{j,k}(x) = \sum_lh_l\,2^{(j+1)/2}\varphi(2^{j+1}x - 2k - l) = \sum_nh_{n - 2k}\,\varphi_{j+1,n}(x)$, with $n = 2k + l$. The same with $g$ gives $\psi_{j,k} = \sum_ng_{n-2k}\varphi_{j+1,n}$.

**Step 2. Take inner products with $f$.** The inner product is linear and the filters are real: $a_j[k] = \sum_nh_{n-2k}\langle f, \varphi_{j+1,n}\rangle = \sum_nh_{n-2k}a_{j+1}[n]$, and likewise $d_j[k]$.

**Step 3. The inverse.** $\{\varphi_{j,k}\}\cup\{\psi_{j,k}\}$ is an orthonormal basis of $V_{j+1}$. Expanding $\varphi_{j+1,n}$ in it, the coefficients are $\langle\varphi_{j+1,n}, \varphi_{j,k}\rangle = h_{n-2k}$ (Step 1) and $\langle\varphi_{j+1,n}, \psi_{j,k}\rangle = g_{n-2k}$. Taking the inner product of that expansion with $f$ gives the third formula.
:::

Read the first line as signal processing: correlate the fine coefficients with the filter $h$, then keep every second output. The second line does the same with $g$. That is a **two-channel filter bank**, and repeating it on the approximation gives every level.

<figure class="sketch-fig">
<svg class="sketch" viewBox="0 0 900 360" role="img" aria-label="A two-channel filter bank. The signal x goes through a low-pass filter h tilde and a high-pass filter g tilde, each followed by keeping every second sample, giving approximation coefficients a and detail coefficients d. Synthesis inserts zeros between samples, filters with h and g, and adds, giving x back. The analysis is repeated on a for the next level."><text x="450" y="36" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--a1)">the fast wavelet transform is a filter bank</text><text x="60" y="191" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">x</text><path d="M79.7 184.6 C86.2 184.1, 91.2 185.2, 108.5 183.4 M80 184.8 C92 185.1, 102.5 183.9, 110.6 184.8" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M109.7 119.6 C109.1 145, 108.1 172, 108.5 248.4 M110 119.8 C109.6 170.8, 108.4 220.3, 110.6 249.8" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M109.7 119.6 C118.2 119.2, 125.2 120.2, 148.5 118.4 M110 119.8 C125.9 120.1, 140.3 118.9, 150.6 119.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M149.8 119.8 C147.9 120.3, 145.2 122, 138.3 124.2 M150 119.9 C145.9 122.1, 141 123.4, 139.5 125.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M149.8 119.8 C148 118.3, 145.2 117.8, 138.3 113.8 M150 119.9 C146 118, 141.1 115.3, 139.5 114.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M149.7 93.6 C166.2 93.4, 181.2 94.5, 228.5 92.4 M150 93.8 C181.5 94.1, 211.5 93, 230.6 93.8 M230.9 93.2 C231.3 108.5, 228.6 119, 229.4 145.8 M228.9 94.5 C230.8 111.4, 230.7 127.3, 229 144.9 M228.8 146.1 C197.5 147.5, 171.1 147.4, 149.9 147.2 M229 146.8 C206.1 146, 180 145.9, 150.8 145.5 M151.5 147.7 C148 134.4, 147.9 122.5, 152.1 93.3 M150.1 147 C148.7 132.3, 149.5 119.8, 150.3 94.6" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="190" y="127" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--a1)">h̃</text><path d="M229.7 119.6 C238.2 119.2, 245.2 120.2, 268.5 118.4 M230 119.8 C245.9 120.1, 260.3 118.9, 270.6 119.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M269.8 119.8 C267.9 120.3, 265.2 122, 258.3 124.2 M270 119.9 C265.9 122.1, 261 123.4, 259.5 125.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M269.8 119.8 C268 118.3, 265.2 117.8, 258.3 113.8 M270 119.9 C266 118, 261.1 115.3, 259.5 114.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M269.7 93.6 C286.2 93.4, 301.2 94.5, 348.5 92.4 M270 93.8 C301.5 94.1, 331.5 93, 350.6 93.8 M350.9 93.2 C351.3 108.5, 348.6 119, 349.4 145.8 M348.9 94.5 C350.8 111.4, 350.7 127.3, 349 144.9 M348.8 146.1 C317.5 147.5, 291.1 147.4, 269.9 147.2 M349 146.8 C326.1 146, 300 145.9, 270.8 145.5 M271.5 147.7 C268 134.4, 267.9 122.5, 272.1 93.3 M270.1 147 C268.7 132.3, 269.5 119.8, 270.3 94.6" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="310" y="127" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">↓2</text><path d="M349.7 119.6 C364.2 119.4, 377.2 120.4, 418.5 118.4 M350 119.8 C377.6 120.1, 403.7 119, 420.6 119.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M419.8 119.8 C417.9 120.3, 415.2 122, 408.3 124.2 M420 119.9 C415.9 122.1, 411 123.4, 409.5 125.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M419.8 119.8 C418 118.3, 415.2 117.8, 408.3 113.8 M420 119.9 C416 118, 411.1 115.3, 409.5 114.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><text x="440" y="127" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--a1)">a</text><path d="M459.7 119.6 C472.2 119.3, 483.2 120.3, 518.5 118.4 M460 119.8 C483.7 120.1, 505.9 119, 520.6 119.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M519.8 119.8 C517.9 120.3, 515.2 122, 508.3 124.2 M520 119.9 C515.9 122.1, 511 123.4, 509.5 125.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M519.8 119.8 C518 118.3, 515.2 117.8, 508.3 113.8 M520 119.9 C516 118, 511.1 115.3, 509.5 114.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M519.7 93.6 C536.2 93.4, 551.2 94.5, 598.5 92.4 M520 93.8 C551.5 94.1, 581.5 93, 600.6 93.8 M600.9 93.2 C601.3 108.5, 598.6 119, 599.4 145.8 M598.9 94.5 C600.8 111.4, 600.7 127.3, 599 144.9 M598.8 146.1 C567.5 147.5, 541.1 147.4, 519.9 147.2 M599 146.8 C576.1 146, 550 145.9, 520.8 145.5 M521.5 147.7 C518 134.4, 517.9 122.5, 522.1 93.3 M520.1 147 C518.7 132.3, 519.5 119.8, 520.3 94.6" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="560" y="127" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">↑2</text><path d="M599.7 119.6 C608.2 119.2, 615.2 120.2, 638.5 118.4 M600 119.8 C615.9 120.1, 630.3 118.9, 640.6 119.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M639.8 119.8 C637.9 120.3, 635.2 122, 628.3 124.2 M640 119.9 C635.9 122.1, 631 123.4, 629.5 125.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M639.8 119.8 C638 118.3, 635.2 117.8, 628.3 113.8 M640 119.9 C636 118, 631.1 115.3, 629.5 114.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M639.7 93.6 C656.2 93.4, 671.2 94.5, 718.5 92.4 M640 93.8 C671.5 94.1, 701.5 93, 720.6 93.8 M720.9 93.2 C721.3 108.5, 718.6 119, 719.4 145.8 M718.9 94.5 C720.8 111.4, 720.7 127.3, 719 144.9 M718.8 146.1 C687.5 147.5, 661.1 147.4, 639.9 147.2 M719 146.8 C696.1 146, 670 145.9, 640.8 145.5 M641.5 147.7 C638 134.4, 637.9 122.5, 642.1 93.3 M640.1 147 C638.7 132.3, 639.5 119.8, 640.3 94.6" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="680" y="127" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--a1)">h</text><path d="M719.7 119.6 C728.2 119.2, 735.2 120.2, 758.5 118.4 M720 119.8 C735.9 120.1, 750.3 118.9, 760.6 119.8" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M109.7 249.6 C118.2 249.2, 125.2 250.2, 148.5 248.4 M110 249.8 C125.9 250.1, 140.3 248.9, 150.6 249.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M149.8 249.8 C147.9 250.3, 145.2 252, 138.3 254.2 M150 249.9 C145.9 252.1, 141 253.4, 139.5 255.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M149.8 249.8 C148 248.3, 145.2 247.8, 138.3 243.8 M150 249.9 C146 248, 141.1 245.3, 139.5 244.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M149.7 223.6 C166.2 223.4, 181.2 224.5, 228.5 222.4 M150 223.8 C181.5 224.1, 211.5 223, 230.6 223.8 M230.9 223.2 C231.3 238.5, 228.6 249, 229.4 275.8 M228.9 224.5 C230.8 241.4, 230.7 257.3, 229 274.9 M228.8 276.1 C197.5 277.5, 171.1 277.4, 149.9 277.2 M229 276.8 C206.1 276, 180 275.9, 150.8 275.5 M151.5 277.7 C148 264.4, 147.9 252.5, 152.1 223.3 M150.1 277 C148.7 262.3, 149.5 249.8, 150.3 224.6" style="stroke:var(--dot);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="190" y="257" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--dot)">g̃</text><path d="M229.7 249.6 C238.2 249.2, 245.2 250.2, 268.5 248.4 M230 249.8 C245.9 250.1, 260.3 248.9, 270.6 249.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M269.8 249.8 C267.9 250.3, 265.2 252, 258.3 254.2 M270 249.9 C265.9 252.1, 261 253.4, 259.5 255.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M269.8 249.8 C268 248.3, 265.2 247.8, 258.3 243.8 M270 249.9 C266 248, 261.1 245.3, 259.5 244.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M269.7 223.6 C286.2 223.4, 301.2 224.5, 348.5 222.4 M270 223.8 C301.5 224.1, 331.5 223, 350.6 223.8 M350.9 223.2 C351.3 238.5, 348.6 249, 349.4 275.8 M348.9 224.5 C350.8 241.4, 350.7 257.3, 349 274.9 M348.8 276.1 C317.5 277.5, 291.1 277.4, 269.9 277.2 M349 276.8 C326.1 276, 300 275.9, 270.8 275.5 M271.5 277.7 C268 264.4, 267.9 252.5, 272.1 223.3 M270.1 277 C268.7 262.3, 269.5 249.8, 270.3 224.6" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="310" y="257" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">↓2</text><path d="M349.7 249.6 C364.2 249.4, 377.2 250.4, 418.5 248.4 M350 249.8 C377.6 250.1, 403.7 249, 420.6 249.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M419.8 249.8 C417.9 250.3, 415.2 252, 408.3 254.2 M420 249.9 C415.9 252.1, 411 253.4, 409.5 255.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M419.8 249.8 C418 248.3, 415.2 247.8, 408.3 243.8 M420 249.9 C416 248, 411.1 245.3, 409.5 244.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><text x="440" y="257" text-anchor="middle" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--dot)">d</text><path d="M459.7 249.6 C472.2 249.3, 483.2 250.3, 518.5 248.4 M460 249.8 C483.7 250.1, 505.9 249, 520.6 249.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M519.8 249.8 C517.9 250.3, 515.2 252, 508.3 254.2 M520 249.9 C515.9 252.1, 511 253.4, 509.5 255.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M519.8 249.8 C518 248.3, 515.2 247.8, 508.3 243.8 M520 249.9 C516 248, 511.1 245.3, 509.5 244.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M519.7 223.6 C536.2 223.4, 551.2 224.5, 598.5 222.4 M520 223.8 C551.5 224.1, 581.5 223, 600.6 223.8 M600.9 223.2 C601.3 238.5, 598.6 249, 599.4 275.8 M598.9 224.5 C600.8 241.4, 600.7 257.3, 599 274.9 M598.8 276.1 C567.5 277.5, 541.1 277.4, 519.9 277.2 M599 276.8 C576.1 276, 550 275.9, 520.8 275.5 M521.5 277.7 C518 264.4, 517.9 252.5, 522.1 223.3 M520.1 277 C518.7 262.3, 519.5 249.8, 520.3 224.6" style="stroke:var(--ink);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="560" y="257" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--ink)">↑2</text><path d="M599.7 249.6 C608.2 249.2, 615.2 250.2, 638.5 248.4 M600 249.8 C615.9 250.1, 630.3 248.9, 640.6 249.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M639.8 249.8 C637.9 250.3, 635.2 252, 628.3 254.2 M640 249.9 C635.9 252.1, 631 253.4, 629.5 255.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M639.8 249.8 C638 248.3, 635.2 247.8, 628.3 243.8 M640 249.9 C636 248, 631.1 245.3, 629.5 244.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M639.7 223.6 C656.2 223.4, 671.2 224.5, 718.5 222.4 M640 223.8 C671.5 224.1, 701.5 223, 720.6 223.8 M720.9 223.2 C721.3 238.5, 718.6 249, 719.4 275.8 M718.9 224.5 C720.8 241.4, 720.7 257.3, 719 274.9 M718.8 276.1 C687.5 277.5, 661.1 277.4, 639.9 277.2 M719 276.8 C696.1 276, 670 275.9, 640.8 275.5 M641.5 277.7 C638 264.4, 637.9 252.5, 642.1 223.3 M640.1 277 C638.7 262.3, 639.5 249.8, 640.3 224.6" style="stroke:var(--dot);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="680" y="257" text-anchor="middle" style="font-family:Kalam,cursive;font-size:20px;font-weight:400;fill:var(--dot)">g</text><path d="M719.7 249.6 C728.2 249.2, 735.2 250.2, 758.5 248.4 M720 249.8 C735.9 250.1, 750.3 248.9, 760.6 249.8" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M759.7 119.6 C759.1 145, 758.1 172, 758.5 248.4 M760 119.8 C759.6 170.8, 758.4 220.3, 760.6 249.8" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M759.7 184.6 C768.2 184.2, 775.2 185.2, 798.5 183.4 M760 184.8 C775.9 185.1, 790.3 183.9, 800.6 184.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M799.8 184.8 C797.9 185.3, 795.2 187, 788.3 189.2 M800 184.9 C795.9 187.1, 791 188.4, 789.5 190.1" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M799.8 184.8 C798 183.3, 795.2 182.8, 788.3 178.8 M800 184.9 C796 183, 791.1 180.3, 789.5 179.7" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><text x="830" y="191" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">x</text><text x="250" y="320" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">analysis: filter, keep every 2nd</text><text x="640" y="320" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">synthesis: insert zeros, filter, add</text><text x="440" y="80" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--a1)">repeat on a for the next level</text></svg>
</figure>

**Cost.** Each level does two filterings of length $L$ on the current approximation and halves it: $2L(N + N/2 + N/4 + \cdots) < 4LN$ multiply-adds. That is $O(N)$, cheaper than an FFT's $O(N\log N)$.

:::question
Where is the wavelet in the code?
:::

:::answer
Nowhere. A discrete wavelet transform is two short filters, a downsampling and a loop. The functions $\varphi$ and $\psi$ appear only if you ask what the filters converge to when the loop is run backwards forever (Section 11's cascade algorithm). Choosing a wavelet means choosing four, eight or twelve numbers.
:::

**Boundaries.** A finite signal needs values beyond its ends, and libraries differ. *Periodisation* wraps around: exactly $N/2$ coefficients per level, but a jump wherever the ends differ. *Symmetric* extension mirrors the signal: no jump, but more coefficients than samples. Measured with PyWavelets: Daubechies-4 (`db2`) on 8 samples gives 4 coefficients per band with periodisation and 5 with symmetric extension; Daubechies-8 (`db4`) on 256 samples gives 128 and 131.

## 7. Theory meets code

Most of the confusion around wavelets lives neither in the theory nor in the code, but in the translation between them. When I first learned this material, the proofs were clear and the library calls were easy; what I could not see was how one became the other. This section is that translation, point by point.

### Which way do the levels count?

Three conventions are in common use, and they disagree about the direction of the index:

- **This post (Daubechies, Strang):** $V_j \subset V_{j+1}$, so a **larger $j$ is finer**, and $\psi_{j,k}(x) = 2^{j/2}\psi(2^jx - k)$ has width $2^{-j}$. The data sit in some finest space $V_J$, and each split moves *down*: $V_J = V_{J-1}\oplus W_{J-1}$.
- **Mallat's book:** $V_{j+1} \subset V_j$, so a **larger $j$ is coarser**, and $\psi_{j,n}(t) = 2^{-j/2}\psi(2^{-j}t - n)$ has width $2^j$. The data sit in $V_0$ and each split moves *up*: $V_0 = V_1 \oplus W_1$. This matches how code counts.
- **Libraries (PyWavelets, MATLAB):** "level" counts **how many splits have been done**. `cD1` is the first, finest detail; `wavedec(x, w, level=n)` returns `[cA_n, cD_n, ..., cD_1]`, coarsest first.

The dictionary: Mallat's $j$ is this post's $-j$, and with data at this post's level $J$, the library's level $\ell$ is this post's $j = J - \ell$. So `cD1` $= d_{J-1}$, `cD2` $= d_{J-2}$, and `cA_n` $= a_{J-n}$. Neither theory nor code is wrong: theory counts resolution upward from coarse, code counts the operations it has performed. When reading anything about wavelets, first check whether $V_j$ grows or shrinks with $j$.

### Why one filter is low-pass and the other high-pass

Nothing in the two-scale equation says "filter". The names come from two facts already proved:

- $H(0) = \sum_kh_k = \sqrt2$ (Section 5): a constant signal passes through $h$ with gain $\sqrt2$, the orthonormal version of averaging.
- $H(\pi) = \sum_k(-1)^kh_k = 0$ (one vanishing moment, Section 10): the fastest possible signal, $+1, -1, +1, \ldots$, is blocked.

So $h$ passes slow signals and stops fast ones: **low-pass**. The alternating flip gives $\lvert G(\omega)\rvert = \lvert H(\omega + \pi)\rvert$, so $g$ does the opposite, $G(0) = 0$ and $\lvert G(\pi)\rvert = \sqrt2$: **high-pass**. Orthonormality, $\lvert H\rvert^2 + \lvert G\rvert^2 = 2$ (Section 9), says that together they split the energy at every frequency exactly. For Haar, $\lvert H(\omega)\rvert = \sqrt2\lvert\cos(\omega/2)\rvert$ and $\lvert G(\omega)\rvert = \sqrt2\lvert\sin(\omega/2)\rvert$; the wavelet zoo in Section 11 plots both for every wavelet. In one line: **low-pass = scaling function = averages = approximation; high-pass = wavelet = differences = detail.**

### Four filters, one list of numbers

A library stores four filters per orthogonal wavelet, and every one of them is the single list $h$ rearranged. With $L$ taps:

| Library name | Is |
|---|---|
| `rec_lo` | $h$ |
| `dec_lo` | $h$ reversed |
| `rec_hi` | $g$, with $g[n] = (-1)^nh[L - 1 - n]$ |
| `dec_hi` | $g$ reversed |

For Daubechies-4 in PyWavelets: `rec_lo` $= (0.4830, 0.8365, 0.2241, -0.1294)$, `dec_lo` $= (-0.1294, 0.2241, 0.8365, 0.4830)$, `rec_hi` $= (-0.1294, -0.2241, 0.8365, -0.4830)$ and `dec_hi` $= (-0.4830, 0.8365, -0.2241, -0.1294)$ (all four checked). Why reversed? Mallat's formula $a_j[k] = \sum_nh_{n-2k}\,a_{j+1}[n]$ is a **correlation** with $h$. Libraries implement every filter as a **convolution**, and convolving with the reversed filter is correlating with the original. So analysis ("decomposition") uses the reversed filters, and synthesis ("reconstruction"), which really is a convolution after inserting zeros, uses $h$ and $g$ as they are.

Two smaller differences you will meet. Section 5 used $g_k = (-1)^kh_{1-k}$, the convention of most books; libraries use $(-1)^nh[L - 1 - n]$. These are the same filter shifted by $L - 2$ places, an even number, so the detail coefficients move by $(L - 2)/2$ positions and nothing else changes. And some texts flip the sign of $g$, which negates every detail coefficient. Magnitudes always agree.

### PyWavelets, rebuilt in NumPy

Here is `pywt.dwt` in periodisation mode, written out with nothing but the numbers above:

```python
import numpy as np, pywt

w = pywt.Wavelet('db2'); h = np.array(w.rec_lo); L = len(h)
g = np.array([(-1)**n * h[L - 1 - n] for n in range(L)])            # the alternating flip
x = np.array([9, 7, 3, 5, 6, 10, 2, 6.]); N = len(x)
s = L // 2 - 1                                                       # PyWavelets centres each window
a = np.array([sum(h[n] * x[(2*k + n - s) % N] for n in range(L)) for k in range(N // 2)])
d = np.array([sum(g[n] * x[(2*k + n - s) % N] for n in range(L)) for k in range(N // 2)])
cA, cD = pywt.dwt(x, 'db2', mode='periodization')
print(np.allclose(a, cA), np.allclose(d, cD))                        # True True
```

The textbook formula starts the $k$-th window at sample $2k$; PyWavelets starts it at $2k - (L/2 - 1)$, which centres it. I checked this offset for Haar, Daubechies-4, 6 and 8, Symlet-8 and Coiflet-12. **Two correct implementations can therefore differ by a shift of the coefficients.** Compare reconstructions or magnitudes, never raw arrays from different libraries. The default boundary mode, `symmetric`, is the same recipe on a mirrored signal: pad $L - 1$ mirrored samples (edge repeated) on each side, convolve with `dec_lo`, and keep every second output starting at index $L$. That yields $\lfloor(N + L - 1)/2\rfloor$ coefficients per band, which is where the 5 (for $N = 8$, $L = 4$) and the 131 (for $N = 256$, $L = 8$) of Section 6 come from.

### Where do the samples enter?

This is the gap that confused me longest. The theory says the finest coefficients are inner products, $a_J[k] = \langle f, \varphi_{J,k}\rangle$. The code is handed samples, $x[k] = f(k2^{-J})$, and treats them *as* the $a_J[k]$. Gilbert Strang calls this "the wavelet crime". How wrong is it?

:::proposition[Samples are scaling coefficients, shifted]
Let $c = \int u\,\varphi(u)\,du$ be the centroid of $\varphi$ (with $\int\varphi = 1$). For smooth $f$,
$$
\langle f, \varphi_{J,k}\rangle = 2^{-J/2}\,f\!\left(\frac{k + c}{2^J}\right) + O\big(2^{-5J/2}\big).
$$
:::

:::proof
**Step 1. Substitute** $u = 2^Jx - k$: $\langle f, \varphi_{J,k}\rangle = \int f(x)\,2^{J/2}\varphi(2^Jx - k)\,dx = 2^{-J/2}\int f\big(\tfrac{k + u}{2^J}\big)\varphi(u)\,du$.

**Step 2. Taylor-expand** $f$ around $x_c = (k + c)/2^J$: $f\big(\tfrac{k + u}{2^J}\big) = f(x_c) + f'(x_c)\frac{u - c}{2^J} + O(2^{-2J})$.

**Step 3. Integrate against $\varphi$.** $\int\varphi = 1$, and $\int(u - c)\varphi(u)\,du = c - c = 0$ by the definition of $c$. Only $f(x_c)$ survives, plus the remainder, which is $2^{-J/2}\cdot O(2^{-2J})$.
:::

So a sample is, up to the factor $2^{-J/2}$, a scaling coefficient **at a shifted position**. For Daubechies-4, $c = 0.634$. Measured on a smooth signal at $J = 6$: using the samples directly as scaling coefficients is 10% off; using samples taken at the shifted positions, 0.02% off. Coiflets are designed so that the higher moments of $\varphi$ also vanish about its centre, which makes the shifted samples an excellent approximation.

Three consequences, worth stating plainly:
1. **Perfect reconstruction never depends on this.** The discrete transform is an exact orthonormal change of basis of whatever vector you give it.
2. **Interpretation does.** Reading the finest detail coefficients as samples of continuous wavelet coefficients is right only up to this shift and a slight smoothing.
3. **The factor $2^{J/2}$ explains a common surprise:** for a constant signal, each level of the orthonormal transform multiplies the approximation coefficients by $\sqrt2$ (Section 4's $6\cdot2^{3/2} = 16.97$).

### Where is each coefficient?

A detail coefficient at library level $\ell$ depends on a window of about $(L - 1)(2^\ell - 1) + 1$ samples, starting near $2^\ell k$ (plus the library's offset). To plot coefficients against time, place coefficient $k$ of level $\ell$ near sample $2^\ell(k + \tfrac12)$. Daubechies filters are *minimum phase*: their energy is concentrated at the start of the filter, so their coefficients lag the features that cause them. Symlets were designed to reduce exactly this lag.

## 8. The lifting scheme

There is a second way to compute the same transform, faster, in place, and able to map integers to integers. Split the signal into even and odd samples, then alternately use each half to **predict** and **update** the other.

<figure class="sketch-fig">
<svg class="sketch" viewBox="0 0 900 330" role="img" aria-label="A lifting ladder for the Haar transform. The signal is split into even and odd samples. A predict step replaces the odd samples by d equals odd minus even. An update step replaces the even samples by a equals even plus d over 2. Inverting runs the steps backwards with the signs flipped."><text x="450" y="36" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--a1)">lifting: split, predict, update (Haar, in place)</text><text x="50" y="181" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--ink)">x</text><path d="M69.1 174.5 C76.8 174.5, 87.2 176, 109.4 175.1 M69.3 175.9 C81 174.4, 91.5 175.2, 110.2 176" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M109.1 119.5 C108.7 141.7, 111.1 165.2, 109.4 230.1 M109.3 120.9 C110.9 149.6, 110.3 180.6, 110.2 231" style="stroke:var(--ink);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><text x="140" y="106" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">even</text><text x="140" y="258" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">odd</text><path d="M109.6 119.8 C255.6 118.8, 402.7 119.4, 839.8 120 M109.7 120.4 C311.1 120.8, 511.9 121.1, 840.1 120.4" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M839.4 119.7 C837.1 120.9, 836.4 122.8, 828.8 125.3 M839.6 120.6 C837.1 121, 833.8 122.8, 829.3 125.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M839.4 119.7 C837.1 118.8, 836.4 118.7, 828.8 114.8 M839.6 120.6 C837 118.1, 833.7 117.1, 829.3 115.4" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M109.6 229.8 C255.6 228.8, 402.7 229.4, 839.8 230 M109.7 230.4 C311.1 230.8, 511.9 231.1, 840.1 230.4" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M839.4 229.7 C837.1 230.9, 836.4 232.8, 828.8 235.3 M839.6 230.6 C837.1 231, 833.8 232.8, 829.3 235.8" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M839.4 229.7 C837.1 228.8, 836.4 228.7, 828.8 224.8 M839.6 230.6 C837 228.1, 833.7 227.1, 829.3 225.4" style="stroke:var(--ink);stroke-width:2.2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M329.1 150.5 C352.8 150.2, 379.2 151.7, 449.4 151.1 M329.3 151.9 C363.1 150.7, 395.6 151.4, 450.2 152 M448.9 151.3 C450.6 164.5, 451 182.3, 451.1 199.4 M450.2 150.2 C450.4 161.6, 450.4 169.6, 450.8 199.5 M449.1 198.2 C419 199.9, 390.7 201.8, 329.2 198.4 M449.3 199.1 C424.9 200.9, 399.7 199.7, 330.5 198.7 M328.3 199.5 C329.8 183.2, 330.9 163.8, 330.3 149.4 M331 200 C330.7 183, 331.5 167.8, 330.1 151.6" style="stroke:var(--dot);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="390" y="182" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--dot)">predict</text><path d="M389.1 123.5 C388.8 128.7, 391.1 135.2, 389.4 149.1 M389.3 124.9 C390.2 130.2, 389.7 137.8, 390.2 150" style="stroke:var(--dot);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M389.4 148.7 C388.2 146.7, 388.6 145.4, 384.4 138.3 M389.6 149.6 C388.5 145.6, 386.7 143, 384.9 138.8" style="stroke:var(--dot);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M389.4 148.7 C390.3 146.6, 392.8 145.4, 394.9 138.3 M389.6 149.6 C391.4 145.6, 392.5 143.1, 395.4 138.8" style="stroke:var(--dot);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M389.1 200.5 C388.8 205.7, 391.1 212.2, 389.4 226.1 M389.3 201.9 C390.2 207.2, 389.7 214.8, 390.2 227" style="stroke:var(--dot);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M389.4 225.7 C388.2 223.7, 388.6 222.4, 384.4 215.3 M389.6 226.6 C388.5 222.6, 386.7 220, 384.9 215.8" style="stroke:var(--dot);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M389.4 225.7 C390.3 223.6, 392.8 222.4, 394.9 215.3 M389.6 226.6 C391.4 222.6, 392.5 220.1, 395.4 215.8" style="stroke:var(--dot);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><text x="470" y="218" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--dot)">d = odd − even</text><path d="M569.1 150.5 C592.8 150.2, 619.2 151.7, 689.4 151.1 M569.3 151.9 C603.1 150.7, 635.6 151.4, 690.2 152 M688.9 151.3 C690.6 164.5, 691 182.3, 691.1 199.4 M690.2 150.2 C690.4 161.6, 690.4 169.6, 690.8 199.5 M689.1 198.2 C659 199.9, 630.7 201.8, 569.2 198.4 M689.3 199.1 C664.9 200.9, 639.7 199.7, 570.5 198.7 M568.3 199.5 C569.8 183.2, 570.9 163.8, 570.3 149.4 M571 200 C570.7 183, 571.5 167.8, 570.1 151.6" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="630" y="182" text-anchor="middle" style="font-family:Kalam,cursive;font-size:18px;font-weight:400;fill:var(--a1)">update</text><path d="M629.1 225.5 C628.8 220.7, 631.1 217.2, 629.4 201.1 M629.3 226.9 C629.8 218.4, 629.3 212.2, 630.2 202" style="stroke:var(--a1);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M629.4 200.7 C630.3 202.9, 632.8 206, 634.9 211.9 M629.6 201.6 C631.5 203.6, 632.7 207, 635.4 212.4" style="stroke:var(--a1);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M629.4 200.7 C628.2 203, 628.6 206, 624.4 211.9 M629.6 201.6 C628.7 203.5, 626.9 206.9, 624.9 212.4" style="stroke:var(--a1);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M629.1 148.5 C628.8 143.7, 631.1 140.2, 629.4 124.1 M629.3 149.9 C629.8 141.4, 629.3 135.2, 630.2 125" style="stroke:var(--a1);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M629.4 123.7 C630.3 125.9, 632.8 129, 634.9 134.9 M629.6 124.6 C631.5 126.6, 632.7 130, 635.4 135.4" style="stroke:var(--a1);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M629.4 123.7 C628.2 126, 628.6 129, 624.4 134.9 M629.6 124.6 C628.7 126.5, 626.9 129.9, 624.9 135.4" style="stroke:var(--a1);stroke-width:2;stroke-opacity:1;fill:none;stroke-linecap:round"/><text x="700" y="108" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--a1)">a = even + d / 2</text><text x="860" y="126" text-anchor="start" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--ink)">a</text><text x="860" y="236" text-anchor="start" style="font-family:Kalam,cursive;font-size:22px;font-weight:700;fill:var(--ink)">d</text><text x="450" y="300" text-anchor="middle" style="font-family:Kalam,cursive;font-size:16px;font-weight:400;fill:var(--muted)">to invert, run the ladder backwards with the signs flipped: every step undoes itself exactly</text></svg>
</figure>

:::example[Haar by lifting]
Same eight numbers: evens $(9, 3, 6, 2)$, odds $(7, 5, 10, 6)$.

**Predict** each odd sample by its even neighbour and keep the error: $d = \text{odd} - \text{even} = (-2, 2, 4, 4)$.

**Update** the evens so that they become averages: $a = \text{even} + d/2 = (8, 4, 8, 4)$.

These are the averages of Section 4, and $d$ is $-2$ times the half-differences (lifting conventions differ by such factors). To invert, run the ladder backwards with flipped signs: $\text{even} = a - d/2$, then $\text{odd} = d + \text{even}$. Each step is undone exactly, whatever was computed inside it.
:::

That last remark is the point. Because each step adds a function of one half to the other, it can be inverted *even if the function rounds*. JPEG 2000's lossless mode uses the **5/3 integer wavelet** this way:

$$
d[n] = x[2n+1] - \Big\lfloor\frac{x[2n] + x[2n+2]}{2}\Big\rfloor, \qquad s[n] = x[2n] + \Big\lfloor\frac{d[n-1] + d[n] + 2}{4}\Big\rfloor .
$$

:::example[5/3 by hand]
With symmetric extension ($x[8] = x[6]$, $d[-1] = d[0]$) on $(9, 7, 3, 5, 6, 10, 2, 6)$:
- $d[0] = 7 - \lfloor(9 + 3)/2\rfloor = 1$, $d[1] = 5 - \lfloor(3 + 6)/2\rfloor = 1$, $d[2] = 10 - \lfloor(6 + 2)/2\rfloor = 6$, $d[3] = 6 - \lfloor(2 + 2)/2\rfloor = 4$;
- $s[0] = 9 + \lfloor(1 + 1 + 2)/4\rfloor = 10$, $s[1] = 3 + \lfloor(1 + 1 + 2)/4\rfloor = 4$, $s[2] = 6 + \lfloor(1 + 6 + 2)/4\rfloor = 8$, $s[3] = 2 + \lfloor(6 + 4 + 2)/4\rfloor = 5$.

Integers in, integers out, and running the steps backwards recovers $(9, 7, 3, 5, 6, 10, 2, 6)$ exactly (checked).
:::

Daubechies and Sweldens (1998) proved that every two-channel perfect-reconstruction filter bank with finite filters factors into such lifting steps, so lifting is not a special trick but a universal way to implement wavelets, typically with about half the arithmetic.

## 9. Perfect reconstruction

When do analysis filters $\tilde h, \tilde g$ followed by synthesis filters $h, g$ return the input exactly? The answer is two equations in the frequency domain, and the key is what "keep every second sample, then put zeros back" does to a spectrum.

:::lemma[Downsampling then upsampling]
If $y$ has transform $Y(\omega)$, the sequence that keeps $y[n]$ at even $n$ and puts $0$ at odd $n$ has transform $\tfrac12\big(Y(\omega) + Y(\omega + \pi)\big)$.
:::

:::proof
That sequence is $y[n]\cdot\tfrac12(1 + (-1)^n)$, and $(-1)^n = e^{i\pi n}$. Multiplying by $e^{i\pi n}$ shifts a spectrum by $\pi$ (the modulation rule of Part 1, in radians), so the transform is $\tfrac12Y(\omega) + \tfrac12Y(\omega + \pi)$.
:::

The term $Y(\omega + \pi)$ is aliasing, the folding of Part 2. Push $X$ through both channels with the lemma, and the output is

$$
\hat X(\omega) = \tfrac12\big[H\tilde H + G\tilde G\big](\omega)\,X(\omega) + \tfrac12\big[H(\omega)\tilde H(\omega + \pi) + G(\omega)\tilde G(\omega + \pi)\big]X(\omega + \pi).
$$

Perfect reconstruction needs the second bracket (aliasing) to vanish and the first (distortion) to equal $2$. Each channel aliases on its own; the two aliases cancel.

:::proposition[Orthonormal filters reconstruct perfectly]
For real $h$, the orthonormality condition $\sum_kh_kh_{k-2m} = \delta_m$ is equivalent to
$$
\lvert H(\omega)\rvert^2 + \lvert H(\omega + \pi)\rvert^2 = 2 \quad\text{for all }\omega .
$$
With $\tilde H = \overline H$, $\tilde G = \overline G$ (analysis filters reversed in time) and $G(\omega) = -e^{-i\omega}\,\overline{H(\omega + \pi)}$ (the alternating flip of Section 5), both conditions of perfect reconstruction hold.
:::

:::proof
**Step 1. Autocorrelation.** Let $r_m = \sum_kh_kh_{k-m}$. Then $\lvert H(\omega)\rvert^2 = H(\omega)\overline{H(\omega)} = \sum_mr_me^{-im\omega}$.

**Step 2. Add the shifted copy.** $\lvert H(\omega)\rvert^2 + \lvert H(\omega+\pi)\rvert^2 = \sum_mr_m\big(1 + (-1)^m\big)e^{-im\omega} = 2\sum_mr_{2m}e^{-2im\omega}$: odd lags cancel.

**Step 3. Equivalence.** This equals the constant $2$ for all $\omega$ exactly when $r_{2m} = \delta_m$, by uniqueness of Fourier coefficients. That is the orthonormality condition.

**Step 4. No aliasing.** The alias bracket is $H(\omega)\overline{H(\omega + \pi)} + G(\omega)\overline{G(\omega + \pi)}$. Using $G(\omega + \pi) = -e^{-i(\omega + \pi)}\overline{H(\omega + 2\pi)} = e^{-i\omega}\overline{H(\omega)}$, the second product is $-e^{-i\omega}\overline{H(\omega + \pi)}\cdot e^{i\omega}H(\omega) = -H(\omega)\overline{H(\omega + \pi)}$, which cancels the first.

**Step 5. No distortion.** $\lvert H(\omega)\rvert^2 + \lvert G(\omega)\rvert^2 = \lvert H(\omega)\rvert^2 + \lvert H(\omega+\pi)\rvert^2 = 2$ by Step 3.
:::

In words: the low-pass and the high-pass responses are mirror images about $\omega = \pi/2$ and their energies add to a constant. These are the "quadrature mirror" and "conjugate quadrature" filters of 1970s and 1980s speech coding, which turned out to be wavelets in disguise.

## 10. Vanishing moments: why wavelets make signals sparse

Haar is orthonormal, but it approximates smooth curves with staircases. What makes a better wavelet? Not smoothness of $\psi$, primarily, but this:

:::definition[Vanishing moments]
$\psi$ has $p$ vanishing moments if $\int x^m\psi(x)\,dx = 0$ for $m = 0, 1, \ldots, p - 1$.
:::

A wavelet with $p$ vanishing moments is blind to polynomials of degree below $p$. Where the signal is smooth it looks like a polynomial at fine scales, so the wavelet coefficients there are tiny.

:::theorem[Decay away from singularities]
Let $\psi$ have $p$ vanishing moments and support in an interval $[0, S]$. If $f$ has $p$ continuous derivatives on the support of $\psi_{j,k}$, then
$$
\lvert d_j[k]\rvert = \lvert\langle f, \psi_{j,k}\rangle\rvert \;\le\; C\,2^{-j(p + 1/2)}\max\lvert f^{(p)}\rvert, \qquad C = \frac{1}{p!}\int\lvert u\rvert^p\lvert\psi(u)\rvert\,du .
$$
:::

:::proof
**Step 1. Taylor's theorem** around $x_0 = k2^{-j}$: $f(x) = P(x) + R(x)$, with $P$ a polynomial of degree $p - 1$ and $\lvert R(x)\rvert \le \frac{\max\lvert f^{(p)}\rvert}{p!}\lvert x - x_0\rvert^p$.

**Step 2. The polynomial is invisible.** $\langle P, \psi_{j,k}\rangle = 0$: after the substitution $u = 2^jx - k$, it is a combination of the moments $\int u^m\psi(u)du$ with $m < p$, all zero.

**Step 3. Bound the remainder.** $\lvert\langle R, \psi_{j,k}\rangle\rvert \le \int\lvert R(x)\rvert\,2^{j/2}\lvert\psi(2^jx - k)\rvert\,dx \le \frac{\max\lvert f^{(p)}\rvert}{p!}\int\lvert x - x_0\rvert^p\,2^{j/2}\lvert\psi(2^jx - k)\rvert\,dx$.

**Step 4. Substitute** $u = 2^jx - k$, so $x - x_0 = 2^{-j}u$ and $dx = 2^{-j}du$: the integral becomes $2^{-jp}\cdot2^{j/2}\cdot2^{-j}\int\lvert u\rvert^p\lvert\psi(u)\rvert du = 2^{-j(p + 1/2)}\int\lvert u\rvert^p\lvert\psi(u)\rvert du$.
:::

So on smooth stretches, each finer level shrinks the coefficients by a factor $2^{p + 1/2}$. Measured on a smooth periodic signal: $2.83$ per level for Haar ($p = 1$), $5.66$ for Daubechies-4 ($p = 2$) and $22.6$ for Daubechies-8 ($p = 4$), matching $2^{p+1/2}$ to three digits. At a jump no number of moments helps: the coefficients shrink only like $2^{-j/2}$. Every jump leaves a trail of a few large coefficients down the scales, and everything else is small. **That is sparsity**, and it is why wavelets compress and denoise piecewise-smooth signals so well. More moments are not free, though: they need longer filters, and a longer filter overlaps each singularity at more positions. In the figure below, coefficients larger than 0.01 number 134 of 240 for Haar, 33 for Daubechies-4 and 42 for Daubechies-8. The best wavelet balances the smoothness of the signal between its singularities against how many singularities there are. (Compare Part 1: Fourier coefficients of a single jump decay like $1/k$ *everywhere in the spectrum*. Wavelets confine the damage to the jump's neighbourhood.)

<figure class="fig">
  <canvas data-figure="moments" data-src="/aboutme/blog/spectral/zoo.json" aria-label="Top: a signal with smooth stretches, a triangular bump and a box. Below: its wavelet detail coefficients at four levels, drawn as stems at their positions, for the chosen wavelet. A slider shifts the signal."></canvas>
  <div class="controls">
    <span class="seg"><button type="button" data-mw="haar">Haar (p = 1)</button><button type="button" data-mw="db2" class="on">D4 (p = 2)</button><button type="button" data-mw="db4">D8 (p = 4)</button></span>
    <label class="slider">shift <input type="range" min="0" max="8" step="1" value="0" data-ms /> <span class="mono" data-msval>0</span></label>
  </div>
  <div class="insight" data-insight></div>
</figure>

Vanishing moments are easy to read off the filter.

:::proposition[Moments from the filter]
$\psi$ has $p$ vanishing moments if and only if $H(\omega)$ has a zero of order $p$ at $\omega = \pi$, that is, $\sum_k(-1)^kk^mh_k = 0$ for $m = 0, \ldots, p - 1$.
:::

:::proof
**Step 1. Moments are derivatives at zero.** By Part 1's rule for multiplication by $x$, $\int x^m\psi = \big(\tfrac{i}{2\pi}\big)^m\hat\psi^{(m)}(0)$. So $p$ vanishing moments means $\hat\psi$ vanishes to order $p$ at $0$.

**Step 2. The wavelet equation in frequency.** Transforming $\psi(x) = \sqrt2\sum_kg_k\varphi(2x - k)$ with the scaling and shift rules gives $\hat\psi(\xi) = \tfrac1{\sqrt2}G(\pi\xi)\,\hat\varphi(\xi/2)$, where $G(\omega) = \sum_kg_ke^{-ik\omega}$.

**Step 3. Only $G$ matters.** $\hat\varphi(0) = \int\varphi = 1 \ne 0$, so by the product rule $\hat\psi$ vanishes to order $p$ at $0$ exactly when $G$ does at $\omega = 0$.

**Step 4. Move to $H$.** $\lvert G(\omega)\rvert = \lvert H(\omega + \pi)\rvert$ for the alternating flip, so $G$ has a zero of order $p$ at $0$ exactly when $H$ has one at $\pi$.

**Step 5. Derivatives of $H$ at $\pi$.** $H^{(m)}(\pi) = \sum_kh_k(-ik)^me^{-ik\pi} = (-i)^m\sum_k(-1)^kk^mh_k$.
:::

## 11. Daubechies-4, by hand

Now design a wavelet. Ask for the shortest filter with **two** vanishing moments. Haar's two taps can give only one, so try four, $h = (h_0, h_1, h_2, h_3)$.

:::example[Solving for the Daubechies-4 filter]
The conditions:
1. $h_0 + h_1 + h_2 + h_3 = \sqrt2$ (normalisation, Section 5);
2. $h_0 - h_1 + h_2 - h_3 = 0$ (first vanishing moment, $m = 0$);
3. $-h_1 + 2h_2 - 3h_3 = 0$ (second, $m = 1$; the $k = 0$ term is zero);
4. $h_0h_2 + h_1h_3 = 0$ (orthogonal to its shift by two).

(The condition $\sum h_k^2 = 1$ follows: square 1 and 2 and add to get $2\sum h_k^2 + 4(h_0h_2 + h_1h_3) = 2$, then use 4.)

**Step 1.** Adding and subtracting 1 and 2: $h_0 + h_2 = h_1 + h_3 = s$, with $s = 1/\sqrt2$.

**Step 2.** Write everything in terms of $d = h_3$: $h_1 = s - d$. Equation 3 becomes $-(s - d) + 2h_2 - 3d = 0$, so $h_2 = d + s/2$, and then $h_0 = s - h_2 = s/2 - d$.

**Step 3.** Substitute into 4: $(s/2 - d)(d + s/2) + (s - d)d = 0$, which expands to $s^2/4 - d^2 + sd - d^2 = 0$, that is, $2d^2 - sd - s^2/4 = 0$.

**Step 4.** The quadratic formula gives $d = \frac{s \pm\sqrt{s^2 + 2s^2}}{4} = \frac{s(1 \pm\sqrt3)}{4}$. Take the minus sign (the plus sign gives the same filter reversed in time), and $s = 1/\sqrt2$:
$$
h = \frac{1}{4\sqrt2}\big(1 + \sqrt3,\;\; 3 + \sqrt3,\;\; 3 - \sqrt3,\;\; 1 - \sqrt3\big) \approx (0.4830,\ 0.8365,\ 0.2241,\ -0.1294).
$$
This is exactly PyWavelets' `db2` reconstruction low-pass filter (checked).
:::

What scaling function does this filter define? Run the two-scale equation as an iteration, starting from a box: the **cascade algorithm**.

<figure class="fig">
<video autoplay muted loop playsinline controls preload="metadata" poster="/aboutme/blog/spectral/cascade.jpg"><source src="/aboutme/blog/spectral/cascade.webm" type="video/webm" /><source src="/aboutme/blog/spectral/cascade.mp4" type="video/mp4" /></video>
<figcaption>The cascade algorithm with the Daubechies-4 filter: starting from a box, each step replaces φ by √2 Σ hₖ φ(2x − k).</figcaption>
</figure>

The result is jagged at every scale: continuous, but rough (its Hölder exponent is only about 0.55, so it has corners at every scale, Daubechies 1992). The zoo below runs the same iteration for other filters.

<figure class="fig">
  <canvas data-figure="zoo" data-src="/aboutme/blog/spectral/zoo.json" aria-label="Left: the scaling function and wavelet of the chosen filter after a number of cascade steps. Right: the magnitude of the low-pass and high-pass frequency responses from 0 to pi."></canvas>
  <div class="controls">
    <span class="seg"><button type="button" data-wv="haar">Haar</button><button type="button" data-wv="db2" class="on">D4</button><button type="button" data-wv="db4">D8</button><button type="button" data-wv="sym4">Sym8</button><button type="button" data-wv="coif2">Coif12</button><button type="button" data-wv="bior4.4">CDF 9/7</button></span>
    <label class="slider">cascade steps <input type="range" min="0" max="6" step="1" value="6" data-lev /> <span class="mono" data-levval>8</span></label>
  </div>
  <div class="insight" data-insight></div>
</figure>

:::question
Why are Daubechies wavelets so ugly?
:::

:::answer
Because beauty was not the specification. They are the *shortest* orthonormal filters with a given number of vanishing moments, which is what makes them fast and sparse. Smoothness comes along slowly, and symmetry not at all. For analysis that needs smooth shapes, use more taps (D8, Symlets) or give up orthonormality (next answer).
:::

There is a theorem behind the lopsidedness. **No real, compactly supported, orthonormal wavelet is symmetric or antisymmetric, except Haar** (Daubechies, *Ten Lectures*, Theorem 8.1.4; we state it without proof). Symmetry matters for images, because an asymmetric filter shifts edges differently on their two sides. The way out is **biorthogonal** wavelets: use one filter pair to analyse and a different pair to synthesise. The perfect-reconstruction conditions of Section 9 still hold, orthonormality is given up, and both pairs can be symmetric. The Cohen–Daubechies–Feauveau **9/7** pair (four vanishing moments on each side) is the lossy wavelet of JPEG 2000, and **5/3** is the lossless one of Section 8.

## 12. Where the filter numbers come from

Every wavelet library contains tables of numbers like $0.4830, 0.8365, 0.2241, -0.1294$. Where do they come from? Except for Haar, **the filter is designed first**, from conditions on $H(\omega)$; the functions $\varphi$ and $\psi$ are whatever the cascade algorithm then produces.

**Haar is the exception: the function comes first.** Start from the box $\varphi = \mathbf{1}_{[0,1)}$. It is the sum of two half-width boxes, $\varphi(x) = \varphi(2x) + \varphi(2x - 1)$, so by the formula of Section 5, $h_k = \langle\varphi, \sqrt2\varphi(2\cdot{} - k)\rangle = \sqrt2\cdot\tfrac12 = \tfrac{1}{\sqrt2}$ for $k = 0, 1$. The alternating flip gives $g = (\tfrac1{\sqrt2}, -\tfrac1{\sqrt2})$: sum and difference, scaled.

**Daubechies wavelets: spectral factorisation.** Section 11 solved four equations by hand. That does not scale to longer filters; Daubechies' construction does.

:::proposition[The Daubechies construction]
A real filter with $p$ vanishing moments and orthonormal shifts has $H(\omega) = \sqrt2\Big(\frac{1 + e^{-i\omega}}{2}\Big)^pL(\omega)$, where $\lvert L(\omega)\rvert^2 = P\big(\sin^2\tfrac\omega2\big)$ for a polynomial $P$ with
$$
(1 - y)^pP(y) + y^pP(1 - y) = 1 .
$$
The lowest-degree solution is $P(y) = \sum_{k=0}^{p-1}\binom{p - 1 + k}{k}y^k$: $1$; $1 + 2y$; $1 + 3y + 6y^2$; $1 + 4y + 10y^2 + 20y^3$; and so on.
:::

:::proof
**Step 1. Vanishing moments are zeros at $\pi$** (Section 10): $H$ has a zero of order $p$ at $\omega = \pi$, so it factors as $\big(\frac{1 + e^{-i\omega}}{2}\big)^pL(\omega)$ times a constant; $H(0) = \sqrt2$ and the factor equals $1$ at $\omega = 0$, so the constant is $\sqrt2$ and $L(0) = 1$.

**Step 2. Magnitudes.** $\big\lvert\frac{1 + e^{-i\omega}}{2}\big\rvert^2 = \cos^2\frac\omega2$. Write $y = \sin^2\frac\omega2$, so $\cos^2\frac\omega2 = 1 - y$. Because $h$ is real, $\lvert L(\omega)\rvert^2$ is a real, even trigonometric polynomial, hence a polynomial in $\cos\omega = 1 - 2y$, hence a polynomial $P(y)$. So $\lvert H(\omega)\rvert^2 = 2(1 - y)^pP(y)$.

**Step 3. Orthonormality.** Replacing $\omega$ by $\omega + \pi$ swaps $\sin^2\frac\omega2$ and $\cos^2\frac\omega2$, that is, $y \leftrightarrow 1 - y$. The condition $\lvert H(\omega)\rvert^2 + \lvert H(\omega + \pi)\rvert^2 = 2$ of Section 9 becomes $(1 - y)^pP(y) + y^pP(1 - y) = 1$.

**Step 4. Check the solution for $p = 2$.** $(1 - y)^2(1 + 2y) + y^2(3 - 2y) = (1 - 2y + y^2)(1 + 2y) + 3y^2 - 2y^3 = 1 - 3y^2 + 2y^3 + 3y^2 - 2y^3 = 1$. (For general $p$ this is Bézout's theorem; see Daubechies, *Ten Lectures*, §6.1.)
:::

The last step, **spectral factorisation**, finds $L$ from $\lvert L\rvert^2$. For $p = 2$: $P = 1 + 2y = 1 + (1 - \cos\omega) = 2 - \cos\omega$. Try $L(\omega) = a + be^{-i\omega}$: then $\lvert L\rvert^2 = a^2 + b^2 + 2ab\cos\omega$, so $a^2 + b^2 = 2$ and $2ab = -1$, with $a + b = L(0) = 1$. The solution is $a = \frac{1 + \sqrt3}{2}$, $b = \frac{1 - \sqrt3}{2}$ (swapping them reverses the filter). Multiply out $\sqrt2\big(\frac{1 + z}{2}\big)^2(a + bz)$ with $z = e^{-i\omega}$: the coefficients are $\frac{\sqrt2}{4}\big(a,\ 2a + b,\ a + 2b,\ b\big) = \frac{1}{4\sqrt2}\big(1 + \sqrt3,\ 3 + \sqrt3,\ 3 - \sqrt3,\ 1 - \sqrt3\big)$: **the same D4 filter as Section 11, by a second route.**

For larger $p$, $\lvert L\rvert^2$ has $p - 1$ pairs of roots $r$ and $1/\bar r$, and building $L$ means choosing one root from each pair. That choice is what separates the families:
- **Daubechies (`dbN`):** every root inside the unit circle (*minimum phase*). The energy is front-loaded and the filter is as asymmetric as possible.
- **Symlets (`symN`):** the combination of roots whose filter is closest to symmetric (linear phase). Same length, same moments, less lag.
- **Coiflets (`coifN`):** a different polynomial, chosen so that $\varphi$ also has vanishing moments, which is what makes samples good scaling coefficients (Section 7).
- **Biorthogonal (`biorN.M`, CDF):** split the factors between two *different* filters, analysis and synthesis, so that both can be symmetric. The 9/7 and 5/3 filters of JPEG 2000 come from this.

The construction fits in a few lines, and it reproduces the library's tables:

```python
from math import comb
import numpy as np, pywt

def daubechies(p):
    """Low-pass filter with p vanishing moments, by spectral factorisation."""
    y = np.poly1d([-0.25, 0.5, -0.25])                                  # z * sin^2(w/2), z = e^{iw}
    Q = sum(comb(p - 1 + k, k) * y**k * np.poly1d([1] + [0] * (p - 1 - k)) for k in range(p))   # z^(p-1) P(y)
    r = Q.r                                                             # roots come in pairs r, 1/r
    L = np.real(np.poly(r[np.abs(r) < 1]))                              # keep those inside the unit circle
    L /= L.sum()                                                        # so that L = 1 at w = 0
    return np.sqrt(2) * np.convolve((np.poly1d([1, 1])**p).c / 2**p, L) # sqrt2 ((1 + z)/2)^p L(z)

for p in (2, 3, 4):
    print(np.allclose(daubechies(p), pywt.Wavelet(f'db{p}').rec_lo))     # True True True
```

Libraries ship these numbers as precomputed tables (PyWavelets) or compute them with the same factorisation (MATLAB's `dbaux`). Nothing about them is arbitrary: each table is the solution of a short list of conditions, *normalisation, orthogonality and vanishing moments*, plus a choice of roots.

## 13. Denoising

Now the payoff. Observe $y = f + \sigma z$, with $f$ piecewise smooth and $z$ white Gaussian noise. Three facts make wavelet denoising work, and each has a short proof.

:::proposition[Three facts behind wavelet denoising]
1. **Noise stays white.** If $\mathbf{W}$ is an orthonormal transform and $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, then $\mathbf{W}\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: in wavelet coordinates the noise is spread evenly over every coefficient.
2. **Soft thresholding is the best shrinkage under an $\ell^1$ penalty.** $\eta_\lambda(y) = \operatorname{sign}(y)\max(\lvert y\rvert - \lambda, 0)$ minimises $\tfrac12(\theta - y)^2 + \lambda\lvert\theta\rvert$ over $\theta$.
3. **The universal threshold catches pure noise.** For $N$ independent $\mathcal{N}(0, \sigma^2)$ values, $P\big(\max_i\lvert z_i\rvert > \sigma\sqrt{2\ln N}\big) \le \sqrt{2/\pi}\,/\sqrt{2\ln N} \to 0$.
:::

:::proof
**Fact 1.** A linear map of a Gaussian vector is Gaussian, with covariance $\mathbf{W}\mathbf{I}\mathbf{W}^{\top} = \mathbf{W}\mathbf{W}^{\top} = \mathbf{I}$ because $\mathbf{W}$ is orthonormal.

**Fact 2.** For $\theta > 0$ the objective is a parabola with minimum at $\theta = y - \lambda$, valid if $y > \lambda$. For $\theta < 0$ the minimum is at $y + \lambda$, valid if $y < -\lambda$. Otherwise ($\lvert y\rvert \le \lambda$) the objective increases away from $\theta = 0$ on both sides, since its slopes there are $\pm\lambda - y$ with the right signs, so $\theta = 0$.

**Fact 3, a Gaussian tail bound.** For $t > 0$, $\int_t^\infty e^{-u^2/2}du \le \int_t^\infty\frac ut e^{-u^2/2}du = \frac{e^{-t^2/2}}{t}$, so $P(\lvert z\rvert > t\sigma) \le \sqrt{2/\pi}\,e^{-t^2/2}/t$.

**Fact 3, the union bound.** $P(\max_i\lvert z_i\rvert > t\sigma) \le N\sqrt{2/\pi}\,e^{-t^2/2}/t$. With $t = \sqrt{2\ln N}$, $e^{-t^2/2} = 1/N$, leaving $\sqrt{2/\pi}/\sqrt{2\ln N}$.
:::

The bound goes to zero, but slowly: measured on 4,000 trials, pure noise of length 1,024 exceeds the threshold $3.72\sigma$ somewhere 19% of the time (the bound says at most 21%). In practice the occasional surviving noise coefficient is small. The noise level itself is estimated from the finest details, which are mostly noise: $\hat\sigma = \operatorname{median}(\lvert d_1\rvert)/0.6745$, since $0.6745$ is the median of $\lvert\mathcal{N}(0, 1)\rvert$ (measured on $10^5$ samples: $\hat\sigma = 1.0005$).

The recipe (Donoho and Johnstone, 1994): transform, estimate $\sigma$, soft-threshold every detail coefficient at $\hat\sigma\sqrt{2\ln N}$, transform back.

<figure class="fig">
  <canvas data-figure="denoise" data-src="/aboutme/blog/spectral/zoo.json" aria-label="A piecewise-smooth signal with jumps and a spike, a noisy version, and the estimate from the chosen method."></canvas>
  <div class="controls">
    <span class="seg"><button type="button" data-dn="soft" class="on">wavelet, soft</button><button type="button" data-dn="hard">wavelet, hard</button><button type="button" data-dn="blur">Gaussian blur</button></span>
    <label class="slider">noise σ <input type="range" min="0.02" max="0.4" step="0.01" value="0.12" data-sig /> <span class="mono" data-sigval>0.12</span></label>
  </div>
  <div class="insight" data-insight></div>
</figure>

A Fourier low-pass filter (the Gaussian blur) treats every place alike, so it must choose one width that suits the smooth wave, the jumps and the spike at once. The figure gives it an unfair advantage, the best possible width chosen by looking at the true signal. The wavelet methods see only the noisy data. Measured (RMS error for soft thresholding, hard thresholding and the oracle blur):

| Noise σ | Soft | Hard | Best blur |
|---|---|---|---|
| 0.05 | 0.046 | 0.023 | 0.038 |
| 0.12 | 0.087 | 0.056 | 0.064 |
| 0.20 | 0.118 | 0.082 | 0.083 |
| 0.40 | 0.161 | 0.148 | 0.116 |

Hard thresholding beats even the oracle blur at low and moderate noise, keeping the jumps and the spike sharp. Soft thresholding at the universal threshold, the classical "VisuShrink", is the most conservative: the threshold is designed so that essentially no noise survives, and soft thresholding also pulls every surviving coefficient toward zero, so it over-smooths. At heavy noise the jumps are buried, and an ideally tuned blur wins. Data-driven thresholds, such as Stein's unbiased risk estimate per level ("SureShrink"; Donoho and Johnstone, 1995), close most of the gap for soft thresholding. [Part 4](/aboutme/blog/wavelets-for-images/) repeats the comparison on images.

## 14. What the discrete transform gets wrong

**It is not shift-invariant.** The discrete transform samples positions on a dyadic grid, so moving the input by one sample does not move the coefficients by one place: they change. Try the shift slider in the vanishing-moments figure and watch the finest-level energy change. Even the decay measurement at a jump shows it: for Haar, the ratio between successive levels at one jump came out $0.71, 2.12, 0.71, 3.06$ instead of a steady $1.41$, because the answer depends on where the jump sits relative to the grid. The cause is the one from Part 2: every level downsamples, and each band aliases a little.

Three remedies, in increasing sophistication:
- **Cycle spinning** (Coifman and Donoho, 1995): denoise every shifted copy, shift back, average. Simple and effective.
- **The undecimated (stationary) transform**: skip the downsampling and dilate the filters instead. Exactly shift-invariant, at $J + 1$ times the storage for $J$ levels.
- **The dual-tree complex wavelet transform** (Kingsbury): two real trees whose filters form approximate Hilbert-transform pairs, giving complex coefficients whose magnitudes barely change under shifts, at only twice the storage in 1-D.

**It knows nothing about direction.** In 1-D there is none to know. In 2-D, [Part 4](/aboutme/blog/wavelets-for-images/) shows that the separable construction sees only horizontal, vertical and a "diagonal" that cannot tell / from \\.

## 15. Images, briefly

Everything here extends to images by doing the 1-D transform along rows and then along columns. Each level then produces one approximation and *three* detail images. [Part 4](/aboutme/blog/wavelets-for-images/) derives why three, does a $2\times2$ and a $4\times4$ example by hand and an $8\times8$ one live, settles the confusing naming of the bands, and shows compression, denoising and the shift problem on my sketches.

## 16. Choosing a tool

<figure class="sketch-fig">
<svg class="sketch" viewBox="0 0 900 460" role="img" aria-label="A decision map. Which frequencies over the whole signal: Fourier. Which frequencies when, at one resolution: short-time Fourier transform. Which scales where, in fine detail: continuous wavelets. A compact, invertible, fast representation: discrete wavelets. A shift-stable, oriented representation of images: pyramids or complex wavelets."><text x="450" y="36" text-anchor="middle" style="font-family:Kalam,cursive;font-size:24px;font-weight:700;fill:var(--a1)">which transform? ask what you need to know</text><text x="40" y="96" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">which frequencies, over the whole signal?</text><path d="M470.9 91.6 C480.1 87.3, 493.9 91.2, 532.1 91 M470.1 89.3 C490.1 90.7, 507.9 90.1, 528.9 90.4" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 91 C526.7 89.8, 525.6 93.2, 520.4 95.8 M530.1 89.6 C527.2 92.4, 522.9 93.7, 518.6 95.5" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 91 C526.7 87.7, 525.6 89.1, 520.4 85.4 M530.1 89.6 C527.1 89.1, 522.9 87.1, 518.6 85" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M540.9 69.6 C576.1 64.2, 616 68.1, 732.1 69 M540.1 67.3 C601.4 67.8, 660.6 67.2, 728.9 68.4 M727.9 68.9 C729.3 84.5, 731.2 100.6, 729.8 110.5 M730.9 67.6 C730.2 81.9, 730.2 97.1, 729.4 111.1 M731.4 110.7 C678.3 112.9, 625.5 113.6, 538.9 110.5 M729.4 111.8 C661 113.4, 591.8 113.8, 540.4 111.8 M541.1 111.9 C538.5 95.5, 540.1 82.4, 541.6 68.2 M539.3 113 C540.5 101.2, 540.2 89.9, 541 68.5" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="635" y="97" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--a1)">Fourier (FFT)</text><text x="745" y="96" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">stationary signals, filters, convolution</text><text x="40" y="168" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">which frequencies, when, at one resolution?</text><path d="M470.9 163.6 C480.1 159.3, 493.9 163.2, 532.1 163 M470.1 161.3 C490.1 162.7, 507.9 162.1, 528.9 162.4" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 163 C526.7 161.8, 525.6 165.2, 520.4 167.8 M530.1 161.6 C527.2 164.4, 522.9 165.7, 518.6 167.5" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 163 C526.7 159.7, 525.6 161.1, 520.4 157.4 M530.1 161.6 C527.1 161.1, 522.9 159.1, 518.6 157" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M540.9 141.6 C576.1 136.2, 616 140.1, 732.1 141 M540.1 139.3 C601.4 139.8, 660.6 139.2, 728.9 140.4 M727.9 140.9 C729.3 156.5, 731.2 172.6, 729.8 182.5 M730.9 139.6 C730.2 153.9, 730.2 169.1, 729.4 183.1 M731.4 182.7 C678.3 184.9, 625.5 185.6, 538.9 182.5 M729.4 183.8 C661 185.4, 591.8 185.8, 540.4 183.8 M541.1 183.9 C538.5 167.5, 540.1 154.4, 541.6 140.2 M539.3 185 C540.5 173.2, 540.2 161.9, 541 140.5" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="635" y="169" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--a1)">STFT / spectrogram</text><text x="745" y="168" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">speech, audio, steady textures</text><text x="40" y="240" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">which scales, where, in fine detail?</text><path d="M470.9 235.6 C480.1 231.3, 493.9 235.2, 532.1 235 M470.1 233.3 C490.1 234.7, 507.9 234.1, 528.9 234.4" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 235 C526.7 233.8, 525.6 237.2, 520.4 239.8 M530.1 233.6 C527.2 236.4, 522.9 237.7, 518.6 239.5" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 235 C526.7 231.7, 525.6 233.1, 520.4 229.4 M530.1 233.6 C527.1 233.1, 522.9 231.1, 518.6 229" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M540.9 213.6 C576.1 208.2, 616 212.1, 732.1 213 M540.1 211.3 C601.4 211.8, 660.6 211.2, 728.9 212.4 M727.9 212.9 C729.3 228.5, 731.2 244.6, 729.8 254.5 M730.9 211.6 C730.2 225.9, 730.2 241.1, 729.4 255.1 M731.4 254.7 C678.3 256.9, 625.5 257.6, 538.9 254.5 M729.4 255.8 C661 257.4, 591.8 257.8, 540.4 255.8 M541.1 255.9 C538.5 239.5, 540.1 226.4, 541.6 212.2 M539.3 257 C540.5 245.2, 540.2 233.9, 541 212.5" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="635" y="241" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--a1)">continuous wavelets</text><text x="745" y="240" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">transients, chirps, analysis by eye</text><text x="40" y="312" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">a compact, invertible, fast representation?</text><path d="M470.9 307.6 C480.1 303.3, 493.9 307.2, 532.1 307 M470.1 305.3 C490.1 306.7, 507.9 306.1, 528.9 306.4" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 307 C526.7 305.8, 525.6 309.2, 520.4 311.8 M530.1 305.6 C527.2 308.4, 522.9 309.7, 518.6 311.5" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 307 C526.7 303.7, 525.6 305.1, 520.4 301.4 M530.1 305.6 C527.1 305.1, 522.9 303.1, 518.6 301" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M540.9 285.6 C576.1 280.2, 616 284.1, 732.1 285 M540.1 283.3 C601.4 283.8, 660.6 283.2, 728.9 284.4 M727.9 284.9 C729.3 300.5, 731.2 316.6, 729.8 326.5 M730.9 283.6 C730.2 297.9, 730.2 313.1, 729.4 327.1 M731.4 326.7 C678.3 328.9, 625.5 329.6, 538.9 326.5 M729.4 327.8 C661 329.4, 591.8 329.8, 540.4 327.8 M541.1 327.9 C538.5 311.5, 540.1 298.4, 541.6 284.2 M539.3 329 C540.5 317.2, 540.2 305.9, 541 284.5" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="635" y="313" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--a1)">discrete wavelets</text><text x="745" y="312" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">compression, denoising, PDE solvers</text><text x="40" y="384" text-anchor="start" style="font-family:Kalam,cursive;font-size:17px;font-weight:400;fill:var(--ink)">a shift-stable, oriented representation of images?</text><path d="M470.9 379.6 C480.1 375.3, 493.9 379.2, 532.1 379 M470.1 377.3 C490.1 378.7, 507.9 378.1, 528.9 378.4" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 379 C526.7 377.8, 525.6 381.2, 520.4 383.8 M530.1 377.6 C527.2 380.4, 522.9 381.7, 518.6 383.5" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M530.6 379 C526.7 375.7, 525.6 377.1, 520.4 373.4 M530.1 377.6 C527.1 377.1, 522.9 375.1, 518.6 373" style="stroke:var(--muted);stroke-width:1.8;stroke-opacity:1;fill:none;stroke-linecap:round"/><path d="M540.9 357.6 C576.1 352.2, 616 356.1, 732.1 357 M540.1 355.3 C601.4 355.8, 660.6 355.2, 728.9 356.4 M727.9 356.9 C729.3 372.5, 731.2 388.6, 729.8 398.5 M730.9 355.6 C730.2 369.9, 730.2 385.1, 729.4 399.1 M731.4 398.7 C678.3 400.9, 625.5 401.6, 538.9 398.5 M729.4 399.8 C661 401.4, 591.8 401.8, 540.4 399.8 M541.1 399.9 C538.5 383.5, 540.1 370.4, 541.6 356.2 M539.3 401 C540.5 389.2, 540.2 377.9, 541 356.5" style="stroke:var(--a1);stroke-width:1.8;stroke-opacity:0.85;fill:none;stroke-linecap:round"/><text x="635" y="385" text-anchor="middle" style="font-family:Kalam,cursive;font-size:17px;font-weight:700;fill:var(--a1)">pyramids, complex wavelets</text><text x="745" y="384" text-anchor="start" style="font-family:Kalam,cursive;font-size:14px;font-weight:400;fill:var(--muted)">texture, motion, features</text></svg>
</figure>

## 17. Implementation and caveats

```python
import numpy as np, pywt

x = np.array([9, 7, 3, 5, 6, 10, 2, 6], float)
cA, cD = pywt.dwt(x, 'haar')                 # one level, orthonormal: (a + b)/sqrt2, (a - b)/sqrt2
print(cA / np.sqrt(2), cD / np.sqrt(2))      # [8. 4. 8. 4.] [ 1. -1. -2. -2.]  -- Section 4
coeffs = pywt.wavedec(x, 'haar', level=3)    # [cA3, cD3, cD2, cD1]: coarsest first (Section 7)
print(pywt.waverec(coeffs, 'haar'))          # [ 9.  7.  3.  5.  6. 10.  2.  6.]
print(pywt.dwt_coeff_len(256, pywt.Wavelet('db4').dec_len, 'symmetric'))   # 131, not 128
```

Run as written; the outputs are in the comments. The traps:

1. **Levels count in opposite directions.** A library's level $\ell$ is this post's $j = J - \ell$, and Mallat's $j$ is this post's $-j$; `cD1` is the *finest* detail (Section 7).
2. **Two correct implementations can disagree by a shift.** PyWavelets centres each filter window; a textbook formula starts it at $2k$. For D4 the coefficients differ by one position. Compare reconstructions or magnitudes, never raw arrays across libraries (Section 7).
3. **Samples are not scaling coefficients.** Feeding samples in place of $\langle f, \varphi_{J,k}\rangle$ shifts everything by the centroid of $\varphi$ (0.634 samples for D4). Reconstruction is unaffected; interpretation is not (Section 7).
4. **Coefficient counts.** The default boundary mode is symmetric extension, which returns more coefficients than samples. Use `mode='periodization'` for exactly $N/2$ per level (and accept the wrap-around jump).
5. **Which filter is which.** `rec_lo` is the $h$ of this post, `dec_lo` is $h$ reversed, `rec_hi` is $g$ and `dec_hi` is $g$ reversed. Books and libraries differ in the sign and shift of $g$; magnitudes do not (Section 7).
6. **Order of the output.** `wavedec` returns the coarsest approximation first, then details from coarse to fine.
7. **How many levels.** Each level needs the signal to be longer than the filter; `pywt.dwt_max_level(256, 'db4')` is 5 while Haar allows 8. Past that, the coefficients are dominated by boundary effects.
8. **Thresholding the approximation by mistake.** Threshold the details only; the coarsest approximation carries the signal's mean and slow trends.
9. **Continuous-transform scales are not frequencies.** Convert with the wavelet's centre frequency (`pywt.scale2frequency`), and check whether the library normalises for energy ($a^{-1/2}$, as here) or for amplitude ($a^{-1}$), which changes how the scalogram's rows compare.
10. **Edges of a scalogram.** Near the ends of the record, long wavelets hang off the data; that region (the "cone of influence") should not be interpreted.
11. **Shift sensitivity.** If features must not depend on alignment, use the stationary transform, cycle spinning or complex wavelets.
12. **Odd lengths.** With periodisation, a level halves the length only if it is even; PyWavelets rounds up (9, 10 and 11 samples give 5, 5 and 6 coefficients per band). For several levels, pad to a multiple of $2^{\text{levels}}$.
13. **Approximations grow.** Each orthonormal level multiplies a constant by $\sqrt2$ in 1-D (a constant 1 becomes $2^{3/2} = 2.83$ after three Haar levels) and by $2$ in 2-D. Divide by that gain before displaying an approximation as an image.

## 18. Where it is used, and what is open

| Use wavelets when | Prefer something else when |
|---|---|
| the signal is piecewise smooth: edges, transients, spikes | it is stationary and periodic: Fourier is simpler and exact |
| you need a compact, invertible, fast representation | you need shift-invariant features: use undecimated or complex transforms |
| events of very different durations matter at once | you need fine direction in images: steerable or complex wavelets (Part 4) |

**In current work.** Image compression (JPEG 2000, used in digital cinema and medical imaging); denoising in imaging, spectroscopy and astronomy; transient detection in ECG, EEG and seismic records; Morlet scalograms as the standard time–frequency tool in neuroscience; multiscale solvers and adaptive grids for PDEs; the **scattering transform** (Bruna and Mallat, 2013), a fixed cascade of wavelet transforms and moduli whose features are stable to small deformations and which explains part of why convolutional networks work; **wavelet neural operators** for learning PDE solution maps (Tripura and Chakraborty, 2023); and [random wavelet features](/aboutme/blog/fourier-to-wavelets/) for scalable non-stationary Gaussian processes.

**Open questions.**
1. **Learned filter banks with guarantees.** Learning the filters of a wavelet transform from data is easy; keeping perfect reconstruction, stability and vanishing moments while doing so is not. Lifting gives a route. Which guarantees matter for downstream learning?
2. **Wavelets off the line.** On graphs, meshes and point clouds there is no dilation, and graph wavelets are defined through the spectrum of a Laplacian. Which properties of the Euclidean theory (fast transforms, vanishing moments, sparsity theorems) survive, and at what cost?
3. **Adaptive bases.** Wavelet packets and best-basis search pick a tiling of the time–frequency plane per signal. When does that adaptivity pay off inside a learning pipeline, compared with a fixed transform followed by a learned model?
4. **Resolution independence in neural operators.** Wavelet-based operators promise to work across discretisations. Under which conditions on the data and the architecture can this be proved, rather than observed?

:::note[Shoulders to stand on]
- **Artem Kirsanov, [*Wavelets: a mathematical microscope*](https://www.youtube.com/watch?v=jnxqHcObNK4)** (YouTube, 2022). The best visual introduction to the continuous wavelet transform.
- **Eric Stollnitz, Tony DeRose, David Salesin, [*Wavelets for Computer Graphics: A Primer*](https://grail.cs.washington.edu/projects/wavelets/article/)** (IEEE CG&A, 1995). Haar by hand in 1-D and 2-D; our eight numbers start from their example.
- **Gilbert Strang, Truong Nguyen, *Wavelets and Filter Banks*** (Wellesley-Cambridge, 1996). The filter-bank view that makes implementation obvious.
- **Stéphane Mallat, *A Wavelet Tour of Signal Processing*** (3rd ed., Academic Press, 2008). The definitive reference, for the full theory.
:::

## Self-check

:::question
A wavelet has 3 vanishing moments. On a smooth stretch, by what factor do its detail coefficients shrink from one level to the next finer one?
:::

:::answer
$2^{p + 1/2} = 2^{3.5} \approx 11.3$ (Section 10).
:::

:::question
Show that the Haar filter satisfies $\lvert H(\omega)\rvert^2 + \lvert H(\omega + \pi)\rvert^2 = 2$.
:::

:::answer
$H(\omega) = \frac{1}{\sqrt2}(1 + e^{-i\omega})$, so $\lvert H(\omega)\rvert^2 = \tfrac12\lvert1 + e^{-i\omega}\rvert^2 = 1 + \cos\omega$. Then $\lvert H(\omega + \pi)\rvert^2 = 1 - \cos\omega$, and the sum is $2$.
:::

:::question
How many vanishing moments does Haar have, and why not two?
:::

:::answer
One: $\int\psi = \tfrac12 - \tfrac12 = 0$, but $\int x\psi(x)dx = \int_0^{1/2}x\,dx - \int_{1/2}^1x\,dx = \tfrac18 - \tfrac38 = -\tfrac14 \ne 0$. In filter terms, $\sum(-1)^kkh_k = -h_1 = -\tfrac1{\sqrt2} \ne 0$.
:::

:::question
A signal of $N = 4096$ samples has noise $\sigma = 0.1$. What is the universal threshold?
:::

:::answer
$\sigma\sqrt{2\ln N} = 0.1\sqrt{2\ln 4096} \approx 0.1 \times 4.08 = 0.408$.
:::

:::question
Why does the fast wavelet transform cost $O(N)$ while the FFT costs $O(N\log N)$?
:::

:::answer
Each level of the wavelet transform works only on the approximation of the previous level, which halves every time: $N + N/2 + N/4 + \cdots < 2N$ samples in total, each touched by a fixed number of filter taps. The FFT does $N$ work at every one of its $\log_2N$ stages (Part 2, Section 8).
:::

:::question
Why can the lifting steps of the 5/3 transform round to integers and still be inverted exactly?
:::

:::answer
Each step adds to one half of the samples a function of the other half only. Undoing it subtracts the same function of the same, unchanged half, so whatever the function computed, including any rounding, is subtracted exactly.
:::

## References

- A. Haar. *Zur Theorie der orthogonalen Funktionensysteme.* Mathematische Annalen 69, 1910.
- A. Grossmann, J. Morlet. *Decomposition of Hardy functions into square integrable wavelets of constant shape.* SIAM Journal on Mathematical Analysis 15(4), 1984.
- S. Mallat. *A theory for multiresolution signal decomposition: the wavelet representation.* IEEE TPAMI 11(7), 1989.
- I. Daubechies. *Orthonormal bases of compactly supported wavelets.* Communications on Pure and Applied Mathematics 41(7), 1988; and *Ten Lectures on Wavelets*, SIAM, 1992.
- A. Cohen, I. Daubechies, J.-C. Feauveau. *Biorthogonal bases of compactly supported wavelets.* CPAM 45(5), 1992.
- W. Sweldens. *The lifting scheme: a custom-design construction of biorthogonal wavelets.* Applied and Computational Harmonic Analysis 3(2), 1996.
- I. Daubechies, W. Sweldens. *Factoring wavelet transforms into lifting steps.* Journal of Fourier Analysis and Applications 4(3), 1998.
- D. L. Donoho, I. M. Johnstone. *Ideal spatial adaptation by wavelet shrinkage.* Biometrika 81(3), 1994.
- D. L. Donoho, I. M. Johnstone. *Adapting to unknown smoothness via wavelet shrinkage.* JASA 90(432), 1995.
- R. R. Coifman, D. L. Donoho. *Translation-invariant de-noising.* In *Wavelets and Statistics*, Springer, 1995.
- I. W. Selesnick, R. G. Baraniuk, N. G. Kingsbury. *The dual-tree complex wavelet transform.* IEEE Signal Processing Magazine 22(6), 2005.
- J. Bruna, S. Mallat. *Invariant scattering convolution networks.* IEEE TPAMI 35(8), 2013.
- T. Tripura, S. Chakraborty. *Wavelet neural operator for solving parametric partial differential equations in computational mechanics problems.* CMAME 404, 2023.
- G. Lee, R. Gommers, F. Wasilewski, K. Wohlfahrt, A. O'Leary. *PyWavelets: a Python package for wavelet analysis.* JOSS 4(36), 2019.
- E. J. Stollnitz, T. D. DeRose, D. H. Salesin. *Wavelets for computer graphics: a primer, part 1.* IEEE Computer Graphics and Applications 15(3), 1995.
- G. Strang, T. Nguyen. *Wavelets and Filter Banks.* Wellesley-Cambridge Press, 1996.
- S. Mallat. *A Wavelet Tour of Signal Processing: The Sparse Way.* 3rd ed., Academic Press, 2008.
