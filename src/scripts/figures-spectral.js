// "Seeing in frequencies" series (Fourier and wavelets). Shared numerics and the interactive figures.
// Conventions follow plans/blog/spectral/PLAN.md: DFT X[k] = sum_n x[n] e^{-2 pi i k n / N} (NumPy), inverse with 1/N.
// Test images are Sawan's own sketches, made by plans/blog/spectral/assets.py.
import { theme, onTheme, fit } from './figures.js';

const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------------------------------------------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------------------------------------------

// In-place radix-2 FFT on Float64Arrays (length a power of two). inverse = true divides by n.
export function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {                       // bit-reversal permutation
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {                   // butterflies
    const half = len >> 1, ang = ((inverse ? 2 : -2) * Math.PI) / len;
    for (let k = 0; k < half; k++) {
      const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
      for (let i = k; i < n; i += len) {
        const b = i + half, tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[i] - tr; im[b] = im[i] - ti; re[i] += tr; im[i] += ti;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// 2-D FFT of an n x n array stored row-major: rows, then columns.
export function fft2(re, im, n, inverse = false) {
  const r = new Float64Array(n), c = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    const o = y * n;
    for (let x = 0; x < n; x++) { r[x] = re[o + x]; c[x] = im[o + x]; }
    fft(r, c, inverse);
    for (let x = 0; x < n; x++) { re[o + x] = r[x]; im[o + x] = c[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; c[y] = im[y * n + x]; }
    fft(r, c, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = c[y]; }
  }
}

// One level of the Haar transform with averages and half-differences: (a, b) -> ((a + b) / 2, (a - b) / 2).
// (The orthonormal version divides by sqrt 2 instead of 2; the post explains the difference.)
export function haarStep(x) {
  const h = x.length >> 1, a = new Array(h), d = new Array(h);
  for (let i = 0; i < h; i++) { a[i] = (x[2 * i] + x[2 * i + 1]) / 2; d[i] = (x[2 * i] - x[2 * i + 1]) / 2; }
  return [a, d];
}
export const haarInverseStep = (a, d) => a.flatMap((ai, i) => [ai + d[i], ai - d[i]]);

// Greyscale image (values in [0, 1]) from a URL, as a Float64Array of size n * n.
const imgCache = new Map();
export function loadGrey(url, n = 256) {
  if (!imgCache.has(url)) {
    imgCache.set(url, new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => {
        const cv = document.createElement('canvas'); cv.width = n; cv.height = n;
        const cx = cv.getContext('2d'); cx.drawImage(im, 0, 0, n, n);
        const px = cx.getImageData(0, 0, n, n).data, out = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) out[i] = px[4 * i] / 255;
        resolve(out);
      };
      im.onerror = reject;
      im.src = url;
    }));
  }
  return imgCache.get(url);
}

// ---------------------------------------------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------------------------------------------

// Instrument colour map for spectra: night -> deep teal -> teal -> amber -> cream (same in light and dark mode).
const STOPS = [[0, [14, 13, 20]], [0.38, [19, 78, 74]], [0.66, [95, 191, 182]], [0.86, [251, 191, 36]], [1, [255, 247, 224]]];
function cmap(v) {
  v = Math.min(1, Math.max(0, v));
  let i = 1; while (i < STOPS.length - 1 && STOPS[i][0] < v) i++;
  const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i], u = (v - t0) / (t1 - t0);
  return [c0[0] + u * (c1[0] - c0[0]), c0[1] + u * (c1[1] - c0[1]), c0[2] + u * (c1[2] - c0[2])];
}

function offscreen(n) { const c = document.createElement('canvas'); c.width = n; c.height = n; return c; }
function putPixels(cv, n, f) {
  const cx = cv.getContext('2d'), img = cx.createImageData(n, n), d = img.data;
  for (let i = 0; i < n * n; i++) { const [r, g, b] = f(i); d[4 * i] = r; d[4 * i + 1] = g; d[4 * i + 2] = b; d[4 * i + 3] = 255; }
  cx.putImageData(img, 0, 0);
}
function label(ctx, c, text, x, y, dpr, { size = 12, alpha = 0.75, align = 'left', mono = true, color } = {}) {
  ctx.fillStyle = color ?? `rgba(${c.ink}, ${alpha})`;
  ctx.font = `${size * dpr}px ${mono ? 'ui-monospace, SFMono-Regular, Menlo, monospace' : 'system-ui, sans-serif'}`;
  ctx.textAlign = align; ctx.fillText(text, x, y); ctx.textAlign = 'left';
}
function observeVisible(el, cb) {
  const io = new IntersectionObserver((es) => es.forEach((e) => cb(e.isIntersecting)), { rootMargin: '80px' });
  io.observe(el);
}

// ---------------------------------------------------------------------------------------------------------------
// F1 · Epicycles: a closed curve (the outline of Sawan's tree sketch) as a sum of rotating circles.
// Data: tree_epicycles.json, coefficients c_k sorted by |c_k|, z(t) = sum_k c_k e^{2 pi i k t}.
// ---------------------------------------------------------------------------------------------------------------

const K_STEPS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300];

export function mountEpicycles(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const kIn = root?.querySelector('input[data-k]'), kOut = root?.querySelector('[data-kval]');
  const playBtn = root?.querySelector('[data-play]');
  let D = null, K = 20, path = [], t = 0, playing = !reduceMotion(), visible = false, last = 0, dpr = 1, raf = 0;
  const SAMPLES = 720, PERIOD = 12;   // seconds per lap

  const size = () => { const w = canvas.getBoundingClientRect().width; canvas.style.height = `${Math.min(400, Math.max(260, w * 0.56))}px`; dpr = fit(canvas); };
  const partial = () => {
    path = new Array(SAMPLES);
    for (let s = 0; s < SAMPLES; s++) {
      const tt = s / SAMPLES; let x = 0, y = 0;
      for (let j = 0; j < K; j++) { const a = 2 * Math.PI * D.k[j] * tt, c = Math.cos(a), si = Math.sin(a); x += D.re[j] * c - D.im[j] * si; y += D.re[j] * si + D.im[j] * c; }
      path[s] = [x, y];
    }
  };
  const draw = () => {
    if (!D) return;
    const c = theme(), W = canvas.width, H = canvas.height;
    // fit the outline, and the largest circle around the origin, into the canvas
    const r1 = Math.hypot(D.re[0], D.im[0]);
    let x0 = -r1, x1 = r1, y0 = -r1, y1 = r1;
    D.outline.forEach(([x, y]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
    const S = Math.min((0.9 * W) / (x1 - x0), (0.84 * H - 30 * dpr) / (y1 - y0));
    const cx = W / 2 - (S * (x0 + x1)) / 2, cy = H / 2 + 14 * dpr + (S * (y0 + y1)) / 2;
    const T = (x, y) => [cx + S * x, cy - S * y];
    ctx.clearRect(0, 0, W, H);
    // target outline, faint
    ctx.strokeStyle = `rgba(${c.ink}, 0.13)`; ctx.lineWidth = 1.5 * dpr; ctx.beginPath();
    D.outline.forEach(([x, y], i) => { const [X, Y] = T(x, y); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.closePath(); ctx.stroke();
    // trace so far
    const upto = Math.floor(t * SAMPLES);
    ctx.strokeStyle = `rgb(${c.a1})`; ctx.lineWidth = 2.4 * dpr; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let s = 0; s <= upto && s < SAMPLES; s++) { const [X, Y] = T(...path[s]); s ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke();
    // the chain of circles at time t, largest first
    let x = 0, y = 0;
    for (let j = 0; j < K; j++) {
      const r = Math.hypot(D.re[j], D.im[j]), a = 2 * Math.PI * D.k[j] * t;
      const nx = x + D.re[j] * Math.cos(a) - D.im[j] * Math.sin(a), ny = y + D.re[j] * Math.sin(a) + D.im[j] * Math.cos(a);
      const [X0, Y0] = T(x, y), [X1, Y1] = T(nx, ny);
      if (r * S > 0.6 * dpr) {
        ctx.strokeStyle = `rgba(${c.ink}, ${j < 12 ? 0.28 : 0.14})`; ctx.lineWidth = dpr;
        ctx.beginPath(); ctx.arc(X0, Y0, r * S, 0, 2 * Math.PI); ctx.stroke();
      }
      ctx.strokeStyle = `rgba(${c.dot}, ${j < 12 ? 0.95 : 0.6})`; ctx.lineWidth = (j < 3 ? 2 : 1.3) * dpr;
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke();
      x = nx; y = ny;
    }
    const [PX, PY] = T(x, y);
    ctx.fillStyle = `rgb(${c.dot})`; ctx.beginPath(); ctx.arc(PX, PY, 4 * dpr, 0, 2 * Math.PI); ctx.fill();
    const err = D.relErr[K - 1] * 100;
    label(ctx, c, `${K} circle${K > 1 ? 's' : ''} · error ${err < 1 ? err.toFixed(2) : err.toFixed(1)}%`, 14 * dpr, 24 * dpr, dpr, { size: 13, alpha: 0.85 });
    label(ctx, c, 'largest circles first', 14 * dpr, 42 * dpr, dpr, { size: 11, alpha: 0.5 });
  };
  const loop = (now) => {
    raf = 0;
    if (!playing || !visible) return;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0; last = now;
    t = (t + dt / PERIOD) % 1;
    draw();
    raf = requestAnimationFrame(loop);
  };
  const kick = () => { if (playing && visible && !raf) { last = 0; raf = requestAnimationFrame(loop); } };
  const setK = () => { K = K_STEPS[Number(kIn?.value ?? 6)]; if (kOut) kOut.textContent = String(K); if (D) { K = Math.min(K, D.k.length); partial(); } draw(); };
  const setPlay = (p) => { playing = p; if (playBtn) playBtn.textContent = p ? 'Pause' : 'Play'; if (!p && reduceMotion()) t = 0.999; draw(); kick(); };

  size();
  fetch(canvas.dataset.src).then((r) => r.json()).then((j) => { D = j; if (reduceMotion()) t = 0.999; setK(); setPlay(playing); });
  kIn?.addEventListener('input', setK);
  playBtn?.addEventListener('click', () => setPlay(!playing));
  observeVisible(canvas, (v) => { visible = v; kick(); });
  new ResizeObserver(() => { size(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F5 · The 2-D spectrum explorer. One of the sketches, its centred log-magnitude spectrum, and the image rebuilt
// from only the frequencies you keep. Modes: low-pass, high-pass, band, one wave (click a frequency), erase (paint).
// ---------------------------------------------------------------------------------------------------------------

const N2 = 256;

export function mountSpectrum(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const q = (s) => root?.querySelector(s);
  const rIn = q('input[data-r]'), rOut = q('[data-rval]'), softIn = q('input[data-soft]'), insight = q('[data-insight]');
  const imgBtns = [...(root?.querySelectorAll('[data-img]') ?? [])], modeBtns = [...(root?.querySelectorAll('[data-mode]') ?? [])];
  const resetBtn = q('[data-reset]');
  const base = canvas.dataset.base;
  let dpr = 1, wide = true, P = null;                       // panel geometry
  let img = null, Fr = null, Fi = null, logMag = null, energyAC = 0;
  let mode = 'low', R = 24, probe = null, erased = new Uint8Array(N2 * N2), name = canvas.dataset.img || 'eye';
  const cvImg = offscreen(N2), cvSpec = offscreen(N2), cvOut = offscreen(N2);
  const radius = new Float64Array(N2 * N2), fu = new Int16Array(N2 * N2), fv = new Int16Array(N2 * N2);
  for (let y = 0; y < N2; y++) for (let x = 0; x < N2; x++) {
    const u = x < N2 / 2 ? x : x - N2, v = y < N2 / 2 ? y : y - N2, i = y * N2 + x;
    fu[i] = u; fv[i] = v; radius[i] = Math.hypot(u, v);
  }
  const idx = (u, v) => (((v % N2) + N2) % N2) * N2 + (((u % N2) + N2) % N2);

  const layout = () => {
    const w = canvas.getBoundingClientRect().width;
    wide = w >= 600;
    const pad = 10, title = 22;
    const side = wide ? (w - 4 * pad) / 3 : (w - 3 * pad) / 2;
    canvas.style.height = `${wide ? side + title + 2 * pad : 2 * (side + title) + 3 * pad}px`;
    dpr = fit(canvas);
    const s = side * dpr, p = pad * dpr, tt = title * dpr;
    const at = (col, row) => ({ x: p + col * (s + p), y: p + tt + row * (s + tt + p), s });
    P = wide ? { img: at(0, 0), spec: at(1, 0), out: at(2, 0), info: null } : { img: at(0, 0), spec: at(1, 0), out: at(0, 1), info: at(1, 1) };
  };

  const mask = () => {
    const m = new Float64Array(N2 * N2), soft = softIn?.checked;
    for (let i = 0; i < N2 * N2; i++) {
      const r = radius[i];
      let k;
      if (mode === 'low' || mode === 'high') {
        k = soft ? Math.exp(-Math.LN2 * (r / Math.max(R, 0.5)) ** 2) : r <= R ? 1 : 0;
        if (mode === 'high') k = 1 - k;
      } else if (mode === 'band') k = Math.abs(r - R) <= 6 ? 1 : 0;
      else if (mode === 'wave') k = probe && ((fu[i] === probe[0] && fv[i] === probe[1]) || (fu[i] === -probe[0] && fv[i] === -probe[1])) ? 1 : 0;
      else k = erased[i] ? 0 : 1;
      m[i] = k;
    }
    return m;
  };

  const recompute = () => {
    if (!img) return;
    const m = mask(), re = new Float64Array(N2 * N2), im = new Float64Array(N2 * N2);
    let kept = 0, eKept = 0;
    for (let i = 0; i < N2 * N2; i++) {
      re[i] = Fr[i] * m[i]; im[i] = Fi[i] * m[i];
      if (m[i] > 0.5) kept++;
      if (i) eKept += (Fr[i] * Fr[i] + Fi[i] * Fi[i]) * m[i] * m[i];
    }
    fft2(re, im, N2, true);
    // display: keep the original grey scale when the mean (DC) survives, else stretch symmetrically around 0
    const dc = m[0] > 0.5;
    let lo = 0, hi = 1;
    if (!dc) { const a = Array.from(re, Math.abs).sort((p, q2) => p - q2); hi = a[Math.floor(0.995 * (a.length - 1))] || 1; lo = -hi; }
    putPixels(cvOut, N2, (i) => { const v = Math.min(1, Math.max(0, (re[i] - lo) / (hi - lo))); return [255 * v, 255 * v, 255 * v]; });
    // spectrum, centred (fftshift), dimmed where removed
    putPixels(cvSpec, N2, (j) => {
      const y = Math.floor(j / N2), x = j % N2, i = idx(x - N2 / 2, y - N2 / 2);
      const col = cmap(logMag[i]), k = 0.3 + 0.7 * m[i];
      return [col[0] * k, col[1] * k, col[2] * k];
    });
    state.kept = kept / (N2 * N2); state.energy = eKept / energyAC; state.dc = dc;
    draw(); explain();
  };
  const state = { kept: 1, energy: 1, dc: true };

  const draw = () => {
    if (!P) return;
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const panel = (cv, p, smooth, title) => {
      ctx.imageSmoothingEnabled = smooth; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(cv, p.x, p.y, p.s, p.s);
      ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(p.x, p.y, p.s, p.s);
      label(ctx, c, title, p.x, p.y - 7 * dpr, dpr, { size: 12, alpha: 0.7 });
    };
    if (!img) { label(ctx, c, 'loading…', 12 * dpr, 30 * dpr, dpr); return; }
    panel(cvImg, P.img, true, 'sketch');
    panel(cvSpec, P.spec, false, 'log |F|, 0 at centre');
    panel(cvOut, P.out, true, state.dc ? 'rebuilt' : 'rebuilt (stretched)');
    // overlays on the spectrum: crosshair at DC, mask boundary, probe
    const sp = P.spec, sc = sp.s / N2, X0 = sp.x + sp.s / 2, Y0 = sp.y + sp.s / 2;
    ctx.save(); ctx.beginPath(); ctx.rect(sp.x, sp.y, sp.s, sp.s); ctx.clip();
    ctx.strokeStyle = 'rgba(236, 235, 245, 0.25)'; ctx.lineWidth = dpr; ctx.setLineDash([3 * dpr, 4 * dpr]);
    ctx.beginPath(); ctx.moveTo(sp.x, Y0); ctx.lineTo(sp.x + sp.s, Y0); ctx.moveTo(X0, sp.y); ctx.lineTo(X0, sp.y + sp.s); ctx.stroke();
    ctx.setLineDash([]);
    if (mode === 'low' || mode === 'high' || mode === 'band') {
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)'; ctx.lineWidth = 1.5 * dpr;
      const rr = mode === 'band' ? [R - 6, R + 6] : [R];
      rr.forEach((r) => { if (r > 0) { ctx.beginPath(); ctx.arc(X0 + sc / 2, Y0 + sc / 2, r * sc, 0, 2 * Math.PI); ctx.stroke(); } });
    }
    if (mode === 'wave' && probe) {
      [[1, 1], [-1, -1]].forEach(([a], k) => {
        const X = X0 + (a * probe[0] + 0.5) * sc, Y = Y0 + (a * probe[1] + 0.5) * sc;
        ctx.strokeStyle = k ? 'rgba(251, 191, 36, 0.6)' : 'rgb(251, 191, 36)'; ctx.lineWidth = 2 * dpr;
        ctx.beginPath(); ctx.arc(X, Y, 7 * dpr, 0, 2 * Math.PI); ctx.stroke();
      });
    }
    ctx.restore();
    if (P.info) {
      const p = P.info;
      ctx.font = `${12 * dpr}px system-ui, sans-serif`;
      const words = `Kept ${(100 * state.kept).toFixed(1)}% of the coefficients, holding ${(100 * state.energy).toFixed(1)}% of the detail energy (the mean excluded).`.split(' ');
      let line = '', y = p.y + 16 * dpr;
      words.forEach((w) => {
        const tryLine = line ? `${line} ${w}` : w;
        if (ctx.measureText(tryLine).width > p.s && line) { label(ctx, c, line, p.x, y, dpr, { size: 12, mono: false }); line = w; y += 18 * dpr; }
        else line = tryLine;
      });
      label(ctx, c, line, p.x, y, dpr, { size: 12, mono: false });
    }
  };

  const explain = () => {
    if (!insight) return;
    const pct = (x) => `${(100 * x).toFixed(x < 0.1 ? 2 : 1)}%`;
    const r = R.toFixed(0);
    const lines = {
      low: `<strong>Low-pass.</strong> Keeping every frequency within ${r} cycles per image (${pct(state.kept)} of the coefficients) keeps ${pct(state.energy)} of the detail energy: the shapes survive, the fine texture goes. ${softIn?.checked ? 'With a smooth edge the result is simply blurred.' : 'The sharp cut-off makes faint ripples beside the edges: that is <em>ringing</em>, the 2-D cousin of the Gibbs phenomenon. Tick “smooth edge” and it disappears.'}`,
      high: `<strong>High-pass.</strong> Removing everything within ${r} cycles per image removes the mean and the slow shading. What is left is edges and texture: ${pct(state.energy)} of the detail energy in ${pct(state.kept)} of the coefficients.`,
      band: `<strong>Band-pass.</strong> Only frequencies near ${r} cycles per image: a wavelength of about ${(N2 / Math.max(R, 1)).toFixed(1)} pixels. Each ring of the spectrum is one scale of detail, in every direction at once.`,
      wave: probe ? waveText() : '<strong>One wave.</strong> Click anywhere in the spectrum.',
      erase: `<strong>Erase.</strong> Drag across the spectrum to delete frequencies. Each stroke also deletes its mirror image through the centre: the spectrum of a real image satisfies F(−u, −v) = conj F(u, v), so the two halves are one wave. ${pct(1 - state.kept)} erased so far.`,
    };
    insight.innerHTML = lines[mode];
  };
  const waveText = () => {
    const [u, v] = probe, f = Math.hypot(u, v), lam = N2 / f, i = idx(u, v);
    const amp = (2 * Math.hypot(Fr[i], Fi[i])) / (N2 * N2), ph = Math.atan2(Fi[i], Fr[i]);
    const ang = (Math.atan2(-v, u) * 180) / Math.PI;
    return `<strong>One wave.</strong> The point (u, v) = (${u}, ${-v}) together with its mirror (${-u}, ${v}) is a single grating: ${f.toFixed(1)} cycles across the image, so a wavelength of ${lam.toFixed(1)} pixels, varying along ${ang.toFixed(0)}° (the stripes run perpendicular to that). In this sketch it has amplitude ${amp.toFixed(4)} and phase ${(ph * 180 / Math.PI).toFixed(0)}°. A ${N2} × ${N2} image is a sum of ${(N2 * N2 - 4) / 2 + 4} such gratings, counting each mirror pair once.`;
  };

  const load = async (nm) => {
    name = nm;
    imgBtns.forEach((b) => b.classList.toggle('on', b.dataset.img === nm));
    img = await loadGrey(`${base}${nm}.png`, N2);
    Fr = Float64Array.from(img); Fi = new Float64Array(N2 * N2);
    fft2(Fr, Fi, N2);
    logMag = new Float64Array(N2 * N2);
    const vals = [];
    energyAC = 0;
    for (let i = 0; i < N2 * N2; i++) { const a = Math.log1p(Math.hypot(Fr[i], Fi[i])); logMag[i] = a; if (i) { vals.push(a); energyAC += Fr[i] * Fr[i] + Fi[i] * Fi[i]; } }
    vals.sort((a, b) => a - b);
    const lo = vals[Math.floor(0.15 * vals.length)], hi = vals[Math.floor(0.999 * (vals.length - 1))];
    for (let i = 0; i < N2 * N2; i++) logMag[i] = Math.pow(Math.min(1, Math.max(0, (logMag[i] - lo) / (hi - lo))), 0.85);
    putPixels(cvImg, N2, (i) => { const v = 255 * img[i]; return [v, v, v]; });
    // a good first probe: the strongest wave away from the centre
    let best = -1, bi = 0;
    for (let i = 1; i < N2 * N2; i++) { if (radius[i] < 3 || fv[i] > 0 || radius[i] > 60) continue; const a = Fr[i] * Fr[i] + Fi[i] * Fi[i]; if (a > best) { best = a; bi = i; } }
    probe = [fu[bi], fv[bi]];
    erased = new Uint8Array(N2 * N2);
    recompute();
  };

  const setMode = (m) => {
    mode = m;
    modeBtns.forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
    const radial = m === 'low' || m === 'high' || m === 'band';
    rIn?.closest('label')?.toggleAttribute('hidden', !radial);
    softIn?.closest('label')?.toggleAttribute('hidden', !(m === 'low' || m === 'high'));
    resetBtn?.toggleAttribute('hidden', m !== 'erase');
    canvas.style.cursor = m === 'wave' || m === 'erase' ? 'crosshair' : 'default';
    recompute();
  };

  // pointer on the spectrum panel -> frequency (u, v), v measured downwards like image rows
  const toFreq = (e) => {
    const b = canvas.getBoundingClientRect(), X = (e.clientX - b.left) * dpr, Y = (e.clientY - b.top) * dpr, sp = P.spec;
    if (X < sp.x || Y < sp.y || X > sp.x + sp.s || Y > sp.y + sp.s) return null;
    return [Math.floor(((X - sp.x) / sp.s) * N2) - N2 / 2, Math.floor(((Y - sp.y) / sp.s) * N2) - N2 / 2];
  };
  let painting = false, pending = false;
  const erase = (f) => {
    const [u0, v0] = f, rad = 5;
    for (let dv = -rad; dv <= rad; dv++) for (let du = -rad; du <= rad; du++) {
      if (du * du + dv * dv > rad * rad) continue;
      erased[idx(u0 + du, v0 + dv)] = 1; erased[idx(-(u0 + du), -(v0 + dv))] = 1;
    }
    if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; recompute(); }); }
  };
  canvas.addEventListener('pointerdown', (e) => {
    const f = toFreq(e); if (!f) return;
    if (mode === 'wave') { if (f[0] || f[1]) { probe = f; recompute(); } }
    else if (mode === 'erase') { painting = true; canvas.setPointerCapture(e.pointerId); erase(f); }
  });
  canvas.addEventListener('pointermove', (e) => { if (!painting) return; const f = toFreq(e); if (f) erase(f); });
  canvas.addEventListener('pointerup', () => { painting = false; });
  canvas.addEventListener('pointercancel', () => { painting = false; });

  rIn?.addEventListener('input', () => { R = Number(rIn.value); if (rOut) rOut.textContent = rIn.value; if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; recompute(); }); } });
  softIn?.addEventListener('change', recompute);
  resetBtn?.addEventListener('click', () => { erased = new Uint8Array(N2 * N2); recompute(); });
  imgBtns.forEach((b) => b.addEventListener('click', () => load(b.dataset.img)));
  modeBtns.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

  if (rIn) { R = Number(rIn.value); if (rOut) rOut.textContent = rIn.value; }
  layout(); draw();
  load(name).then(() => setMode(mode));
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F13 · Haar by hand. Eight numbers, three rounds of "average and half-difference", then throw away small details.
// Drag the bars in the top row to change the signal.
// ---------------------------------------------------------------------------------------------------------------

export function mountHaar(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const stepBtns = [...(root?.querySelectorAll('[data-step]') ?? [])];
  const tIn = root?.querySelector('input[data-t]'), tOut = root?.querySelector('[data-tval]'), insight = root?.querySelector('[data-insight]');
  let x = [9, 7, 3, 5, 6, 10, 2, 6], step = 3, thr = 0, dpr = 1, geo = null;
  const VMAX = 12, BASE = 0.8, UNITF = 0.5;     // baseline height and bar scale, as fractions of a row

  // rows[s] = coefficient vector after s steps, with group boundaries
  const transform = () => {
    const rows = [{ v: x.slice(), groups: [[0, 8, 'x']] }];
    let a = x.slice(), details = [];
    for (let s = 1; s <= 3; s++) {
      const [na, d] = haarStep(a);
      details = [d, ...details];
      a = na;
      const v = [...a, ...details.flat()];
      const groups = [[0, a.length, `a${s}`]]; let o = a.length;
      details.forEach((dd, k) => { groups.push([o, o + dd.length, `d${s - k}`]); o += dd.length; });
      rows.push({ v, groups });
    }
    return rows;
  };
  const reconstruct = (coef) => {
    // coef = [a3, d3, d2(2), d1(4)]
    let a = [coef[0]];
    a = haarInverseStep(a, [coef[1]]);
    a = haarInverseStep(a, coef.slice(2, 4));
    a = haarInverseStep(a, coef.slice(4, 8));
    return a;
  };

  const layout = () => {
    const w = canvas.getBoundingClientRect().width;
    canvas.style.height = `${Math.round(Math.min(420, 4 * 86 + 20))}px`;
    dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, left = 64 * dpr, right = 12 * dpr, top = 8 * dpr;
    const rowH = (H - top - 6 * dpr) / 4, slot = (W - left - right) / 8;
    geo = { left, top, rowH, slot, w };
  };
  const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2).replace(/0$/, ''));

  const draw = () => {
    if (!geo) return;
    const c = theme(), W = canvas.width, H = canvas.height, { left, top, rowH, slot } = geo;
    ctx.clearRect(0, 0, W, H);
    const rows = transform();
    const final = rows[3].v, kept = final.map((v, i) => (i === 0 || Math.abs(v) >= thr - 1e-9 ? v : 0));
    const rec = reconstruct(kept);
    const unit = (UNITF * rowH) / VMAX;
    for (let s = 0; s <= 3; s++) {
      const y0 = top + s * rowH, base = y0 + BASE * rowH, active = s <= step;
      const row = rows[s];
      label(ctx, c, s === 0 ? 'signal' : `step ${s}`, 8 * dpr, base - 0.3 * rowH, dpr, { size: 12, alpha: active ? 0.8 : 0.3 });
      ctx.strokeStyle = `rgba(${c.ink}, ${active ? 0.25 : 0.08})`; ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(left, base); ctx.lineTo(W - 12 * dpr, base); ctx.stroke();
      if (!active) continue;
      row.groups.forEach(([g0, g1, name], gi) => {
        const isA = name[0] === 'a' || name === 'x';
        for (let i = g0; i < g1; i++) {
          let v = row.v[i];
          const dropped = s === 3 && !isA && Math.abs(v) < thr - 1e-9;
          const X = left + i * slot + 0.16 * slot, bw = 0.68 * slot, h = v * unit;
          const col = name === 'x' ? c.ink : isA ? c.a1 : c.dot;
          ctx.fillStyle = `rgba(${col}, ${dropped ? 0.12 : name === 'x' ? 0.55 : 0.85})`;
          ctx.fillRect(X, h >= 0 ? base - h : base, bw, Math.max(Math.abs(h), dpr));
          label(ctx, c, fmt(v), X + bw / 2, h >= 0 ? base - h - 5 * dpr : base - h + 13 * dpr, dpr, { size: 11, alpha: dropped ? 0.3 : 0.85, align: 'center' });
        }
        // group label and separator
        if (name !== 'x') label(ctx, c, name, left + g0 * slot + 6 * dpr, y0 + 12 * dpr, dpr, { size: 11.5, color: `rgba(${isA ? c.a1 : c.dot}, 0.95)` });
        if (gi > 0) { ctx.strokeStyle = `rgba(${c.ink}, 0.2)`; ctx.setLineDash([3 * dpr, 3 * dpr]); ctx.beginPath(); ctx.moveTo(left + g0 * slot, y0 + 6 * dpr); ctx.lineTo(left + g0 * slot, base + 0.2 * rowH); ctx.stroke(); ctx.setLineDash([]); }
      });
      // pair brackets on the row that the next step reads
      if (s === step - 1 && step > 0) {
        const n = 8 >> s;
        ctx.strokeStyle = `rgba(${c.a2}, 0.7)`; ctx.lineWidth = 1.4 * dpr;
        for (let p = 0; p < n / 2; p++) {
          const X0 = left + 2 * p * slot + 0.1 * slot, X1 = left + (2 * p + 2) * slot - 0.1 * slot, Y = base + 0.1 * rowH;
          ctx.beginPath(); ctx.moveTo(X0, Y - 4 * dpr); ctx.lineTo(X0, Y); ctx.lineTo(X1, Y); ctx.lineTo(X1, Y - 4 * dpr); ctx.stroke();
        }
      }
    }
    // reconstruction from thresholded coefficients, drawn over the signal
    if (thr > 0 && step === 3) {
      const base = top + BASE * rowH;
      ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 2 * dpr;
      rec.forEach((v, i) => { const X = left + i * slot + 0.16 * slot, bw = 0.68 * slot; ctx.strokeRect(X, base - v * unit, bw, v * unit); });
    }
    explain(rows, kept, rec);
  };

  const explain = (rows, kept, rec) => {
    if (!insight) return;
    const [p, q2] = [x[0], x[1]];
    const nz = kept.filter((v) => Math.abs(v) > 1e-9).length, err = Math.max(...rec.map((v, i) => Math.abs(v - x[i])));
    const msg = [
      `<strong>Eight numbers.</strong> Drag the bars. Then press <em>step 1</em>.`,
      `<strong>Step 1.</strong> Take the numbers in pairs. Each pair (${fmt(p)}, ${fmt(q2)}) becomes its average ${fmt((p + q2) / 2)} (teal) and its half-difference ${fmt((p - q2) / 2)} (amber). Nothing is lost: average ± half-difference gives the pair back.`,
      `<strong>Step 2.</strong> Repeat on the four averages only. The details of step 1 are kept as they are. Coarser averages, coarser details.`,
      `<strong>Step 3.</strong> One number is left: the overall mean ${fmt(rows[3].v[0])}. The other seven say how the signal departs from it, at three scales. ${thr > 0 ? `With details below ${fmt(thr)} set to zero, ${nz} of 8 numbers remain and the rebuilt signal (amber outline) is off by at most ${fmt(err)}.` : 'Now raise the threshold: small details are the cheapest thing to throw away.'}`,
    ];
    insight.innerHTML = msg[step];
  };

  const setStep = (s) => { step = s; stepBtns.forEach((b) => b.classList.toggle('on', Number(b.dataset.step) === s)); tIn?.closest('label')?.toggleAttribute('hidden', s !== 3); draw(); };
  const dragTo = (e) => {
    const b = canvas.getBoundingClientRect(), X = (e.clientX - b.left) * dpr, Y = (e.clientY - b.top) * dpr, { left, top, rowH, slot } = geo;
    if (Y > top + rowH) return false;
    const i = Math.floor((X - left) / slot); if (i < 0 || i > 7) return false;
    const base = top + BASE * rowH, unit = (UNITF * rowH) / VMAX;
    x[i] = Math.max(0, Math.min(VMAX, Math.round(((base - Y) / unit) * 2) / 2));
    draw(); return true;
  };
  let dragging = false;
  canvas.addEventListener('pointerdown', (e) => { if (dragTo(e)) { dragging = true; canvas.setPointerCapture(e.pointerId); } });
  canvas.addEventListener('pointermove', (e) => { if (dragging) dragTo(e); });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  stepBtns.forEach((b) => b.addEventListener('click', () => setStep(Number(b.dataset.step))));
  tIn?.addEventListener('input', () => { thr = Number(tIn.value); if (tOut) tOut.textContent = tIn.value; draw(); });

  layout(); setStep(step);
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F19 · 2-D Haar by hand on an 8 x 8 patch. Rows first (average | half-difference of each pair), then columns,
// then the same again on the top-left quarter. Click a coefficient to see the block it came from and its arithmetic.
// Layout follows PyWavelets' coeffs_to_array: top-right = left - right (cV), bottom-left = top - bottom (cH).
// ---------------------------------------------------------------------------------------------------------------

const EYE8 = [
  [6, 2, 2, 3, 3, 2, 2, 1], [7, 3, 2, 3, 3, 1, 1, 1], [8, 4, 1, 2, 2, 2, 2, 2], [8, 6, 2, 2, 2, 2, 1, 2],
  [8, 7, 5, 2, 2, 2, 1, 1], [8, 7, 7, 7, 5, 4, 4, 4], [7, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 6, 6, 6, 6],
];   // a 32 x 32 patch of the Eye sketch (x = 88, y = 144), averaged to 8 x 8 and rounded to 0..9 (assets check)
const PRESETS2D = {
  vedge: () => grid((y, x) => (x < 3 ? 8 : 0)),
  ongrid: () => grid((y, x) => (x < 2 ? 8 : 0)),
  hedge: () => grid((y, x) => (y < 3 ? 8 : 0)),
  diag: () => grid((y, x) => (x > y ? 8 : 0)),
  checker: () => grid((y, x) => ((x + y) % 2 ? 0 : 8)),
  eye: () => EYE8.map((r) => r.slice()),
};
function grid(f) { return Array.from({ length: 8 }, (_, y) => Array.from({ length: 8 }, (_, x) => f(y, x))); }

// one pass of pairs along rows (dir = 'r') or columns (dir = 'c') inside the top-left n x n block
function pass2d(A, n, dir) {
  const B = A.map((r) => r.slice()), h = n / 2;
  for (let i = 0; i < n; i++) for (let k = 0; k < h; k++) {
    if (dir === 'r') { const p = A[i][2 * k], q = A[i][2 * k + 1]; B[i][k] = (p + q) / 2; B[i][h + k] = (p - q) / 2; }
    else { const p = A[2 * k][i], q = A[2 * k + 1][i]; B[k][i] = (p + q) / 2; B[h + k][i] = (p - q) / 2; }
  }
  return B;
}

export function mountHaar2d(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const stepBtns = [...(root?.querySelectorAll('[data-step2]') ?? [])], presetBtns = [...(root?.querySelectorAll('[data-preset]') ?? [])];
  const shiftBtn = root?.querySelector('[data-shift]'), insight = root?.querySelector('[data-insight]');
  let preset = canvas.dataset.preset in PRESETS2D ? canvas.dataset.preset : 'vedge', X = PRESETS2D[preset](), step = Number(canvas.dataset.step ?? 2), pick = null, dpr = 1, G = null;
  const NEG = '139, 92, 246';   // violet for negative details; amber (dot) for positive

  const stages = () => {
    const s = [X];
    s.push(pass2d(s[0], 8, 'r'));
    s.push(pass2d(s[1], 8, 'c'));
    s.push(pass2d(s[2], 4, 'r'));
    s.push(pass2d(s[3], 4, 'c'));
    return s;   // 0 image, 1 rows, 2 level 1, 3 level 2 rows, 4 level 2
  };
  const stageIndex = () => [0, 1, 2, 4][step];

  const layout = () => {
    const w = canvas.getBoundingClientRect().width, wide = w >= 600, lab = 20;
    const cell = Math.floor(wide ? (w - 70) / 16 : (w - 24) / 8);
    canvas.style.height = `${wide ? 8 * cell + 2 * lab + 26 : 16 * cell + 4 * lab + 52}px`;
    dpr = fit(canvas);
    const c = cell * dpr, L = lab * dpr, gap = 38 * dpr;
    const x0 = (canvas.width - (wide ? 16 * c + gap : 8 * c)) / 2;
    G = wide ? { c, inp: { x: x0, y: L + 6 * dpr }, out: { x: x0 + 8 * c + gap, y: L + 6 * dpr }, L, wide }
             : { c, inp: { x: x0, y: L + 6 * dpr }, out: { x: x0, y: 8 * c + 3 * L + 30 * dpr }, L, wide };
  };
  const fmt = (v) => { const r = Math.round(v * 100) / 100; return Math.abs(r) < 1e-9 ? '0' : String(r).replace('-', '−'); };

  // which region of the output a cell belongs to: 'avg' | 'lr' | 'tb' | 'diag', and its level
  const region = (y, x) => {
    if (step === 0) return 'img';
    if (step === 1) return x < 4 ? 'avg' : 'lr';
    if (step === 3 && y < 4 && x < 4) return y < 2 ? (x < 2 ? 'avg' : 'lr') : x < 2 ? 'tb' : 'diag';
    return y < 4 ? (x < 4 ? 'avg' : 'lr') : x < 4 ? 'tb' : 'diag';
  };
  const levelOf = (y, x) => (step === 3 && y < 4 && x < 4 ? 2 : 1);

  const draw = () => {
    if (!G) return;
    const c = theme(), W = canvas.width, H = canvas.height, S = stages(), O = S[stageIndex()], { c: cs } = G;
    ctx.clearRect(0, 0, W, H);
    const cellAt = (gx, gy, y, x, fill, txt, txtCol) => {
      ctx.fillStyle = fill; ctx.fillRect(gx + x * cs, gy + y * cs, cs, cs);
      ctx.fillStyle = txtCol; ctx.font = `${Math.min(12, cs / dpr / 3.3) * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(txt, gx + (x + 0.5) * cs, gy + (y + 0.5) * cs); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
    };
    const grey = (v) => { const g = Math.round(255 * (0.08 + 0.84 * Math.min(1, Math.max(0, v / 9)))); return [`rgb(${g}, ${g}, ${g})`, g > 140 ? '#111' : '#eee']; };
    // input
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const [f, t] = grey(X[y][x]); cellAt(G.inp.x, G.inp.y, y, x, f, fmt(X[y][x]), t); }
    label(ctx, c, 'the image (click to paint)', G.inp.x, G.inp.y - 8 * dpr, dpr, { size: 12 });
    // output
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const v = O[y][x], r = region(y, x);
      let fill, tc;
      if (r === 'img' || r === 'avg') [fill, tc] = grey(v);
      else { const a = Math.min(1, Math.abs(v) / 6); fill = Math.abs(v) < 1e-9 ? `rgba(${c.ink}, 0.04)` : `rgba(${v > 0 ? c.dot : NEG}, ${0.18 + 0.7 * a})`; tc = `rgba(${c.ink}, ${Math.abs(v) < 1e-9 ? 0.3 : 0.95})`; }
      cellAt(G.out.x, G.out.y, y, x, fill, fmt(v), tc);
    }
    // grid lines and band boundaries
    const lines = (g, bands) => {
      ctx.strokeStyle = `rgba(${c.ink}, 0.12)`; ctx.lineWidth = dpr;
      for (let i = 0; i <= 8; i++) { ctx.beginPath(); ctx.moveTo(g.x + i * cs, g.y); ctx.lineTo(g.x + i * cs, g.y + 8 * cs); ctx.moveTo(g.x, g.y + i * cs); ctx.lineTo(g.x + 8 * cs, g.y + i * cs); ctx.stroke(); }
      ctx.strokeStyle = `rgb(${c.a1})`; ctx.lineWidth = 2.2 * dpr;
      bands.forEach(([x1, y1, x2, y2]) => { ctx.beginPath(); ctx.moveTo(g.x + x1 * cs, g.y + y1 * cs); ctx.lineTo(g.x + x2 * cs, g.y + y2 * cs); ctx.stroke(); });
      ctx.strokeRect(g.x, g.y, 8 * cs, 8 * cs);
    };
    lines(G.inp, []);
    const bands = step === 0 ? [] : step === 1 ? [[4, 0, 4, 8]] : [[4, 0, 4, 8], [0, 4, 8, 4], ...(step === 3 ? [[2, 0, 2, 4], [0, 2, 4, 2]] : [])];
    lines(G.out, bands);
    const title = ['the same image', 'after the rows: average | half-difference', 'after the columns: one level', 'two levels'][step];
    label(ctx, c, title, G.out.x, G.out.y - 8 * dpr, dpr, { size: 12 });
    if (step >= 2) {
      const yb = G.out.y + 8 * cs + 15 * dpr;
      label(ctx, c, '↖ average      ↗ left − right', G.out.x, yb, dpr, { size: 11, alpha: 0.7 });
      label(ctx, c, '↙ top − bottom  ↘ checker', G.out.x, yb + 15 * dpr, dpr, { size: 11, alpha: 0.7 });
    }
    // the picked coefficient and its source block
    if (pick && step >= 2) {
      const [py, px] = pick, lev = levelOf(py, px), size = lev === 1 ? 2 : 4, half = lev === 1 ? 4 : 2;
      const by = (py % half) * size, bx = (px % half) * size;
      ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 3 * dpr;
      ctx.strokeRect(G.out.x + px * cs, G.out.y + py * cs, cs, cs);
      ctx.strokeRect(G.inp.x + bx * cs, G.inp.y + by * cs, size * cs, size * cs);
    }
    explain(O);
  };

  const explain = (O) => {
    if (!insight) return;
    if (pick && step >= 2) {
      const [py, px] = pick, lev = levelOf(py, px), r = region(py, px), half = lev === 1 ? 4 : 2;
      if (lev === 1) {
        const by = (py % half) * 2, bx = (px % half) * 2;
        const a = X[by][bx], b = X[by][bx + 1], cc = X[by + 1][bx], d = X[by + 1][bx + 1];
        const f = { avg: ['(a + b + c + d) / 4', a + b + cc + d, 'the average of the block'], lr: ['(a − b + c − d) / 4', a - b + cc - d, 'left column minus right column: it responds to <em>vertical</em> edges (PyWavelets calls it cV)'], tb: ['(a + b − c − d) / 4', a + b - cc - d, 'top row minus bottom row: it responds to <em>horizontal</em> edges (PyWavelets: cH)'], diag: ['(a − b − c + d) / 4', a - b - cc + d, 'one diagonal minus the other: a checkerboard detector (PyWavelets: cD)'] }[r];
        insight.innerHTML = `<strong>One 2 × 2 block, by hand.</strong> The highlighted block is a = ${fmt(a)}, b = ${fmt(b)} (top row), c = ${fmt(cc)}, d = ${fmt(d)} (bottom row). This coefficient is ${f[0]} = ${fmt(f[1])} / 4 = <strong>${fmt(f[1] / 4)}</strong>: ${f[2]}.`;
      } else {
        insight.innerHTML = `<strong>Level 2.</strong> This coefficient sees a 4 × 4 block: the same four formulas, applied to the four averages of its 2 × 2 sub-blocks. Coarser blocks catch coarser structure, including edges that fell between the level-1 blocks.`;
      }
      return;
    }
    const det = O.flat().filter((v, i) => region(Math.floor(i / 8), i % 8) !== 'avg' && region(Math.floor(i / 8), i % 8) !== 'img');
    const nz = det.filter((v) => Math.abs(v) > 1e-9).length;
    const msgs = [
      'An 8 × 8 patch, one number per pixel. Pick a preset or paint, then step through.',
      '<strong>Rows.</strong> Every row is treated like the eight bars above: four averages on the left, four half-differences on the right. Only vertical structure (change along a row) leaves anything on the right.',
      `<strong>Columns.</strong> The same on every column of that result. Each 2 × 2 block of the image has become four numbers, one in each quarter. ${nz} of the 48 detail numbers are non-zero. Click any coefficient to see its arithmetic.`,
      `<strong>Level 2.</strong> Repeat on the top-left quarter (the averages) only: a 4 × 4 picture at half the resolution. ${nz} detail numbers are non-zero in total.`,
    ];
    let extra = '';
    if (preset === 'ongrid' && step === 2 && nz === 0) extra = ' <strong>Notice:</strong> the edge lies exactly between two blocks, so every level-1 detail is zero. The transform cannot see it. Press <em>shift right</em>, or go to level 2.';
    if (preset === 'ongrid' && step === 3) extra = ' The edge that level 1 missed now shows up: level-2 blocks straddle it.';
    insight.innerHTML = msgs[step] + extra;
  };

  const setStep = (s) => { step = s; pick = null; stepBtns.forEach((b) => b.classList.toggle('on', Number(b.dataset.step2) === s)); draw(); };
  const setPreset = (p) => { preset = p; X = PRESETS2D[p](); pick = null; presetBtns.forEach((b) => b.classList.toggle('on', b.dataset.preset === p)); draw(); };
  // shift right by one pixel, repeating the left column (not a cyclic shift, which would add a second edge)
  shiftBtn?.addEventListener('click', () => { X = X.map((r) => [r[0], ...r.slice(0, 7)]); preset = 'shifted'; pick = null; presetBtns.forEach((b) => b.classList.remove('on')); draw(); });

  const hit = (e, g) => {
    const b = canvas.getBoundingClientRect(), px = (e.clientX - b.left) * dpr, py = (e.clientY - b.top) * dpr;
    const x = Math.floor((px - g.x) / G.c), y = Math.floor((py - g.y) / G.c);
    return x >= 0 && x < 8 && y >= 0 && y < 8 ? [y, x] : null;
  };
  let painting = null;
  canvas.addEventListener('pointerdown', (e) => {
    const o = hit(e, G.out); if (o && step >= 2) { pick = o; draw(); return; }
    const i = hit(e, G.inp); if (!i) return;
    painting = X[i[0]][i[1]] < 4.5 ? 8 : 0; X[i[0]][i[1]] = painting; preset = 'custom'; pick = null;
    presetBtns.forEach((b) => b.classList.remove('on')); canvas.setPointerCapture(e.pointerId); draw();
  });
  canvas.addEventListener('pointermove', (e) => { if (painting === null) return; const i = hit(e, G.inp); if (i && X[i[0]][i[1]] !== painting) { X[i[0]][i[1]] = painting; draw(); } });
  canvas.addEventListener('pointerup', () => { painting = null; });
  stepBtns.forEach((b) => b.addEventListener('click', () => setStep(Number(b.dataset.step2))));
  presetBtns.forEach((b) => b.addEventListener('click', () => setPreset(b.dataset.preset)));

  presetBtns.forEach((b) => b.classList.toggle('on', b.dataset.preset === preset));
  layout(); setStep(step);
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// Shared 1-D plotting
// ---------------------------------------------------------------------------------------------------------------

// A plot box: maps data (x in [x0, x1], y in [y0, y1]) into the rectangle r = {x, y, w, h} (device pixels).
function box1d(r, x0, x1, y0, y1) {
  return { ...r, X: (x) => r.x + ((x - x0) / (x1 - x0)) * r.w, Y: (y) => r.y + r.h - ((y - y0) / (y1 - y0)) * r.h };
}
function polyline(ctx, B, xs, ys, stroke, width, dpr, dash) {
  ctx.strokeStyle = stroke; ctx.lineWidth = width * dpr; ctx.lineJoin = 'round'; ctx.setLineDash(dash ? dash.map((d) => d * dpr) : []);
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) { const X = B.X(xs[i]), Y = B.Y(ys[i]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
  ctx.stroke(); ctx.setLineDash([]);
}
function axisLine(ctx, c, B, y, dpr) { ctx.strokeStyle = `rgba(${c.ink}, 0.25)`; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(B.x, B.Y(y)); ctx.lineTo(B.x + B.w, B.Y(y)); ctx.stroke(); }
function shadeSigned(ctx, c, B, xs, ys, dpr) {   // area between curve and zero: teal above, amber below
  for (const [sign, col] of [[1, c.a1], [-1, c.dot]]) {
    ctx.fillStyle = `rgba(${col}, 0.32)`; ctx.beginPath(); ctx.moveTo(B.X(xs[0]), B.Y(0));
    for (let i = 0; i < xs.length; i++) ctx.lineTo(B.X(xs[i]), B.Y(sign > 0 ? Math.max(0, ys[i]) : Math.min(0, ys[i])));
    ctx.lineTo(B.X(xs[xs.length - 1]), B.Y(0)); ctx.closePath(); ctx.fill();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// F2 · The probe. A Fourier coefficient is an inner product: multiply the signal by a probe wave and integrate.
// Signal f(x) = 0.9 cos(2 pi 3x - 0.7) + 0.5 cos(2 pi 7x + 1.2). The inset traces c(k) in the complex plane.
// ---------------------------------------------------------------------------------------------------------------

export function mountProbe(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const kIn = root?.querySelector('input[data-k]'), kOut = root?.querySelector('[data-kval]'), insight = root?.querySelector('[data-insight]');
  const M = 1200, xs = Array.from({ length: M + 1 }, (_, i) => i / M);
  const f = xs.map((x) => 0.9 * Math.cos(2 * Math.PI * 3 * x - 0.7) + 0.5 * Math.cos(2 * Math.PI * 7 * x + 1.2));
  const coef = (k) => {   // trapezoid rule for c(k) = int_0^1 f(x) e^{-2 pi i k x} dx
    let re = 0, im = 0;
    for (let i = 0; i <= M; i++) { const w = i === 0 || i === M ? 0.5 : 1, a = 2 * Math.PI * k * xs[i]; re += w * f[i] * Math.cos(a); im -= w * f[i] * Math.sin(a); }
    return [re / M, im / M];
  };
  const trace = Array.from({ length: 401 }, (_, i) => coef(i / 40));   // k from 0 to 10
  let k = 3, dpr = 1, wide = true;

  const layout = () => { const w = canvas.getBoundingClientRect().width; wide = w >= 600; canvas.style.height = `${wide ? 340 : 3 * 104 + 250}px`; dpr = fit(canvas); };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const pad = 10 * dpr, insetS = wide ? Math.min(0.3 * W, H - 2 * pad) : Math.min(W - 2 * pad, 230 * dpr);
    const plotW = wide ? W - insetS - 3 * pad : W - 2 * pad, rowH = wide ? (H - 2 * pad) / 3 : 104 * dpr;
    const rows = [0, 1, 2].map((r) => box1d({ x: pad, y: pad + r * rowH + 16 * dpr, w: plotW, h: rowH - 24 * dpr }, 0, 1, -1.5, 1.5));
    const probe = xs.map((x) => Math.cos(2 * Math.PI * k * x)), sprobe = xs.map((x) => Math.sin(2 * Math.PI * k * x));
    const pc = f.map((v, i) => v * probe[i]), ps = f.map((v, i) => v * sprobe[i]);
    const [re, im] = coef(k);
    // row 1: signal and probe
    axisLine(ctx, c, rows[0], 0, dpr);
    polyline(ctx, rows[0], xs, probe, `rgba(${c.a1}, 0.7)`, 1.4, dpr, [4, 3]);
    polyline(ctx, rows[0], xs, f, `rgb(${c.ink})`, 2, dpr);
    label(ctx, c, wide ? `signal f(x), and the probe cos(2π·${k.toFixed(2)}·x) (dashed)` : `f(x), probe dashed (k = ${k.toFixed(2)})`, rows[0].x, rows[0].y - 4 * dpr, dpr, { size: 11.5 });
    // rows 2 and 3: products, shaded
    [[pc, rows[1], `f(x)·cos(2πkx): area = ${re.toFixed(3)}`], [ps, rows[2], `f(x)·sin(2πkx): area = ${(-im).toFixed(3)}`]].forEach(([p, B, t]) => {
      axisLine(ctx, c, B, 0, dpr); shadeSigned(ctx, c, B, xs, p, dpr); polyline(ctx, B, xs, p, `rgba(${c.ink}, 0.8)`, 1.2, dpr);
      label(ctx, c, t, B.x, B.y - 4 * dpr, dpr, { size: 11.5 });
    });
    // inset: c(k) = (area under f cos) - i (area under f sin), in the complex plane
    const ix = wide ? W - pad - insetS : pad, iy = wide ? (H - insetS) / 2 : 3 * rowH + pad;
    const R = 0.5, I = box1d({ x: ix + 10 * dpr, y: iy + 18 * dpr, w: insetS - 20 * dpr, h: insetS - 28 * dpr }, -R, R, -R, R);
    ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(ix, iy, insetS, insetS);
    ctx.strokeStyle = `rgba(${c.ink}, 0.2)`; ctx.beginPath(); ctx.moveTo(I.X(-R), I.Y(0)); ctx.lineTo(I.X(R), I.Y(0)); ctx.moveTo(I.X(0), I.Y(-R)); ctx.lineTo(I.X(0), I.Y(R)); ctx.stroke();
    polyline(ctx, I, trace.map((p) => p[0]), trace.map((p) => p[1]), `rgba(${c.a1}, 0.55)`, 1.3, dpr);
    ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 2.2 * dpr; ctx.beginPath(); ctx.moveTo(I.X(0), I.Y(0)); ctx.lineTo(I.X(re), I.Y(im)); ctx.stroke();
    ctx.fillStyle = `rgb(${c.dot})`; ctx.beginPath(); ctx.arc(I.X(re), I.Y(im), 4.5 * dpr, 0, 2 * Math.PI); ctx.fill();
    label(ctx, c, 'c(k) in the complex plane', ix + 8 * dpr, iy + 14 * dpr, dpr, { size: 11 });
    label(ctx, c, `|c| = ${Math.hypot(re, im).toFixed(3)}`, ix + 8 * dpr, iy + insetS - 8 * dpr, dpr, { size: 11.5, alpha: 0.85 });
    explain(re, im);
  };
  const explain = (re, im) => {
    if (!insight) return;
    const mag = Math.hypot(re, im), near = Math.abs(k - Math.round(k)) < 1e-9, n = Math.round(k);
    let t;
    if (near && n === 3) t = `<strong>k = 3: a match.</strong> The product stays mostly on one side of zero, so the areas add up: c₃ = ${re.toFixed(3)} ${im < 0 ? '−' : '+'} ${Math.abs(im).toFixed(3)}i, with |c₃| = ${mag.toFixed(3)}. That is <em>half</em> the amplitude 0.9 of the 3-cycle wave, and its angle is the wave's phase, −0.7. The other half lives at k = −3.`;
    else if (near && n === 7) t = `<strong>k = 7: the second wave.</strong> |c₇| = ${mag.toFixed(3)}, half of its amplitude 0.5, with angle ${Math.atan2(im, re).toFixed(2)} (its phase is 1.2).`;
    else if (near) t = `<strong>k = ${n}: no match.</strong> Over a whole number of cycles the positive and negative areas cancel exactly: |c${n === 0 ? '₀' : ''}| = ${mag.toFixed(3)}. This cancellation is orthogonality, and it is why each coefficient sees only its own frequency.`;
    else t = `<strong>Between integers</strong> the probe does not finish whole cycles, so the areas do not quite cancel: |c(k)| = ${mag.toFixed(3)}. A Fourier <em>series</em> only uses whole numbers k; the spill-over at in-between frequencies is what Part 2 calls leakage.`;
    insight.innerHTML = t;
  };
  kIn?.addEventListener('input', () => { k = Number(kIn.value); if (kOut) kOut.textContent = k.toFixed(2); draw(); });
  if (kIn) { k = Number(kIn.value); if (kOut) kOut.textContent = k.toFixed(2); }
  layout(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F3 · Series builder. Partial sums S_N f = sum_{|k| <= N} c_k e^{2 pi i k x} of a square, sawtooth or triangle
// wave, or of a curve you draw. Coefficients by FFT on 4096 samples. Shows Gibbs, and coefficient decay on a log scale.
// ---------------------------------------------------------------------------------------------------------------

export function mountSeries(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const nIn = root?.querySelector('input[data-n]'), nOut = root?.querySelector('[data-nval]'), insight = root?.querySelector('[data-insight]');
  const shapeBtns = [...(root?.querySelectorAll('[data-shape]') ?? [])];
  const S = 4096, P = 1024;   // FFT samples, plot samples per period
  const shapes = {
    square: (x) => (x === 0 || x === 0.5 ? 0 : x < 0.5 ? 1 : -1),
    sawtooth: (x) => (x === 0 ? 0 : 2 * x - 1),
    triangle: (x) => 1 - 4 * Math.abs(x - 0.5),
  };
  let shape = 'square', N = 9, dpr = 1, drawn = null, G = null;
  let target = new Float64Array(S), cre = new Float64Array(S), cim = new Float64Array(S);

  const setTarget = () => {
    for (let i = 0; i < S; i++) { const x = i / S; target[i] = drawn ? drawn[Math.floor(x * drawn.length)] : shapes[shape](x); }
    cre = Float64Array.from(target); cim = new Float64Array(S); fft(cre, cim);
    for (let i = 0; i < S; i++) { cre[i] /= S; cim[i] /= S; }
  };
  const partial = () => {   // S_N on P points per period, real because c_{-k} = conj c_k
    const out = new Float64Array(P);
    for (let j = 0; j < P; j++) {
      const x = j / P; let s = cre[0];
      for (let k = 1; k <= N; k++) { const a = 2 * Math.PI * k * x; s += 2 * (cre[k] * Math.cos(a) - cim[k] * Math.sin(a)); }
      out[j] = s;
    }
    return out;
  };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width; canvas.style.height = `${w >= 600 ? 360 : 380}px`; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr;
    G = { top: box1d({ x: pad, y: 22 * dpr, w: W - 2 * pad, h: 0.58 * H - 30 * dpr }, 0, 2, -1.45, 1.45),
          bot: box1d({ x: pad + 34 * dpr, y: 0.62 * H + 14 * dpr, w: W - 2 * pad - 34 * dpr, h: 0.38 * H - 34 * dpr }, -0.5, 40.5, -4.3, 0.1) };
  };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height, { top, bot } = G;
    ctx.clearRect(0, 0, W, H);
    const s = partial(), xs = [], ys = [], ts = [];
    for (let j = 0; j < 2 * P; j++) { xs.push(j / P); ys.push(s[j % P]); ts.push(target[Math.floor(((j % P) / P) * S)]); }
    axisLine(ctx, c, top, 0, dpr);
    polyline(ctx, top, xs, ts, `rgba(${c.ink}, 0.35)`, 2, dpr);
    polyline(ctx, top, xs, ys, `rgb(${c.a1})`, 2.2, dpr);
    label(ctx, c, drawn ? 'your curve (drag to redraw)' : `${shape} wave, two periods`, top.x, top.y - 6 * dpr, dpr, { size: 11.5 });
    // stems of log10 |c_k|, k = 0..40
    label(ctx, c, 'log₁₀ |cₖ|, k = 0 … 40 (teal: used)', bot.x, bot.y - 6 * dpr, dpr, { size: 11.5 });
    for (const v of [0, -2, -4]) { ctx.strokeStyle = `rgba(${c.ink}, 0.1)`; ctx.beginPath(); ctx.moveTo(bot.x, bot.Y(v)); ctx.lineTo(bot.x + bot.w, bot.Y(v)); ctx.stroke(); label(ctx, c, String(v), bot.x - 26 * dpr, bot.Y(v) + 4 * dpr, dpr, { size: 10, alpha: 0.5 }); }
    for (let k = 0; k <= 40; k++) {
      const m = Math.hypot(cre[k], cim[k]); if (m < 1e-9) continue;
      const v = Math.max(-4.3, Math.log10(m)), used = k <= N;
      ctx.strokeStyle = used ? `rgb(${c.a1})` : `rgba(${c.ink}, 0.25)`; ctx.lineWidth = 2 * dpr;
      ctx.beginPath(); ctx.moveTo(bot.X(k), bot.Y(-4.3)); ctx.lineTo(bot.X(k), bot.Y(v)); ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(bot.X(k), bot.Y(v), 2.6 * dpr, 0, 2 * Math.PI); ctx.fill();
    }
    explain(s);
  };
  const explain = (s) => {
    if (!insight) return;
    let e2 = 0, t2 = 0; for (let i = 0; i < S; i++) t2 += target[i] * target[i];
    let kept = cre[0] ** 2; for (let k = 1; k <= N; k++) kept += 2 * (cre[k] ** 2 + cim[k] ** 2);
    e2 = Math.max(0, t2 / S - kept);
    const rel = Math.sqrt(e2 / (t2 / S)) * 100, mx = Math.max(...s);
    const terms = `With harmonics up to k = ${N}, the relative L² error is ${rel.toFixed(rel < 1 ? 2 : 1)}%.`;
    let t;
    if (drawn) t = `${terms} Your curve's coefficients decay as fast as its smoothness allows: every jump you drew forces a slow 1/k tail.`;
    else if (shape === 'triangle') t = `${terms} The triangle has kinks but no jumps, so its coefficients fall like 1/k² (the stems drop two decades per decade of k) and the partial sums converge everywhere, with no overshoot.`;
    else t = `${terms} Near each jump the sum overshoots to ${mx.toFixed(3)}, ${(((mx - 1) / 2) * 100).toFixed(1)}% of the jump of 2. Add more terms: the overshoot gets narrower but not lower. That is the Gibbs phenomenon (Section 4). The coefficients fall only like 1/k.`;
    insight.innerHTML = t;
  };
  const setShape = (sh) => { shape = sh; drawn = sh === 'draw' ? (drawn ?? Array.from({ length: 512 }, (_, i) => 0.7 * Math.sin(2 * Math.PI * i / 512) + (i > 300 && i < 360 ? 0.6 : 0))) : null; shapeBtns.forEach((b) => b.classList.toggle('on', b.dataset.shape === sh)); setTarget(); draw(); };
  // drawing: drag across the top plot (either period) to set the curve
  let drawing = false, last = null;
  const paint = (e) => {
    const b = canvas.getBoundingClientRect(), X = (e.clientX - b.left) * dpr, Y = (e.clientY - b.top) * dpr, { top } = G;
    if (Y < top.y - 10 * dpr || Y > top.y + top.h + 10 * dpr) return;
    const x = ((X - top.x) / top.w) * 2, y = Math.max(-1.4, Math.min(1.4, 1.45 - ((Y - top.y) / top.h) * 2.9));
    const i = Math.floor((((x % 1) + 1) % 1) * 512);
    if (last && Math.abs(last[0] - i) < 256) { const [i0, y0] = last, n = Math.abs(i - i0); for (let s = 0; s <= n; s++) drawn[i0 + Math.sign(i - i0) * s] = y0 + ((y - y0) * s) / Math.max(1, n); }
    else drawn[i] = y;
    last = [i, y]; setTarget(); draw();
  };
  canvas.addEventListener('pointerdown', (e) => { if (shape !== 'draw') setShape('draw'); drawing = true; last = null; canvas.setPointerCapture(e.pointerId); paint(e); });
  canvas.addEventListener('pointermove', (e) => { if (drawing) paint(e); });
  canvas.addEventListener('pointerup', () => { drawing = false; last = null; });
  nIn?.addEventListener('input', () => { N = Number(nIn.value); if (nOut) nOut.textContent = String(N); draw(); });
  shapeBtns.forEach((b) => b.addEventListener('click', () => setShape(b.dataset.shape)));
  if (nIn) { N = Number(nIn.value); if (nOut) nOut.textContent = String(N); }
  layout(); setTarget(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// The song: the series' shared 1-D signal. 2048 samples on [0, 1): a chord (40 and 56 Hz) from 0.04 to 0.40,
// a chirp (60 -> 240 Hz) from 0.46 to 0.84, and a click at 0.92. Fully synthetic, so every number is exact.
// ---------------------------------------------------------------------------------------------------------------

export const SONG_N = 2048;
export function song() {
  const N = SONG_N, x = new Float64Array(N);
  const env = (t, a, b) => { if (t < a || t > b) return 0; const r = 0.03, u = Math.min((t - a) / r, (b - t) / r, 1); return 0.5 - 0.5 * Math.cos(Math.PI * u); };
  for (let i = 0; i < N; i++) {
    const t = i / N, tau = t - 0.46;
    x[i] = env(t, 0.04, 0.40) * 0.55 * (Math.cos(2 * Math.PI * 40 * t) + Math.cos(2 * Math.PI * 56 * t))
      + env(t, 0.46, 0.84) * 0.8 * Math.cos(2 * Math.PI * (60 * tau + (180 / (2 * 0.38)) * tau * tau))
      + 1.6 * Math.exp(-(((t - 0.92) / 0.0015) ** 2));
  }
  return x;
}

// ---------------------------------------------------------------------------------------------------------------
// F4 · Filter lab on the song. Spectrum (dB), a filter's frequency response, and the filtered signal.
// ---------------------------------------------------------------------------------------------------------------

export function mountFilter(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const fIn = root?.querySelector('input[data-fc]'), fOut = root?.querySelector('[data-fcval]'), insight = root?.querySelector('[data-insight]');
  const typeBtns = [...(root?.querySelectorAll('[data-ftype]') ?? [])];
  const N = SONG_N, x = song(), Xr = Float64Array.from(x), Xi = new Float64Array(N);
  fft(Xr, Xi);
  const mag = Array.from({ length: N / 2 }, (_, k) => Math.hypot(Xr[k], Xi[k]));
  const mmax = Math.max(...mag);
  let type = 'ideal-low', fc = 80, dpr = 1, G = null, wide = true;
  const H = (f) => {   // frequency response at |f| Hz
    if (type === 'ideal-low') return f <= fc ? 1 : 0;
    if (type === 'gauss-low') return Math.exp(-Math.LN2 * (f / fc) ** 2);
    if (type === 'ideal-high') return f >= fc ? 1 : 0;
    return Math.abs(f - fc) <= 20 ? 1 : 0;   // band of width 40 Hz
  };
  const filtered = () => {
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let k = 0; k < N; k++) { const f = k <= N / 2 ? k : N - k, h = H(f); re[k] = Xr[k] * h; im[k] = Xi[k] * h; }
    fft(re, im, true); return re;
  };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width; wide = w >= 600; canvas.style.height = `${wide ? 380 : 400}px`; dpr = fit(canvas);
    const W = canvas.width, Hh = canvas.height, pad = 12 * dpr;
    G = { t: box1d({ x: pad, y: 22 * dpr, w: W - 2 * pad, h: 0.46 * Hh - 30 * dpr }, 0, 1, -2, 2),
          f: box1d({ x: pad, y: 0.52 * Hh + 16 * dpr, w: W - 2 * pad, h: 0.48 * Hh - 40 * dpr }, 0, 400, -60, 3) };
  };
  const draw = () => {
    const c = theme(), W = canvas.width, Hh = canvas.height, { t, f } = G;
    ctx.clearRect(0, 0, W, Hh);
    const y = filtered(), ts = Array.from({ length: N }, (_, i) => i / N);
    axisLine(ctx, c, t, 0, dpr);
    polyline(ctx, t, ts, Array.from(x), `rgba(${c.ink}, 0.28)`, 1, dpr);
    polyline(ctx, t, ts, Array.from(y), `rgb(${c.a1})`, 1.5, dpr);
    label(ctx, c, wide ? 'the song (grey) and what the filter lets through (teal)' : 'song (grey), filtered (teal)', t.x, t.y - 6 * dpr, dpr, { size: 11.5 });
    [['chord', 0.22], ['chirp', 0.65], ['click', 0.92]].forEach(([n, at]) => label(ctx, c, n, t.X(at), t.y + t.h + 12 * dpr, dpr, { size: 11, alpha: 0.55, align: 'center' }));
    // spectrum in dB and the response
    const fs = [], db = [];
    for (let k = 0; k <= 400; k++) { fs.push(k); db.push(Math.max(-60, 20 * Math.log10(mag[k] / mmax + 1e-12))); }
    ctx.fillStyle = `rgba(${c.dot}, 0.12)`;
    for (let k = 0; k < 400; k++) { const h = H(k + 0.5); if (h > 0.01) { ctx.globalAlpha = h; ctx.fillRect(f.X(k), f.y, f.X(k + 1) - f.X(k) + 0.5, f.h); } }
    ctx.globalAlpha = 1;
    polyline(ctx, f, fs, db, `rgba(${c.ink}, 0.75)`, 1.2, dpr);
    polyline(ctx, f, fs, fs.map((k) => -60 + 63 * H(k)), `rgb(${c.dot})`, 2, dpr);
    label(ctx, c, wide ? '|spectrum| in dB, 0 to 400 Hz; amber: the filter’s response' : '|spectrum| (dB), Hz; amber: filter', f.x, f.y - 6 * dpr, dpr, { size: 11.5 });
    for (const hz of [0, 100, 200, 300, 400]) label(ctx, c, `${hz}`, f.X(hz), f.y + f.h + 13 * dpr, dpr, { size: 10, alpha: 0.5, align: 'center' });
    explain(y);
  };
  const explain = (y) => {
    if (!insight) return;
    const i0 = Math.floor(0.86 * N), i1 = Math.floor(0.98 * N);
    let peak = -Infinity, dip = Infinity; for (let i = i0; i < i1; i++) { peak = Math.max(peak, y[i]); dip = Math.min(dip, y[i]); }
    const ring = peak > 1e-6 ? (-dip / peak) * 100 : 0;
    const lines = {
      'ideal-low': `<strong>Ideal low-pass at ${fc} Hz.</strong> Every frequency above ${fc} Hz is removed and every one below kept. The click becomes a bump of peak ${peak.toFixed(2)} with ripples that dip to ${ring.toFixed(0)}% of the peak below zero: the sharp cut in frequency is a sinc in time, and the sinc rings.`,
      'gauss-low': `<strong>Gaussian low-pass, half-power at ${fc} Hz.</strong> The response falls smoothly, its kernel in time is a positive Gaussian, and the click becomes a smooth bump of peak ${peak.toFixed(2)} with ${ring < 0.5 ? 'no' : `${ring.toFixed(1)}%`} undershoot. No ringing.`,
      'ideal-high': `<strong>Ideal high-pass at ${fc} Hz.</strong> The slow parts go; what is left is detail. The click survives (it has energy at every frequency, Section 6), and so does the top of the chirp.`,
      band: `<strong>Band-pass, ${Math.max(0, fc - 20)} to ${fc + 20} Hz.</strong> Watch the chirp: only the moment when its frequency sweeps through the band survives. The spectrum knew the chirp contained these frequencies; only the time plot shows when.`,
    };
    insight.innerHTML = lines[type];
  };
  typeBtns.forEach((b) => b.addEventListener('click', () => { type = b.dataset.ftype; typeBtns.forEach((q) => q.classList.toggle('on', q === b)); draw(); }));
  fIn?.addEventListener('input', () => { fc = Number(fIn.value); if (fOut) fOut.textContent = String(fc); draw(); });
  if (fIn) { fc = Number(fIn.value); if (fOut) fOut.textContent = String(fc); }
  layout(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F6 · Amplitude versus phase. Rebuild an image from the amplitude of one sketch and the phase of another,
// or from its own amplitude with random phase. Correlations are measured live.
// ---------------------------------------------------------------------------------------------------------------

export function mountPhase(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const base = canvas.dataset.base, insight = root?.querySelector('[data-insight]');
  const aBtns = [...(root?.querySelectorAll('[data-a]') ?? [])], bBtns = [...(root?.querySelectorAll('[data-b]') ?? [])], modeBtns = [...(root?.querySelectorAll('[data-pmode]') ?? [])];
  const n = N2, spectra = {};
  let A = 'portrait', B = 'tree', mode = 'swap', dpr = 1, wide = true, panels = [];
  const cvs = [0, 1, 2, 3].map(() => offscreen(n));
  let randPhase = null;

  const get = async (name) => {
    if (!spectra[name]) {
      const img = await loadGrey(`${base}${name}.png`, n), re = Float64Array.from(img), im = new Float64Array(n * n);
      fft2(re, im, n); spectra[name] = { img, re, im };
    }
    return spectra[name];
  };
  const rebuild = (amp, ph) => {   // amplitude of one spectrum, phase of another (both Hermitian, so the result is real)
    const re = new Float64Array(n * n), im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) { const a = Math.hypot(amp.re[i], amp.im[i]), p = Math.atan2(ph.im[i], ph.re[i]); re[i] = a * Math.cos(p); im[i] = a * Math.sin(p); }
    fft2(re, im, n, true); return re;
  };
  const show = (cv, v) => {
    const s = Array.from(v).sort((p, q) => p - q), lo = s[Math.floor(0.01 * s.length)], hi = s[Math.floor(0.99 * s.length)];
    putPixels(cv, n, (i) => { const g = 255 * Math.min(1, Math.max(0, (v[i] - lo) / (hi - lo))); return [g, g, g]; });
  };
  const corr = (p, q) => { let mp = 0, mq = 0; for (let i = 0; i < p.length; i++) { mp += p[i]; mq += q[i]; } mp /= p.length; mq /= q.length; let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < p.length; i++) { const a = p[i] - mp, b = q[i] - mq; sxy += a * b; sxx += a * a; syy += b * b; } return sxy / Math.sqrt(sxx * syy); };
  let titles = [], stats = {};
  const compute = async () => {
    const sa = await get(A), sb = await get(B);
    show(cvs[0], sa.img); show(cvs[1], sb.img);
    if (mode === 'swap') {
      const ab = rebuild(sa, sb), ba = rebuild(sb, sa);
      show(cvs[2], ab); show(cvs[3], ba);
      titles = [A, B, wide ? `|${A}|, phase of ${B}` : `|${A}| ∠${B}`, wide ? `|${B}|, phase of ${A}` : `|${B}| ∠${A}`];
      stats = { ab: [corr(ab, sb.img), corr(ab, sa.img)], ba: [corr(ba, sa.img), corr(ba, sb.img)] };
    } else {
      if (!randPhase) { const g = (() => { let s = 7; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 - 0.5; }; })(); const re = Float64Array.from({ length: n * n }, g), im = new Float64Array(n * n); fft2(re, im, n); randPhase = { re, im }; }
      const ra = rebuild(sa, randPhase), rb = rebuild(sb, randPhase);
      show(cvs[2], ra); show(cvs[3], rb);
      titles = [A, B, wide ? `|${A}|, random phase` : `|${A}| ∠random`, wide ? `|${B}|, random phase` : `|${B}| ∠random`];
      stats = { ra: corr(ra, sa.img), rb: corr(rb, sb.img) };
    }
    draw(); explain();
  };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width; wide = w >= 600;
    const pad = 10, title = 20, side = wide ? (w - 5 * pad) / 4 : (w - 3 * pad) / 2;
    canvas.style.height = `${wide ? side + title + 2 * pad : 2 * (side + title) + 3 * pad}px`;
    dpr = fit(canvas);
    const s = side * dpr, p = pad * dpr, t = title * dpr;
    panels = [0, 1, 2, 3].map((i) => (wide ? { x: p + i * (s + p), y: p + t, s } : { x: p + (i % 2) * (s + p), y: p + t + Math.floor(i / 2) * (s + t + p), s }));
  };
  const draw = () => {
    const c = theme(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    panels.forEach((p, i) => {
      ctx.imageSmoothingEnabled = true; ctx.drawImage(cvs[i], p.x, p.y, p.s, p.s);
      ctx.strokeStyle = i >= 2 ? `rgb(${c.a1})` : c.line; ctx.lineWidth = (i >= 2 ? 2 : 1) * dpr; ctx.strokeRect(p.x, p.y, p.s, p.s);
      if (titles[i]) label(ctx, c, titles[i], p.x, p.y - 6 * dpr, dpr, { size: 11.5 });
    });
  };
  const explain = () => {
    if (!insight) return;
    if (mode === 'swap') {
      const [abB, abA] = stats.ab;
      insight.innerHTML = `<strong>Phase wins.</strong> The third image has every amplitude of the ${A} and every phase of the ${B}, and you see the ${B}: its correlation with the ${B} is ${abB.toFixed(2)}, with the ${A} only ${abA.toFixed(2)}. The amplitudes of natural images are much alike (they fall roughly like 1/|ξ|); where edges and shapes sit is written in the phases.`;
    } else {
      insight.innerHTML = `<strong>Amplitude alone is texture.</strong> Keep each sketch's amplitudes and draw the phases at random: the correlation with the original drops to ${stats.ra.toFixed(2)} and ${stats.rb.toFixed(2)}. The amplitude spectrum says how much of each scale and orientation there is, not where.`;
    }
  };
  const setSel = (btns, attr, v) => btns.forEach((b) => b.classList.toggle('on', b.dataset[attr] === v));
  aBtns.forEach((b) => b.addEventListener('click', () => { if (b.dataset.a === B) return; A = b.dataset.a; setSel(aBtns, 'a', A); compute(); }));
  bBtns.forEach((b) => b.addEventListener('click', () => { if (b.dataset.b === A) return; B = b.dataset.b; setSel(bBtns, 'b', B); compute(); }));
  modeBtns.forEach((b) => b.addEventListener('click', () => { mode = b.dataset.pmode; setSel(modeBtns, 'pmode', mode); compute(); }));
  layout(); compute();
  let lastWide = null;
  new ResizeObserver(() => { layout(); if (lastWide !== null && lastWide !== wide) compute(); else draw(); lastWide = wide; }).observe(canvas);
  onTheme(draw);
}

// ===============================================================================================================
// Part 2 · The discrete Fourier world
// ===============================================================================================================

function zonePlate(n) {   // phase pi r^2 / n, so the local frequency is r / n cycles per pixel: 0.5 (Nyquist) at the edge, r = n / 2
  const z = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const dx = x - n / 2, dy = y - n / 2; z[y * n + x] = 0.5 + 0.5 * Math.cos((Math.PI * (dx * dx + dy * dy)) / n); }
  return z;
}
function blurGauss(img, n, sigma) {   // separable Gaussian blur with periodic boundary
  const r = Math.ceil(3 * sigma), k = []; let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float64Array(n * n), out = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * img[y * n + ((x + i + n) % n)]; tmp[y * n + x] = a; }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * tmp[((y + i + n) % n) * n + x]; out[y * n + x] = a; }
  return out;
}

// F7 · Moiré: downsample by d, with and without a low-pass first.
export function mountMoire(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure');
  const base = canvas.dataset.base, insight = root?.querySelector('[data-insight]');
  const imgBtns = [...(root?.querySelectorAll('[data-mimg]') ?? [])], dBtns = [...(root?.querySelectorAll('[data-d]') ?? [])];
  const n = N2; let name = 'zone', d = 4, src = null, dpr = 1, panels = [], wide = true;
  const cvs = [offscreen(n), offscreen(n), offscreen(n)];
  const grey = (cv, arr, m) => { cv.width = m; cv.height = m; putPixels(cv, m, (i) => { const g = 255 * Math.min(1, Math.max(0, arr[i])); return [g, g, g]; }); };
  const compute = async () => {
    src = name === 'zone' ? zonePlate(n) : await loadGrey(`${base}${name}.png`, n);
    const m = Math.floor(n / d), naive = new Float64Array(m * m), filt = new Float64Array(m * m), bl = blurGauss(src, n, 0.45 * d);
    for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) { naive[y * m + x] = src[(y * d) * n + x * d]; filt[y * m + x] = bl[(y * d) * n + x * d]; }
    grey(cvs[0], src, n); grey(cvs[1], naive, m); grey(cvs[2], filt, m);
    draw(); explain();
  };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width; wide = w >= 600;
    const pad = 10, title = 20, side = wide ? (w - 4 * pad) / 3 : (w - 3 * pad) / 2;
    canvas.style.height = `${wide ? side + title + 2 * pad : 2 * (side + title) + 3 * pad}px`; dpr = fit(canvas);
    const s = side * dpr, p = pad * dpr, t = title * dpr;
    panels = [0, 1, 2].map((i) => (wide ? { x: p + i * (s + p), y: p + t, s } : { x: p + (i % 2) * (s + p), y: p + t + Math.floor(i / 2) * (s + t + p), s }));
  };
  const draw = () => {
    const c = theme(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    const titles = ['original', `every ${d}th pixel`, `blur, then every ${d}th`];
    panels.forEach((p, i) => {
      ctx.imageSmoothingEnabled = i === 0; ctx.drawImage(cvs[i], p.x, p.y, p.s, p.s);
      ctx.strokeStyle = i === 2 ? `rgb(${c.a1})` : c.line; ctx.lineWidth = (i === 2 ? 2 : 1) * dpr; ctx.strokeRect(p.x, p.y, p.s, p.s);
      label(ctx, c, titles[i], p.x, p.y - 6 * dpr, dpr, { size: 11.5 });
    });
  };
  const explain = () => {
    if (!insight) return;
    const ny = 1 / (2 * d);
    insight.innerHTML = name === 'zone'
      ? `<strong>Zone plate, every ${d}th pixel.</strong> The rings get finer outward: at radius r pixels their frequency is r/256 cycles per pixel. Keeping every ${d}th pixel can only represent frequencies up to 1/(2·${d}) = ${ny.toFixed(3)} cycles per original pixel, so every ring beyond radius ${Math.round(128 / d)} folds back into slow false rings: aliasing. Blurring first removes those frequencies before they can fold.`
      : `<strong>${name[0].toUpperCase() + name.slice(1)}, every ${d}th pixel.</strong> Fine texture above ${ny.toFixed(3)} cycles per pixel cannot survive the subsampling. Without a blur it does not disappear: it reappears as coarser, false texture (look at the hatching and the wave crests). With the blur it is simply gone.`;
  };
  imgBtns.forEach((b) => b.addEventListener('click', () => { name = b.dataset.mimg; imgBtns.forEach((q) => q.classList.toggle('on', q === b)); compute(); }));
  dBtns.forEach((b) => b.addEventListener('click', () => { d = Number(b.dataset.d); dBtns.forEach((q) => q.classList.toggle('on', q === b)); compute(); }));
  layout(); compute();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F8 · Aliasing lab: a cosine, its samples, the band-limited signal those samples define, and the folding diagram.
export function mountAlias(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), q = (s) => root?.querySelector(s);
  const fIn = q('input[data-f]'), fsIn = q('input[data-fs]'), phIn = q('input[data-ph]'), insight = q('[data-insight]');
  const fOut = q('[data-fval]'), fsOut = q('[data-fsval]'), phOut = q('[data-phval]');
  let f = 7, fs = 10, ph = 0.6, dpr = 1, G = null;
  const layout = () => {
    canvas.style.height = '330px'; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr;
    G = { t: box1d({ x: pad, y: 22 * dpr, w: W - 2 * pad, h: 0.5 * H - 30 * dpr }, 0, 1.2, -1.3, 1.3),
          f: box1d({ x: pad, y: 0.6 * H + 6 * dpr, w: W - 2 * pad, h: 0.4 * H - 34 * dpr }, -25, 25, 0, 1.25) };
  };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height, { t, f: F } = G;
    ctx.clearRect(0, 0, W, H);
    const m = Math.round(f / fs), fa = f - m * fs;             // the alias in [-fs/2, fs/2]
    const ts = Array.from({ length: 1201 }, (_, i) => (1.2 * i) / 1200);
    axisLine(ctx, c, t, 0, dpr);
    polyline(ctx, t, ts, ts.map((x) => Math.cos(2 * Math.PI * f * x + ph)), `rgba(${c.ink}, 0.55)`, 1.3, dpr);
    polyline(ctx, t, ts, ts.map((x) => Math.cos(2 * Math.PI * fa * x + ph)), `rgb(${c.a1})`, 2.2, dpr, [6, 4]);
    for (let k = 0; k / fs <= 1.2 + 1e-9; k++) { const x = k / fs, y = Math.cos(2 * Math.PI * f * x + ph); ctx.fillStyle = `rgb(${c.dot})`; ctx.beginPath(); ctx.arc(t.X(x), t.Y(y), 4 * dpr, 0, 2 * Math.PI); ctx.fill(); }
    label(ctx, c, canvas.getBoundingClientRect().width < 600 ? `signal (grey), samples (amber), alias (teal)` : `signal at ${f.toFixed(1)} Hz (grey), samples at ${fs.toFixed(1)} Hz (amber), what they define (teal, dashed)`, t.x, t.y - 6 * dpr, dpr, { size: 11 });
    // folding diagram
    ctx.fillStyle = `rgba(${c.a1}, 0.1)`; ctx.fillRect(F.X(-fs / 2), F.y, F.X(fs / 2) - F.X(-fs / 2), F.h);
    axisLine(ctx, c, F, 0, dpr);
    const stem = (x, h, col, w) => { if (x < -25 || x > 25) return; ctx.strokeStyle = col; ctx.lineWidth = w * dpr; ctx.beginPath(); ctx.moveTo(F.X(x), F.Y(0)); ctx.lineTo(F.X(x), F.Y(h)); ctx.stroke(); };
    for (let k = -6; k <= 6; k++) { if (k === 0) continue; stem(f + k * fs, 0.75, `rgba(${c.dot}, 0.55)`, 2); stem(-f + k * fs, 0.75, `rgba(${c.dot}, 0.55)`, 2); }
    stem(f, 1, `rgb(${c.ink})`, 2.6); stem(-f, 1, `rgb(${c.ink})`, 2.6);
    stem(fa, 0.55, `rgb(${c.a1})`, 4); stem(-fa, 0.55, `rgb(${c.a1})`, 4);
    for (const hz of [-20, -10, 0, 10, 20]) label(ctx, c, `${hz}`, F.X(hz), F.y + F.h + 13 * dpr, dpr, { size: 10, alpha: 0.5, align: 'center' });
    label(ctx, c, canvas.getBoundingClientRect().width < 600 ? `spectrum, copies (amber), band |ξ| ≤ fs/2` : `spectrum: ±${f.toFixed(1)} Hz (dark), copies every ${fs.toFixed(1)} Hz (amber); shaded: |ξ| ≤ fs/2`, F.x, F.y - 6 * dpr, dpr, { size: 11 });
    explain(fa);
  };
  const explain = (fa) => {
    if (!insight) return;
    const ny = fs / 2, a = Math.abs(fa);
    let t;
    if (f < ny - 1e-9) t = `<strong>${f.toFixed(1)} Hz, sampled at ${fs.toFixed(1)} Hz.</strong> Below the Nyquist frequency ${ny.toFixed(1)} Hz, so the samples pin down the signal: the teal curve lies on the grey one.`;
    else if (Math.abs(f - ny) < 1e-9) t = `<strong>Exactly at the Nyquist frequency.</strong> The samples are (−1)ⁿ cos(phase) = ±${Math.abs(Math.cos(ph)).toFixed(2)}: they depend on the phase, and for a sine (phase 90°) they are all zero. Sampling at exactly twice the frequency is not enough.`;
    else t = `<strong>Aliasing.</strong> A ${f.toFixed(1)} Hz wave sampled at ${fs.toFixed(1)} Hz gives exactly the samples of a ${a.toFixed(1)} Hz wave${fa < 0 ? ' (running backwards, phase flipped)' : ''}. Nothing in the samples can tell them apart: every frequency f − m·fs, for whole m, fits the same dots, and reconstruction picks the one inside ±${ny.toFixed(1)} Hz.`;
    insight.innerHTML = t;
  };
  const sync = () => { f = Number(fIn?.value ?? f); fs = Number(fsIn?.value ?? fs); ph = Number(phIn?.value ?? ph); if (fOut) fOut.textContent = f.toFixed(1); if (fsOut) fsOut.textContent = fs.toFixed(1); if (phOut) phOut.textContent = `${Math.round((ph * 180) / Math.PI)}°`; draw(); };
  [fIn, fsIn, phIn].forEach((el) => el?.addEventListener('input', sync));
  layout(); sync();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F9 · Stride-2 subsampling is not shift-invariant: plain vs blur-then-stride (Zhang 2019).
export function mountStride(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), sIn = root?.querySelector('input[data-shift]'), sOut = root?.querySelector('[data-shiftval]'), insight = root?.querySelector('[data-insight]');
  const L = 64, nb = Array.from({ length: L + 16 }, (_, i) => i);
  const base = nb.map((i) => Math.cos(2 * Math.PI * 0.05 * i) + 0.6 * Math.cos(2 * Math.PI * 0.47 * i));
  const hann = Array.from({ length: L }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (L - 1)));
  const input = (s) => hann.map((w, i) => w * base[s + i]);
  const stride = (v) => v.filter((_, i) => i % 2 === 0);
  const blurStride = (v) => stride(v.map((_, i) => 0.25 * (v[i - 1] ?? 0) + 0.5 * v[i] + 0.25 * (v[i + 1] ?? 0)));
  const energy = (v) => v.reduce((a, b) => a + b * b, 0);
  const E = Array.from({ length: 8 }, (_, s) => [energy(stride(input(s))), energy(blurStride(input(s)))]);
  const cv = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return (Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length) / m) * 100; };
  const cvS = cv(E.map((e) => e[0])), cvB = cv(E.map((e) => e[1]));
  let s = 0, dpr = 1, G = null;
  const layout = () => {
    const w = canvas.getBoundingClientRect().width; canvas.style.height = `${w >= 600 ? 320 : 380}px`; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr, wide = w >= 600;
    const leftW = wide ? 0.66 * W : W - 2 * pad;
    G = { a: box1d({ x: pad, y: 22 * dpr, w: leftW - pad, h: (wide ? 0.5 * H : 0.33 * H) - 34 * dpr }, 0, 63, -1.8, 1.8),
          b: box1d({ x: pad, y: (wide ? 0.5 * H : 0.36 * H) + 18 * dpr, w: leftW - pad, h: (wide ? 0.5 * H : 0.3 * H) - 40 * dpr }, 0, 31, -1.8, 1.8),
          e: wide ? { x: leftW + 2 * pad, y: 22 * dpr, w: W - leftW - 3 * pad, h: H - 50 * dpr } : { x: pad, y: 0.7 * H + 16 * dpr, w: W - 2 * pad, h: 0.3 * H - 40 * dpr } };
  };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height, { a, b, e } = G;
    ctx.clearRect(0, 0, W, H);
    const v = input(s), idx = v.map((_, i) => i);
    axisLine(ctx, c, a, 0, dpr); polyline(ctx, a, idx, v, `rgb(${c.ink})`, 1.4, dpr);
    label(ctx, c, canvas.getBoundingClientRect().width < 600 ? `input, shift ${s}` : `input, shifted by ${s} sample${s === 1 ? '' : 's'}: slow wave + texture near Nyquist`, a.x, a.y - 6 * dpr, dpr, { size: 11 });
    const o1 = stride(v), o2 = blurStride(v), j = o1.map((_, i) => i);
    axisLine(ctx, c, b, 0, dpr); polyline(ctx, b, j, o1, `rgb(${c.dot})`, 1.8, dpr); polyline(ctx, b, j, o2, `rgb(${c.a1})`, 2.2, dpr);
    label(ctx, c, canvas.getBoundingClientRect().width < 600 ? 'stride (amber), blur + stride (teal)' : 'stride 2 (amber) vs blur [1,2,1]/4 then stride 2 (teal)', b.x, b.y - 6 * dpr, dpr, { size: 11 });
    // energy bars by shift
    const mx = Math.max(...E.flat()), bw = e.w / 8;
    label(ctx, c, 'output energy vs shift', e.x, e.y - 6 * dpr, dpr, { size: 11 });
    E.forEach(([es, eb], k) => {
      const x0 = e.x + k * bw + 0.12 * bw, w2 = 0.36 * bw;
      ctx.fillStyle = `rgba(${c.dot}, ${k === s ? 0.95 : 0.45})`; ctx.fillRect(x0, e.y + e.h - (es / mx) * e.h, w2, (es / mx) * e.h);
      ctx.fillStyle = `rgba(${c.a1}, ${k === s ? 0.95 : 0.45})`; ctx.fillRect(x0 + w2 + 2 * dpr, e.y + e.h - (eb / mx) * e.h, w2, (eb / mx) * e.h);
      label(ctx, c, String(k), x0 + w2, e.y + e.h + 13 * dpr, dpr, { size: 10, alpha: 0.55, align: 'center' });
    });
    if (insight) insight.innerHTML = `<strong>Shift ${s}.</strong> Plain stride-2 keeps every other sample, so the texture near 0.47 cycles per sample folds to a slow false wave whose size depends on where the samples land: across the eight shifts its output energy varies by ${cvS.toFixed(0)}% (standard deviation over mean). Blurring first with [1, 2, 1]/4 passes the slow wave (response ${(Math.cos(Math.PI * 0.05) ** 2).toFixed(2)}) and nearly erases the texture (response ${(Math.cos(Math.PI * 0.47) ** 2).toFixed(3)}): variation ${cvB.toFixed(0)}%. This is the whole idea of anti-aliased CNNs.`;
  };
  sIn?.addEventListener('input', () => { s = Number(sIn.value); if (sOut) sOut.textContent = String(s); draw(); });
  layout(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F10 · Leakage lab: a strong tone between bins, a weak tone nearby, three windows, zero padding.
function windowFn(name, N) {
  return Array.from({ length: N }, (_, n) => {
    const a = (2 * Math.PI * n) / N;
    if (name === 'hann') return 0.5 - 0.5 * Math.cos(a);
    if (name === 'blackman') return 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
    return 1;
  });
}
export function mountLeakage(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), q = (s) => root?.querySelector(s);
  const f1In = q('input[data-f1]'), dIn = q('input[data-df]'), f1Out = q('[data-f1val]'), dOut = q('[data-dfval]'), insight = q('[data-insight]');
  const winBtns = [...(root?.querySelectorAll('[data-win]') ?? [])], padBtns = [...(root?.querySelectorAll('[data-pad]') ?? [])];
  const N = 64, weak = 0.01;   // weak tone at -40 dB
  let f1 = 10.5, df = 4, win = 'rect', pad = 4, dpr = 1, G = null;
  const spectrum = (withWeak, P) => {
    const w = windowFn(win, N), re = new Float64Array(N * P), im = new Float64Array(N * P);
    let wsum = 0; for (let n = 0; n < N; n++) wsum += w[n];
    for (let n = 0; n < N; n++) re[n] = w[n] * (Math.cos((2 * Math.PI * f1 * n) / N) + (withWeak ? weak * Math.cos((2 * Math.PI * (f1 + df) * n) / N) : 0));
    fft(re, im);
    return Array.from({ length: (N * P) / 2 + 1 }, (_, k) => 20 * Math.log10(Math.hypot(re[k], im[k]) / (wsum / 2) + 1e-12));
  };
  const layout = () => {
    canvas.style.height = '300px'; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad0 = 12 * dpr;
    G = box1d({ x: pad0 + 26 * dpr, y: 22 * dpr, w: W - 2 * pad0 - 26 * dpr, h: H - 50 * dpr }, 0, 32, -110, 6);
  };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height, B = G;
    ctx.clearRect(0, 0, W, H);
    for (const db of [0, -40, -80]) { ctx.strokeStyle = `rgba(${c.ink}, 0.1)`; ctx.beginPath(); ctx.moveTo(B.x, B.Y(db)); ctx.lineTo(B.x + B.w, B.Y(db)); ctx.stroke(); label(ctx, c, `${db}`, B.x - 26 * dpr, B.Y(db) + 4 * dpr, dpr, { size: 10, alpha: 0.5 }); }
    [[f1, 'strong'], [f1 + df, '−40 dB']].forEach(([fx, t]) => { if (fx > 32) return; ctx.strokeStyle = `rgba(${c.ink}, 0.35)`; ctx.setLineDash([3 * dpr, 4 * dpr]); ctx.beginPath(); ctx.moveTo(B.X(fx), B.y); ctx.lineTo(B.X(fx), B.y + B.h); ctx.stroke(); ctx.setLineDash([]); label(ctx, c, t, B.X(fx) + 4 * dpr, B.y + 12 * dpr, dpr, { size: 10.5, alpha: 0.7 }); });
    const S = spectrum(true, pad), xs = S.map((_, k) => k / pad);
    polyline(ctx, B, xs, S.map((v) => Math.max(-110, v)), `rgb(${c.a1})`, 1.6, dpr);
    const S1 = spectrum(true, 1);
    S1.forEach((v, k) => { ctx.fillStyle = `rgb(${c.dot})`; ctx.beginPath(); ctx.arc(B.X(k), B.Y(Math.max(-110, v)), 3.2 * dpr, 0, 2 * Math.PI); ctx.fill(); });
    for (const k of [0, 8, 16, 24, 32]) label(ctx, c, `${k}`, B.X(k), B.y + B.h + 13 * dpr, dpr, { size: 10, alpha: 0.5, align: 'center' });
    label(ctx, c, canvas.getBoundingClientRect().width < 600 ? `dB vs bins; dots: DFT; line: padded ×${pad}` : `dB vs frequency in bins (64 samples); dots: the DFT bins; line: zero-padded ×${pad}`, B.x, B.y - 6 * dpr, dpr, { size: 11 });
    explain();
  };
  const explain = () => {
    if (!insight) return;
    const P = 16, a = spectrum(true, P), b = spectrum(false, P), kW = Math.round((f1 + df) * P), half = P / 2;
    // leakage envelope near the weak tone (the leakage pattern has exact zeros at whole-bin distances, so take the
    // maximum over +-0.5 bins), against the weak tone's own peak (maximum over +-0.25 bins)
    let env = -Infinity, weakPk = -Infinity;
    for (let k = kW - half; k <= kW + half; k++) if (k >= 0 && k < b.length) env = Math.max(env, b[k]);
    for (let k = kW - P / 4; k <= kW + P / 4; k++) if (k >= 0 && k < a.length) weakPk = Math.max(weakPk, a[k]);
    const S1 = spectrum(true, 1), bin = Math.max(S1[Math.floor(f1)], S1[Math.ceil(f1)]);
    const lift = weakPk - env, visible = lift > 6;
    const name = { rect: 'No window (rectangular)', hann: 'Hann window', blackman: 'Blackman window' }[win];
    const onBin = Math.abs(f1 - Math.round(f1)) < 0.01;
    insight.innerHTML = `<strong>${name}.</strong> The strong tone sits at ${f1.toFixed(2)} bins${onBin ? ', exactly on a bin' : ', between two bins, so its energy leaks into the others'}; the nearest DFT bin reads ${bin.toFixed(1)} dB of its true level${onBin ? '' : ' (the scalloping loss)'}. Around the weak tone (${df.toFixed(1)} bins away, −40 dB) the strong tone's leakage reaches ${env.toFixed(0)} dB, so the weak tone ${visible ? `stands out by ${lift.toFixed(0)} dB` : 'is buried in it'}. ${win === 'rect' ? 'Try Hann or Blackman: lower side lobes, wider main lobe.' : ''} Zero-padding only draws the curve more finely; it never separates what the window merged.`;
  };
  winBtns.forEach((b) => b.addEventListener('click', () => { win = b.dataset.win; winBtns.forEach((q2) => q2.classList.toggle('on', q2 === b)); draw(); }));
  padBtns.forEach((b) => b.addEventListener('click', () => { pad = Number(b.dataset.pad); padBtns.forEach((q2) => q2.classList.toggle('on', q2 === b)); draw(); }));
  const sync = () => { f1 = Number(f1In?.value ?? f1); df = Number(dIn?.value ?? df); if (f1Out) f1Out.textContent = f1.toFixed(2); if (dOut) dOut.textContent = df.toFixed(1); draw(); };
  [f1In, dIn].forEach((el) => el?.addEventListener('input', sync));
  layout(); sync();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F11 · Spectrogram of the song: short-time Fourier transform with a Hann window of adjustable length.
export function mountSpectrogram(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), lBtns = [...(root?.querySelectorAll('[data-wl]') ?? [])], insight = root?.querySelector('[data-insight]');
  const x = song(), N = SONG_N, FMAX = 300;
  let L = 128, dpr = 1, G = null, img = null, meta = null;
  const stft = () => {
    const hop = Math.max(4, L / 8), w = windowFn('hann', L), frames = [], centers = [];
    for (let s = -L / 2; s + L / 2 <= N; s += hop) {
      const re = new Float64Array(L), im = new Float64Array(L);
      for (let n = 0; n < L; n++) { const i = s + n; re[n] = i >= 0 && i < N ? x[i] * w[n] : 0; }
      fft(re, im);
      const kmax = Math.floor((FMAX * L) / N), col = [];
      for (let k = 0; k <= kmax; k++) col.push(Math.hypot(re[k], im[k]));
      frames.push(col); centers.push((s + L / 2) / N);
    }
    const mx = Math.max(...frames.flat());
    const T = frames.length, K = frames[0].length, cv = document.createElement('canvas'); cv.width = T; cv.height = K;
    const id = cv.getContext('2d').createImageData(T, K);
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) {
      const db = 20 * Math.log10(frames[t][k] / mx + 1e-9), v = Math.max(0, (db + 70) / 70), [r, g, b] = cmap(v), o = 4 * ((K - 1 - k) * T + t);
      id.data[o] = r; id.data[o + 1] = g; id.data[o + 2] = b; id.data[o + 3] = 255;
    }
    cv.getContext('2d').putImageData(id, 0, 0);
    img = cv;
    // measurements: is the chord resolved? (zero-padded spectrum of the frame at 0.22 s, count peaks between 30 and 66 Hz)
    const P8 = 8, c0 = Math.round(0.22 * N) - L / 2, re = new Float64Array(L * P8), im = new Float64Array(L * P8);
    for (let n = 0; n < L; n++) re[n] = x[c0 + n] * w[n];
    fft(re, im);
    const fr = (k) => (k * N) / (L * P8), mags = [];
    for (let k = 0; fr(k) <= 66; k++) if (fr(k) >= 30) mags.push(Math.hypot(re[k], im[k]));
    const mmax = Math.max(...mags); let peaks = 0;
    for (let i = 1; i < mags.length - 1; i++) if (mags[i] > mags[i - 1] && mags[i] >= mags[i + 1] && mags[i] > 0.3 * mmax) {
      let lmin = mags[i], rmin = mags[i];
      for (let j = i - 1; j >= 0 && mags[j] <= mags[i]; j--) lmin = Math.min(lmin, mags[j]);
      for (let j = i + 1; j < mags.length && mags[j] <= mags[i]; j++) rmin = Math.min(rmin, mags[j]);
      if (mags[i] - Math.max(lmin, rmin) > 0.05 * mmax) peaks++;
    }
    const resolved = peaks >= 2;
    const band = frames.map((c) => c.slice(Math.floor((150 * L) / N)).reduce((a, b) => a + b * b, 0));
    const near = band.map((e, i) => (Math.abs(centers[i] - 0.92) < 0.06 ? e : 0)), pk = Math.max(...near);
    const above = centers.filter((c, i) => near[i] >= pk / 4);
    const width = above.length ? (Math.max(...above) - Math.min(...above) + Math.max(4, L / 8) / N) * 1000 : 0;
    meta = { resolved, width, dt: (L / N) * 1000, df: N / L };
  };
  const layout = () => {
    canvas.style.height = '300px'; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr;
    G = { x: pad + 30 * dpr, y: 22 * dpr, w: W - 2 * pad - 30 * dpr, h: H - 50 * dpr };
  };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (img) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, G.x, G.y, G.w, G.h); }
    ctx.strokeStyle = c.line; ctx.strokeRect(G.x, G.y, G.w, G.h);
    for (const hz of [0, 100, 200, 300]) label(ctx, c, `${hz}`, G.x - 6 * dpr, G.y + G.h - (hz / FMAX) * G.h + 4 * dpr, dpr, { size: 10, alpha: 0.55, align: 'right' });
    for (const s of [0, 0.25, 0.5, 0.75, 1]) label(ctx, c, `${s}`, G.x + s * G.w, G.y + G.h + 13 * dpr, dpr, { size: 10, alpha: 0.55, align: 'center' });
    label(ctx, c, canvas.getBoundingClientRect().width < 600 ? `window ${L} (${meta.dt.toFixed(0)} ms); Hz vs s` : `spectrogram of the song, window ${L} samples (${meta.dt.toFixed(0)} ms); Hz vs seconds`, G.x, G.y - 6 * dpr, dpr, { size: 11 });
    if (insight) insight.innerHTML = `<strong>Window ${L} samples = ${meta.dt.toFixed(0)} ms; bins ${meta.df.toFixed(0)} Hz apart.</strong> The chord's two tones (16 Hz apart) are ${meta.resolved ? '<em>resolved</em>' : '<em>merged into one band</em>'}, and the click is smeared over about ${meta.width.toFixed(0)} ms. ${L >= 512 ? 'Long windows see frequency sharply and time vaguely.' : L <= 64 ? 'Short windows see time sharply and frequency vaguely.' : 'Try the extremes.'} No single window gets both: Section 10 proves it cannot.`;
  };
  lBtns.forEach((b) => b.addEventListener('click', () => { L = Number(b.dataset.wl); lBtns.forEach((q2) => q2.classList.toggle('on', q2 === b)); stft(); draw(); }));
  layout(); stft(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F-dft4 · The 4 x 4 DFT by hand: every coefficient is a sum of pixels times 1, -i, -1 or i.
export function mountDft4(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), presetBtns = [...(root?.querySelectorAll('[data-p4]') ?? [])], insight = root?.querySelector('[data-insight]');
  const P = {
    vstripes: () => grid4((m, n) => (n % 2 === 0 ? 1 : 0)), hstripes: () => grid4((m, n) => (m % 2 === 0 ? 1 : 0)),
    checker: () => grid4((m, n) => ((m + n) % 2 === 0 ? 1 : 0)), pixel: () => grid4((m, n) => (m === 0 && n === 0 ? 1 : 0)),
    constant: () => grid4(() => 1), block: () => grid4((m, n) => (m < 2 && n < 2 ? 1 : 0)),
  };
  function grid4(f) { return Array.from({ length: 4 }, (_, m) => Array.from({ length: 4 }, (_, n) => f(m, n))); }
  const W4 = [[1, 0], [0, -1], [-1, 0], [0, 1]];   // (-i)^p for p = 0..3: 1, -i, -1, i
  const SYM = ['1', '−i', '−1', 'i'];
  let x = P.vstripes(), sel = [0, 2], dpr = 1, G = null;
  const dft = () => {
    const X = [];
    for (let k = 0; k < 4; k++) { X.push([]); for (let l = 0; l < 4; l++) { let re = 0, im = 0; for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) { const [a, b] = W4[(k * m + l * n) % 4]; re += x[m][n] * a; im += x[m][n] * b; } X[k].push([re, im]); } }
    return X;
  };
  const fmt = (re, im) => {
    const r = Math.round(re * 100) / 100, i = Math.round(im * 100) / 100;
    if (Math.abs(i) < 1e-9) return String(r).replace('-', '−');
    if (Math.abs(r) < 1e-9) return `${i === 1 ? '' : i === -1 ? '−' : String(i).replace('-', '−')}i`;
    return `${String(r).replace('-', '−')}${i > 0 ? '+' : '−'}${Math.abs(i) === 1 ? '' : Math.abs(i)}i`;
  };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width, wide = w >= 600, lab = 22;
    const cell = Math.floor(wide ? (w - 80) / 12 : Math.min(48, (w - 30) / 8));
    canvas.style.height = `${wide ? 4 * cell + lab + 14 : 8 * cell + 2 * lab + 40}px`; dpr = fit(canvas);
    const c = cell * dpr, L = lab * dpr, gap = 30 * dpr;
    G = wide ? { c, g: [{ x: 10 * dpr, y: L }, { x: 10 * dpr + 4 * c + gap, y: L }, { x: 10 * dpr + 8 * c + 2 * gap, y: L }] }
             : { c, g: [{ x: 10 * dpr, y: L }, { x: 10 * dpr + 4 * c + gap, y: L }, { x: 10 * dpr, y: 4 * c + 2 * L + 14 * dpr }] };
  };
  const draw = () => {
    const c = theme(), X = dft(), { c: cs, g } = G;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cell = (gx, gy, m, n, fill, txt, tc) => {
      ctx.fillStyle = fill; ctx.fillRect(gx + n * cs, gy + m * cs, cs, cs);
      ctx.strokeStyle = `rgba(${c.ink}, 0.15)`; ctx.lineWidth = dpr; ctx.strokeRect(gx + n * cs, gy + m * cs, cs, cs);
      ctx.fillStyle = tc; ctx.font = `${Math.min(13, cs / dpr / 3.4) * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, gx + (n + 0.5) * cs, gy + (m + 0.5) * cs); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    };
    let mx = 0; X.flat().forEach(([a, b]) => { mx = Math.max(mx, Math.hypot(a, b)); });
    for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) {
      const v = x[m][n]; cell(g[0].x, g[0].y, m, n, v ? 'rgb(236,235,245)' : 'rgb(28,26,39)', String(v), v ? '#111' : '#eee');
      const [a, b] = X[m][n], r = mx ? Math.hypot(a, b) / mx : 0, col = cmap(0.15 + 0.85 * r);
      cell(g[1].x, g[1].y, m, n, Math.hypot(a, b) < 1e-9 ? `rgba(${c.ink}, 0.04)` : `rgb(${col.map(Math.round).join(',')})`, fmt(a, b), Math.hypot(a, b) < 1e-9 ? `rgba(${c.ink}, 0.35)` : r > 0.6 ? '#111' : '#eee');
      const p = (sel[0] * m + sel[1] * n) % 4;
      cell(g[2].x, g[2].y, m, n, ['rgba(95,191,182,0.35)', 'rgba(251,191,36,0.3)', 'rgba(139,92,246,0.3)', 'rgba(47,143,134,0.15)'][p], SYM[p], `rgb(${c.ink})`);
    }
    ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 3 * dpr; ctx.strokeRect(g[1].x + sel[1] * cs, g[1].y + sel[0] * cs, cs, cs);
    label(ctx, c, 'image x[m, n] (click to toggle)', g[0].x, g[0].y - 7 * dpr, dpr, { size: 11.5 });
    label(ctx, c, 'DFT X[k, l] (click one)', g[1].x, g[1].y - 7 * dpr, dpr, { size: 11.5 });
    label(ctx, c, `multiply by, for X[${sel[0]}, ${sel[1]}]`, g[2].x, g[2].y - 7 * dpr, dpr, { size: 11.5 });
    if (insight) {
      const [k, l] = sel, terms = []; let re = 0, im = 0;
      for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) if (x[m][n]) { const p = (k * m + l * n) % 4; terms.push(SYM[p]); re += W4[p][0]; im += W4[p][1]; }
      const nz = X.flat().filter(([a, b]) => Math.hypot(a, b) > 1e-9).length;
      insight.innerHTML = `<strong>X[${k}, ${l}] by hand.</strong> With N = 4, every factor e<sup>−2πi(km + ln)/4</sup> is one of 1, −i, −1, i (right grid). Add the factors at the lit pixels: ${terms.length ? terms.join(' + ').replace(/\+ −/g, '− ') : '(no lit pixels)'} = <strong>${fmt(re, im)}</strong>. ${nz} of 16 coefficients are non-zero. Index k counts cycles down the columns (vertical frequency), l across the rows (horizontal), as in NumPy's fft2.`;
    }
  };
  const hit = (e, gg) => { const b = canvas.getBoundingClientRect(), px = (e.clientX - b.left) * dpr, py = (e.clientY - b.top) * dpr, n = Math.floor((px - gg.x) / G.c), m = Math.floor((py - gg.y) / G.c); return m >= 0 && m < 4 && n >= 0 && n < 4 ? [m, n] : null; };
  canvas.addEventListener('pointerdown', (e) => {
    const a = hit(e, G.g[0]); if (a) { x[a[0]][a[1]] = x[a[0]][a[1]] ? 0 : 1; presetBtns.forEach((b) => b.classList.remove('on')); draw(); return; }
    const o = hit(e, G.g[1]); if (o) { sel = o; draw(); }
  });
  presetBtns.forEach((b) => b.addEventListener('click', () => { x = P[b.dataset.p4](); presetBtns.forEach((q2) => q2.classList.toggle('on', q2 === b)); draw(); }));
  layout(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ===============================================================================================================
// Part 3 · Wavelets, from first principles
// ===============================================================================================================

const zooCache = new Map();
const loadZoo = (url) => { if (!zooCache.has(url)) zooCache.set(url, fetch(url).then((r) => r.json())); return zooCache.get(url); };

// Periodized orthogonal DWT with low-pass h (sum sqrt 2) and high-pass g[n] = (-1)^n h[L-1-n].
// a[k] = sum_n h[n] x[(2k + n - off) mod N], d[k] = sum_n g[n] x[(2k + n - off) mod N]; the inverse is the transpose.
// off = 0 is the textbook alignment; off = L/2 - 1 reproduces pywt mode='periodization' exactly (checks.py).
export function dwtStep(x, h, off = 0) {
  const N = x.length, L = h.length, a = new Float64Array(N / 2), d = new Float64Array(N / 2);
  for (let k = 0; k < N / 2; k++) {
    let sa = 0, sd = 0;
    for (let n = 0; n < L; n++) { const v = x[(((2 * k + n - off) % N) + N) % N]; sa += h[n] * v; sd += (n % 2 ? -1 : 1) * h[L - 1 - n] * v; }
    a[k] = sa; d[k] = sd;
  }
  return [a, d];
}
export function idwtStep(a, d, h, off = 0) {
  const N = 2 * a.length, L = h.length, x = new Float64Array(N);
  for (let k = 0; k < a.length; k++) for (let n = 0; n < L; n++) x[(((2 * k + n - off) % N) + N) % N] += h[n] * a[k] + (n % 2 ? -1 : 1) * h[L - 1 - n] * d[k];
  return x;
}
export function wavedec(x, h, levels) {   // returns [a_J, d_J, ..., d_1] (coarsest first), like pywt
  let a = Float64Array.from(x); const ds = [];
  for (let j = 0; j < levels; j++) { const [na, d] = dwtStep(a, h); ds.unshift(d); a = na; }
  return [a, ...ds];
}
export function waverec(c, h) {
  let a = c[0];
  for (let j = 1; j < c.length; j++) a = idwtStep(a, c[j], h);
  return a;
}

// F12 · Scalogram of the song: an analytic Morlet wavelet transform, computed through the FFT.
export function mountScalogram(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), cIn = root?.querySelector('input[data-nc]'), cOut = root?.querySelector('[data-ncval]'), insight = root?.querySelector('[data-insight]');
  const x = song(), N = SONG_N, Xr = Float64Array.from(x), Xi = new Float64Array(N);
  fft(Xr, Xi);
  const ROWS = 72, COLS = 256, F0 = 20, F1 = 320, freqs = Array.from({ length: ROWS }, (_, r) => F0 * Math.pow(F1 / F0, r / (ROWS - 1)));
  let nc = 8, dpr = 1, G = null, img = null, M = null, pick = [0.92, 250], meta = null;
  const compute = () => {
    M = [];
    for (let r = 0; r < ROWS; r++) {
      const f = freqs[r], sf = f / nc, re = new Float64Array(N), im = new Float64Array(N);
      for (let k = 1; k < N / 2; k++) { const g = Math.exp(-0.5 * ((k - f) / sf) ** 2); re[k] = 2 * Xr[k] * g; im[k] = 2 * Xi[k] * g; }   // analytic: positive frequencies only
      fft(re, im, true);
      const row = new Float64Array(N); for (let i = 0; i < N; i++) row[i] = Math.hypot(re[i], im[i]);
      M.push(row);
    }
    let mx = 0; M.forEach((row) => row.forEach((v) => { mx = Math.max(mx, v); }));
    const cv = document.createElement('canvas'); cv.width = COLS; cv.height = ROWS;
    const id = cv.getContext('2d').createImageData(COLS, ROWS);
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const v = M[r][Math.floor((c * N) / COLS)], db = 20 * Math.log10(v / mx + 1e-9), [R, Gc, B] = cmap(Math.max(0, (db + 50) / 50)), o = 4 * ((ROWS - 1 - r) * COLS + c);
      id.data[o] = R; id.data[o + 1] = Gc; id.data[o + 2] = B; id.data[o + 3] = 255;
    }
    cv.getContext('2d').putImageData(id, 0, 0); img = cv;
    // measurements: chord resolved at t = 0.22? click width at the row nearest 250 Hz?
    const i22 = Math.round(0.22 * N), col = freqs.map((f, r) => [f, M[r][i22]]).filter(([f]) => f >= 30 && f <= 66);
    let peaks = 0; const mm = Math.max(...col.map((p) => p[1]));
    for (let i = 1; i < col.length - 1; i++) if (col[i][1] > col[i - 1][1] && col[i][1] >= col[i + 1][1] && col[i][1] > 0.3 * mm) {
      let lmin = col[i][1], rmin = col[i][1];
      for (let j = i - 1; j >= 0 && col[j][1] <= col[i][1]; j--) lmin = Math.min(lmin, col[j][1]);
      for (let j = i + 1; j < col.length && col[j][1] <= col[i][1]; j++) rmin = Math.min(rmin, col[j][1]);
      if (col[i][1] - Math.max(lmin, rmin) > 0.05 * mm) peaks++;
    }
    const r250 = freqs.reduce((b, f, r) => (Math.abs(f - 250) < Math.abs(freqs[b] - 250) ? r : b), 0), row = M[r250];
    let pk = 0, ip = 0; for (let i = Math.round(0.87 * N); i < Math.round(0.97 * N); i++) if (row[i] > pk) { pk = row[i]; ip = i; }
    let lo = ip, hi = ip; while (lo > 0 && row[lo] > pk / 2) lo--; while (hi < N - 1 && row[hi] > pk / 2) hi++;
    meta = { resolved: peaks >= 2, clickMs: ((hi - lo) / N) * 1000, f250: freqs[r250] };
  };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width; canvas.style.height = `${w >= 600 ? 380 : 400}px`; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr;
    G = { s: box1d({ x: pad + 30 * dpr, y: 20 * dpr, w: W - 2 * pad - 30 * dpr, h: 0.28 * H - 26 * dpr }, 0, 1, -2, 2),
          m: { x: pad + 30 * dpr, y: 0.28 * H + 18 * dpr, w: W - 2 * pad - 30 * dpr, h: 0.72 * H - 44 * dpr } };
  };
  const yOf = (f) => G.m.y + G.m.h * (1 - Math.log(f / F0) / Math.log(F1 / F0));
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height, { s, m } = G, narrow = canvas.getBoundingClientRect().width < 600;
    ctx.clearRect(0, 0, W, H);
    const ts = Array.from({ length: N }, (_, i) => i / N);
    axisLine(ctx, c, s, 0, dpr); polyline(ctx, s, ts, Array.from(x), `rgba(${c.ink}, 0.45)`, 1, dpr);
    // the chosen atom, real part, drawn on the song
    const [b, f] = pick, st = nc / (2 * Math.PI * f), ax = [], ay = [];
    for (let t = Math.max(0, b - 4 * st); t <= Math.min(1, b + 4 * st); t += 0.0004) { ax.push(t); ay.push(1.6 * Math.exp(-((t - b) ** 2) / (2 * st * st)) * Math.cos(2 * Math.PI * f * (t - b))); }
    if (ax.length > 1) polyline(ctx, s, ax, ay, `rgb(${c.dot})`, 1.8, dpr);
    label(ctx, c, narrow ? 'the song; amber: one wavelet' : 'the song, and (amber) the wavelet you picked: same number of wiggles at every scale', s.x, s.y - 5 * dpr, dpr, { size: 11 });
    if (img) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, m.x, m.y, m.w, m.h); }
    ctx.strokeStyle = c.line; ctx.strokeRect(m.x, m.y, m.w, m.h);
    for (const hz of [20, 40, 80, 160, 320]) label(ctx, c, `${hz}`, m.x - 6 * dpr, yOf(hz) + 4 * dpr, dpr, { size: 10, alpha: 0.55, align: 'right' });
    for (const t of [0, 0.25, 0.5, 0.75, 1]) label(ctx, c, `${t}`, m.x + t * m.w, m.y + m.h + 13 * dpr, dpr, { size: 10, alpha: 0.55, align: 'center' });
    label(ctx, c, narrow ? 'scalogram (log Hz) — tap it' : 'scalogram |W(t, f)|, frequency on a log scale: click to pick a wavelet', m.x, m.y - 5 * dpr, dpr, { size: 11 });
    ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.arc(m.x + b * m.w, yOf(f), 6 * dpr, 0, 2 * Math.PI); ctx.stroke();
    if (insight) insight.innerHTML = `<strong>${nc} cycles per wavelet.</strong> The window shrinks as the frequency grows: at ${f.toFixed(0)} Hz this wavelet lasts about ${(2 * st * 1000).toFixed(0)} ms (two standard deviations). In one picture the chord's two tones, 16 Hz apart, are ${meta.resolved ? '<em>resolved</em>' : '<em>merged</em>'}, and the click at ${meta.f250.toFixed(0)} Hz is ${meta.clickMs.toFixed(0)} ms wide (half height). The spectrogram of Part 2 needed a 256-sample window to separate the chord, and then smeared the click over about 63 ms.`;
  };
  canvas.addEventListener('pointerdown', (e) => {
    const r = canvas.getBoundingClientRect(), X = (e.clientX - r.left) * dpr, Y = (e.clientY - r.top) * dpr, { m } = G;
    if (X < m.x || X > m.x + m.w || Y < m.y || Y > m.y + m.h) return;
    pick = [(X - m.x) / m.w, F0 * Math.pow(F1 / F0, 1 - (Y - m.y) / m.h)]; draw();
  });
  cIn?.addEventListener('input', () => { nc = Number(cIn.value); if (cOut) cOut.textContent = String(nc); compute(); draw(); });
  if (cIn) { nc = Number(cIn.value); if (cOut) cOut.textContent = String(nc); }
  layout(); compute(); draw();
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F15 · Wavelet zoo and the cascade algorithm (shapes precomputed with PyWavelets in plans/blog/spectral/zoo.py).
export function mountZoo(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), wBtns = [...(root?.querySelectorAll('[data-wv]') ?? [])], lIn = root?.querySelector('input[data-lev]'), lOut = root?.querySelector('[data-levval]'), insight = root?.querySelector('[data-insight]');
  const LEVELS = ['1', '2', '3', '4', '5', '6', '8'];
  let Z = null, name = 'db2', li = 6, dpr = 1, G = null;
  const layout = () => {
    const w = canvas.getBoundingClientRect().width, wide = w >= 600; canvas.style.height = `${wide ? 300 : 460}px`; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr;
    G = wide ? { a: { x: pad, y: 22 * dpr, w: 0.62 * W - 2 * pad, h: H - 46 * dpr }, b: { x: 0.62 * W + pad, y: 22 * dpr, w: 0.38 * W - 2 * pad, h: H - 46 * dpr } }
             : { a: { x: pad, y: 22 * dpr, w: W - 2 * pad, h: 0.6 * H - 40 * dpr }, b: { x: pad, y: 0.6 * H + 18 * dpr, w: W - 2 * pad, h: 0.4 * H - 44 * dpr } };
  };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!Z) { label(ctx, c, 'loading…', 12 * dpr, 30 * dpr, dpr); return; }
    const z = Z[name], S = z.shapes[LEVELS[li]], xs = S.x;
    const lo = Math.min(...S.phi, ...S.psi), hi = Math.max(...S.phi, ...S.psi), A = box1d(G.a, xs[0], xs[xs.length - 1], lo - 0.1, hi + 0.1);
    axisLine(ctx, c, A, 0, dpr);
    polyline(ctx, A, xs, S.phi, `rgb(${c.a1})`, 2, dpr); polyline(ctx, A, xs, S.psi, `rgb(${c.dot})`, 2, dpr);
    label(ctx, c, `${z.label}: φ (teal), ψ (amber), after ${LEVELS[li]} cascade step${LEVELS[li] === '1' ? '' : 's'}`, G.a.x, G.a.y - 6 * dpr, dpr, { size: 11 });
    // |H| and |G| on [0, pi]
    const h = z.h, om = Array.from({ length: 200 }, (_, i) => (Math.PI * i) / 199);
    const Hm = (o) => { let re = 0, im = 0; h.forEach((v, k) => { re += v * Math.cos(k * o); im -= v * Math.sin(k * o); }); return Math.hypot(re, im); };
    const B = box1d(G.b, 0, Math.PI, 0, 1.55);
    axisLine(ctx, c, B, 0, dpr);
    polyline(ctx, B, om, om.map(Hm), `rgb(${c.a1})`, 2, dpr); polyline(ctx, B, om, om.map((o) => Hm(o + Math.PI)), `rgb(${c.dot})`, 2, dpr);
    label(ctx, c, '|H| (teal), |G| (amber), ω from 0 to π', G.b.x, G.b.y - 6 * dpr, dpr, { size: 11 });
    label(ctx, c, '0', B.X(0), B.y + B.h + 13 * dpr, dpr, { size: 10, alpha: 0.55, align: 'center' }); label(ctx, c, 'π', B.X(Math.PI), B.y + B.h + 13 * dpr, dpr, { size: 10, alpha: 0.55, align: 'center' });
    if (insight) {
      const taps = name === 'bior4.4' ? '9 analysis and 7 synthesis taps' : `${h.length} taps`;
      const notes = {
        haar: 'Discontinuous: one vanishing moment. The only compactly supported orthonormal wavelet that is symmetric.',
        db2: 'The shortest filter with two vanishing moments. Its scaling function is continuous but rough (Hölder exponent about 0.55), with corners at every scale; the jagged, self-similar shape is real, not a plotting artefact.',
        db4: 'Four vanishing moments from eight taps. Smoother than D4, still lopsided.',
        sym4: 'Same length and moments as db4, chosen to be as close to symmetric as an orthonormal wavelet can be.',
        coif2: 'Coiflet: vanishing moments for φ as well as ψ, so the scaling coefficients approximate samples of the signal.',
        'bior4.4': 'Biorthogonal: two different filter pairs for analysis and synthesis, which buys exact symmetry (linear phase). This is the 9/7 pair of JPEG 2000.',
      };
      insight.innerHTML = `<strong>${z.label}.</strong> ${taps}, ${z.moments} vanishing moment${z.moments === 1 ? '' : 's'}, ${z.orthogonal ? 'orthonormal' : 'not orthonormal'}, ${z.symmetry}. ${notes[name]} Slide the steps: the cascade algorithm starts from a box and applies the two-scale equation again and again.`;
    }
  };
  wBtns.forEach((b) => b.addEventListener('click', () => { name = b.dataset.wv; wBtns.forEach((q) => q.classList.toggle('on', q === b)); draw(); }));
  lIn?.addEventListener('input', () => { li = Number(lIn.value); if (lOut) lOut.textContent = LEVELS[li]; draw(); });
  if (lIn) { li = Number(lIn.value); if (lOut) lOut.textContent = LEVELS[li]; }
  layout(); draw();
  loadZoo(canvas.dataset.src).then((j) => { Z = j; draw(); });
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F14 · Vanishing moments and shift variance. A periodic signal with smooth stretches, a triangle bump (kinks)
// and a box (jumps); detail coefficients at four levels for Haar, D4, D8; a shift slider.
export function mountMoments(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), wBtns = [...(root?.querySelectorAll('[data-mw]') ?? [])], sIn = root?.querySelector('input[data-ms]'), sOut = root?.querySelector('[data-msval]'), insight = root?.querySelector('[data-insight]');
  const N = 256, base = Array.from({ length: N }, (_, i) => { const t = i / N; return 0.5 * Math.sin(2 * Math.PI * t) + 1.2 * Math.max(0, 0.15 - Math.abs(t - 0.35)) * 4 + (t > 0.7 && t < 0.85 ? 0.6 : 0); });
  let Z = null, name = 'db2', s = 0, dpr = 1, G = null;
  const layout = () => {
    canvas.style.height = '380px'; dpr = fit(canvas);
    const W = canvas.width, H = canvas.height, pad = 12 * dpr, top = 0.26 * H;
    G = { sig: box1d({ x: pad, y: 20 * dpr, w: W - 2 * pad, h: top - 26 * dpr }, 0, N, -0.7, 1.4), rows: [0, 1, 2, 3].map((j) => ({ x: pad, y: top + 14 * dpr + j * ((H - top - 20 * dpr) / 4), w: W - 2 * pad, h: (H - top - 20 * dpr) / 4 - 16 * dpr })) };
  };
  const coeffs = (h, shift) => { const x = Float64Array.from({ length: N }, (_, i) => base[(i - shift + N) % N]); return wavedec(x, h, 4); };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const xs = Array.from({ length: N }, (_, i) => i), sig = xs.map((i) => base[(i - s + N) % N]);
    axisLine(ctx, c, G.sig, 0, dpr); polyline(ctx, G.sig, xs, sig, `rgb(${c.ink})`, 1.6, dpr);
    label(ctx, c, 'signal: smooth, a bump with three kinks, a box with two jumps', G.sig.x, G.sig.y - 5 * dpr, dpr, { size: 11 });
    if (!Z) return;
    const h = Z[name].h, L = h.length, C = coeffs(h, s), ds = C.slice(1).reverse();   // ds[0] = finest level
    let mx = 0; ds.forEach((d) => d.forEach((v) => { mx = Math.max(mx, Math.abs(v)); }));
    const tiny = 0.01;
    ds.forEach((d, j) => {
      const r = G.rows[j], lev = j + 1, base0 = r.y + r.h;
      ctx.strokeStyle = `rgba(${c.ink}, 0.15)`; ctx.beginPath(); ctx.moveTo(r.x, base0); ctx.lineTo(r.x + r.w, base0); ctx.stroke();
      d.forEach((v, k) => {
        const pos = ((2 ** lev * (k + (L - 1) / 2) - (L - 1) / 2) % N + N) % N, X = r.x + (pos / N) * r.w, hh = (Math.abs(v) / mx) * r.h;
        ctx.strokeStyle = Math.abs(v) > tiny ? `rgb(${c.dot})` : `rgba(${c.ink}, 0.25)`; ctx.lineWidth = Math.max(1, 2.4 - 0.4 * j) * dpr;
        ctx.beginPath(); ctx.moveTo(X, base0); ctx.lineTo(X, base0 - Math.max(hh, 0.5 * dpr)); ctx.stroke();
      });
      label(ctx, c, `level ${lev}: ${d.length} details`, r.x, r.y + 2 * dpr, dpr, { size: 10.5, alpha: 0.6 });
    });
    if (insight) {
      const countFor = (nm) => coeffs(Z[nm].h, s).slice(1).reduce((n, d) => n + d.filter((v) => Math.abs(v) > tiny).length, 0);
      const total = ds.reduce((n, d) => n + d.length, 0), cH = countFor('haar'), c4 = countFor('db2'), c8 = countFor('db4');
      const e1 = (sh) => { const d1 = coeffs(h, sh)[4]; let e = 0; d1.forEach((v) => { e += v * v; }); return e; };
      const p = Z[name].moments;
      insight.innerHTML = `<strong>${Z[name].label}, ${p} vanishing moment${p === 1 ? '' : 's'}.</strong> Detail coefficients larger than ${tiny}, out of ${total}: Haar ${cH}, D4 ${c4}, D8 ${c8}. On smooth stretches more moments mean smaller coefficients (Haar leaves many there; D4 and D8 almost none). Near the kinks and jumps it goes the other way: a longer filter overlaps each singularity at more positions, so D8 is not automatically sparser than D4. That is the trade-off between vanishing moments and support length. Shift the signal by one sample: the finest-level energy goes from ${e1(s).toFixed(3)} to ${e1(s + 1).toFixed(3)}; the coefficients change, not just move (Section 14).`;
    }
  };
  wBtns.forEach((b) => b.addEventListener('click', () => { name = b.dataset.mw; wBtns.forEach((q) => q.classList.toggle('on', q === b)); draw(); }));
  sIn?.addEventListener('input', () => { s = Number(sIn.value); if (sOut) sOut.textContent = String(s); draw(); });
  layout(); draw();
  loadZoo(canvas.dataset.src).then((j) => { Z = j; draw(); });
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F16 · Denoising a 1-D signal: wavelet thresholding (soft or hard, universal threshold, sigma from the median
// absolute deviation) against the best Gaussian blur (chosen with the truth, so an oracle).
export function mountDenoise(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), mBtns = [...(root?.querySelectorAll('[data-dn]') ?? [])], nIn = root?.querySelector('input[data-sig]'), nOut = root?.querySelector('[data-sigval]'), insight = root?.querySelector('[data-insight]');
  const N = 1024, truth = Float64Array.from({ length: N }, (_, i) => {
    const t = i / N;
    return (t < 0.45 ? 0.7 * Math.sin(6 * Math.PI * t) * Math.exp(-((t - 0.22) ** 2) / 0.02) : 0) + (t >= 0.5 && t < 0.62 ? 1.0 : 0) + (t >= 0.62 && t < 0.8 ? -0.5 : 0) + (t >= 0.8 ? 0.35 : 0) + 0.8 * Math.exp(-(((t - 0.9) / 0.006) ** 2));
  });
  const gauss = (() => { let st = 12345; const u = () => { st = (st * 1664525 + 1013904223) >>> 0; return (st + 0.5) / 4294967296; }; return Array.from({ length: N }, () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())); })();
  let Z = null, method = 'soft', sigma = 0.12, dpr = 1, G = null;
  const rmse = (a) => { let s = 0; for (let i = 0; i < N; i++) s += (a[i] - truth[i]) ** 2; return Math.sqrt(s / N); };
  const blur = (y, sd) => { const r = Math.ceil(3 * sd), k = []; let s = 0; for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sd * sd)); k.push(v); s += v; } const out = new Float64Array(N); for (let i = 0; i < N; i++) { let a = 0; for (let j = -r; j <= r; j++) a += k[j + r] * y[(i + j + N) % N]; out[i] = a / s; } return out; };
  const estimate = (y, m) => {
    if (m === 'blur') { let best = null; for (let q = 0; q < 24; q++) { const sd = 0.5 * Math.pow(60, q / 23), e = blur(y, sd), r = rmse(e); if (!best || r < best.r) best = { e, r, sd }; } return best; }
    const h = Z.sym4.h, c = wavedec(y, h, 6), d1 = Array.from(c[c.length - 1]).map(Math.abs).sort((a, b) => a - b);
    const sh = d1[Math.floor(d1.length / 2)] / 0.6745, thr = sh * Math.sqrt(2 * Math.log(N));
    const cc = c.map((band, i) => (i === 0 ? band : band.map((v) => (m === 'hard' ? (Math.abs(v) > thr ? v : 0) : Math.sign(v) * Math.max(0, Math.abs(v) - thr)))));
    const e = waverec(cc, h); return { e, r: rmse(e), thr, sh, kept: cc.slice(1).reduce((n, b) => n + b.filter((v) => v !== 0).length, 0) };
  };
  const layout = () => { canvas.style.height = '300px'; dpr = fit(canvas); const W = canvas.width, H = canvas.height, pad = 12 * dpr; G = box1d({ x: pad, y: 20 * dpr, w: W - 2 * pad, h: H - 36 * dpr }, 0, N, -1.2, 1.6); };
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!Z) return;
    const y = Float64Array.from(truth, (v, i) => v + sigma * gauss[i]), xs = Array.from({ length: N }, (_, i) => i);
    polyline(ctx, G, xs, Array.from(y), `rgba(${c.ink}, 0.22)`, 1, dpr);
    polyline(ctx, G, xs, Array.from(truth), `rgba(${c.ink}, 0.8)`, 1.2, dpr, [4, 3]);
    const res = estimate(y, method); polyline(ctx, G, xs, Array.from(res.e), `rgb(${c.a1})`, 2, dpr);
    label(ctx, c, 'noisy (grey), truth (dashed), estimate (teal)', G.x, G.y - 5 * dpr, dpr, { size: 11 });
    if (insight) {
      const soft = estimate(y, 'soft'), hard = estimate(y, 'hard'), bl = estimate(y, 'blur');
      const head = { soft: 'Wavelet, soft threshold.', hard: 'Wavelet, hard threshold.', blur: 'Gaussian blur, best width.' }[method];
      const best = [['soft thresholding', soft.r], ['hard thresholding', hard.r], ['the best blur', bl.r]].sort((p, q) => p[1] - q[1])[0][0];
      insight.innerHTML = `<strong>${head}</strong> Noise σ = ${sigma.toFixed(2)}; the median rule estimates ${soft.sh.toFixed(3)}, giving the threshold ${soft.thr.toFixed(3)} = σ̂·√(2 ln ${N}). Error (RMS): soft ${soft.r.toFixed(3)}, hard ${hard.r.toFixed(3)}, best Gaussian blur ${bl.r.toFixed(3)} (width ${bl.sd.toFixed(1)} samples, chosen by peeking at the truth, which no real method can do). Lowest here: ${best}. ${soft.kept} of ${N - N / 64} wavelet details survive the threshold. The universal threshold is built to remove essentially all noise, so it errs toward over-smoothing, and soft thresholding also shrinks the survivors; hard thresholding keeps them intact.`;
    }
  };
  mBtns.forEach((b) => b.addEventListener('click', () => { method = b.dataset.dn; mBtns.forEach((q) => q.classList.toggle('on', q === b)); draw(); }));
  nIn?.addEventListener('input', () => { sigma = Number(nIn.value); if (nOut) nOut.textContent = sigma.toFixed(2); draw(); });
  if (nIn) { sigma = Number(nIn.value); if (nOut) nOut.textContent = sigma.toFixed(2); }
  layout(); draw();
  loadZoo(canvas.dataset.src).then((j) => { Z = j; draw(); });
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// F-pyramid · The classic picture: an image split into an approximation and three detail bands, then the
// approximation split again. Real 2-D periodized DWT of the Eye sketch (rows, then columns), animated.
export function mountPyramid(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), q = (s) => root?.querySelector(s);
  const lIn = q('input[data-plev]'), lOut = q('[data-plevval]'), playBtn = q('[data-pplay]'), insight = q('[data-insight]');
  const wBtns = [...(root?.querySelectorAll('[data-pw]') ?? [])];
  const n = N2, MAXL = 4;
  let Z = null, img = null, wname = 'haar', stats = [], pos = 0, playing = !reduceMotion(), visible = false, raf = 0, last = 0, dpr = 1, side = 0, hold = 0;
  const cvs = Array.from({ length: MAXL + 1 }, () => offscreen(n));

  const transform = () => {   // cvs[L]: display mosaic after L levels; stats[L]: energy and sparsity
    const h = Z[wname].h, C = Float64Array.from(img), total = C.reduce((a, v) => a + v * v, 0);
    const disp = new Float64Array(n * n), bands = [];             // bands: [x0, y0, size]
    const render = (L) => {
      const s = n >> L;
      for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) disp[y * n + x] = C[y * n + x] / 2 ** L;   // approximation: gain 2 per level
      bands.forEach(([x0, y0, sz]) => {
        const vals = []; for (let y = 0; y < sz; y++) for (let x = 0; x < sz; x++) vals.push(Math.abs(C[(y0 + y) * n + x0 + x]));
        vals.sort((a, b) => a - b); const top = vals[Math.floor(0.999 * (vals.length - 1))] || 1;
        for (let y = 0; y < sz; y++) for (let x = 0; x < sz; x++) disp[(y0 + y) * n + x0 + x] = Math.pow(Math.min(1, Math.abs(C[(y0 + y) * n + x0 + x]) / top), 1.1);
      });
      putPixels(cvs[L], n, (i) => { const g = 255 * Math.min(1, Math.max(0, disp[i])); return [g, g, g]; });
      let eA = 0; for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) eA += C[y * n + x] ** 2;
      const det = []; for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (y >= s || x >= s) det.push(C[y * n + x] ** 2);
      det.sort((a, b) => b - a); const eD = det.reduce((a, b) => a + b, 0); let acc = 0, k = 0; while (k < det.length && acc < 0.9 * eD) acc += det[k++];
      stats[L] = { eA: eA / total, k99: det.length ? k / det.length : 0, nDet: det.length, size: s };
    };
    render(0);
    for (let L = 1; L <= MAXL; L++) {
      const s = n >> (L - 1), hs = s >> 1, line = new Float64Array(s);
      for (let y = 0; y < s; y++) { for (let x = 0; x < s; x++) line[x] = C[y * n + x]; const [a, d] = dwtStep(line, h); for (let x = 0; x < hs; x++) { C[y * n + x] = a[x]; C[y * n + hs + x] = d[x]; } }
      for (let x = 0; x < s; x++) { for (let y = 0; y < s; y++) line[y] = C[y * n + x]; const [a, d] = dwtStep(line, h); for (let y = 0; y < hs; y++) { C[y * n + x] = a[y]; C[(hs + y) * n + x] = d[y]; } }
      bands.push([hs, 0, hs], [0, hs, hs], [hs, hs, hs]);
      render(L);
    }
  };
  const layout = () => { const w = canvas.getBoundingClientRect().width, sz = Math.min(w, 520); canvas.style.height = `${sz + 8}px`; dpr = fit(canvas); side = sz * dpr; };
  const ease = (t) => t * t * (3 - 2 * t);
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!img || !stats.length) { label(ctx, c, 'loading…', 12 * dpr, 30 * dpr, dpr); return; }
    const x0 = (W - side) / 2, y0 = 4 * dpr, sc = side / n, L = Math.min(MAXL, Math.floor(pos)), e = ease(pos - L);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(cvs[L], x0, y0, side, side);
    if (e > 0 && L < MAXL) {
      const s = (n >> L) * sc;   // region of the current approximation
      ctx.globalAlpha = e; ctx.drawImage(cvs[L + 1], 0, 0, n >> L, n >> L, x0, y0, s, s); ctx.globalAlpha = 1;   // new bands fade in
      const ss = s * (1 - e / 2);
      ctx.drawImage(cvs[L], 0, 0, n >> L, n >> L, x0, y0, ss, ss);                                                // old approximation shrinks
    }
    // band boundaries
    const levels = L + (e > 0.5 && L < MAXL ? 1 : 0);
    ctx.strokeStyle = `rgba(${c.a1}, 0.9)`; ctx.lineWidth = 1.5 * dpr;
    for (let j = 1; j <= levels; j++) { const s = (n >> (j - 1)) * sc, hs = s / 2; ctx.beginPath(); ctx.moveTo(x0 + hs, y0); ctx.lineTo(x0 + hs, y0 + s); ctx.moveTo(x0, y0 + hs); ctx.lineTo(x0 + s, y0 + hs); ctx.stroke(); }
    ctx.strokeStyle = c.line; ctx.strokeRect(x0, y0, side, side);
    if (levels >= 1 && e === 0) {
      const hs = side / 2, tag = (t, x, y) => { ctx.font = `${10.5 * dpr}px ui-monospace, monospace`; const w = ctx.measureText(t).width + 8 * dpr; ctx.fillStyle = 'rgba(14,13,20,0.7)'; ctx.fillRect(x, y, w, 16 * dpr); ctx.fillStyle = '#ecebf5'; ctx.fillText(t, x + 4 * dpr, y + 12 * dpr); };
      tag('vertical edges', x0 + hs + 6 * dpr, y0 + 6 * dpr); tag('horizontal edges', x0 + 6 * dpr, y0 + hs + 6 * dpr); tag('diagonal', x0 + hs + 6 * dpr, y0 + hs + 6 * dpr);
    }
    if (insight && stats[L]) {
      const st = stats[L];
      insight.innerHTML = L === 0
        ? `<strong>The Eye, 256 × 256 pixels.</strong> Press play, or slide the levels. Each level splits the current approximation into a half-size approximation (top-left) and three detail images.`
        : `<strong>${L} level${L > 1 ? 's' : ''}.</strong> One ${st.size} × ${st.size} approximation and ${3 * L} detail bands: still exactly 65,536 numbers, an orthonormal change of basis. The approximation holds ${(100 * st.eA).toFixed(2)}% of the energy. 90% of the remaining detail energy sits in just ${(100 * st.k99).toFixed(1)}% of the ${st.nDet.toLocaleString()} detail coefficients, mostly along the edges; much of the rest is the grain of the pencil and paper. That sparsity is what compression and denoising exploit. (Details are shown as magnitudes, scaled per band.)`;
    }
  };
  const loop = (now) => {
    raf = 0; if (!playing || !visible) return;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0; last = now;
    if (hold > 0) hold -= dt;
    else { const prev = Math.floor(pos); pos += dt / 1.1; if (Math.floor(pos) > prev) { pos = Math.floor(pos); hold = pos >= MAXL ? 2.2 : 0.7; } if (pos >= MAXL && hold <= 0) { pos = 0; hold = 0.8; } }
    if (pos > MAXL) pos = MAXL;
    if (lIn) { lIn.value = String(Math.floor(pos)); if (lOut) lOut.textContent = String(Math.floor(pos)); }
    draw(); raf = requestAnimationFrame(loop);
  };
  const kick = () => { if (playing && visible && !raf) { last = 0; raf = requestAnimationFrame(loop); } };
  const setPlay = (p) => { playing = p; if (playBtn) playBtn.textContent = p ? 'Pause' : 'Play'; kick(); draw(); };
  lIn?.addEventListener('input', () => { pos = Number(lIn.value); if (lOut) lOut.textContent = lIn.value; setPlay(false); });
  playBtn?.addEventListener('click', () => { if (!playing && pos >= MAXL) pos = 0; setPlay(!playing); });
  wBtns.forEach((b) => b.addEventListener('click', () => { wname = b.dataset.pw; wBtns.forEach((q2) => q2.classList.toggle('on', q2 === b)); if (Z && img) { transform(); draw(); } }));
  layout(); draw();
  Promise.all([loadZoo(canvas.dataset.src), loadGrey(`${canvas.dataset.base}eye.png`, n)]).then(([z, im]) => { Z = z; img = im; transform(); if (!playing) pos = 2; setPlay(playing); });
  observeVisible(canvas, (v) => { visible = v; kick(); });
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// Part 4 · Wavelets for images. Shared 2-D numerics.
// ---------------------------------------------------------------------------------------------------------------

const PYWT_OFF = (h) => h.length / 2 - 1;   // filter alignment of pywt mode='periodization'

// One level of the separable 2-D DWT on the top-left s x s block of the n x n array C, in place: rows, then columns.
// Layout as pywt.coeffs_to_array: approximation top-left, cV (left - right) top-right, cH (top - bottom) bottom-left,
// cD bottom-right. With PYWT_OFF the numbers equal pywt.wavedec2(..., mode='periodization').
function dwt2Level(C, n, s, h) {
  const hs = s >> 1, line = new Float64Array(s), off = PYWT_OFF(h);
  for (let y = 0; y < s; y++) { for (let x = 0; x < s; x++) line[x] = C[y * n + x]; const [a, d] = dwtStep(line, h, off); for (let x = 0; x < hs; x++) { C[y * n + x] = a[x]; C[y * n + hs + x] = d[x]; } }
  for (let x = 0; x < s; x++) { for (let y = 0; y < s; y++) line[y] = C[y * n + x]; const [a, d] = dwtStep(line, h, off); for (let y = 0; y < hs; y++) { C[y * n + x] = a[y]; C[(hs + y) * n + x] = d[y]; } }
}
function idwt2Level(C, n, s, h) {
  const hs = s >> 1, off = PYWT_OFF(h), a = new Float64Array(hs), d = new Float64Array(hs);
  for (let x = 0; x < s; x++) { for (let y = 0; y < hs; y++) { a[y] = C[y * n + x]; d[y] = C[(hs + y) * n + x]; } const r = idwtStep(a, d, h, off); for (let y = 0; y < s; y++) C[y * n + x] = r[y]; }
  for (let y = 0; y < s; y++) { for (let x = 0; x < hs; x++) { a[x] = C[y * n + x]; d[x] = C[y * n + hs + x]; } const r = idwtStep(a, d, h, off); for (let x = 0; x < s; x++) C[y * n + x] = r[x]; }
}
export function wavedec2(img, n, h, L) { const C = Float64Array.from(img); for (let j = 0; j < L; j++) dwt2Level(C, n, n >> j, h); return C; }
export function waverec2(C0, n, h, L) { const C = Float64Array.from(C0); for (let j = L - 1; j >= 0; j--) idwt2Level(C, n, n >> j, h); return C; }
function detailBands(n, L) {   // [{ j, name, x0, y0, s }], j = 1 finest (the library's level numbering)
  const out = [];
  for (let j = 1; j <= L; j++) { const hs = n >> j; out.push({ j, name: 'cV', x0: hs, y0: 0, s: hs }, { j, name: 'cH', x0: 0, y0: hs, s: hs }, { j, name: 'cD', x0: hs, y0: hs, s: hs }); }
  return out;
}

// Orthonormal 8 x 8 block DCT-II (scipy.fft.dctn(..., norm='ortho') on each block), the transform inside JPEG.
const DCT8 = (() => { const M = new Float64Array(64); for (let u = 0; u < 8; u++) for (let x = 0; x < 8; x++) M[u * 8 + x] = (u ? Math.sqrt(2 / 8) : Math.sqrt(1 / 8)) * Math.cos(((2 * x + 1) * u * Math.PI) / 16); return M; })();
function dct8x8(img, n, inverse = false) {
  const out = new Float64Array(n * n), B = new Float64Array(64), R = new Float64Array(64);
  const P = (i, j) => (inverse ? DCT8[j * 8 + i] : DCT8[i * 8 + j]);   // out = P B P^T
  for (let by = 0; by < n; by += 8) for (let bx = 0; bx < n; bx += 8) {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) B[y * 8 + x] = img[(by + y) * n + bx + x];
    for (let y = 0; y < 8; y++) for (let v = 0; v < 8; v++) { let a = 0; for (let x = 0; x < 8; x++) a += B[y * 8 + x] * P(v, x); R[y * 8 + v] = a; }
    for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) { let a = 0; for (let y = 0; y < 8; y++) a += P(u, y) * R[y * 8 + v]; out[(by + u) * n + bx + v] = a; }
  }
  return out;
}
const psnr = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return 10 * Math.log10(a.length / s); };
const paintGrey = (cv, n, v) => putPixels(cv, n, (i) => { const g = 255 * Math.min(1, Math.max(0, v[i])); return [g, g, g]; });
function seededNormals(len, seed) {
  let st = seed >>> 0; const u = () => { st = (st * 1664525 + 1013904223) >>> 0; return (st + 0.5) / 4294967296; };
  const out = new Float64Array(len);
  for (let i = 0; i < len; i += 2) { const r = Math.sqrt(-2 * Math.log(u())), t = 2 * Math.PI * u(); out[i] = r * Math.cos(t); if (i + 1 < len) out[i + 1] = r * Math.sin(t); }
  return out;
}

// Two image panels side by side (stacked on a phone), each with a title line; click to zoom both on the same spot.
function twoPanels(canvas, n) {
  const P = { dpr: 1, side: 0, at: [], wide: true, zoom: null, Z: 64 };
  P.layout = () => {
    const w = canvas.getBoundingClientRect().width, wide = w >= 540, gap = 16, top = 22;
    const side = Math.floor(Math.min(wide ? (w - gap) / 2 : w, 300));
    canvas.style.height = `${wide ? side + top + 6 : 2 * (side + top) + 14}px`;
    P.dpr = fit(canvas); const d = P.dpr, S = side * d, x0 = wide ? (canvas.width - 2 * S - gap * d) / 2 : (canvas.width - S) / 2;
    P.side = S; P.wide = wide;
    P.at = wide ? [{ x: x0, y: top * d }, { x: x0 + S + gap * d, y: top * d }] : [{ x: x0, y: top * d }, { x: x0, y: (2 * top + 8) * d + S }];
  };
  P.draw = (ctx, cv, k) => {
    const { x, y } = P.at[k], S = P.side;
    ctx.imageSmoothingEnabled = !P.zoom;
    if (P.zoom) ctx.drawImage(cv, P.zoom[0], P.zoom[1], P.Z, P.Z, x, y, S, S); else ctx.drawImage(cv, x, y, S, S);
    ctx.imageSmoothingEnabled = true;
  };
  P.hit = (e) => {   // returns image coordinates of a click, or null
    const b = canvas.getBoundingClientRect(), px = (e.clientX - b.left) * P.dpr, py = (e.clientY - b.top) * P.dpr;
    for (const a of P.at) if (px >= a.x && px < a.x + P.side && py >= a.y && py < a.y + P.side) return [((px - a.x) / P.side) * n, ((py - a.y) / P.side) * n];
    return null;
  };
  P.click = (e) => {
    if (P.zoom) { P.zoom = null; return true; }
    const h = P.hit(e); if (!h) return false;
    P.zoom = [Math.max(0, Math.min(n - P.Z, Math.round(h[0] - P.Z / 2))), Math.max(0, Math.min(n - P.Z, Math.round(h[1] - P.Z / 2)))];
    return true;
  };
  return P;
}

// ---------------------------------------------------------------------------------------------------------------
// F20 · Compression: keep the largest k% of the numbers. 2-D wavelets (4 levels, periodized) vs the 8 x 8 block DCT.
// Same rule as checks.py: both transforms are orthonormal, so the error energy is the energy of what is dropped.
// ---------------------------------------------------------------------------------------------------------------

export function mountCompress(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), q = (s) => root?.querySelector(s);
  const imgBtns = [...(root?.querySelectorAll('[data-cimg]') ?? [])], wBtns = [...(root?.querySelectorAll('[data-cw]') ?? [])];
  const kIn = q('input[data-keep]'), kOut = q('[data-keepval]'), insight = q('[data-insight]');
  const KEEP = [0.5, 1, 2, 3, 5, 7, 10, 15, 20], NAMES = { haar: 'Haar', db2: 'D4', db4: 'D8' };
  const n = N2, LEV = 4, base = canvas.dataset.base, P = twoPanels(canvas, n), cvW = offscreen(n), cvD = offscreen(n);
  let Z = null, name = 'eye', wname = 'db2', ki = Number(kIn?.value ?? 2), res = null;
  const cache = new Map();
  const sortAbs = (C) => Float64Array.from(C, Math.abs).sort();
  const get = async () => {
    if (!cache.has(name)) { const img = await loadGrey(`${base}${name}.png`, n), D = dct8x8(img, n); cache.set(name, { img, D, sD: sortAbs(D) }); }
    const key = `${name}:${wname}`;
    if (!cache.has(key)) { const C = wavedec2(cache.get(name).img, n, Z[wname].h, LEV); cache.set(key, { C, sC: sortAbs(C) }); }
    return [cache.get(name), cache.get(key)];
  };
  const compute = async () => {
    if (!Z) return;
    const [I, W] = await get(), K = Math.floor((KEEP[ki] / 100) * n * n), tW = W.sC[n * n - K], tD = I.sD[n * n - K];
    const rw = waverec2(W.C.map((v) => (Math.abs(v) >= tW ? v : 0)), n, Z[wname].h, LEV);
    const rd = dct8x8(I.D.map((v) => (Math.abs(v) >= tD ? v : 0)), n, true);
    paintGrey(cvW, n, rw); paintGrey(cvD, n, rd);
    res = { pw: psnr(I.img, rw), pd: psnr(I.img, rd), K };
    draw();
  };
  const draw = () => {
    const c = theme(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!res) { label(ctx, c, 'loading…', 12 * P.dpr, 30 * P.dpr, P.dpr); return; }
    P.draw(ctx, cvW, 0); P.draw(ctx, cvD, 1);
    const d = P.dpr;
    label(ctx, c, `${NAMES[wname]} wavelets · ${res.pw.toFixed(1)} dB`, P.at[0].x, P.at[0].y - 7 * d, d, { size: 12, alpha: 0.85 });
    label(ctx, c, `8 × 8 DCT · ${res.pd.toFixed(1)} dB`, P.at[1].x, P.at[1].y - 7 * d, d, { size: 12, alpha: 0.85 });
    P.at.forEach((a) => { ctx.strokeStyle = c.line; ctx.lineWidth = d; ctx.strokeRect(a.x, a.y, P.side, P.side); });
    if (insight) {
      const k = KEEP[ki], diff = res.pw - res.pd, wl = NAMES[wname];
      let verdict = Math.abs(diff) < 0.3 ? 'About even.' : diff > 0 ? `The wavelets win by ${diff.toFixed(1)} dB.` : `The DCT wins by ${(-diff).toFixed(1)} dB.`;
      if (diff > 0.3 && k <= 3) verdict += ' At a small budget the DCT has to spend its numbers block by block, and the 8 × 8 grid shows. The wavelet image has no fixed grid: its errors are smudges of many sizes around the edges.';
      if (diff < -0.3 && name === 'sunset') verdict += ' The Sunset is mostly oscillating texture (the waves), and once there is budget to spare the DCT\'s cosines fit it better.';
      insight.innerHTML = `<strong>Keep ${k}% of the numbers</strong> (${res.K.toLocaleString()} of 65,536), set the rest to zero, invert. ${verdict} ${wl === 'Haar' ? 'Haar has one vanishing moment, so smooth shading costs it more numbers than D4 or D8.' : ''} ${P.zoom ? 'Zoomed 4×: click again to zoom out.' : 'Click either image to zoom in on the same spot in both.'} PSNR = 10 log₁₀(1 / mean squared error), higher is better.`;
    }
  };
  canvas.addEventListener('click', (e) => { if (P.click(e)) draw(); });
  kIn?.addEventListener('input', () => { ki = Number(kIn.value); if (kOut) kOut.textContent = `${KEEP[ki]}%`; compute(); });
  if (kOut) kOut.textContent = `${KEEP[ki]}%`;
  imgBtns.forEach((b) => b.addEventListener('click', () => { name = b.dataset.cimg; imgBtns.forEach((x) => x.classList.toggle('on', x === b)); P.zoom = null; compute(); }));
  wBtns.forEach((b) => b.addEventListener('click', () => { wname = b.dataset.cw; wBtns.forEach((x) => x.classList.toggle('on', x === b)); compute(); }));
  P.layout(); draw();
  loadZoo(canvas.dataset.src).then((z) => { Z = z; compute(); });
  new ResizeObserver(() => { P.layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F21 · Denoising an image: oracle Gaussian blur vs wavelet thresholding (universal, BayesShrink, cycle spinning).
// D4, 4 levels, periodized, the same rules as checks.py. The noise is fixed (seeded) and scaled by sigma.
// ---------------------------------------------------------------------------------------------------------------

export function mountDenoise2d(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), q = (s) => root?.querySelector(s);
  const imgBtns = [...(root?.querySelectorAll('[data-dimg]') ?? [])], mBtns = [...(root?.querySelectorAll('[data-dm]') ?? [])];
  const sIn = q('input[data-dsig]'), sOut = q('[data-dsigval]'), insight = q('[data-insight]');
  const n = N2, LEV = 4, base = canvas.dataset.base, P = twoPanels(canvas, n), cvN = offscreen(n), cvE = offscreen(n);
  const noise = seededNormals(n * n, 2026);
  let Z = null, name = 'portrait', method = 'bayes', sigma = Number(sIn?.value ?? 0.1), R = null, timer = 0;
  const TITLES = { blur: 'best Gaussian blur (oracle)', visu: 'universal threshold, soft', visuhard: 'universal threshold, hard', bayes: 'BayesShrink', cycle: 'BayesShrink + 16 shifts' };

  const shrink = (y, h, rule) => {
    const C = wavedec2(y, n, h, LEV), d1 = [];
    for (let yy = n / 2; yy < n; yy++) for (let x = n / 2; x < n; x++) d1.push(Math.abs(C[yy * n + x]));   // finest cD
    d1.sort((a, b) => a - b);
    const sh = (d1[d1.length / 2 - 1] + d1[d1.length / 2]) / 2 / 0.6745, tU = sh * Math.sqrt(2 * Math.log(n * n));
    for (const b of detailBands(n, LEV)) {
      let e2 = 0; for (let yy = 0; yy < b.s; yy++) for (let x = 0; x < b.s; x++) e2 += C[(b.y0 + yy) * n + b.x0 + x] ** 2;
      const t = rule === 'bayes' ? (sh * sh) / Math.sqrt(Math.max(e2 / (b.s * b.s) - sh * sh, 1e-12)) : tU;
      for (let yy = 0; yy < b.s; yy++) for (let x = 0; x < b.s; x++) {
        const i = (b.y0 + yy) * n + b.x0 + x, v = C[i];
        C[i] = rule === 'visuhard' ? (Math.abs(v) > t ? v : 0) : Math.sign(v) * Math.max(0, Math.abs(v) - t);
      }
    }
    return { e: waverec2(C, n, h, LEV), sh, tU };
  };
  const roll = (a, dy, dx) => { const o = new Float64Array(n * n); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) o[((y + dy + n) % n) * n + ((x + dx + n) % n)] = a[y * n + x]; return o; };
  const run = async () => {
    if (!Z) return;
    const img = await loadGrey(`${base}${name}.png`, n), h = Z.db2.h, y = Float64Array.from(img, (v, i) => v + sigma * noise[i]);
    const out = { noisy: { e: y, p: psnr(img, y) } };
    let best = null;
    for (let k = 0; k < 20; k++) { const s = 0.3 * Math.pow(4 / 0.3, k / 19), e = blurGauss(y, n, s), p = psnr(img, e); if (!best || p > best.p) best = { e, p, s }; }
    out.blur = best;
    for (const r of ['visu', 'visuhard', 'bayes']) { const z = shrink(y, h, r); out[r] = { e: z.e, p: psnr(img, z.e), sh: z.sh, tU: z.tU }; }
    const acc = new Float64Array(n * n);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { const e = roll(shrink(roll(y, i, j), h, 'bayes').e, -i, -j); for (let t = 0; t < n * n; t++) acc[t] += e[t] / 16; }
    out.cycle = { e: acc, p: psnr(img, acc) };
    R = out; paintGrey(cvN, n, y); paint();
  };
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 40); };
  const paint = () => { if (R) paintGrey(cvE, n, R[method].e); draw(); };
  const draw = () => {
    const c = theme(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!R) { label(ctx, c, 'denoising…', 12 * P.dpr, 30 * P.dpr, P.dpr); return; }
    const d = P.dpr;
    P.draw(ctx, cvN, 0); P.draw(ctx, cvE, 1);
    label(ctx, c, `noisy, σ = ${sigma.toFixed(2)} · ${R.noisy.p.toFixed(1)} dB`, P.at[0].x, P.at[0].y - 7 * d, d, { size: 12, alpha: 0.85 });
    label(ctx, c, `${TITLES[method]} · ${R[method].p.toFixed(1)} dB`, P.at[1].x, P.at[1].y - 7 * d, d, { size: 12, alpha: 0.85 });
    P.at.forEach((a) => { ctx.strokeStyle = c.line; ctx.lineWidth = d; ctx.strokeRect(a.x, a.y, P.side, P.side); });
    if (insight) {
      const what = {
        blur: `The Gaussian blur whose width (${R.blur.s.toFixed(2)} px) was chosen by comparing with the clean image: an oracle that no real method has. It removes noise everywhere, from the edges too.`,
        visu: `Soft thresholding of every detail band at the universal threshold σ̂√(2 ln N) = ${R.visu.tU.toFixed(3)}, the rule proved in Part 3 (σ̂ = ${R.visu.sh.toFixed(3)} from the median of the finest diagonal band). On a photograph-like image it removes the noise and most of the texture with it.`,
        visuhard: 'The same threshold, but the survivors are kept whole instead of shrunk. Better, still too harsh for an image.',
        bayes: 'A separate soft threshold for every band, σ̂² / σ̂ₓ, where σ̂ₓ estimates how much signal the band holds (Chang, Yu and Vetterli, 2000). Bands rich in signal get a small threshold, bands of mostly noise a large one.',
        cycle: 'BayesShrink applied to 16 shifted copies of the noisy image (shifts of 0 to 3 pixels in each direction), each shifted back and averaged (Coifman and Donoho, 1995). Every copy leaves its artefacts in different places, and averaging cancels them.',
      }[method];
      insight.innerHTML = `<strong>${TITLES[method][0].toUpperCase() + TITLES[method].slice(1)}.</strong> ${what} PSNR in dB, noisy ${R.noisy.p.toFixed(1)}: oracle blur ${R.blur.p.toFixed(1)} · universal soft ${R.visu.p.toFixed(1)} · universal hard ${R.visuhard.p.toFixed(1)} · BayesShrink ${R.bayes.p.toFixed(1)} · + 16 shifts ${R.cycle.p.toFixed(1)}. ${P.zoom ? 'Zoomed 4×: click again to zoom out.' : 'Click an image to zoom in.'}`;
    }
  };
  canvas.addEventListener('click', (e) => { if (P.click(e)) draw(); });
  sIn?.addEventListener('input', () => { sigma = Number(sIn.value); if (sOut) sOut.textContent = sigma.toFixed(2); schedule(); });
  if (sOut) sOut.textContent = sigma.toFixed(2);
  imgBtns.forEach((b) => b.addEventListener('click', () => { name = b.dataset.dimg; imgBtns.forEach((x) => x.classList.toggle('on', x === b)); P.zoom = null; schedule(); }));
  mBtns.forEach((b) => b.addEventListener('click', () => { method = b.dataset.dm; mBtns.forEach((x) => x.classList.toggle('on', x === b)); paint(); }));
  P.layout(); draw();
  let started = false;
  observeVisible(canvas, (v) => { if (v && !started) { started = true; loadZoo(canvas.dataset.src).then((z) => { Z = z; run(); }); } });
  new ResizeObserver(() => { P.layout(); draw(); }).observe(canvas);
  onTheme(draw);
}

// ---------------------------------------------------------------------------------------------------------------
// F22 · Orientation. Where a grating's energy goes in one level of the separable 2-D DWT. For a grating with
// frequency (kx, ky) in cycles per pixel the shares follow from the filter responses alone:
//   approximation |H(wx)|^2 |H(wy)|^2 / 4, cH |H(wx)|^2 |G(wy)|^2 / 4, cV |G(wx)|^2 |H(wy)|^2 / 4, cD |G(wx)|^2 |G(wy)|^2 / 4,
// with w = 2 pi k. Exact for every phase unless both kx and ky are in {0, 1/4, 1/2} (checks.py: 1e-15).
// ---------------------------------------------------------------------------------------------------------------

export function mountOrient(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('figure'), q = (s) => root?.querySelector(s);
  const aIn = q('input[data-oang]'), aOut = q('[data-oangval]'), fIn = q('input[data-ofreq]'), fOut = q('[data-ofreqval]'), insight = q('[data-insight]');
  const wBtns = [...(root?.querySelectorAll('[data-ow]') ?? [])];
  const VIOLET = '139, 92, 246', GREY = '150, 150, 165';
  let Z = null, wname = 'haar', ang = Number(aIn?.value ?? 45), f = Number(fIn?.value ?? 0.375), dpr = 1, G = null;
  const plane = offscreen(128), grat = offscreen(192), PATCH = 24;   // a 24 x 24 pixel patch, drawn smooth at 8 samples per pixel
  const resp = (h, w) => {   // |H(w)|^2, |G(w)|^2 with g[n] = (-1)^n h[L-1-n]
    let hr = 0, hi = 0, gr = 0, gi = 0; const L = h.length;
    for (let k = 0; k < L; k++) { const c = Math.cos(w * k), s = Math.sin(w * k), g = (k % 2 ? -1 : 1) * h[L - 1 - k]; hr += h[k] * c; hi -= h[k] * s; gr += g * c; gi -= g * s; }
    return [hr * hr + hi * hi, gr * gr + gi * gi];
  };
  const shares = (kx, ky) => { const h = Z[wname].h, [Hx, Gx] = resp(h, 2 * Math.PI * kx), [Hy, Gy] = resp(h, 2 * Math.PI * ky); return { A: (Hx * Hy) / 4, H: (Hx * Gy) / 4, V: (Gx * Hy) / 4, D: (Gx * Gy) / 4 }; };
  const kOf = (a, fr) => { const t = (a * Math.PI) / 180; return [fr * Math.sin(t), fr * Math.cos(t)]; };   // stripes at angle a (from horizontal, counter-clockwise on screen)
  let colors = null;
  const renderPlane = () => {
    const c = theme(), cols = { A: GREY.split(',').map(Number), H: c.a2.split(',').map(Number), V: c.dot.split(',').map(Number), D: VIOLET.split(',').map(Number) };
    colors = cols;
    putPixels(plane, 128, (i) => {
      const kx = ((i % 128) + 0.5) / 128 - 0.5, ky = (Math.floor(i / 128) + 0.5) / 128 - 0.5, s = shares(kx, ky);
      return [0, 1, 2].map((ch) => s.A * cols.A[ch] + s.H * cols.H[ch] + s.V * cols.V[ch] + s.D * cols.D[ch]);
    });
  };
  const renderGrating = () => { const [kx, ky] = kOf(ang, f), m = 192 / PATCH; putPixels(grat, 192, (i) => { const x = (i % 192) / m, y = Math.floor(i / 192) / m, g = 127.5 + 127.5 * Math.cos(2 * Math.PI * (kx * x + ky * y)); return [g, g, g]; }); };
  const layout = () => {
    const w = canvas.getBoundingClientRect().width, wide = w >= 620;
    const sq = Math.floor(wide ? Math.min(200, (w - 300) / 2 - 12) : (w - 16) / 2), chartH = 150;
    canvas.style.height = `${wide ? sq + 24 + chartH + 56 : sq + 24 + 120 + chartH + 60}px`;
    dpr = fit(canvas); const d = dpr, S = sq * d;
    G = wide
      ? { wide, S, g: { x: 0, y: 20 * d }, p: { x: S + 20 * d, y: 20 * d }, bars: { x: 2 * S + 44 * d, y: 26 * d, w: canvas.width - 2 * S - 44 * d, h: S - 10 * d }, chart: { x: 30 * d, y: S + 52 * d, w: canvas.width - 40 * d, h: chartH * d } }
      : { S, g: { x: 0, y: 20 * d }, p: { x: S + 16 * d, y: 20 * d }, bars: { x: 0, y: S + 34 * d, w: canvas.width, h: 104 * d }, chart: { x: 30 * d, y: S + 178 * d, w: canvas.width - 40 * d, h: chartH * d } };
  };
  const draw = () => {
    const c = theme(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!Z || !G) { label(ctx, c, 'loading…', 12 * dpr, 30 * dpr, dpr); return; }
    const d = dpr, S = G.S, [kx, ky] = kOf(ang, f), sh = shares(kx, ky);
    // grating
    ctx.drawImage(grat, G.g.x, G.g.y, S, S);
    ctx.strokeStyle = c.line; ctx.lineWidth = d; ctx.strokeRect(G.g.x, G.g.y, S, S);
    label(ctx, c, `stripes at ${ang}°`, G.g.x, G.g.y - 7 * d, d, { size: 11.5 });
    // frequency plane
    const { x: px, y: py } = G.p;
    ctx.drawImage(plane, px, py, S, S);
    ctx.setLineDash([4 * d, 3 * d]); ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1.2 * d;
    [0.25, 0.75].forEach((t) => { ctx.beginPath(); ctx.moveTo(px + t * S, py); ctx.lineTo(px + t * S, py + S); ctx.moveTo(px, py + t * S); ctx.lineTo(px + S, py + t * S); ctx.stroke(); });
    ctx.setLineDash([]);
    const tag = (t, u, v) => { ctx.font = `${10.5 * d}px ui-monospace, monospace`; ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.textAlign = 'center'; ctx.fillText(t, px + u * S, py + v * S + 4 * d); ctx.textAlign = 'left'; };
    tag('A', 0.5, 0.5); tag('cV', 0.125, 0.5); tag('cV', 0.875, 0.5); tag('cH', 0.5, 0.125); tag('cH', 0.5, 0.875);
    [[0.1, 0.05], [0.9, 0.05], [0.1, 0.95], [0.9, 0.95]].forEach(([u, v]) => tag('cD', u, v));
    [[kx, ky], [-kx, -ky]].forEach(([u, v]) => { ctx.beginPath(); ctx.arc(px + (u + 0.5) * S, py + (v + 0.5) * S, 5 * d, 0, 2 * Math.PI); ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#111'; ctx.lineWidth = 1.5 * d; ctx.stroke(); });
    // the mirror grating (180 - a), hollow
    const [mx, my] = kOf(180 - ang, f);
    if (Math.abs(mx - kx) > 1e-3 || Math.abs(my - ky) > 1e-3) [[mx, my], [-mx, -my]].forEach(([u, v]) => { ctx.beginPath(); ctx.arc(px + (u + 0.5) * S, py + (v + 0.5) * S, 5 * d, 0, 2 * Math.PI); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5 * d; ctx.setLineDash([2 * d, 2 * d]); ctx.stroke(); ctx.setLineDash([]); });
    ctx.strokeStyle = c.line; ctx.lineWidth = d; ctx.strokeRect(px, py, S, S);
    label(ctx, c, 'frequency plane', px, py - 7 * d, d, { size: 11.5 });
    // bars
    const B = G.bars, rows = [['A', 'approximation', sh.A, colors.A], ['H', 'cH  top − bottom', sh.H, colors.H], ['V', 'cV  left − right', sh.V, colors.V], ['D', 'cD  checker', sh.D, colors.D]];
    const rh = B.h / 4, lw = 112 * d, bw = B.w - lw - 46 * d;
    rows.forEach(([, name, v, col], i) => {
      const y = B.y + i * rh;
      label(ctx, c, name, B.x, y + rh / 2 + 4 * d, d, { size: 11 });
      ctx.fillStyle = `rgba(${c.ink}, 0.07)`; ctx.fillRect(B.x + lw, y + rh * 0.22, bw, rh * 0.56);
      ctx.fillStyle = `rgb(${col.join(',')})`; ctx.fillRect(B.x + lw, y + rh * 0.22, bw * v, rh * 0.56);
      label(ctx, c, `${(100 * v).toFixed(1)}%`, B.x + lw + bw + 6 * d, y + rh / 2 + 4 * d, d, { size: 11 });
    });
    // shares against the stripe angle
    const C = G.chart, X = (a) => C.x + (a / 180) * C.w, Y = (v) => C.y + C.h - v * C.h;
    ctx.strokeStyle = `rgba(${c.ink}, 0.25)`; ctx.lineWidth = d; ctx.strokeRect(C.x, C.y, C.w, C.h);
    [45, 90, 135].forEach((a) => { ctx.beginPath(); ctx.moveTo(X(a), C.y); ctx.lineTo(X(a), C.y + C.h); ctx.strokeStyle = `rgba(${c.ink}, 0.08)`; ctx.stroke(); });
    ['A', 'H', 'V', 'D'].forEach((k, i) => {
      ctx.beginPath();
      for (let a = 0; a <= 180; a += 1) { const [u, v] = kOf(a, f), s = shares(u, v)[k]; a ? ctx.lineTo(X(a), Y(s)) : ctx.moveTo(X(a), Y(s)); }
      ctx.strokeStyle = `rgb(${rows[i][3].join(',')})`; ctx.lineWidth = 2 * d; ctx.stroke();
    });
    ctx.setLineDash([4 * d, 3 * d]); ctx.strokeStyle = `rgb(${c.a1})`; ctx.lineWidth = 1.5 * d;
    [ang, 180 - ang].forEach((a) => { ctx.beginPath(); ctx.moveTo(X(a), C.y); ctx.lineTo(X(a), C.y + C.h); ctx.stroke(); });
    ctx.setLineDash([]);
    label(ctx, c, G.wide ? 'energy share against the stripe angle (dashed: your angle and its mirror image, 180° − angle)' : 'share against angle (dashed: yours and 180° − yours)', C.x, C.y - 7 * d, d, { size: 11 });
    [0, 45, 90, 135, 180].forEach((a) => label(ctx, c, `${a}°`, X(a), C.y + C.h + 14 * d, d, { size: 10.5, align: a === 180 ? 'right' : a === 0 ? 'left' : 'center', alpha: 0.6 }));
    [0, 0.5, 1].forEach((v) => label(ctx, c, v.toFixed(1), C.x - 5 * d, Y(v) + 4 * d, d, { size: 10, align: 'right', alpha: 0.55 }));
    if (insight) {
      const glyph = ang % 180 === 0 ? '—' : ang === 90 ? '|' : ang < 90 ? '/' : '\\';
      const near = (k) => [0, 0.25, 0.5].some((t) => Math.abs(Math.abs(k) - t) < 0.006), edge = near(kx) && near(ky);
      insight.innerHTML = `<strong>Stripes at ${ang}° (${glyph}), ${f.toFixed(3)} cycles per pixel (one stripe every ${(1 / f).toFixed(1)} px).</strong> One level of ${{ haar: 'Haar', db2: 'D4', db4: 'D8' }[wname]} puts ${(100 * sh.A).toFixed(1)}% of the energy in the approximation, ${(100 * sh.H).toFixed(1)}% in cH, ${(100 * sh.V).toFixed(1)}% in cV and ${(100 * sh.D).toFixed(1)}% in cD. `
        + `The mirror-image stripes at ${180 - ang}° (hollow dots) give exactly the same four numbers: the filters are real, so |H| and |G| are even in each frequency, and no band can tell / from \\. The transform is still invertible (the signs of the coefficients differ), but any feature built from band energies sees one texture where there are two.`
        + (edge ? ' <em>Caveat:</em> when both components of the frequency are 0, ¼ or ½ cycle per pixel, keeping every second sample folds the wave onto itself, and the real split also depends on where the stripes start; the bars then show its average over positions.' : '');
    }
  };
  const sync = () => { if (aOut) aOut.textContent = `${ang}°`; if (fOut) fOut.textContent = f.toFixed(3); };
  aIn?.addEventListener('input', () => { ang = Number(aIn.value); sync(); renderGrating(); draw(); });
  fIn?.addEventListener('input', () => { f = Number(fIn.value); sync(); renderGrating(); draw(); });
  wBtns.forEach((b) => b.addEventListener('click', () => { wname = b.dataset.ow; wBtns.forEach((x) => x.classList.toggle('on', x === b)); renderPlane(); draw(); }));
  sync(); layout(); draw();
  loadZoo(canvas.dataset.src).then((z) => { Z = z; renderPlane(); renderGrating(); draw(); });
  new ResizeObserver(() => { layout(); draw(); }).observe(canvas);
  onTheme(() => { if (Z) renderPlane(); draw(); });
}
