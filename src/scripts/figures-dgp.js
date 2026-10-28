// Deep GP post: (1) samples of composed GPs, to see what depth does to the prior;
// (2) an exact GP and a two-layer deep GP fitted to the same step data (fit precomputed by plans/blog/deep-gp/fit.py);
// (3) the joint density of (y(x), y(x')) for a two-layer deep GP next to the GP with the same covariance.
import { theme, onTheme, fit, cholesky, seededGauss } from './figures.js';

// Two-layer deep GP of the post's covariance proposition: sigma1 = sigma2 = 1, ell1 = 0.3, ell2 = 0.5.
const J = { l1: 0.3, l2: 0.5, sn2: 0.01, lim: 3, n: 110 };

export function mountDgpJoint(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-dgpjoint]');
  const slider = root?.querySelector('input[data-delta]');
  const out = root?.querySelector('[data-out]');
  // two square panels side by side: on narrow screens shrink the canvas height to fit them
  const size = () => {
    const w = canvas.getBoundingClientRect().width;
    canvas.style.height = `${Math.min(330, (w - 40) / 2 + 50)}px`;
    return fit(canvas);
  };
  let dpr = size();

  // density of a zero-mean bivariate Gaussian with unit-ish variances v and covariance c
  const bvn = (a, b, v, c) => {
    const det = v * v - c * c;
    return Math.exp(-0.5 * (v * a * a - 2 * c * a * b + v * b * b) / det) / (2 * Math.PI * Math.sqrt(det));
  };

  const draw = () => {
    const c = theme();
    const [r, g, b] = c.a1.split(',').map(Number);
    const delta = Number(slider?.value ?? 0.3);
    const s2 = 2 * (1 - Math.exp(-0.5 * delta * delta / (J.l1 * J.l1)));     // Var[f1(x) - f1(x')]
    const s = Math.sqrt(s2);
    // d = f1(x) - f1(x') ~ N(0, s2): integrate over d >= 0 (the density depends on d^2), trapezoid on [0, 5s]
    const K = 40;
    const nodes = Array.from({ length: K }, (_, k) => {
      const d = (5 * s * k) / (K - 1);
      const w = (k === 0 || k === K - 1 ? 0.5 : 1) * (5 * s / (K - 1)) * 2 * Math.exp(-0.5 * d * d / s2) / Math.sqrt(2 * Math.PI * s2);
      return [Math.exp(-0.5 * d * d / (J.l2 * J.l2)), w];
    });
    const wsum = nodes.reduce((t, [, w]) => t + w, 0);
    const Erho = 1 / Math.sqrt(1 + s2 / (J.l2 * J.l2));                         // the closed form of the post
    const v = 1 + J.sn2;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const side = Math.min((W - 40 * dpr) / 2, H - 46 * dpr);
    const narrow = W / dpr < 520;
    const panels = [
      [narrow ? 'GP, same covariance' : 'GP with the same covariance', (a, bb) => bvn(a, bb, v, Erho)],
      ['two-layer deep GP', (a, bb) => nodes.reduce((t, [rho, w]) => t + w * bvn(a, bb, v, rho), 0) / wsum],
    ];
    panels.forEach(([name, dens], p) => {
      const x0 = 10 * dpr + p * (side + 20 * dpr), y0 = 26 * dpr;
      const off = document.createElement('canvas'); off.width = J.n; off.height = J.n;
      const octx = off.getContext('2d'); const img = octx.createImageData(J.n, J.n);
      const vals = new Float64Array(J.n * J.n);
      let mx = 0;
      for (let i = 0; i < J.n; i++) for (let j = 0; j < J.n; j++) {
        const a = -J.lim + (2 * J.lim * i) / (J.n - 1), bb = J.lim - (2 * J.lim * j) / (J.n - 1);
        const d = dens(a, bb); vals[j * J.n + i] = d; if (d > mx) mx = d;
      }
      for (let k = 0; k < vals.length; k++) {
        img.data[4 * k] = r; img.data[4 * k + 1] = g; img.data[4 * k + 2] = b;
        img.data[4 * k + 3] = Math.round(255 * Math.pow(vals[k] / mx, 0.45));
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, x0, y0, side, side);
      ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(x0, y0, side, side);
      ctx.fillStyle = `rgba(${c.ink}, 0.8)`;
      ctx.font = `${(narrow ? 10.5 : 12) * dpr}px ui-monospace, monospace`;
      ctx.fillText(name, x0 + 2 * dpr, 16 * dpr);
      ctx.fillStyle = `rgba(${c.ink}, 0.55)`;
      ctx.font = `${11 * dpr}px ui-monospace, monospace`;
      ctx.fillText('y(x) →', x0 + side - 52 * dpr, y0 + side + 14 * dpr);
      ctx.save(); ctx.translate(x0 - 2 * dpr, y0 + 48 * dpr); ctx.rotate(-Math.PI / 2); ctx.fillText("y(x′) →", 0, 0); ctx.restore();
    });
    // kurtosis of the increment y(x) - y(x'): Gaussian 3; deep GP 3 E[v^2] / E[v]^2 in closed form
    const Ev = 1 - Erho, Erho2 = 1 / Math.sqrt(1 + 2 * s2 / (J.l2 * J.l2));
    const kurt = (3 * (1 - 2 * Erho + Erho2)) / (Ev * Ev);
    if (out) out.textContent = `|x − x′| = ${delta.toFixed(2)} · correlation ${Erho.toFixed(2)} in both · kurtosis of y(x) − y(x′): GP 3.0, deep GP ${kurt.toFixed(1)}`;
  };
  slider?.addEventListener('input', draw);
  addEventListener('resize', () => { dpr = size(); draw(); });
  onTheme(draw);
  draw();
}

const N = 160;
const U = Array.from({ length: N }, (_, i) => -2 + (4 * i) / (N - 1));   // inputs on [-2, 2]
const COLS = ['a1', 'dot', 'a2'];

// One draw of an L-layer composition at the grid inputs. Draws are seeded per (path, layer),
// so adding a layer composes onto the same lower layers.
function compose(L, ell, arch, path, seed) {
  let h = [...U];
  for (let l = 0; l < L; l++) {
    const K = h.map((a, i) => h.map((b, j) => {
      let d2 = (a - b) ** 2;
      if (arch === 'input') d2 += (U[i] - U[j]) ** 2;
      return Math.exp(-0.5 * d2 / (ell * ell)) + (i === j ? 1e-6 : 0);
    }));
    const C = cholesky(K);
    const g = seededGauss(seed + 7919 * path + 104729 * l);
    const z = Array.from({ length: N }, () => g());
    const f = C.map((row) => row.reduce((s, v, j) => s + v * z[j], 0));   // f = L z
    h = arch === 'identity' ? h.map((v, i) => v + f[i]) : f;
  }
  return h;
}

export function mountDgp(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-dgp]');
  const depth = root?.querySelector('input[data-depth]');
  const ellIn = root?.querySelector('input[data-ell]');
  const out = root?.querySelector('[data-out]');
  let arch = 'plain';
  let seed = 11;
  let dpr = fit(canvas);
  let paths = [];

  const sample = () => {
    const L = Number(depth?.value ?? 1);
    const ell = Number(ellIn?.value ?? 1);
    paths = [0, 1, 2].map((p) => compose(L, ell, arch, p, seed));
  };

  const draw = () => {
    const c = theme();
    const W = canvas.width, H = canvas.height;
    const L = Number(depth?.value ?? 1);
    ctx.clearRect(0, 0, W, H);
    const ymax = Math.max(2.5, ...paths.flat().map(Math.abs)) * 1.08;
    const X_ = (i) => 10 * dpr + (i / (N - 1)) * (W - 20 * dpr);
    const Y_ = (v) => H / 2 - (v / ymax) * (H / 2 - 14 * dpr);
    ctx.strokeStyle = c.line; ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(X_(0), Y_(0)); ctx.lineTo(X_(N - 1), Y_(0)); ctx.stroke();
    paths.forEach((h, p) => {
      ctx.strokeStyle = `rgb(${c[COLS[p]]})`;
      ctx.lineWidth = 2.2 * dpr;
      ctx.beginPath();
      h.forEach((v, i) => ctx.lineTo(X_(i), Y_(v)));
      ctx.stroke();
    });
    // how much of the axis is nearly flat, and the steepest slope (per unit of input)
    const du = 4 / (N - 1);
    let flat = 0, steep = 0;
    paths.forEach((h) => {
      for (let i = 1; i < N; i++) {
        const s = Math.abs(h[i] - h[i - 1]) / du;
        if (s < 0.1) flat += 1;
        steep = Math.max(steep, s);
      }
    });
    const share = (100 * flat) / (3 * (N - 1));
    ctx.fillStyle = `rgba(${c.ink}, 0.6)`;
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.fillText(`three draws of f${L > 1 ? `${L} ∘ … ∘ f1` : '1'}(x)`, X_(0) + 4 * dpr, 16 * dpr);
    if (out) out.textContent = `L = ${L} · nearly flat (|slope| < 0.1) on ${share.toFixed(0)}% of the axis · steepest slope ${steep.toFixed(1)}`;
  };

  const redo = () => { sample(); draw(); };
  depth?.addEventListener('input', redo);
  ellIn?.addEventListener('input', redo);
  root?.querySelectorAll('[data-arch]').forEach((b) => b.addEventListener('click', () => {
    arch = b.dataset.arch;
    root.querySelectorAll('[data-arch]').forEach((o) => o.classList.toggle('on', o === b));
    redo();
  }));
  root?.querySelector('[data-resample]')?.addEventListener('click', () => { seed += 1; redo(); });
  addEventListener('resize', () => { dpr = fit(canvas); draw(); });
  onTheme(draw);
  redo();
}

export function mountDgpFit(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-dgpfit]');
  const out = root?.querySelector('[data-out]');
  let dpr = fit(canvas);
  let mode = 'dgp';
  let xstar = 0.02;
  let F = null;
  let dens = null;
  let heat = null;

  const grid = (spec) => Array.from({ length: spec[2] }, (_, i) => spec[0] + ((spec[1] - spec[0]) * i) / (spec[2] - 1));

  // the deep GP density as an image, coloured with the current theme
  const buildHeat = () => {
    const c = theme();
    const [r, g, b] = c.a1.split(',').map(Number);
    const nx = F.xs[2], ny = F.ys[2];
    const off = document.createElement('canvas');
    off.width = nx; off.height = ny;
    const img = off.getContext('2d').createImageData(nx, ny);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const v = dens[i * ny + j] / 255;
      const k = 4 * ((ny - 1 - j) * nx + i);
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = Math.round(255 * Math.min(1, v ** 0.6) * 0.95);
    }
    off.getContext('2d').putImageData(img, 0, 0);
    heat = off;
  };

  const draw = () => {
    if (!F) return;
    const c = theme();
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const xs = grid(F.xs), ys = grid(F.ys);
    const pw = W * 0.7, top = 10 * dpr, bot = H - 22 * dpr;
    const X_ = (x) => 8 * dpr + ((x - F.xs[0]) / (F.xs[1] - F.xs[0])) * (pw - 16 * dpr);
    const Y_ = (v) => top + ((F.ys[1] - v) / (F.ys[1] - F.ys[0])) * (bot - top);
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;

    if (mode === 'dgp') {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(heat, X_(F.xs[0]), Y_(F.ys[1]), X_(F.xs[1]) - X_(F.xs[0]), Y_(F.ys[0]) - Y_(F.ys[1]));
    } else if (mode === 'gp') {
      const sd = F.gp_sd.map((s) => Math.sqrt(s * s + F.gp_sn * F.gp_sn));
      ctx.fillStyle = `rgba(${c.a1}, 0.18)`;
      ctx.beginPath();
      xs.forEach((x, i) => ctx.lineTo(X_(x), Y_(F.gp_mu[i] + 2 * sd[i])));
      for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X_(xs[i]), Y_(F.gp_mu[i] - 2 * sd[i]));
      ctx.fill();
      ctx.strokeStyle = `rgb(${c.a1})`; ctx.lineWidth = 2.2 * dpr;
      ctx.beginPath(); xs.forEach((x, i) => ctx.lineTo(X_(x), Y_(F.gp_mu[i]))); ctx.stroke();
    } else {
      ctx.fillStyle = `rgba(${c.dot}, 0.2)`;
      ctx.beginPath();
      xs.forEach((x, i) => ctx.lineTo(X_(x), Y_(F.h_mu[i] + 2 * F.h_sd[i])));
      for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X_(xs[i]), Y_(F.h_mu[i] - 2 * F.h_sd[i]));
      ctx.fill();
      ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 2.4 * dpr;
      ctx.beginPath(); xs.forEach((x, i) => ctx.lineTo(X_(x), Y_(F.h_mu[i]))); ctx.stroke();
      ctx.fillStyle = `rgba(${c.ink}, 0.7)`;
      ctx.fillText('hidden layer h(x), mean ± 2 sd', X_(F.xs[0]) + 4 * dpr, top + 12 * dpr);
    }
    if (mode !== 'hidden') {
      ctx.fillStyle = `rgba(${c.ink}, 0.85)`;
      F.x.forEach((x, i) => { ctx.beginPath(); ctx.arc(X_(x), Y_(F.y[i]), 2.2 * dpr, 0, 2 * Math.PI); ctx.fill(); });
    }
    // x* marker
    ctx.strokeStyle = `rgba(${c.ink}, 0.5)`; ctx.setLineDash([4 * dpr, 4 * dpr]); ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(X_(xstar), top); ctx.lineTo(X_(xstar), bot); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = `rgba(${c.ink}, 0.7)`;
    ctx.fillText(`x* = ${xstar.toFixed(2)}`, X_(xstar) + 4 * dpr, bot + 14 * dpr);

    // predictive density at x*, drawn sideways on the same y-axis: GP (dashed) and deep GP (filled)
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(((xstar - F.xs[0]) / (F.xs[1] - F.xs[0])) * (xs.length - 1))));
    const dy = ys[1] - ys[0];
    const col = ys.map((_, j) => dens[i * ys.length + j]);
    const area = col.reduce((s, v) => s + v, 0) * dy || 1;
    const pd = col.map((v) => v / area);
    const s2 = F.gp_sd[i] ** 2 + F.gp_sn ** 2;
    const pg = ys.map((v) => Math.exp(-0.5 * (v - F.gp_mu[i]) ** 2 / s2) / Math.sqrt(2 * Math.PI * s2));
    const peak = Math.max(...pd, ...pg);
    const rx = pw + 14 * dpr, rw = W - rx - 8 * dpr;
    const D_ = (p) => rx + (p / peak) * rw;
    ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(rx, top); ctx.lineTo(rx, bot); ctx.stroke();
    ctx.fillStyle = `rgba(${c.a1}, 0.35)`;
    ctx.beginPath(); ctx.moveTo(rx, Y_(ys[0]));
    ys.forEach((v, j) => ctx.lineTo(D_(pd[j]), Y_(v)));
    ctx.lineTo(rx, Y_(ys[ys.length - 1])); ctx.fill();
    ctx.strokeStyle = `rgb(${c.a1})`; ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath(); ys.forEach((v, j) => ctx.lineTo(D_(pd[j]), Y_(v))); ctx.stroke();
    ctx.strokeStyle = `rgba(${c.ink}, 0.8)`; ctx.setLineDash([5 * dpr, 3 * dpr]);
    ctx.beginPath(); ys.forEach((v, j) => ctx.lineTo(D_(pg[j]), Y_(v))); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = `rgba(${c.ink}, 0.7)`;
    ctx.fillText('p(y | x*)', rx + 4 * dpr, top + 12 * dpr);
    ctx.fillStyle = `rgb(${c.a1})`; ctx.fillText('deep GP', rx + 4 * dpr, top + 26 * dpr);
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.fillText('- - GP', rx + 4 * dpr, top + 40 * dpr);
  };

  const setX = (e) => {
    const r = canvas.getBoundingClientRect();
    const px = (e.clientX - r.left) * dpr;
    const pw = canvas.width * 0.7;
    if (px > pw) return;
    const t = (px - 8 * dpr) / (pw - 16 * dpr);
    xstar = Math.max(F.xs[0], Math.min(F.xs[1], F.xs[0] + t * (F.xs[1] - F.xs[0])));
    draw();
  };
  canvas.addEventListener('pointermove', (e) => { if (F) setX(e); });
  canvas.addEventListener('pointerdown', (e) => { if (F) setX(e); });
  root?.querySelectorAll('[data-model]').forEach((b) => b.addEventListener('click', () => {
    mode = b.dataset.model;
    root.querySelectorAll('[data-model]').forEach((o) => o.classList.toggle('on', o === b));
    draw();
  }));
  addEventListener('resize', () => { dpr = fit(canvas); draw(); });
  onTheme(() => { if (F) { buildHeat(); draw(); } });

  fetch(canvas.dataset.src).then((r) => r.json()).then((json) => {
    F = json;
    dens = Uint8Array.from(atob(F.dens), (ch) => ch.charCodeAt(0));
    buildHeat();
    const s = F.stats;
    if (out) out.textContent = `held-out log density per point: GP ${s.gp_tlpd.toFixed(2)}, deep GP ${s.dgp_tlpd.toFixed(2)} · learned noise sd: GP ${F.gp_sn.toFixed(3)}, deep GP ${F.dgp_sn.toFixed(3)} (true 0.05)`;
    draw();
  });
}
