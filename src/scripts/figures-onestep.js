// One-step generators post. Data: the series' ring of 8 Gaussians; linear path x_t = (1 - t) x0 + t eps,
// data at t = 0. mountChord computes the exact flow live (closed-form velocity, RK4). mountOneStep and
// mountDrift read samples precomputed by plans/blog/one-step/onestep.py; the drifting field itself is
// computed live from those samples with the published kernel.
import { theme, onTheme, fit } from './figures.js';

const K = 8, R = 2, S = 0.1;
const MU = Array.from({ length: K }, (_, k) => [R * Math.cos((2 * Math.PI * k) / K), R * Math.sin((2 * Math.PI * k) / K)]);
let cache = null;
const load = (url) => (cache ??= fetch(url).then((r) => r.json()));
const square = (canvas, max) => { const w = canvas.getBoundingClientRect().width; canvas.style.height = `${Math.min(max, w)}px`; return fit(canvas); };
const onMode = (P) => P.filter(([a, b]) => MU.some(([m, n]) => Math.hypot(a - m, b - n) < 3 * S)).length / P.length;

function frame(ctx, c, W, H, dpr) {
  const side = Math.min(W, H) - 20 * dpr, x0 = (W - side) / 2, y0 = 10 * dpr;
  const T = (a, b) => [x0 + ((a + 3.2) / 6.4) * side, y0 + ((3.2 - b) / 6.4) * side];
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(x0, y0, side, side);
  MU.forEach((m) => { const [X, Y] = T(m[0], m[1]); ctx.strokeStyle = `rgba(${c.ink}, 0.25)`; ctx.beginPath(); ctx.arc(X, Y, (0.3 / 6.4) * side, 0, 2 * Math.PI); ctx.stroke(); });
  return { T, x0, y0, side, inv: (X, Y) => [((X - x0) / side) * 6.4 - 3.2, 3.2 - ((Y - y0) / side) * 6.4] };
}

function arrow(ctx, X1, Y1, X2, Y2, dpr) {
  ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X2, Y2); ctx.stroke();
  const a = Math.atan2(Y2 - Y1, X2 - X1), L = Math.min(7 * dpr, 0.45 * Math.hypot(X2 - X1, Y2 - Y1));
  ctx.beginPath(); ctx.moveTo(X2, Y2); ctx.lineTo(X2 - L * Math.cos(a - 0.45), Y2 - L * Math.sin(a - 0.45));
  ctx.moveTo(X2, Y2); ctx.lineTo(X2 - L * Math.cos(a + 0.45), Y2 - L * Math.sin(a + 0.45)); ctx.stroke();
}

// exact marginal velocity v(x, t) = E[eps - x0 | x_t = x] for the linear path
function vel(x, y, t) {
  t = Math.max(t, 1e-5);
  const a = 1 - t, v = a * a * S * S + t * t, g = (a * S * S) / v;
  let lw = new Array(K), mx = -Infinity;
  for (let k = 0; k < K; k++) { const dx = x - a * MU[k][0], dy = y - a * MU[k][1]; lw[k] = (-0.5 * (dx * dx + dy * dy)) / v; mx = Math.max(mx, lw[k]); }
  let z = 0, Dx = 0, Dy = 0;
  for (let k = 0; k < K; k++) {
    const w = Math.exp(lw[k] - mx); z += w;
    Dx += w * (MU[k][0] + g * (x - a * MU[k][0])); Dy += w * (MU[k][1] + g * (y - a * MU[k][1]));
  }
  Dx /= z; Dy /= z;
  return [(x - Dx) / t, (y - Dy) / t];
}

// RK4 from t = 1 to t = 0 in n steps; returns positions at every step
function trajectory(x, y, n) {
  const out = [[x, y]], h = 1 / n;
  let t = 1;
  for (let i = 0; i < n; i++) {
    const k1 = vel(x, y, t), k2 = vel(x - 0.5 * h * k1[0], y - 0.5 * h * k1[1], t - 0.5 * h);
    const k3 = vel(x - 0.5 * h * k2[0], y - 0.5 * h * k2[1], t - 0.5 * h), k4 = vel(x - h * k3[0], y - h * k3[1], t - h);
    x -= (h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])) / 6; y -= (h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])) / 6; t -= h;
    out.push([x, y]);
  }
  return out;
}

function gauss(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6d2b79f5) >>> 0; let q = s; q = Math.imul(q ^ (q >>> 15), q | 1); q ^= q + Math.imul(q ^ (q >>> 7), q | 61); return ((q ^ (q >>> 14)) >>> 0) / 4294967296; };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
}

export function mountChord(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-chord]');
  const rIn = root?.querySelector('input[data-r]');
  const rOut = root?.querySelector('[data-rval]');
  const out = root?.querySelector('[data-out]');
  const N = 100, g = gauss(11);
  const Z = Array.from({ length: 500 }, () => [g(), g()]);
  const P = Z.map(([a, b]) => trajectory(a, b, N));               // P[i][k] = X_{1 -> 1 - k/N}(z_i)
  let dpr = square(canvas, 440), mode = 'map', pick = trajectory(0.35, 0.9, N), view = null;
  const draw = () => {
    const c = theme(), W = canvas.width, H = canvas.height;
    const r = Number(rIn?.value ?? 0) / 100, k = Math.round((1 - r) * N);
    if (rOut) rOut.textContent = r.toFixed(2);
    view = frame(ctx, c, W, H, dpr);
    const { T } = view;
    const pts = mode === 'map' ? P.map((p) => p[k]) : Z.map(([a, b]) => [r * a, r * b]);   // Euler: z - (1 - r) v(z, 1), with v(z, 1) = z
    ctx.fillStyle = mode === 'map' ? `rgba(${c.dot}, 0.8)` : `rgba(${c.ink}, 0.55)`;
    pts.forEach(([a, b]) => { const [X, Y] = T(a, b); ctx.fillRect(X - 1.4 * dpr, Y - 1.4 * dpr, 2.8 * dpr, 2.8 * dpr); });
    // the chosen point: its path, the tangent at t = 1, and the chord to time r
    const z = pick[0], end = pick[k], v1 = vel(z[0], z[1], 1);
    ctx.lineWidth = 2 * dpr; ctx.strokeStyle = `rgba(${c.a2}, 0.9)`; ctx.beginPath();
    pick.forEach((p, i) => { const [X, Y] = T(...p); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke();
    const [Z0, Z1] = T(...z);
    ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.strokeStyle = `rgba(${c.ink}, 0.85)`;
    const [E0, E1] = T(z[0] - (1 - r) * v1[0], z[1] - (1 - r) * v1[1]); if (r < 1) arrow(ctx, Z0, Z1, E0, E1, dpr);
    ctx.setLineDash([]); ctx.strokeStyle = `rgb(${c.dot})`; ctx.lineWidth = 2.4 * dpr;
    const [C0, C1] = T(...end); if (r < 1) arrow(ctx, Z0, Z1, C0, C1, dpr);
    ctx.fillStyle = `rgb(${c.ink})`; ctx.beginPath(); ctx.arc(Z0, Z1, 3.5 * dpr, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillText(mode === 'map' ? `flow map X(1→${r.toFixed(2)})` : `one Euler step to ${r.toFixed(2)}`, 18 * dpr, 26 * dpr);
    const u = [(z[0] - end[0]) / Math.max(1 - r, 1e-9), (z[1] - end[1]) / Math.max(1 - r, 1e-9)];
    const cos = (u[0] * v1[0] + u[1] * v1[1]) / (Math.hypot(...u) * Math.hypot(...v1) + 1e-12);
    const share = (100 * onMode(pts)).toFixed(0);
    if (out) out.textContent = `r = ${r.toFixed(2)} · ${mode === 'map' ? 'flow map' : 'Euler step'}: ${share}% of 500 points on a mode`
      + (r < 0.995 ? ` · chosen point: |u| = ${Math.hypot(...u).toFixed(2)}, |v| = ${Math.hypot(...v1).toFixed(2)}, angle ${(Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI).toFixed(0)}°` : '');
  };
  canvas.addEventListener('click', (e) => {
    if (!view) return;
    const b = canvas.getBoundingClientRect(), [a, c] = view.inv((e.clientX - b.left) * (canvas.width / b.width), (e.clientY - b.top) * (canvas.height / b.height));
    pick = trajectory(a, c, N); draw();
  });
  rIn?.addEventListener('input', draw);
  root?.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => { mode = b.dataset.mode; root.querySelectorAll('[data-mode]').forEach((o) => o.classList.toggle('on', o === b)); draw(); }));
  addEventListener('resize', () => { dpr = square(canvas, 440); draw(); });
  onTheme(draw);
  draw();
}

export function mountOneStep(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-onestep]');
  const out = root?.querySelector('[data-out]');
  let dpr = square(canvas, 420), model = 'exact', N = '1', F = null;
  const names = { exact: 'exact flow map', distill: 'distilled map', cd: 'consistency distillation', ct: 'consistency training', shortcut: 'shortcut model', meanflow: 'MeanFlow', drift: 'drifting model' };
  const draw = () => {
    if (!F) return;
    const c = theme(), W = canvas.width, H = canvas.height;
    const has = F.samples[model][N] ? N : '1';
    root?.querySelectorAll('[data-steps]').forEach((b) => { b.disabled = !F.samples[model][b.dataset.steps]; b.classList.toggle('on', b.dataset.steps === has); });
    const { T } = frame(ctx, c, W, H, dpr);
    ctx.fillStyle = `rgba(${c.dot}, 0.85)`;
    F.samples[model][has].forEach(([a, b]) => { const [X, Y] = T(a, b); ctx.fillRect(X - 1.5 * dpr, Y - 1.5 * dpr, 3 * dpr, 3 * dpr); });
    const s = F.stats[model][has];
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillText(`${names[model]} · ${has} step${has === '1' ? '' : 's'}`, 18 * dpr, 26 * dpr);
    const pct = 100 * s.on, on = pct >= 99.5 && pct < 100 ? pct.toFixed(1) : pct.toFixed(0);
    if (out) out.textContent = `${names[model]}, ${has} network call${has === '1' ? '' : 's'}: ${s.modes}/8 modes, ${on}% on a mode, energy distance ${Number(s.ed).toFixed(1)} ×10⁻³ (4000 samples; data against data: ${F.floor})`;
  };
  root?.querySelectorAll('[data-model]').forEach((b) => b.addEventListener('click', () => { model = b.dataset.model; root.querySelectorAll('[data-model]').forEach((o) => o.classList.toggle('on', o === b)); draw(); }));
  root?.querySelectorAll('[data-steps]').forEach((b) => b.addEventListener('click', () => { N = b.dataset.steps; draw(); }));
  addEventListener('resize', () => { dpr = square(canvas, 420); draw(); });
  onTheme(draw);
  load(canvas.dataset.src).then((j) => { F = j; draw(); });
}

// drifting field (Deng et al., 2026): V = V+ - V-, each a kernel-weighted mean shift with k(x, y) = exp(-|x - y| / tau),
// normalised over the samples (a softmax). Positives: data. Negatives: the other generated samples.
function shift(x, Y, tau, skip) {
  let mx = -Infinity; const l = new Float64Array(Y.length);
  for (let j = 0; j < Y.length; j++) { l[j] = j === skip ? -Infinity : -Math.hypot(x[0] - Y[j][0], x[1] - Y[j][1]) / tau; if (l[j] > mx) mx = l[j]; }
  let z = 0, a = 0, b = 0;
  for (let j = 0; j < Y.length; j++) { if (j === skip) continue; const w = Math.exp(l[j] - mx); z += w; a += w * Y[j][0]; b += w * Y[j][1]; }
  return [a / z - x[0], b / z - x[1]];
}

export function mountDrift(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-drift]');
  const sIn = root?.querySelector('input[data-snap]');
  const sOut = root?.querySelector('[data-snapval]');
  const out = root?.querySelector('[data-out]');
  let dpr = square(canvas, 440), part = 'net', equal = false, F = null;
  const draw = () => {
    if (!F) return;
    const c = theme(), W = canvas.width, H = canvas.height;
    const i = Number(sIn?.value ?? 0), [step, gen] = F.snaps[Math.min(i, F.snaps.length - 1)];
    const Q = equal ? F.data2 : gen, Pd = F.data;
    if (sOut) sOut.textContent = equal ? 'q = p' : String(step);
    const { T } = frame(ctx, c, W, H, dpr);
    ctx.fillStyle = `rgba(${c.a2}, 0.35)`;
    Pd.forEach(([a, b]) => { const [X, Y] = T(a, b); ctx.fillRect(X - 1.2 * dpr, Y - 1.2 * dpr, 2.4 * dpr, 2.4 * dpr); });
    let tot = 0; const show = Q.length > 200 ? 2 : 1;
    ctx.lineWidth = 1.3 * dpr; ctx.strokeStyle = part === 'minus' ? `rgba(${c.ink}, 0.7)` : `rgba(${c.dot}, 0.9)`;
    Q.forEach((x, j) => {
      const vp = shift(x, Pd, F.tau, -1), vq = shift(x, Q, F.tau, j);
      const V = part === 'plus' ? vp : part === 'minus' ? [-vq[0], -vq[1]] : [vp[0] - vq[0], vp[1] - vq[1]];
      tot += Math.hypot(vp[0] - vq[0], vp[1] - vq[1]);
      if (j % show) return;
      const [X1, Y1] = T(...x), [X2, Y2] = T(x[0] + V[0], x[1] + V[1]);
      if (Math.hypot(X2 - X1, Y2 - Y1) > 1.5 * dpr) arrow(ctx, X1, Y1, X2, Y2, dpr);
    });
    ctx.fillStyle = `rgb(${c.ink})`;
    Q.forEach(([a, b], j) => { if (j % show) return; const [X, Y] = T(a, b); ctx.beginPath(); ctx.arc(X, Y, 1.8 * dpr, 0, 2 * Math.PI); ctx.fill(); });
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillText(equal ? 'generated = fresh data' : `training step ${step}`, 18 * dpr, 26 * dpr);
    const lab = { plus: 'attraction V⁺ (toward data)', minus: 'repulsion −V⁻ (away from other samples)', net: 'drift V = V⁺ − V⁻' }[part];
    if (out) out.textContent = `${equal ? 'q = p' : `step ${step}`} · arrows: ${lab} · mean |V| = ${(tot / Q.length).toFixed(3)} · ${(100 * onMode(Q)).toFixed(0)}% of generated points on a mode`;
  };
  sIn?.addEventListener('input', () => { equal = false; root?.querySelector('[data-equal]')?.classList.remove('on'); draw(); });
  root?.querySelectorAll('[data-part]').forEach((b) => b.addEventListener('click', () => { part = b.dataset.part; root.querySelectorAll('[data-part]').forEach((o) => o.classList.toggle('on', o === b)); draw(); }));
  root?.querySelector('[data-equal]')?.addEventListener('click', (e) => { equal = !equal; e.currentTarget.classList.toggle('on', equal); draw(); });
  addEventListener('resize', () => { dpr = square(canvas, 440); draw(); });
  onTheme(draw);
  load(canvas.dataset.src).then((j) => { F = j; if (sIn) sIn.max = String(F.snaps.length - 1); draw(); });
}
