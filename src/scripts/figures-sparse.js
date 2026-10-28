// Sparse variational GP post: exact GP vs the Titsias (2009) sparse GP, with the four evidence bounds.
import { theme, onTheme, fit, cholesky, forward, seededGauss } from './figures.js';

const SF2 = 1;      // kernel amplitude σ_f²
const ELL = 0.07;   // length-scale
const SN = 0.25;    // noise standard deviation σ_n
const S2 = SN * SN;
const kern = (a, b) => SF2 * Math.exp(-0.5 * ((a - b) / ELL) ** 2);
const LOG2PI = Math.log(2 * Math.PI);

function makeData() {
  const g = seededGauss(7);
  const xs = [];
  for (let i = 0; i < 26; i++) xs.push(0.03 + (0.42 * (i + 0.5 + 0.35 * g())) / 26);
  for (let i = 0; i < 24; i++) xs.push(0.6 + (0.37 * (i + 0.5 + 0.35 * g())) / 24);
  const f = (x) => 0.9 * Math.sin(2 * Math.PI * 1.5 * x) + 0.45 * Math.cos(2 * Math.PI * 4.2 * x);
  return { X: xs, y: xs.map((x) => f(x) + SN * g()) };
}

// Exact GP: log evidence and predictive moments on a grid (data are fixed, so computed once).
function exactGp(X, y, grid) {
  const N = X.length;
  const K = X.map((a) => X.map((b) => kern(a, b)));
  K.forEach((r, i) => (r[i] += S2));
  const L = cholesky(K);
  const w = forward(L, y);
  const logp = -0.5 * w.reduce((s, v) => s + v * v, 0) - L.reduce((s, r, i) => s + Math.log(r[i]), 0) - 0.5 * N * LOG2PI;
  // alpha = K^{-1} y by a back-substitution on w
  const alpha = [...w];
  for (let i = N - 1; i >= 0; i--) {
    for (let p = i + 1; p < N; p++) alpha[i] -= L[p][i] * alpha[p];
    alpha[i] /= L[i][i];
  }
  const pred = grid.map((x) => {
    const ks = X.map((a) => kern(a, x));
    const v = forward(L, ks);
    return [ks.reduce((s, k, i) => s + k * alpha[i], 0), Math.max(0, SF2 - v.reduce((s, t) => s + t * t, 0))];
  });
  return { logp, pred };
}

// Titsias (2009) collapsed bound, the Titsias (2025) bound and the upper bound, all in O(NM^2).
// Â = L⁻¹K_uf/σ_n, B = I + ÂÂᵀ, c = L_B⁻¹Ây/σ_n (the GPflow SGPR route).
function sparse(X, y, Z) {
  const N = X.length, M = Z.length;
  const Kuu = Z.map((a) => Z.map((b) => kern(a, b)));
  Kuu.forEach((r, i) => (r[i] += 1e-6));
  const L = cholesky(Kuu);
  const Acol = X.map((x) => forward(L, Z.map((z) => kern(z, x))).map((v) => v / SN));
  const gram = (scale) => {
    const B = Array.from({ length: M }, (_, a) => Array.from({ length: M }, (_, b) => (a === b ? 1 : 0)));
    for (let i = 0; i < N; i++) {
      const col = Acol[i];
      for (let a = 0; a < M; a++) for (let b = 0; b <= a; b++) B[a][b] += scale * col[a] * col[b];
    }
    for (let a = 0; a < M; a++) for (let b = a + 1; b < M; b++) B[a][b] = B[b][a];
    return B;
  };
  const Ay = Array.from({ length: M }, (_, a) => Acol.reduce((s, col, i) => s + col[a] * y[i], 0));
  const yy = y.reduce((s, v) => s + v * v, 0);
  const LB = cholesky(gram(1));
  const cvec = forward(LB, Ay).map((v) => v / SN);
  const logdetB = 2 * LB.reduce((s, r, i) => s + Math.log(r[i]), 0);
  const cc = cvec.reduce((s, v) => s + v * v, 0);
  const ci = Acol.map((col) => Math.max(0, SF2 - S2 * col.reduce((s, v) => s + v * v, 0)));
  const t = ci.reduce((s, v) => s + v, 0);
  const logN = -0.5 * N * LOG2PI - 0.5 * N * Math.log(S2) - 0.5 * logdetB - yy / (2 * S2) + 0.5 * cc;
  const F09 = logN - t / (2 * S2);
  const F25 = logN - 0.5 * ci.reduce((s, c) => s + Math.log1p(c / S2), 0);
  // upper bound: same determinant, quadratic term with noise σ_n² + t
  const s2p = S2 + t, r = S2 / s2p;
  const LBp = cholesky(gram(r));
  const cp = forward(LBp, Ay.map((v) => v * Math.sqrt(r))).map((v) => v / Math.sqrt(s2p));
  const U = -0.5 * N * LOG2PI - 0.5 * (N * Math.log(S2) + logdetB) - 0.5 * (yy / s2p - cp.reduce((s, v) => s + v * v, 0));
  const predict = (x) => {
    const t1 = forward(L, Z.map((z) => kern(z, x)));
    const t2 = forward(LB, t1);
    const mean = t2.reduce((s, v, i) => s + v * cvec[i], 0);
    const v = SF2 - t1.reduce((s, q) => s + q * q, 0) + t2.reduce((s, q) => s + q * q, 0);
    return [mean, Math.max(0, v)];
  };
  return { F09, F25, U, ci, predict };
}

const spread = (M) => Array.from({ length: M }, (_, j) => (j + 0.5) / M);

export function mountSvgp(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-svgp]');
  const mInput = root?.querySelector('input[data-m]');
  const mOut = root?.querySelector('[data-mval]');
  const out = root?.querySelector('[data-out]');
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let dpr = fit(canvas);
  const { X, y } = makeData();
  const grid = Array.from({ length: 160 }, (_, i) => i / 159);
  const exact = exactGp(X, y, grid);
  let Z = spread(Number(mInput?.value ?? 6));
  let drag = -1;
  let anim = 0;

  const layout = () => {
    const W = canvas.width, H = canvas.height;
    const pw = W * 0.72;
    const top = 8 * dpr, base = H - 18 * dpr, plotBottom = H - 62 * dpr;
    const X_ = (x) => 10 * dpr + x * (pw - 20 * dpr);
    const Y_ = (v) => top + ((2.6 - v) / 5.2) * (plotBottom - top);
    return { W, H, pw, top, base, plotBottom, X_, Y_ };
  };

  const draw = () => {
    const c = theme();
    const { W, H, pw, base, plotBottom, X_, Y_ } = layout();
    const narrow = W / dpr < 520;
    const s = sparse(X, y, Z);
    ctx.clearRect(0, 0, W, H);
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;

    // sparse posterior band and mean
    const sp = grid.map((x) => s.predict(x));
    ctx.fillStyle = `rgba(${c.a1}, 0.16)`;
    ctx.beginPath();
    grid.forEach((x, i) => ctx.lineTo(X_(x), Y_(sp[i][0] + 2 * Math.sqrt(sp[i][1]))));
    for (let i = grid.length - 1; i >= 0; i--) ctx.lineTo(X_(grid[i]), Y_(sp[i][0] - 2 * Math.sqrt(sp[i][1])));
    ctx.fill();
    ctx.strokeStyle = `rgb(${c.a1})`;
    ctx.lineWidth = 2.2 * dpr;
    ctx.beginPath();
    grid.forEach((x, i) => ctx.lineTo(X_(x), Y_(sp[i][0])));
    ctx.stroke();

    // exact posterior: dashed mean, dotted ±2σ
    ctx.strokeStyle = `rgba(${c.ink}, 0.75)`;
    ctx.lineWidth = 1.4 * dpr;
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.beginPath();
    grid.forEach((x, i) => ctx.lineTo(X_(x), Y_(exact.pred[i][0])));
    ctx.stroke();
    ctx.setLineDash([1.5 * dpr, 3 * dpr]);
    [1, -1].forEach((sg) => {
      ctx.beginPath();
      grid.forEach((x, i) => ctx.lineTo(X_(x), Y_(exact.pred[i][0] + sg * 2 * Math.sqrt(exact.pred[i][1]))));
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // data
    ctx.fillStyle = `rgb(${c.dot})`;
    X.forEach((x, i) => { ctx.beginPath(); ctx.arc(X_(x), Y_(y[i]), 3 * dpr, 0, 2 * Math.PI); ctx.fill(); });

    // q(u): inducing values with ±2 sd, and the rail of inducing inputs
    Z.forEach((z) => {
      const [m, v] = s.predict(z);
      const sd = Math.sqrt(v);
      ctx.strokeStyle = `rgba(${c.a1}, 0.35)`;
      ctx.lineWidth = dpr;
      ctx.setLineDash([2 * dpr, 3 * dpr]);
      ctx.beginPath(); ctx.moveTo(X_(z), Y_(m)); ctx.lineTo(X_(z), base); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = `rgb(${c.a1})`;
      ctx.lineWidth = 1.6 * dpr;
      ctx.beginPath(); ctx.moveTo(X_(z), Y_(m - 2 * sd)); ctx.lineTo(X_(z), Y_(m + 2 * sd)); ctx.stroke();
      diamond(X_(z), Y_(m), 4.5 * dpr, `rgb(${c.a1})`);
    });

    // unexplained variance strip: c_i = k_ii − q_ii at every training input
    const stripH = 34 * dpr;
    ctx.strokeStyle = c.line;
    ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(X_(0), base); ctx.lineTo(X_(1), base); ctx.stroke();
    ctx.fillStyle = `rgba(${c.ink}, 0.32)`;
    X.forEach((x, i) => {
      const h = (s.ci[i] / SF2) * stripH;
      ctx.fillRect(X_(x) - 1.5 * dpr, base - h, 3 * dpr, h);
    });
    Z.forEach((z, j) => diamond(X_(z), base, (j === drag ? 7 : 6) * dpr, `rgb(${c.a1})`, true));
    ctx.fillStyle = `rgba(${c.ink}, 0.6)`;
    ctx.fillText(narrow ? 'kᵢᵢ − qᵢᵢ' : 'unexplained variance kᵢᵢ − qᵢᵢ', X_(0), plotBottom + 12 * dpr);
    if (!narrow) ctx.fillText('drag ◆', X_(1) - 48 * dpr, plotBottom + 12 * dpr);

    // legend
    ctx.fillStyle = `rgba(${c.ink}, 0.75)`;
    ctx.fillText(narrow ? '- - exact' : '- - exact GP', X_(0.02), 18 * dpr);
    ctx.fillStyle = `rgb(${c.a1})`;
    ctx.fillText(narrow ? '— sparse' : '— sparse GP (Titsias)', X_(0.02) + (narrow ? 66 : 92) * dpr, 18 * dpr);

    // bound ladder
    const rx = pw + 18 * dpr, rw = W - rx - 6 * dpr;
    const rows = [
      [narrow ? '𝒰' : 'upper bound 𝒰', s.U, c.ink, true],
      [narrow ? 'exact' : 'log p(y), exact', exact.logp, c.ink],
      [narrow ? '2025' : 'Titsias 2025', s.F25, c.a1],
      [narrow ? '2009 ℱ' : 'Titsias 2009 ℱ', s.F09, c.dot],
    ];
    const vals = rows.map((r) => r[1]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = Math.max(2, 0.08 * (hi - lo));
    const top = 26 * dpr, bot = H - 26 * dpr;
    const LY = (v) => top + ((hi + pad - v) / (hi - lo + 2 * pad)) * (bot - top);
    ctx.strokeStyle = c.line;
    ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(rx + 8 * dpr, top); ctx.lineTo(rx + 8 * dpr, bot); ctx.stroke();
    ctx.fillStyle = `rgba(${c.ink}, 0.6)`;
    ctx.fillText('nats', rx, 14 * dpr);
    // labels pushed apart so they never overlap
    const ys = rows.map((r) => LY(r[1]));
    const order = ys.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const placed = [];
    order.forEach(([v, i], k) => { placed[i] = k === 0 ? v : Math.max(v, placed[order[k - 1][1]] + 30 * dpr); });
    rows.forEach(([name, v, col, dashed], i) => {
      ctx.strokeStyle = `rgba(${col}, ${dashed ? 0.6 : 1})`;
      ctx.lineWidth = (i === 1 ? 2.4 : 2) * dpr;
      ctx.setLineDash(dashed ? [4 * dpr, 3 * dpr] : []);
      ctx.beginPath(); ctx.moveTo(rx, ys[i]); ctx.lineTo(rx + 18 * dpr, ys[i]); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = `rgba(${col}, ${dashed ? 0.7 : 1})`;
      ctx.fillText(name, rx + 24 * dpr, placed[i] + 1 * dpr);
      ctx.fillStyle = `rgba(${c.ink}, 0.7)`;
      ctx.fillText(v.toFixed(1), rx + 24 * dpr, placed[i] + 13 * dpr);
    });
    void rw;

    const gap = exact.logp - s.F09;
    const closed = gap > 1e-6 ? (100 * (s.F25 - s.F09)) / gap : 100;
    if (mOut) mOut.textContent = String(Z.length);
    if (out) out.textContent = `M = ${Z.length} · gap log p(y) − ℱ = ${gap.toFixed(2)} nats · the 2025 bound closes ${closed.toFixed(0)}% of it · 𝒰 − ℱ = ${(s.U - s.F09).toFixed(2)}`;
  };

  function diamond(x, yy, r, fill, outline = false) {
    ctx.beginPath();
    ctx.moveTo(x, yy - r); ctx.lineTo(x + r, yy); ctx.lineTo(x, yy + r); ctx.lineTo(x - r, yy); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    if (outline) { ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim() || '#fff'; ctx.lineWidth = 1.5 * dpr; ctx.stroke(); }
  }

  const stop = () => { cancelAnimationFrame(anim); anim = 0; };

  // gradient ascent on ℱ (Titsias 2009) with central differences; each step moves a point at most 0.01
  const optimise = () => {
    stop();
    let k = 0;
    const step = () => {
      const h = 1e-4;
      const g = Z.map((_, j) => {
        const zp = [...Z], zm = [...Z];
        zp[j] += h; zm[j] -= h;
        return (sparse(X, y, zp).F09 - sparse(X, y, zm).F09) / (2 * h);
      });
      const gmax = Math.max(...g.map(Math.abs), 1e-9);
      Z = Z.map((z, j) => Math.min(1, Math.max(0, z + (0.01 * g[j]) / gmax)));
      k += 1;
      return k < 140;
    };
    if (still) { while (step()); draw(); return; }
    const loop = () => { const more = step(); draw(); anim = more ? requestAnimationFrame(loop) : 0; };
    anim = requestAnimationFrame(loop);
  };

  // greedy: repeatedly add the training input with the largest unexplained variance c_i
  const greedy = (M) => {
    let z = [];
    for (let j = 0; j < M; j++) {
      if (!z.length) { z = [X[Math.floor(X.length / 2)]]; continue; }
      const ci = sparse(X, y, z).ci;
      const i = ci.reduce((bi, v, k) => (v > ci[bi] ? k : bi), 0);
      z = [...z, X[i]];
    }
    return z;
  };

  const xFromEvent = (e) => {
    const r = canvas.getBoundingClientRect();
    const { X_ } = layout();
    const px = (e.clientX - r.left) * dpr;
    return { px, x: (px - X_(0)) / (X_(1) - X_(0)) };
  };
  canvas.addEventListener('pointerdown', (e) => {
    const { px } = xFromEvent(e);
    const { X_ } = layout();
    let best = -1, bd = 16 * dpr;
    Z.forEach((z, j) => { const d = Math.abs(X_(z) - px); if (d < bd) { bd = d; best = j; } });
    if (best < 0) return;
    stop();
    drag = best;
    canvas.setPointerCapture(e.pointerId);
    draw();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (drag < 0) return;
    Z[drag] = Math.min(1, Math.max(0, xFromEvent(e).x));
    draw();
  });
  const end = () => { if (drag >= 0) { drag = -1; draw(); } };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  mInput?.addEventListener('input', () => { stop(); Z = spread(Number(mInput.value)); draw(); });
  root?.querySelector('[data-opt]')?.addEventListener('click', optimise);
  root?.querySelector('[data-greedy]')?.addEventListener('click', () => { stop(); Z = greedy(Number(mInput?.value ?? Z.length)); draw(); });
  root?.querySelector('[data-spread]')?.addEventListener('click', () => { stop(); Z = spread(Number(mInput?.value ?? Z.length)); draw(); });
  addEventListener('resize', () => { dpr = fit(canvas); draw(); });
  onTheme(draw);
  draw();
}
