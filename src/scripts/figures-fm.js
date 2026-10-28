// Flow-matching post. Data: the series' ring of 8 Gaussians; linear interpolant x_t = (1 - t) x0 + t eps.
// Everything shown is precomputed by plans/blog/flow-matching/fm.py: the exact marginal flow, and three
// trained networks (independent coupling, minibatch optimal transport, reflow).
import { theme, onTheme, fit } from './figures.js';

const K = 8, R = 2;
const MU = Array.from({ length: K }, (_, k) => [R * Math.cos((2 * Math.PI * k) / K), R * Math.sin((2 * Math.PI * k) / K)]);
let cache = null;
const load = (url) => (cache ??= fetch(url).then((r) => r.json()));
const square = (canvas, max) => { const w = canvas.getBoundingClientRect().width; canvas.style.height = `${Math.min(max, w)}px`; return fit(canvas); };

function frame(ctx, c, W, H, dpr) {
  const side = Math.min(W, H) - 20 * dpr, x0 = (W - side) / 2, y0 = 10 * dpr;
  const T = (a, b) => [x0 + ((a + 3.2) / 6.4) * side, y0 + ((3.2 - b) / 6.4) * side];
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(x0, y0, side, side);
  MU.forEach((m) => { const [X, Y] = T(m[0], m[1]); ctx.strokeStyle = `rgba(${c.ink}, 0.25)`; ctx.beginPath(); ctx.arc(X, Y, (0.3 / 6.4) * side, 0, 2 * Math.PI); ctx.stroke(); });
  return { T, x0, y0, side };
}

export function mountFmPaths(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-fmpaths]');
  const out = root?.querySelector('[data-out]');
  let dpr = square(canvas, 420), mode = 'pairs', F = null;
  const notes = {
    pairs: 'training pairs: each data point is joined to an independent noise point by a straight line, and the lines cross everywhere',
    exact: 'the flow those lines define (exact marginal velocity): curved paths that never cross',
    ot: 'trained with minibatch optimal-transport pairs: the flow is almost straight',
    reflow: 'after one round of reflow: almost perfectly straight',
  };
  const draw = () => {
    if (!F) return;
    const c = theme(), W = canvas.width, H = canvas.height;
    const { T, x0, y0, side } = frame(ctx, c, W, H, dpr);
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, side, side); ctx.clip();
    ctx.lineWidth = 1.2 * dpr;
    if (mode === 'pairs') {
      ctx.strokeStyle = `rgba(${c.a2}, 0.55)`;
      F.cond_pairs.forEach(([a, b]) => { const [X1, Y1] = T(...a), [X2, Y2] = T(...b); ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X2, Y2); ctx.stroke(); });
    } else {
      const tr = F[`traj_${mode}`];
      ctx.strokeStyle = `rgba(${c.a2}, 0.6)`;
      tr.forEach((path) => { ctx.beginPath(); path.forEach((p, i) => { const [X, Y] = T(...p); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); });
    }
    const ends = mode === 'pairs' ? F.cond_pairs.map(([a, b]) => [b, a]) : F[`traj_${mode}`].map((p) => [p[0], p[p.length - 1]]);
    ends.forEach(([z, x]) => {
      let [X, Y] = T(...z); ctx.fillStyle = `rgba(${c.ink}, 0.55)`; ctx.beginPath(); ctx.arc(X, Y, 2.2 * dpr, 0, 2 * Math.PI); ctx.fill();
      [X, Y] = T(...x); ctx.fillStyle = `rgb(${c.dot})`; ctx.beginPath(); ctx.arc(X, Y, 2.6 * dpr, 0, 2 * Math.PI); ctx.fill();
    });
    ctx.restore();
    const st = { pairs: null, exact: F.straight.exact, ot: F.straight.ot, reflow: F.straight.reflow }[mode];
    if (out) out.textContent = notes[mode] + (st ? ` · straightness (chord / arc length) ${st.toFixed(3)}` : '');
  };
  root?.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => { mode = b.dataset.mode; root.querySelectorAll('[data-mode]').forEach((o) => o.classList.toggle('on', o === b)); draw(); }));
  addEventListener('resize', () => { dpr = square(canvas, 420); draw(); });
  onTheme(draw);
  load(canvas.dataset.src).then((j) => { F = j; draw(); });
}

export function mountFmSteps(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-fmsteps]');
  const nIn = root?.querySelector('input[data-n]');
  const nOut = root?.querySelector('[data-nval]');
  const out = root?.querySelector('[data-out]');
  const NS = [1, 2, 3, 4, 6, 8, 16, 32];
  let dpr = square(canvas, 400), model = 'exact', F = null;
  const names = { exact: 'exact flow (independent pairs)', indep: 'network, independent pairs', ot: 'network, minibatch OT pairs', reflow: 'network, after reflow' };
  const draw = () => {
    if (!F) return;
    const c = theme(), W = canvas.width, H = canvas.height;
    const N = NS[Number(nIn?.value ?? 0)];
    if (nOut) nOut.textContent = String(N);
    const { T } = frame(ctx, c, W, H, dpr);
    ctx.fillStyle = `rgba(${c.dot}, 0.85)`;
    F.samples[model][String(N)].forEach(([a, b]) => { const [X, Y] = T(a, b); ctx.fillRect(X - 1.5 * dpr, Y - 1.5 * dpr, 3 * dpr, 3 * dpr); });
    const [m, on] = F.stats[model][String(N)];
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillText(`${names[model]} · ${N} Euler step${N > 1 ? 's' : ''}`, 18 * dpr, 26 * dpr);
    if (out) out.textContent = `${N} step${N > 1 ? 's' : ''} · ${names[model]}: ${m}/8 modes, ${(100 * on).toFixed(0)}% of samples on a mode (4000 samples)`;
  };
  nIn?.addEventListener('input', draw);
  root?.querySelectorAll('[data-model]').forEach((b) => b.addEventListener('click', () => { model = b.dataset.model; root.querySelectorAll('[data-model]').forEach((o) => o.classList.toggle('on', o === b)); draw(); }));
  addEventListener('resize', () => { dpr = square(canvas, 400); draw(); });
  onTheme(draw);
  load(canvas.dataset.src).then((j) => { F = j; draw(); });
}
