// Generative-models opener: (1) a pushforward of Gaussian noise through an invertible map, with the
// change-of-variables formula live; (2) forward vs reverse KL fits of one Gaussian to two modes;
// (3) a flow, a VAE and a GAN trained on the series' toy (precomputed by plans/blog/generative/families.py).
import { theme, onTheme, fit, seededGauss } from './figures.js';

// ---------- (1) pushforward ----------
const K = 0.25, OMEGA = 2.2, BETA = 0.45;      // radial stretch, swirl, shear
function forward(z1, z2, t) {
  const r = Math.hypot(z1, z2);
  const g = r + t * K * r * r;
  let u1 = r > 0 ? (z1 * g) / r : 0, u2 = r > 0 ? (z2 * g) / r : 0;
  const phi = t * OMEGA * Math.exp(-(g * g) / 4);
  const v1 = u1 * Math.cos(phi) - u2 * Math.sin(phi), v2 = u1 * Math.sin(phi) + u2 * Math.cos(phi);
  return [v1, v2 + t * BETA * (v1 * v1 / 2 - 1)];
}
function inverse(x1, x2, t) {
  const v1 = x1, v2 = x2 - t * BETA * (x1 * x1 / 2 - 1);
  const rho = Math.hypot(v1, v2);
  const phi = t * OMEGA * Math.exp(-(rho * rho) / 4);
  const u1 = v1 * Math.cos(phi) + v2 * Math.sin(phi), u2 = -v1 * Math.sin(phi) + v2 * Math.cos(phi);
  const r = t > 0 ? (-1 + Math.sqrt(1 + 4 * t * K * rho)) / (2 * t * K) : rho;
  const s = rho > 0 ? r / rho : 0;
  return [u1 * s, u2 * s, r];
}
const logN2 = (a, b) => -0.5 * (a * a + b * b) - Math.log(2 * Math.PI);
const logDet = (r, t) => Math.log(1 + 2 * t * K * r) + Math.log(1 + t * K * r);   // radial part; swirl and shear have det 1

export function mountPush(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-push]');
  const tIn = root?.querySelector('input[data-t]');
  const out = root?.querySelector('[data-out]');
  const size = () => { const w = canvas.getBoundingClientRect().width; canvas.style.height = `${Math.min(340, (w - 40) / 2 + 44)}px`; return fit(canvas); };
  let dpr = size();
  const g = seededGauss(3);
  const Z = Array.from({ length: 700 }, () => [g(), g()]);
  let hover = null;   // a point in x-space
  const ZL = 3.2, XL = 5.5;

  const draw = () => {
    const c = theme();
    const t = Number(tIn?.value ?? 1);
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const side = Math.min((W - 40 * dpr) / 2, H - 40 * dpr);
    const P = [10 * dpr, 10 * dpr + side + 20 * dpr], y0 = 24 * dpr;
    const toZ = (a, b) => [P[0] + ((a + ZL) / (2 * ZL)) * side, y0 + ((ZL - b) / (2 * ZL)) * side];
    const toX = (a, b) => [P[1] + ((a + XL) / (2 * XL)) * side, y0 + ((XL - b) / (2 * XL)) * side];
    // density heat in x-space from the change-of-variables formula
    const n = 90, off = document.createElement('canvas'); off.width = n; off.height = n;
    const octx = off.getContext('2d'), img = octx.createImageData(n, n);
    const [r0, g0, b0] = c.a1.split(',').map(Number);
    const vals = new Float64Array(n * n); let mx = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x1 = -XL + (2 * XL * (i + 0.5)) / n, x2 = XL - (2 * XL * (j + 0.5)) / n;
      const [z1, z2, r] = inverse(x1, x2, t);
      const v = Math.exp(logN2(z1, z2) - logDet(r, t)); vals[j * n + i] = v; if (v > mx) mx = v;
    }
    for (let k = 0; k < n * n; k++) { img.data[4 * k] = r0; img.data[4 * k + 1] = g0; img.data[4 * k + 2] = b0; img.data[4 * k + 3] = Math.round(200 * Math.pow(vals[k] / mx, 0.5)); }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, P[1], y0, side, side);
    // frames and titles
    ctx.strokeStyle = c.line; ctx.lineWidth = dpr;
    ctx.strokeRect(P[0], y0, side, side); ctx.strokeRect(P[1], y0, side, side);
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${11.5 * dpr}px ui-monospace, monospace`;
    ctx.fillText('noise z ~ N(0, I)', P[0] + 2 * dpr, 15 * dpr);
    ctx.fillText(`data x = f(z), t = ${t.toFixed(2)}`, P[1] + 2 * dpr, 15 * dpr);
    // the same grid lines, before and after the map (clipped to the two panels)
    ctx.save(); ctx.beginPath(); ctx.rect(P[0], y0, side, side); ctx.rect(P[1], y0, side, side); ctx.clip();
    ctx.strokeStyle = `rgba(${c.ink}, 0.22)`; ctx.lineWidth = dpr;
    for (let k = -6; k <= 6; k++) {
      const a = k * 0.5;
      [[(s) => [a, s]], [(s) => [s, a]]].forEach(([line]) => {
        ctx.beginPath();
        for (let q = 0; q <= 60; q++) { const s = -ZL + (2 * ZL * q) / 60; const [p1, p2] = line(s); const [X, Y] = toZ(p1, p2); q ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
        ctx.stroke();
        ctx.beginPath();
        for (let q = 0; q <= 60; q++) { const s = -ZL + (2 * ZL * q) / 60; const [p1, p2] = line(s); const [f1, f2] = forward(p1, p2, t); const [X, Y] = toX(f1, f2); q ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
        ctx.stroke();
      });
    }
    ctx.restore();
    // samples, coloured by their half-plane in z so you can follow them
    ctx.save(); ctx.beginPath(); ctx.rect(P[1], y0, side, side); ctx.rect(P[0], y0, side, side); ctx.clip();
    Z.forEach(([a, b]) => {
      const col = Math.atan2(b, a) > 0 ? c.dot : c.a2;
      ctx.fillStyle = `rgba(${col}, 0.7)`;
      const [X, Y] = toZ(a, b); ctx.beginPath(); ctx.arc(X, Y, 1.6 * dpr, 0, 2 * Math.PI); ctx.fill();
      const [f1, f2] = forward(a, b, t); const [U, V] = toX(f1, f2); ctx.beginPath(); ctx.arc(U, V, 1.6 * dpr, 0, 2 * Math.PI); ctx.fill();
    });
    ctx.restore();
    // a hovered point and its preimage, with the formula
    const hx = hover ?? [1.6, 1.2];
    const [z1, z2, r] = inverse(hx[0], hx[1], t);
    const lp = logN2(z1, z2), ld = logDet(r, t);
    [[toX(hx[0], hx[1])], [toZ(z1, z2)]].forEach(([[X, Y]]) => { ctx.strokeStyle = `rgb(${c.ink})`; ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.arc(X, Y, 6 * dpr, 0, 2 * Math.PI); ctx.stroke(); });
    if (out) out.textContent = `at the circled point: log p_x(x) = log p_z(z) − log|det ∂f/∂z| = ${lp.toFixed(2)} − ${ld.toFixed(2)} = ${(lp - ld).toFixed(2)}`;
    canvas._xmap = { P, y0, side };
  };
  canvas.addEventListener('pointermove', (e) => {
    const m = canvas._xmap; if (!m) return;
    const r = canvas.getBoundingClientRect();
    const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
    if (px < m.P[1] || px > m.P[1] + m.side || py < m.y0 || py > m.y0 + m.side) return;
    hover = [-XL + ((px - m.P[1]) / m.side) * 2 * XL, XL - ((py - m.y0) / m.side) * 2 * XL];
    draw();
  });
  tIn?.addEventListener('input', draw);
  addEventListener('resize', () => { dpr = size(); draw(); });
  onTheme(draw);
  draw();
}

// ---------- (2) forward vs reverse KL ----------
export function mountKL(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-kl]');
  const mIn = root?.querySelector('input[data-sep]');
  const out = root?.querySelector('[data-out]');
  let dpr = fit(canvas);
  const W1 = 0.6, W2 = 0.4, S = 0.5;
  const xs = Array.from({ length: 401 }, (_, i) => -8 + (16 * i) / 400);
  const dx = xs[1] - xs[0];
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (Math.sqrt(2 * Math.PI) * s);
  const fitAll = (m) => {
    const p = xs.map((x) => W1 * npdf(x, -m, S) + W2 * npdf(x, m, S));
    const mean = W1 * -m + W2 * m;
    const fwd = [mean, Math.sqrt(W1 * (S * S + m * m) + W2 * (S * S + m * m) - mean * mean)];
    const rkl = (mu, sd) => xs.reduce((acc, x, i) => { const q = npdf(x, mu, sd); return q > 1e-12 ? acc + q * Math.log(q / Math.max(p[i], 1e-300)) * dx : acc; }, 0);
    let best = [0, 1, Infinity];
    for (let mu = -4; mu <= 4; mu += 0.1) for (let sd = 0.1; sd <= 3; sd += 0.075) { const v = rkl(mu, sd); if (v < best[2]) best = [mu, sd, v]; }
    const [m0, s0] = best;
    for (let mu = m0 - 0.1; mu <= m0 + 0.1; mu += 0.01) for (let sd = Math.max(0.05, s0 - 0.08); sd <= s0 + 0.08; sd += 0.005) { const v = rkl(mu, sd); if (v < best[2]) best = [mu, sd, v]; }
    return { p, fwd, rev: best };
  };
  const draw = () => {
    const c = theme();
    const m = Number(mIn?.value ?? 2);
    const { p, fwd, rev } = fitAll(m);
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const X_ = (x) => 10 * dpr + ((x + 5) / 10) * (W - 20 * dpr);
    const top = 28 * dpr, bot = H - 16 * dpr;
    const qf = xs.map((x) => npdf(x, fwd[0], fwd[1])), qr = xs.map((x) => npdf(x, rev[0], rev[1]));
    const ymax = Math.max(...p, ...qf, ...qr) * 1.05;
    const Y_ = (v) => bot - (v / ymax) * (bot - top);
    ctx.fillStyle = `rgba(${c.ink}, 0.13)`; ctx.beginPath(); ctx.moveTo(X_(-5), Y_(0));
    xs.forEach((x, i) => { if (x >= -5 && x <= 5) ctx.lineTo(X_(x), Y_(p[i])); }); ctx.lineTo(X_(5), Y_(0)); ctx.fill();
    [[qf, c.a1, false], [qr, c.dot, true]].forEach(([q, col, dash]) => {
      ctx.strokeStyle = `rgb(${col})`; ctx.lineWidth = 2.4 * dpr; ctx.setLineDash(dash ? [6 * dpr, 4 * dpr] : []);
      ctx.beginPath(); xs.forEach((x, i) => { if (x >= -5 && x <= 5) ctx.lineTo(X_(x), Y_(q[i])); }); ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.font = `${11.5 * dpr}px ui-monospace, monospace`;
    ctx.fillStyle = `rgba(${c.ink}, 0.7)`; ctx.fillText('p (two modes)', X_(-4.9), 16 * dpr);
    ctx.fillStyle = `rgb(${c.a1})`; ctx.fillText('— maximum likelihood: min KL(p‖q)', X_(-4.9) + 118 * dpr, 16 * dpr);
    ctx.fillStyle = `rgb(${c.dot})`; ctx.fillText('- - min KL(q‖p)', X_(-4.9) + 396 * dpr, 16 * dpr);
    if (out) out.textContent = `separation ±${m.toFixed(1)} · KL(p‖q) fit: mean ${fwd[0].toFixed(2)}, sd ${fwd[1].toFixed(2)} · KL(q‖p) fit: mean ${rev[0].toFixed(2)}, sd ${rev[1].toFixed(2)}`;
  };
  mIn?.addEventListener('input', draw);
  addEventListener('resize', () => { dpr = fit(canvas); draw(); });
  onTheme(draw);
  draw();
}

// ---------- (3) same data, three families ----------
export function mountFamilies(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-families]');
  let dpr = fit(canvas);
  let F = null;
  const size = () => {
    const w = canvas.getBoundingClientRect().width;
    const cols = w < 560 ? 2 : 4, side = (w - 20 - (cols - 1) * 14) / cols;
    canvas.style.height = `${(4 / cols) * (side + 62) + 6}px`;
    return fit(canvas);
  };
  const draw = () => {
    if (!F) return;
    const c = theme();
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const cols = W / dpr < 560 ? 2 : 4, gap = 14 * dpr;
    const side = (W - 20 * dpr - (cols - 1) * gap) / cols;
    const panels = [['data', 'the data'], ['flow', 'normalising flow'], ['vae', 'VAE'], ['gan', 'GAN']];
    panels.forEach(([key, name], i) => {
      const x0 = 10 * dpr + (i % cols) * (side + gap), y0 = 20 * dpr + Math.floor(i / cols) * (side + 62 * dpr);
      const T = (a, b) => [x0 + ((a + 3) / 6) * side, y0 + ((3 - b) / 6) * side];
      if (key === 'flow') {
        const n = F.grid[2], off = document.createElement('canvas'); off.width = n; off.height = n;
        const octx = off.getContext('2d'), img = octx.createImageData(n, n);
        const [r0, g0, b0] = c.a1.split(',').map(Number);
        let mx = -Infinity; F.flow_logp.forEach((row) => row.forEach((v) => { if (v > mx) mx = v; }));
        F.flow_logp.forEach((row, j) => row.forEach((v, k) => {
          const q = 4 * ((n - 1 - j) * n + k);
          img.data[q] = r0; img.data[q + 1] = g0; img.data[q + 2] = b0; img.data[q + 3] = Math.round(150 * Math.pow(Math.exp(v - mx), 0.35));
        }));
        octx.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, x0, y0, side, side);
      }
      ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(x0, y0, side, side);
      ctx.fillStyle = `rgba(${key === 'data' ? c.ink : c.dot}, 0.75)`;
      F[key].forEach(([a, b]) => { const [X, Y] = T(a, b); ctx.fillRect(X - 1.1 * dpr, Y - 1.1 * dpr, 2.2 * dpr, 2.2 * dpr); });
      const s = F.stats[key];
      ctx.fillStyle = `rgba(${c.ink}, 0.85)`; ctx.font = `${11.5 * dpr}px ui-monospace, monospace`;
      ctx.fillText(name, x0 + 2 * dpr, y0 - 6 * dpr);
      ctx.fillStyle = `rgba(${c.ink}, 0.65)`; ctx.font = `${10.5 * dpr}px ui-monospace, monospace`;
      const narrow = cols === 2;
      ctx.fillText(narrow ? `${s.modes}/8 modes · ${(100 * s.on_mode).toFixed(0)}% on` : `modes ${s.modes}/8 · on a mode ${(100 * s.on_mode).toFixed(0)}%`, x0 + 2 * dpr, y0 + side + 15 * dpr);
      const lik = key === 'data' ? `NLL ${F.stats.true_nll} (true)` : key === 'flow' ? `NLL ${s.nll}` : key === 'vae' ? (narrow ? `−ELBO ${s.nelbo}` : `−ELBO ${s.nelbo} ≥ NLL`) : 'no likelihood';
      ctx.fillText(lik, x0 + 2 * dpr, y0 + side + 30 * dpr);
    });
  };
  addEventListener('resize', () => { dpr = size(); draw(); });
  onTheme(draw);
  dpr = size();
  fetch(canvas.dataset.src).then((r) => r.json()).then((j) => { F = j; draw(); });
  void root;
}
