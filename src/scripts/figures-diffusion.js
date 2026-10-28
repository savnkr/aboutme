// Diffusion post. The data are the series' ring of 8 Gaussians (radius 2, sd 0.1), so every noised
// marginal is a Gaussian mixture and the denoiser E[x0 | x_t] is exact. Schedule: alpha = cos(pi t/2),
// sigma = sin(pi t/2), data at t = 0. (1) a lab: forward noising, the score field, and ODE / SDE sampling
// with the exact denoiser or a trained network (plans/blog/diffusion/train.py); (2) classifier-free guidance.
import { theme, onTheme, fit, seededGauss } from './figures.js';

const K = 8, R = 2, SD = 0.1;
const MU = Array.from({ length: K }, (_, k) => [R * Math.cos((2 * Math.PI * k) / K), R * Math.sin((2 * Math.PI * k) / K)]);
const al = (t) => Math.cos((Math.PI * t) / 2), sg = (t) => Math.sin((Math.PI * t) / 2);

// E[x0 | x_t] for a mixture over the components idx (exact)
function denoiseExact(x1, x2, t, idx = null) {
  const a = al(t), s = sg(t), v = a * a * SD * SD + s * s, c = (a * SD * SD) / v;
  const ks = idx ?? [0, 1, 2, 3, 4, 5, 6, 7];
  let mx = -Infinity; const lw = ks.map((k) => { const d = (x1 - a * MU[k][0]) ** 2 + (x2 - a * MU[k][1]) ** 2; const l = -0.5 * d / v; if (l > mx) mx = l; return l; });
  let z = 0, m1 = 0, m2 = 0;
  ks.forEach((k, j) => { const w = Math.exp(lw[j] - mx); z += w; m1 += w * (MU[k][0] + c * (x1 - a * MU[k][0])); m2 += w * (MU[k][1] + c * (x2 - a * MU[k][1])); });
  return [m1 / z, m2 / z];
}

// the trained noise-prediction MLP, evaluated in the browser
function makeNet(json) {
  const L = json.layers.map((l) => ({ W: Float32Array.from(l.W.flat()), b: Float32Array.from(l.b), nin: l.W.length, nout: l.b.length }));
  const freq = json.freq;
  return (x1, x2, t) => {
    let h = new Float32Array(18); h[0] = x1; h[1] = x2;
    freq.forEach((f, i) => { h[2 + i] = Math.sin((Math.PI * t * f) / 2); h[10 + i] = Math.cos((Math.PI * t * f) / 2); });
    L.forEach((l, li) => {
      const o = new Float32Array(l.nout);
      for (let j = 0; j < l.nout; j++) o[j] = l.b[j];
      for (let i = 0; i < l.nin; i++) { const hi = h[i]; if (hi === 0) continue; const row = i * l.nout; for (let j = 0; j < l.nout; j++) o[j] += hi * l.W[row + j]; }
      if (li < L.length - 1) for (let j = 0; j < l.nout; j++) o[j] = o[j] / (1 + Math.exp(-o[j]));   // SiLU
      h = o;
    });
    return [h[0], h[1]];   // predicted noise
  };
}

function metrics(P) {
  const counts = new Array(K).fill(0); let near = 0;
  P.forEach(([a, b]) => {
    let best = 0, bd = Infinity;
    MU.forEach((m, k) => { const d = Math.hypot(a - m[0], b - m[1]); if (d < bd) { bd = d; best = k; } });
    if (bd < 3 * SD) { near++; counts[best]++; }
  });
  return { modes: counts.filter((c) => c / P.length > 0.02).length, on: near / P.length, counts };
}

export function mountDiffLab(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-difflab]');
  const tIn = root?.querySelector('input[data-t]');
  const nIn = root?.querySelector('input[data-steps]');
  const nOut = root?.querySelector('[data-nval]');
  const out = root?.querySelector('[data-out]');
  const field = root?.querySelector('input[data-field]');
  let kind = 'ode', src = 'exact', net = null, anim = 0, particles = null, label = '';
  const size = () => { const w = canvas.getBoundingClientRect().width; canvas.style.height = `${Math.min(440, w)}px`; return fit(canvas); };
  let dpr = size();
  const g = seededGauss(5), n = 500;
  const X0 = Array.from({ length: n }, (_, i) => { const m = MU[i % K]; return [m[0] + SD * g(), m[1] + SD * g()]; });
  const E = Array.from({ length: n }, () => [g(), g()]);

  const denoiser = (x1, x2, t) => {
    if (src === 'exact' || !net) return denoiseExact(x1, x2, t);
    const [e1, e2] = net(x1, x2, t), a = al(t), s = sg(t);
    const c = (v) => Math.max(-3, Math.min(3, v));
    return [c((x1 - s * e1) / a), c((x2 - s * e2) / a)];
  };

  const draw = (pts, t) => {
    const c = theme();
    const W = canvas.width, H = canvas.height, side = Math.min(W, H) - 20 * dpr, x0 = (W - side) / 2, y0 = 10 * dpr;
    const T = (a, b) => [x0 + ((a + 3) / 6) * side, y0 + ((3 - b) / 6) * side];
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(x0, y0, side, side);
    MU.forEach((m) => { const [X, Y] = T(m[0], m[1]); ctx.strokeStyle = `rgba(${c.ink}, 0.25)`; ctx.beginPath(); ctx.arc(X, Y, (3 * SD / 6) * side, 0, 2 * Math.PI); ctx.stroke(); });
    if (field?.checked && t > 0.01) {
      // the score, scaled by sigma: sigma * grad log p_t = -(predicted noise), arrows of length 0.25 per unit
      ctx.strokeStyle = `rgba(${c.a2}, 0.6)`; ctx.lineWidth = 1.2 * dpr;
      for (let i = 0; i <= 14; i++) for (let j = 0; j <= 14; j++) {
        const a = -2.8 + (5.6 * i) / 14, b = -2.8 + (5.6 * j) / 14;
        const [d1, d2] = denoiseExact(a, b, t), A = al(t), S2 = sg(t);
        const s1 = (A * d1 - a) / S2, s2 = (A * d2 - b) / S2;   // sigma * score
        const m = Math.hypot(s1, s2) || 1, len = (0.2 * Math.min(1, m / 2)) / m;   // capped arrow length
        const [X, Y] = T(a, b), [U, V] = T(a + len * s1, b + len * s2);
        ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(U, V); ctx.stroke();
        ctx.beginPath(); ctx.arc(U, V, 1.3 * dpr, 0, 2 * Math.PI); ctx.fillStyle = `rgba(${c.a2}, 0.9)`; ctx.fill();
      }
    }
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, side, side); ctx.clip();
    ctx.fillStyle = `rgba(${c.dot}, 0.85)`;
    pts.forEach(([a, b]) => { const [X, Y] = T(a, b); ctx.fillRect(X - 1.4 * dpr, Y - 1.4 * dpr, 2.8 * dpr, 2.8 * dpr); });
    ctx.restore();
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillText(`${label}t = ${t.toFixed(2)}   α = ${al(t).toFixed(2)}, σ = ${sg(t).toFixed(2)}`, x0 + 6 * dpr, y0 + 16 * dpr);
  };

  const forwardView = () => {
    if (anim) return;
    const t = Number(tIn?.value ?? 0);
    label = 'forward: ';
    particles = null;
    draw(X0.map(([a, b], i) => [al(t) * a + sg(t) * E[i][0], al(t) * b + sg(t) * E[i][1]]), t);
    if (out) out.textContent = 'x_t = α_t x_0 + σ_t ε. Slide t to noise the data; press Sample to run the reverse process.';
  };

  const run = async () => {
    cancelAnimationFrame(anim); anim = 0;
    if (src === 'net' && !net) {
      if (out) out.textContent = 'loading the trained network…';
      net = makeNet(await (await fetch(canvas.dataset.net)).json());
    }
    const N = Number(nIn?.value ?? 30), ng = seededGauss(9);
    if (out) out.textContent = 'sampling…';
    const T = Array.from({ length: N + 1 }, (_, i) => 0.995 * (1 - i / N));
    let P = Array.from({ length: n }, () => [ng(), ng()]), k = 0, frame = 0;
    const sync = document.hidden || matchMedia('(prefers-reduced-motion: reduce)').matches;
    const per = sync ? 1 : Math.max(1, Math.round(80 / N));
    label = `${kind === 'ode' ? 'ODE (DDIM)' : 'SDE (DDPM)'}, ${src === 'exact' ? 'exact score' : 'network'}: `;
    const step = () => {
      if (frame++ % per === 0) {
        const t = T[k], s = T[k + 1], a = al(t), st = sg(t), as = al(s), ss = sg(s);
        P = P.map(([x1, x2]) => {
          const [d1, d2] = denoiser(x1, x2, t);
          if (kind === 'ode') { const e1 = (x1 - a * d1) / st, e2 = (x2 - a * d2) / st; return [as * d1 + ss * e1, as * d2 + ss * e2]; }
          const ats = a / as, vts = st * st - ats * ats * ss * ss;
          const c1 = (ats * ss * ss) / (st * st), c2 = (as * vts) / (st * st), sdv = s > 0 ? Math.sqrt((vts * ss * ss) / (st * st)) : 0;
          return [c1 * x1 + c2 * d1 + sdv * ng(), c1 * x2 + c2 * d2 + sdv * ng()];
        });
        k++;
        if (tIn) tIn.value = String(s);
      }
      if (k < N) { if (!sync) { draw(P, T[k]); anim = requestAnimationFrame(step); } return; }
      draw(P, T[k]);
      anim = 0; particles = P;
      const m = metrics(P);
      if (out) out.textContent = `${N} steps · ${kind === 'ode' ? 'ODE' : 'SDE'} · ${src === 'exact' ? 'exact score' : 'trained network'}: ${m.modes}/8 modes, ${(100 * m.on).toFixed(0)}% of samples on a mode`;
    };
    // no animation for readers who prefer reduced motion, or when the page is hidden
    if (sync) { while (k < N) { frame = 0; step(); } return; }
    anim = requestAnimationFrame(step);
  };

  tIn?.addEventListener('input', () => { cancelAnimationFrame(anim); anim = 0; forwardView(); });
  field?.addEventListener('change', () => (particles ? draw(particles, 0) : forwardView()));
  nIn?.addEventListener('input', () => { if (nOut) nOut.textContent = nIn.value; });
  root?.querySelectorAll('[data-kind]').forEach((b) => b.addEventListener('click', () => { kind = b.dataset.kind; root.querySelectorAll('[data-kind]').forEach((o) => o.classList.toggle('on', o === b)); }));
  root?.querySelectorAll('[data-src]').forEach((b) => b.addEventListener('click', () => { src = b.dataset.src; root.querySelectorAll('[data-src]').forEach((o) => o.classList.toggle('on', o === b)); }));
  root?.querySelector('[data-run]')?.addEventListener('click', run);
  addEventListener('resize', () => { dpr = size(); particles ? draw(particles, 0) : forwardView(); });
  onTheme(() => (particles ? draw(particles, 0) : forwardView()));
  forwardView();
}

export function mountGuidance(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-guidance]');
  const wIn = root?.querySelector('input[data-w]');
  const out = root?.querySelector('[data-out]');
  const CLS = [1, 2, 3];
  const size = () => { const w = canvas.getBoundingClientRect().width; canvas.style.height = `${Math.min(380, w)}px`; return fit(canvas); };
  let dpr = size();
  const draw = () => {
    const c = theme(), w = Number(wIn?.value ?? 0);
    const g = seededGauss(4), N = 40, n = 1200;
    const T = Array.from({ length: N + 1 }, (_, i) => 0.995 * (1 - i / N));
    let P = Array.from({ length: n }, () => [g(), g()]);
    for (let k = 0; k < N; k++) {
      const t = T[k], s = T[k + 1], a = al(t), st = sg(t), as = al(s), ss = sg(s);
      P = P.map(([x1, x2]) => {
        const [c1, c2] = denoiseExact(x1, x2, t, CLS), [u1, u2] = denoiseExact(x1, x2, t);
        const d1 = (1 + w) * c1 - w * u1, d2 = (1 + w) * c2 - w * u2;            // guided denoiser
        const e1 = (x1 - a * d1) / st, e2 = (x2 - a * d2) / st;
        return [as * d1 + ss * e1, as * d2 + ss * e2];
      });
    }
    const W = canvas.width, H = canvas.height, side = Math.min(W, H) - 20 * dpr, x0 = (W - side) / 2, y0 = 10 * dpr;
    const Tr = (a, b) => [x0 + ((a + 3) / 6) * side, y0 + ((3 - b) / 6) * side];
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = c.line; ctx.lineWidth = dpr; ctx.strokeRect(x0, y0, side, side);
    MU.forEach((m, k) => {
      const [X, Y] = Tr(m[0], m[1]);
      ctx.strokeStyle = CLS.includes(k) ? `rgba(${c.a1}, 0.9)` : `rgba(${c.ink}, 0.25)`; ctx.lineWidth = (CLS.includes(k) ? 2 : 1) * dpr;
      ctx.beginPath(); ctx.arc(X, Y, (3 * SD / 6) * side, 0, 2 * Math.PI); ctx.stroke();
    });
    ctx.fillStyle = `rgba(${c.dot}, 0.8)`;
    P.forEach(([a, b]) => { const [X, Y] = Tr(a, b); ctx.fillRect(X - 1.3 * dpr, Y - 1.3 * dpr, 2.6 * dpr, 2.6 * dpr); });
    ctx.fillStyle = `rgba(${c.ink}, 0.8)`; ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillText(`class y: the three teal modes · w = ${w.toFixed(2)}`, x0 + 6 * dpr, y0 + 16 * dpr);
    const m = metrics(P), share = CLS.map((k) => Math.round((100 * m.counts[k]) / n));
    const rad = P.reduce((acc, [a, b]) => acc + Math.hypot(a, b), 0) / n;
    if (out) out.textContent = `share of samples at the three class modes: ${share.join('% · ')}% · mean radius ${rad.toFixed(2)} (data: 2.00)`;
  };
  wIn?.addEventListener('input', draw);
  addEventListener('resize', () => { dpr = size(); draw(); });
  onTheme(draw);
  draw();
}
