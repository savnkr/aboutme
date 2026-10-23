const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// Colours come from the CSS tokens so every figure follows the light/dark theme.
function theme() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return {
    ink: g('--ink-rgb') || '28, 26, 39',
    a1: g('--a1-rgb') || '79, 70, 229',
    a2: g('--a2-rgb') || '8, 145, 178',
    dot: g('--dot-rgb') || '245, 158, 11',
    line: g('--line') || '#e5e3ef',
  };
}
const onTheme = (fn) => {
  addEventListener('themechange', fn);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', fn);
};

function fit(canvas) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  return dpr;
}

export function mountPrior(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = fit(canvas);
  const paths = 5;
  const n = 72;
  const xs = Array.from({ length: n }, (_, i) => i / (n - 1));
  const ell = 0.16;
  const K = xs.map((a) => xs.map((b) => Math.exp(-0.5 * ((a - b) / ell) ** 2)));
  for (let i = 0; i < n; i++) K[i][i] += 1e-6;
  const L = cholesky(K);
  let seed = 20261023;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-9))) * Math.cos(2 * Math.PI * rnd());
  const prior = Array.from({ length: paths }, () => {
    const z = Array.from({ length: n }, gauss);
    return xs.map((_, i) => {
      let s = 0;
      for (let j = 0; j <= i; j++) s += L[i][j] * z[j];
      return s;
    });
  });
  const mean = xs.map((x) => 0.08 * Math.sin(Math.PI * x));
  let start = 0;
  let tries = 0;
  const draw = (now) => {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const hero = canvas.getBoundingClientRect();
    const photo = canvas.parentElement?.querySelector('.portrait-frame')?.getBoundingClientRect();
    const sx = w / Math.max(hero.width, 1);
    const sy = h / Math.max(hero.height, 1);
    const copy = canvas.parentElement?.querySelector('.hero-copy')?.getBoundingClientRect();
    const hitsCopy = (box) =>
      copy &&
      box.left < copy.right - 12 &&
      box.right > copy.left + 12 &&
      box.top < copy.bottom - 12 &&
      box.bottom > copy.top + 12;
    let band = null;
    if (photo && photo.width > 8) {
      const columnLeft = copy ? Math.max(photo.left, copy.right + 18) : photo.left;
      const boxes = [
        { left: columnLeft, right: photo.right, top: hero.top + 16, bottom: photo.top - 14 },
        { left: photo.left, right: photo.right, top: hero.top + 28, bottom: photo.top - 24 },
        {
          left: photo.right + 18,
          right: hero.right - 36,
          top: photo.top + 10,
          bottom: photo.top + photo.height * 0.72,
        },
      ];
      if (copy && photo.left - copy.right > 72) {
        boxes.push({
          left: copy.right + 16,
          right: photo.left - 16,
          top: photo.top + 8,
          bottom: photo.top + Math.min(photo.height * 0.7, 280),
        });
      }
      const box = boxes.find((item) => item.right - item.left > 88 && item.bottom - item.top > 48 && !hitsCopy(item));
      if (box) {
        band = {
          left: (box.left - hero.left) * sx,
          right: (box.right - hero.left) * sx,
          top: (box.top - hero.top) * sy,
          bottom: (box.bottom - hero.top) * sy,
        };
      }
    }
    if (!band || band.right - band.left < 40 * dpr || band.bottom - band.top < 48 * dpr) {
      if (tries++ < 40) requestAnimationFrame(draw);
      return;
    }
    if (!start) start = now;
    const t = reduce() ? 1 : Math.min(1, (now - start) / 2000);
    const ease = t * t * (3 - 2 * t);
    const { left: bandLeft, right: bandRight, top: bandTop, bottom: bandBottom } = band;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(1, 0.85 * dpr);
    const spread = 0.62 * (1 - ease) + 0.28;
    for (let i = 0; i < paths; i++) {
      ctx.beginPath();
      xs.forEach((x, k) => {
        const y = mean[k] + prior[i][k] * spread;
        const X = bandLeft + x * (bandRight - bandLeft);
        const Y = (bandTop + bandBottom) / 2 - y * (bandBottom - bandTop) * 0.42;
        if (k === 0) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      });
      ctx.strokeStyle = `rgba(28, 71, 56, ${0.28 + 0.14 * ease})`;
      ctx.stroke();
    }
    if (t < 1) requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

function kernel(a, b) {
  const ell = 0.12;
  return Math.exp(-0.5 * ((a - b) / ell) ** 2);
}

function cholesky(K) {
  const n = K.length;
  const L = K.map((row) => row.map(() => 0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = K[i][j];
      for (let p = 0; p < j; p++) s -= L[i][p] * L[j][p];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-9)) : s / L[j][j];
    }
  }
  return L;
}

function solve(L, b) {
  const n = b.length;
  const y = [...b];
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < i; p++) y[i] -= L[i][p] * y[p];
    y[i] /= L[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let p = i + 1; p < n; p++) y[i] -= L[p][i] * y[p];
    y[i] /= L[i][i];
  }
  return y;
}

export function mountGp(canvas) {
  const ctx = canvas.getContext('2d');
  let obs = [
    [0.12, 0.35],
    [0.3, -0.45],
    [0.5, 0.7],
    [0.68, 0.1],
    [0.86, -0.35],
  ];
  const paint = () => {
    const dpr = fit(canvas);
    const W = canvas.width;
    const H = canvas.height;
    const N = 80;
    const xs = Array.from({ length: N }, (_, i) => i / (N - 1));
    const n = obs.length;
    const K = obs.map((a) => obs.map((b) => kernel(a[0], b[0])));
    for (let i = 0; i < n; i++) K[i][i] += 0.02;
    const L = cholesky(K);
    const alpha = solve(L, obs.map((o) => o[1]));
    const Ks = xs.map((x) => obs.map((o) => kernel(x, o[0])));
    const mu = Ks.map((row) => row.reduce((s, v, j) => s + v * alpha[j], 0));
    const v = Ks.map((row) => {
      const solved = solve(L, row);
      return Math.max(0, 1 - row.reduce((s, val, j) => s + val * solved[j], 0));
    });
    const X = (x) => 28 * dpr + x * (W - 56 * dpr);
    const Y = (y) => H / 2 - y * H * 0.28;
    const c = theme();
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = c.line;
    ctx.lineWidth = dpr;
    for (let g = 0; g <= 8; g++) {
      ctx.beginPath();
      ctx.moveTo(X(g / 8), 0);
      ctx.lineTo(X(g / 8), H);
      ctx.stroke();
    }
    ctx.fillStyle = `rgba(${c.a1}, 0.16)`;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(mu[i] + 2 * Math.sqrt(v[i]))));
    for (let i = N - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Y(mu[i] - 2 * Math.sqrt(v[i])));
    ctx.fill();
    ctx.strokeStyle = `rgb(${c.ink})`;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(mu[i])));
    ctx.stroke();
    obs.forEach((o) => {
      ctx.fillStyle = `rgb(${c.dot})`;
      ctx.beginPath();
      ctx.arc(X(o[0]), Y(o[1]), 4.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    });
  };
  paint();
  onTheme(paint);
  canvas.addEventListener('click', (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (0.5 - (event.clientY - rect.top) / rect.height) / 0.28;
    if (x < 0.02 || x > 0.98 || obs.length > 16) return;
    obs = [...obs, [x, Math.max(-1.4, Math.min(1.4, y))]];
    paint();
  });
  canvas.parentElement?.querySelector('[data-reset]')?.addEventListener('click', () => {
    obs = [
      [0.12, 0.35],
      [0.5, 0.7],
      [0.86, -0.35],
    ];
    paint();
  });
}

export function mountFlow(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = fit(canvas);
  const W = canvas.width;
  const H = canvas.height;
  const steps = 7;
  ctx.clearRect(0, 0, W, H);
  for (let s = 0; s < steps; s++) {
    const t = s / (steps - 1);
    ctx.strokeStyle = `rgba(${theme().a1}, ${0.15 + 0.7 * t})`;
    ctx.lineWidth = (1 + t * 1.4) * dpr;
    ctx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const x = i / 80;
      const noise = Math.sin(i * 0.7 + s) * (1 - t) * 0.35;
      const y = Math.sin(x * Math.PI) * 0.7 + noise;
      const X = 24 * dpr + x * (W - 48 * dpr);
      const Y = H * 0.72 - y * H * 0.45;
      if (i === 0) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    }
    ctx.stroke();
  }
  ctx.fillStyle = `rgb(${theme().a2})`;
  [[0.2, 0.35], [0.48, 0.72], [0.8, 0.4]].forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(24 * dpr + x * (W - 48 * dpr), H * 0.72 - y * H * 0.45, 4 * dpr, 0, 7);
    ctx.fill();
  });
}

export function mountRobot(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = fit(canvas);
  const W = canvas.width;
  const H = canvas.height;
  const P = (t) => [0.12 * W + t * 0.76 * W, H * 0.62 - Math.sin(t * Math.PI) * H * 0.34];
  ctx.clearRect(0, 0, W, H);
  for (let w = 42; w > 0; w -= 6) {
    ctx.strokeStyle = `rgba(${theme().a1}, ${0.04 + (42 - w) / 400})`;
    ctx.lineWidth = w * dpr * 0.35;
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let t = 0; t <= 1; t += 0.02) {
      const [x, y] = P(t);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.strokeStyle = `rgb(${theme().ink})`;
  ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath();
  for (let t = 0; t <= 1; t += 0.02) {
    const [x, y] = P(t);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
  const base = [0.12 * W, H * 0.86];
  const joint = [0.28 * W, H * 0.48];
  const end = P(0.18);
  ctx.strokeStyle = `rgb(${theme().a2})`;
  ctx.lineWidth = 5 * dpr;
  ctx.beginPath();
  ctx.moveTo(base[0], base[1]);
  ctx.lineTo(joint[0], joint[1]);
  ctx.lineTo(end[0], end[1]);
  ctx.stroke();
  [base, joint, end].forEach((p) => {
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = `rgb(${theme().a2})`;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.arc(p[0], p[1], 6 * dpr, 0, 7);
    ctx.fill();
    ctx.stroke();
  });
}

// Scroll-told story: one canvas, four scenes, crossfaded by scroll progress.
// scene 0 race-car chassis + airflow, 1 welding robot on a rigid path,
// 2 GP prior collapsing to a posterior, 3 robot with an uncertainty tube that makes it stop.
export function mountStory(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-story]');
  const chapters = root ? [...root.querySelectorAll('[data-chapter]')] : [];
  let ink = '';
  let green = '';
  let rust = '';
  let dpr = fit(canvas);
  let W = canvas.width;
  let H = canvas.height;
  let progress = 0;
  let shown = 0;
  let visible = false;
  let raf = 0;
  const still = reduce();

  const resize = () => {
    dpr = fit(canvas);
    W = canvas.width;
    H = canvas.height;
  };
  addEventListener('resize', resize);

  const readProgress = () => {
    const vis = root?.querySelector('.saga-visual')?.getBoundingClientRect();
    const stacked = vis && vis.top <= 1 && vis.bottom < innerHeight * 0.6;
    const mid = stacked ? (vis.bottom + innerHeight) / 2 : innerHeight / 2;
    let p = 0;
    chapters.forEach((el, i) => {
      const r = el.getBoundingClientRect();
      const centre = r.top + r.height / 2;
      if (centre < mid + r.height / 2) p = Math.max(p, i + Math.min(1, Math.max(0, (mid - centre) / r.height + 0.5)) - 0.5);
      el.classList.toggle('active', r.top < mid && r.bottom > mid);
    });
    progress = Math.max(0, Math.min(chapters.length - 1, p));
    root?.querySelectorAll('.saga-dots li').forEach((d, i) => d.classList.toggle('on', Math.round(progress) === i));
  };

  const line = (pts, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width * dpr;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  };
  const dot = (x, y, r, fill, stroke) => {
    ctx.beginPath();
    ctx.arc(x, y, r * dpr, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
    }
  };

  // 0: a space-frame chassis drawn in, with streamlines flowing over it.
  const car = (a, t, draw) => {
    const s = Math.min(W, H * 1.6);
    const ox = W / 2 - s * 0.42;
    const oy = H * 0.58;
    const P = ([x, y]) => [ox + x * s * 0.84, oy - y * s * 0.3];
    const nodes = [[0, 0.05], [0.18, 0], [0.42, 0], [0.66, 0], [0.9, 0.02], [1, 0.12], [0.08, 0.3], [0.3, 0.42], [0.46, 0.75], [0.6, 0.45], [0.82, 0.32]];
    const bars = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 6], [6, 1], [1, 7], [6, 7], [7, 2], [7, 8], [8, 9], [2, 9], [9, 3], [9, 10], [3, 10], [10, 4], [10, 5]];
    const k = Math.floor(draw * bars.length);
    bars.slice(0, Math.max(1, k + 1)).forEach(([i, j], n) => {
      const f = n < k ? 1 : (draw * bars.length) % 1;
      const [x1, y1] = P(nodes[i]);
      const [x2, y2] = P(nodes[j]);
      line([[x1, y1], [x1 + (x2 - x1) * f, y1 + (y2 - y1) * f]], `rgba(${ink}, ${0.75 * a})`, 1.6);
    });
    [[0.17, -0.02], [0.8, -0.02]].forEach((w) => {
      const [x, y] = P(w);
      ctx.beginPath();
      ctx.arc(x, y, s * 0.075, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${ink}, ${0.8 * a})`;
      ctx.lineWidth = 2.5 * dpr;
      ctx.stroke();
    });
    for (let r = 0; r < 7; r++) {
      const base = 0.15 + r * 0.14;
      const pts = [];
      for (let i = 0; i <= 60; i++) {
        const x = -0.15 + (i / 60) * 1.3;
        const bump = Math.exp(-(((x - 0.46) / 0.32) ** 2)) * Math.max(0, 0.95 - base) * 0.55;
        pts.push(P([x, base + bump + 0.02 * Math.sin(x * 9 + t * 2 + r)]));
      }
      ctx.setLineDash([10 * dpr, 14 * dpr]);
      ctx.lineDashOffset = -t * 60 * dpr;
      line(pts, `rgba(${green}, ${0.45 * a})`, 1.2);
      ctx.setLineDash([]);
    }
  };

  // Two-link arm helper.
  const arm = (base, target, l1, l2, color, a) => {
    const dx = target[0] - base[0];
    const dy = target[1] - base[1];
    const d = Math.min(Math.hypot(dx, dy), l1 + l2 - 1);
    const q = Math.acos((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d));
    const ang = Math.atan2(dy, dx) - q;
    const elbow = [base[0] + l1 * Math.cos(ang), base[1] + l1 * Math.sin(ang)];
    ctx.lineCap = 'round';
    line([base, elbow, target], `rgba(${color}, ${a})`, 7);
    [base, elbow].forEach((p) => dot(p[0], p[1], 6, `rgba(255, 255, 255, ${a})`, `rgba(${color}, ${a})`));
    ctx.fillStyle = `rgba(${ink}, ${0.85 * a})`;
    ctx.fillRect(base[0] - 26 * dpr, base[1] + 6 * dpr, 52 * dpr, 10 * dpr);
  };
  const seam = (u) => [W * (0.25 + 0.55 * u), H * 0.68 - Math.sin(u * Math.PI) * H * 0.12];

  // 1: a welding robot following a programmed seam, exactly and blindly.
  const welder = (a, t) => {
    const pts = Array.from({ length: 50 }, (_, i) => seam(i / 49));
    ctx.setLineDash([4 * dpr, 6 * dpr]);
    line(pts, `rgba(${ink}, ${0.35 * a})`, 1.2);
    ctx.setLineDash([]);
    const u = (Math.sin(t * 0.8) + 1) / 2;
    line(pts.slice(0, Math.max(1, Math.round(u * 49))), `rgba(${rust}, ${0.9 * a})`, 3);
    const tip = seam(u);
    const base = [W * 0.2, H * 0.84];
    const reach = Math.hypot(seam(1)[0] - base[0], seam(1)[1] - base[1]);
    arm(base, tip, reach * 0.55, reach * 0.5, ink, a);
    for (let s = 0; s < 6; s++) {
      const ang = t * 7 + s;
      const r = (4 + 6 * ((t * 3 + s / 6) % 1)) * dpr;
      dot(tip[0] + Math.cos(ang) * r, tip[1] + Math.sin(ang) * r, 1.3, `rgba(${rust}, ${0.8 * a})`);
    }
  };

  // 2: prior samples settle into a posterior as observations arrive.
  const xs = Array.from({ length: 70 }, (_, i) => i / 69);
  const obsAll = [[0.15, 0.4], [0.38, -0.3], [0.55, 0.55], [0.82, -0.2]];
  const samples = Array.from({ length: 6 }, (_, k) => xs.map((x) => 0.5 * Math.sin(6 * x + k * 1.7) + 0.3 * Math.sin(13 * x + k)));
  const posterior = (obs, x) => {
    if (!obs.length) return [0, 1];
    const K = obs.map((a) => obs.map((b) => kernel(a[0], b[0])));
    K.forEach((row, i) => (row[i] += 0.01));
    const L = cholesky(K);
    const alpha = solve(L, obs.map((o) => o[1]));
    const ks = obs.map((o) => kernel(x, o[0]));
    const v = solve(L, ks);
    return [ks.reduce((s, k, j) => s + k * alpha[j], 0), Math.max(0, 1 - ks.reduce((s, k, j) => s + k * v[j], 0))];
  };
  const gp = (a, t) => {
    // Observations arrive one by one, hold, then the loop restarts from the prior.
    const n = still ? obsAll.length : Math.min(obsAll.length, Math.floor((t * 0.8) % (obsAll.length + 3)));
    const obs = obsAll.slice(0, n);
    const X = (x) => W * 0.12 + x * W * 0.76;
    const Y = (y) => H * 0.5 - y * H * 0.28;
    const post = xs.map((x) => posterior(obs, x));
    ctx.fillStyle = `rgba(${green}, ${0.14 * a})`;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(post[i][0] + 2 * Math.sqrt(post[i][1]))));
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Y(post[i][0] - 2 * Math.sqrt(post[i][1])));
    ctx.fill();
    samples.forEach((smp, k) => {
      const pts = xs.map((x, i) => {
        const wob = 0.15 * Math.sin(t * 0.9 + k + i * 0.05);
        return [X(x), Y(post[i][0] + (smp[i] + wob) * Math.sqrt(post[i][1]))];
      });
      line(pts, `rgba(${green}, ${0.3 * a})`, 1);
    });
    line(xs.map((x, i) => [X(x), Y(post[i][0])]), `rgba(${ink}, ${0.9 * a})`, 2);
    obs.forEach(([x, y]) => dot(X(x), Y(y), 5, `rgba(${rust}, ${a})`));
  };

  // 3: a planned path with a tube that widens off-data; the arm stops where the tube is too wide.
  const tube = (a, t) => {
    const P = (u) => [W * (0.25 + 0.6 * u), H * 0.6 - Math.sin(u * Math.PI * 0.9) * H * 0.22];
    const width = (u) => (8 + 70 * Math.max(0, u - 0.45) ** 1.6 * 3) * dpr;
    for (let i = 0; i < 60; i++) {
      const u = i / 60;
      const [x, y] = P(u);
      dot(x, y, width(u) / dpr, `rgba(${u > 0.5 ? rust : green}, ${0.04 * a})`);
    }
    line(Array.from({ length: 60 }, (_, i) => P(i / 59)), `rgba(${ink}, ${0.5 * a})`, 1.2);
    const stop = 0.62;
    const u = Math.min(stop, ((t * 0.18) % 1) * 1.25);
    const tip = P(u);
    const base = [W * 0.2, H * 0.86];
    const far = P(stop);
    const reach = Math.hypot(far[0] - base[0], far[1] - base[1]);
    const halted = u >= stop;
    arm(base, tip, reach * 0.58, reach * 0.52, halted ? rust : ink, a);
    if (halted) {
      const pulse = 8 + 6 * Math.sin(t * 6);
      ctx.beginPath();
      ctx.arc(tip[0], tip[1], pulse * dpr, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${rust}, ${0.8 * a})`;
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
      ctx.font = `italic ${15 * dpr}px 'Source Serif 4', serif`;
      ctx.fillStyle = `rgba(${rust}, ${a})`;
      ctx.fillText('not sure, asking for help', tip[0] + 16 * dpr, tip[1] - 14 * dpr);
    }
  };

  const scenes = [car, welder, gp, tube];
  const frame = (now) => {
    raf = 0;
    if (!canvas.isConnected) return;
    readProgress();
    ({ ink, a1: green, dot: rust } = theme());
    shown += (progress - shown) * (still ? 1 : 0.12);
    const t = still ? 2.2 : now / 1000;
    ctx.clearRect(0, 0, W, H);
    scenes.forEach((scene, i) => {
      const a = Math.max(0, 1 - Math.abs(shown - i) * 1.6);
      if (a <= 0.01) return;
      const local = Math.min(1, Math.max(0, shown - i + 0.6));
      scene(a, t, i === 0 ? Math.min(1, 0.25 + local) : local);
    });
    if (visible && !still) raf = requestAnimationFrame(frame);
  };
  const kick = () => {
    if (!raf) raf = requestAnimationFrame(frame);
  };
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) kick();
  }).observe(root || canvas);
  addEventListener('scroll', kick, { passive: true });
  kick();
}

// Hero: a live posterior. The cursor is a "what if I observed here?" point;
// click pins it as data. With no pointer, a ghost point wanders on its own.
export function mountLive(canvas) {
  const ctx = canvas.getContext('2d');
  const frame = canvas.closest('[data-live]');
  const count = frame?.querySelector('[data-count]');
  const truth = (x) => 0.55 * Math.sin(7 * x) + 0.35 * Math.cos(15 * x + 1) * Math.exp(-2 * x);
  const seed = [0.06, 0.22, 0.38, 0.55, 0.9].map((x) => [x, truth(x)]);
  let obs = [...seed];
  let hover = null;
  let raf = 0;
  let visible = true;
  const still = reduce();
  let dpr = fit(canvas);
  addEventListener('resize', () => (dpr = fit(canvas)));
  const N = 140;
  const xs = Array.from({ length: N }, (_, i) => i / (N - 1));
  const ell = 0.095;
  const k = (a, b) => Math.exp(-0.5 * ((a - b) / ell) ** 2);
  const post = (pts) => {
    const K = pts.map((a) => pts.map((b) => k(a[0], b[0])));
    K.forEach((row, i) => (row[i] += 0.004));
    const L = cholesky(K);
    const alpha = solve(L, pts.map((p) => p[1]));
    return xs.map((x) => {
      const ks = pts.map((p) => k(x, p[0]));
      const v = solve(L, ks);
      return [ks.reduce((s, q, j) => s + q * alpha[j], 0), Math.max(0, 1 - ks.reduce((s, q, j) => s + q * v[j], 0))];
    });
  };
  const toData = (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width;
    const y = (0.5 - (ev.clientY - r.top) / r.height) / 0.24;
    return [Math.min(0.99, Math.max(0.01, x)), Math.max(-1.5, Math.min(1.5, y))];
  };
  canvas.addEventListener('pointermove', (ev) => { if (ev.pointerType === 'mouse') { hover = toData(ev); kick(); } });
  canvas.addEventListener('pointerleave', () => { hover = null; kick(); });
  canvas.addEventListener('click', (ev) => {
    if (obs.length >= 18) obs = obs.slice(seed.length);
    obs = [...obs, toData(ev)];
    if (count) count.textContent = String(obs.length);
    kick();
  });
  frame?.querySelector('[data-reset]')?.addEventListener('click', () => {
    obs = [...seed];
    if (count) count.textContent = String(obs.length);
    kick();
  });
  const draw = (now) => {
    raf = 0;
    if (!canvas.isConnected) return;
    const c = theme();
    const W = canvas.width;
    const H = canvas.height;
    const t = still ? 0 : now / 1000;
    const ghost = hover ?? (still ? null : [0.62 + 0.3 * Math.sin(t * 0.35), 0.9 * Math.sin(t * 0.6)]);
    const P = post(ghost ? [...obs, ghost] : obs);
    const X = (x) => x * W;
    const Y = (y) => H * 0.5 - y * H * 0.24;
    ctx.clearRect(0, 0, W, H);
    // faint grid
    ctx.strokeStyle = c.line;
    ctx.lineWidth = dpr;
    for (let g = 1; g < 12; g++) {
      ctx.beginPath();
      ctx.moveTo(X(g / 12), 0);
      ctx.lineTo(X(g / 12), H);
      ctx.stroke();
    }
    // the band: indigo where the model is sure, coral where it isn't
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    xs.forEach((x, i) => {
      if (i % 10) return;
      const u = Math.min(1, Math.sqrt(P[i][1]) * 1.3);
      grad.addColorStop(x, `rgba(${u > 0.6 ? c.a2 : c.a1}, ${0.08 + 0.1 * u})`);
    });
    ctx.fillStyle = grad;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(P[i][0] + 2 * Math.sqrt(P[i][1]))));
    for (let i = N - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Y(P[i][0] - 2 * Math.sqrt(P[i][1])));
    ctx.closePath();
    ctx.fill();
    // a few wandering "samples"
    for (let s = 0; s < 4; s++) {
      ctx.strokeStyle = `rgba(${c.a1}, 0.28)`;
      ctx.lineWidth = dpr;
      ctx.beginPath();
      xs.forEach((x, i) => {
        const wave = Math.sin(9 * x + s * 2.1 + t * 0.7) * 0.7 + Math.sin(23 * x - s + t * 0.4) * 0.5;
        ctx.lineTo(X(x), Y(P[i][0] + wave * Math.sqrt(P[i][1])));
      });
      ctx.stroke();
    }
    // mean
    ctx.strokeStyle = `rgb(${c.ink})`;
    ctx.lineWidth = 2.2 * dpr;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(P[i][0])));
    ctx.stroke();
    // data
    obs.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(X(x), Y(y), 5 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${c.a2})`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
    });
    if (ghost) {
      ctx.beginPath();
      ctx.arc(X(ghost[0]), Y(ghost[1]), 7 * dpr, 0, Math.PI * 2);
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.strokeStyle = `rgb(${c.a1})`;
      ctx.lineWidth = 1.8 * dpr;
      ctx.stroke();
      ctx.setLineDash([]);
      if (hover) {
        ctx.font = `600 ${12 * dpr}px 'JetBrains Mono Variable', monospace`;
        ctx.fillStyle = `rgb(${c.a1})`;
        const label = 'click to observe';
        const lx = Math.min(X(ghost[0]) + 12 * dpr, W - ctx.measureText(label).width - 8 * dpr);
        ctx.fillText(label, lx, Y(ghost[1]) - 12 * dpr);
      }
    }
    if (visible && !still && !hover) raf = requestAnimationFrame(draw);
  };
  const kick = () => { if (!raf) raf = requestAnimationFrame(draw); };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); }).observe(canvas);
  onTheme(kick);
  kick();
}

// "Uncertainty about Sawan": every section read is an observation.
// Posterior sd after n observations ~ 1 / sqrt(1 + 1.5 n^2), shown as a band that tightens.
export function mountMeter(el) {
  const canvas = el.querySelector('canvas');
  const pct = el.querySelector('.pct');
  const what = el.querySelector('.what');
  const line = document.querySelector('.read-line');
  const ctx = canvas.getContext('2d');
  const sections = [...document.querySelectorAll('[data-obs]')];
  let dpr = fit(canvas);
  let n = -1;
  let shown = 1;
  let target = 1;
  let raf = 0;
  const xs = Array.from({ length: 60 }, (_, i) => i / 59);
  const draw = () => {
    raf = 0;
    if (!canvas.isConnected) return;
    const c = theme();
    const W = canvas.width;
    const H = canvas.height;
    shown += (target - shown) * (reduce() ? 1 : 0.12);
    ctx.clearRect(0, 0, W, H);
    const mean = (x) => Math.sin(x * 6 + 0.5) * 0.25;
    ctx.beginPath();
    xs.forEach((x) => ctx.lineTo(x * W, H / 2 - (mean(x) + shown * 0.9) * H * 0.5));
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(xs[i] * W, H / 2 - (mean(xs[i]) - shown * 0.9) * H * 0.5);
    const g = ctx.createLinearGradient(0, 0, W, 0);
    g.addColorStop(0, `rgba(${c.a1}, 0.35)`);
    g.addColorStop(1, `rgba(${c.a2}, ${0.15 + 0.3 * shown})`);
    ctx.fillStyle = g;
    ctx.fill();
    ctx.strokeStyle = `rgb(${c.ink})`;
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    xs.forEach((x) => ctx.lineTo(x * W, H / 2 - mean(x) * H * 0.5));
    ctx.stroke();
    for (let k = 0; k < Math.max(0, n); k++) {
      const x = (k + 0.5) / sections.length;
      ctx.beginPath();
      ctx.arc(x * W, H / 2 - mean(x) * H * 0.5, 2.6 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${c.dot})`;
      ctx.fill();
    }
    if (Math.abs(target - shown) > 0.002) raf = requestAnimationFrame(draw);
  };
  const kick = () => { if (!raf) raf = requestAnimationFrame(draw); };
  const update = () => {
    if (!el.isConnected) return removeEventListener('scroll', update);
    const mid = innerHeight * 0.6;
    const seen = sections.filter((s) => s.getBoundingClientRect().top < mid);
    const doc = document.documentElement;
    line?.style.setProperty('--p', String(Math.min(1, scrollY / Math.max(1, doc.scrollHeight - innerHeight))));
    el.classList.toggle('show', scrollY > innerHeight * 0.5);
    if (seen.length === n) return;
    n = seen.length;
    target = 1 / Math.sqrt(1 + 1.5 * n * n);
    pct.textContent = `±${Math.round(target * 100)}%`;
    what.textContent = n === 0 ? 'prior: just a name' : seen[n - 1].dataset.obs;
    kick();
  };
  addEventListener('scroll', update, { passive: true });
  addEventListener('resize', () => { dpr = fit(canvas); kick(); });
  onTheme(kick);
  update();
}

// Hero: a flow model, live. Particles start as Gaussian noise and are carried
// along x_t = (1 - t) x0 + t x1 onto samples from one of my pencil sketches.
// The cursor re-noises nearby particles (pushes their t back), and they denoise again.
export function mountGenesis(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-genesis]');
  const label = root?.querySelector('[data-t]');
  const which = root?.querySelector('[data-which]');
  const sources = JSON.parse(canvas.dataset.sketches || '[]');
  const still = reduce();
  let dpr = fit(canvas);
  let W = canvas.width;
  let H = canvas.height;
  const area = (W * H) / (dpr * dpr);
  const N = Math.round(Math.min(6000, Math.max(3000, area / 120)));
  const px = new Float32Array(N * 2);
  const x0 = new Float32Array(N * 2);
  const x1 = new Float32Array(N * 2);
  const lag = new Float32Array(N);
  const local = new Float32Array(N).fill(1);
  const targets = [];
  let current = 0;
  let clock = 0;
  let last = 0;
  let pointer = null;
  let visible = true;
  let raf = 0;
  const gauss = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());

  // Sample points in proportion to pencil darkness.
  const sample = (img) => {
    const w = 220;
    const h = Math.round((img.naturalHeight / img.naturalWidth) * w);
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const o = off.getContext('2d', { willReadFrequently: true });
    o.drawImage(img, 0, 0, w, h);
    const data = o.getImageData(0, 0, w, h).data;
    const weights = new Float32Array(w * h);
    let total = 0;
    let bg = 0;
    for (let i = 0; i < w * h; i++) bg = Math.max(bg, (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 765);
    const thr = bg * 0.82;
    for (let i = 0; i < w * h; i++) {
      const lum = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 765;
      const x = i % w;
      const y = (i / w) | 0;
      const edge = x < 4 || y < 4 || x > w - 5 || y > h - 5;
      const d = edge ? 0 : Math.max(0, (thr - lum) / thr);
      total += d * d;
      weights[i] = total;
    }
    const pts = new Float32Array(N * 2);
    const aspect = h / w;
    pts.aspect = aspect;
    for (let n = 0; n < N; n++) {
      const r = Math.random() * total;
      let lo = 0;
      let hi = weights.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (weights[mid] < r) lo = mid + 1;
        else hi = mid;
      }
      pts[n * 2] = ((lo % w) + Math.random()) / w - 0.5;
      pts[n * 2 + 1] = (((lo / w) | 0) + Math.random()) / h * aspect - aspect / 2;
    }
    return pts;
  };

  const place = () => {
    const wide = W / dpr > 820;
    const aspect = targets[current % targets.length]?.aspect ?? 1;
    const size = Math.min(wide ? W * 0.5 : W * 0.96, (H * 0.9) / aspect);
    return { cx: wide ? W * 0.68 : W * 0.5, cy: H * 0.5, size };
  };
  const renoise = () => {
    for (let i = 0; i < N; i++) {
      x0[i * 2] = gauss() * 0.22;
      x0[i * 2 + 1] = gauss() * 0.22;
      lag[i] = Math.random() * 0.35;
    }
  };
  const setTarget = () => {
    const t = targets[current % targets.length];
    if (t) x1.set(t);
    if (which) which.textContent = sources[current % sources.length]?.name ?? '';
  };

  // One cycle: generate (0 -> 1), hold, dissolve back to noise, next sketch.
  const GEN = 5.2, HOLD = 5, BACK = 1.8;
  const schedule = (c) => {
    const cyc = GEN + HOLD + BACK;
    const u = c % cyc;
    if (u < GEN) return u / GEN;
    if (u < GEN + HOLD) return 1;
    return 1 - (u - GEN - HOLD) / BACK;
  };

  const frame = (now) => {
    raf = 0;
    if (!canvas.isConnected) return;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    const before = Math.floor(clock / (GEN + HOLD + BACK));
    clock += still ? 0 : dt;
    if (Math.floor(clock / (GEN + HOLD + BACK)) !== before) {
      current++;
      renoise();
      setTarget();
    }
    const T = still ? 1 : schedule(clock);
    const c = theme();
    const { cx, cy, size } = place();
    ctx.clearRect(0, 0, W, H);
    const r2 = (90 * dpr) ** 2;
    const buckets = [[], [], [], []];
    for (let i = 0; i < N; i++) {
      let t = Math.min(1, Math.max(0, (T - lag[i]) / (1 - 0.35)));
      t = t * t * (3 - 2 * t);
      // cursor: push nearby particles back toward noise, then let them relax
      const ex = cx + px[i * 2];
      const ey = cy + px[i * 2 + 1];
      if (pointer) {
        const dx = ex - pointer[0];
        const dy = ey - pointer[1];
        const d2 = dx * dx + dy * dy;
        if (d2 < r2) local[i] = Math.min(local[i], 0.15 + 0.85 * (d2 / r2));
      }
      local[i] += (1 - local[i]) * (still ? 1 : 0.025);
      const tt = t * local[i];
      const bend = Math.sin(Math.PI * tt) * 0.06;
      const ax = x0[i * 2], ay = x0[i * 2 + 1], bx = x1[i * 2], by = x1[i * 2 + 1];
      const x = (1 - tt) * ax + tt * bx - (by - ay) * bend;
      const y = (1 - tt) * ay + tt * by + (bx - ax) * bend;
      px[i * 2] = x * size;
      px[i * 2 + 1] = y * size;
      buckets[Math.min(3, (tt * 4) | 0)].push(i);
    }
    // colour by how "certain" each particle is: cyan noise -> indigo sketch
    const cols = [c.a2, c.a2, c.a1, c.ink];
    const alphas = [0.4, 0.55, 0.7, 0.8];
    buckets.forEach((list, b) => {
      ctx.fillStyle = `rgba(${cols[b]}, ${alphas[b]})`;
      const s = (b === 3 ? 1.15 : 1.5) * dpr;
      for (const i of list) ctx.fillRect(cx + px[i * 2] - s / 2, cy + px[i * 2 + 1] - s / 2, s, s);
    });
    if (label) label.textContent = `t = ${T.toFixed(2)}`;
    if (visible && !still) raf = requestAnimationFrame(frame);
  };
  const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };

  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    pointer = [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr];
  });
  canvas.addEventListener('pointerleave', () => (pointer = null));
  addEventListener('resize', () => {
    dpr = fit(canvas);
    W = canvas.width;
    H = canvas.height;
  });
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) { last = 0; kick(); } }).observe(canvas);
  onTheme(kick);

  renoise();
  Promise.all(
    sources.map((src) => new Promise((ok) => {
      const img = new Image();
      img.onload = () => ok(sample(img));
      img.onerror = () => ok(null);
      img.src = src.src;
    })),
  ).then((list) => {
    list.forEach((pts) => pts && targets.push(pts));
    if (!targets.length) return;
    setTarget();
    kick();
  });
}

// Research page: a GP "lab". Pick a kernel, click to add data, drag a point.
// Samples are exact posterior draws, z(t) = cos(t) z1 + sin(t) z2, so every frame is a valid sample.
const KERNELS = {
  rbf: (a, b, l) => Math.exp(-0.5 * ((a - b) / l) ** 2),
  matern12: (a, b, l) => Math.exp(-Math.abs(a - b) / l),
  matern52: (a, b, l) => {
    const r = (Math.sqrt(5) * Math.abs(a - b)) / l;
    return (1 + r + (r * r) / 3) * Math.exp(-r);
  },
  periodic: (a, b, l) => Math.exp((-2 * Math.sin((Math.PI * Math.abs(a - b)) / 0.25) ** 2) / (l * 6) ** 2),
  // Gibbs kernel: the length-scale grows from left to right (non-stationary).
  gibbs: (a, b, l) => {
    const la = l * (0.12 + 3.2 * a * a);
    const lb = l * (0.12 + 3.2 * b * b);
    const s = la * la + lb * lb;
    return Math.sqrt((2 * la * lb) / s) * Math.exp(-((a - b) ** 2) / s);
  },
};

function forward(L, b) {
  const y = [...b];
  for (let i = 0; i < y.length; i++) {
    for (let p = 0; p < i; p++) y[i] -= L[i][p] * y[p];
    y[i] /= L[i][i];
  }
  return y;
}

export function mountGpLab(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-lab]');
  const insight = root?.querySelector('[data-insight]');
  const still = reduce();
  let dpr = fit(canvas);
  const N = 110;
  const S = 5;
  const xs = Array.from({ length: N }, (_, i) => i / (N - 1));
  const start = [[0.12, 0.5], [0.3, -0.35], [0.47, 0.6], [0.78, -0.2]];
  let obs = start.map((o) => [...o]);
  let kernel = 'rbf';
  let ell = 0.09;
  let mean = [];
  let sd = [];
  let Lp = [];
  let drag = -1;
  let moved = false;
  let raf = 0;
  let visible = true;
  const z1 = Array.from({ length: S }, () => xs.map(() => gaussRand()));
  const z2 = Array.from({ length: S }, () => xs.map(() => gaussRand()));
  function gaussRand() {
    return Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
  }
  const noise = 0.004;
  const compute = () => {
    const k = (a, b) => KERNELS[kernel](a, b, ell);
    const n = obs.length;
    if (!n) {
      mean = xs.map(() => 0);
      const C = xs.map((a) => xs.map((b) => k(a, b)));
      C.forEach((r, i) => (r[i] += 1e-5));
      Lp = cholesky(C);
      sd = xs.map((x) => Math.sqrt(k(x, x)));
      return;
    }
    const K = obs.map((a) => obs.map((b) => k(a[0], b[0])));
    K.forEach((r, i) => (r[i] += noise));
    const L = cholesky(K);
    const alpha = solve(L, obs.map((o) => o[1]));
    const V = xs.map((x) => forward(L, obs.map((o) => k(x, o[0]))));
    mean = xs.map((x, i) => obs.reduce((s, o, j) => s + k(x, o[0]) * alpha[j], 0));
    const C = xs.map((a, i) => xs.map((b, j) => k(a, b) - V[i].reduce((s, v, q) => s + v * V[j][q], 0)));
    C.forEach((r, i) => (r[i] += 1e-5));
    sd = C.map((r, i) => Math.sqrt(Math.max(0, r[i])));
    Lp = cholesky(C);
  };
  const X = (x) => x * canvas.width;
  const Y = (y) => canvas.height * 0.5 - y * canvas.height * 0.26;
  const toData = (e) => {
    const r = canvas.getBoundingClientRect();
    return [Math.min(0.99, Math.max(0.01, (e.clientX - r.left) / r.width)), Math.max(-1.6, Math.min(1.6, (0.5 - (e.clientY - r.top) / r.height) / 0.26))];
  };
  const draw = (now) => {
    raf = 0;
    if (!canvas.isConnected) return;
    const c = theme();
    const W = canvas.width;
    const H = canvas.height;
    const t = still ? 0.6 : now / 2600;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = c.line;
    ctx.lineWidth = dpr;
    for (let g = 1; g < 16; g++) {
      ctx.beginPath();
      ctx.moveTo(X(g / 16), 0);
      ctx.lineTo(X(g / 16), H);
      ctx.stroke();
    }
    // ±2σ band, coloured by how unsure the model is
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    xs.forEach((x, i) => {
      if (i % 11) return;
      const u = Math.min(1, sd[i]);
      grad.addColorStop(x, `rgba(${u > 0.55 ? c.a2 : c.a1}, ${0.08 + 0.12 * u})`);
    });
    ctx.fillStyle = grad;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(mean[i] + 2 * sd[i])));
    for (let i = N - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Y(mean[i] - 2 * sd[i]));
    ctx.fill();
    // exact posterior samples moving on a great circle
    for (let s = 0; s < S; s++) {
      const cs = Math.cos(t + s);
      const sn = Math.sin(t + s);
      ctx.strokeStyle = `rgba(${s % 2 ? c.a2 : c.a1}, 0.45)`;
      ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        let f = mean[i];
        const row = Lp[i];
        for (let j = 0; j <= i; j++) f += row[j] * (cs * z1[s][j] + sn * z2[s][j]);
        ctx.lineTo(X(xs[i]), Y(f));
      }
      ctx.stroke();
    }
    ctx.strokeStyle = `rgb(${c.ink})`;
    ctx.lineWidth = 2.4 * dpr;
    ctx.beginPath();
    xs.forEach((x, i) => ctx.lineTo(X(x), Y(mean[i])));
    ctx.stroke();
    obs.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(X(x), Y(y), (i === drag ? 8 : 6) * dpr, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${c.dot})`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,.95)';
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
    });
    if (visible && !still) raf = requestAnimationFrame(draw);
  };
  const kick = () => { if (!raf) raf = requestAnimationFrame(draw); };
  const say = (key) => {
    if (!insight) return;
    const tpl = root.querySelector(`template[data-for="${key}"]`);
    if (tpl) insight.innerHTML = tpl.innerHTML;
  };
  canvas.addEventListener('pointerdown', (e) => {
    const [x, y] = toData(e);
    const r = canvas.getBoundingClientRect();
    drag = obs.findIndex(([ox, oy]) => Math.hypot((ox - x) * r.width, (oy - y) * r.height * 0.26) < 14);
    moved = false;
    if (drag >= 0) {
      canvas.setPointerCapture(e.pointerId);
    } else if (obs.length < 24) {
      obs.push([x, y]);
      compute();
      kick();
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (drag < 0) return;
    obs[drag][1] = toData(e)[1];
    moved = true;
    compute();
    kick();
  });
  const end = () => {
    if (drag >= 0 && moved) say('drag');
    drag = -1;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  root?.querySelectorAll('[data-kernel]').forEach((b) =>
    b.addEventListener('click', () => {
      kernel = b.dataset.kernel;
      root.querySelectorAll('[data-kernel]').forEach((o) => o.classList.toggle('on', o === b));
      say(kernel);
      compute();
      kick();
    }),
  );
  root?.querySelector('[data-ell]')?.addEventListener('input', (e) => {
    ell = Number(e.target.value);
    compute();
    kick();
  });
  root?.querySelector('[data-clear]')?.addEventListener('click', () => {
    obs = [];
    say('prior');
    compute();
    kick();
  });
  root?.querySelector('[data-reset]')?.addEventListener('click', () => {
    obs = start.map((o) => [...o]);
    compute();
    kick();
  });
  addEventListener('resize', () => { dpr = fit(canvas); kick(); });
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); }).observe(canvas);
  onTheme(kick);
  compute();
  say('rbf');
  kick();
}

// Research page: real spherical harmonics Y_l^m, the eigenfunctions of the Laplacian on the sphere.
export function mountSphere(canvas) {
  const ctx = canvas.getContext('2d');
  const root = canvas.closest('[data-sphere]');
  const lIn = root?.querySelector('[data-l]');
  const mIn = root?.querySelector('[data-m]');
  const out = root?.querySelector('[data-eig]');
  const still = reduce();
  const R = 220;
  const img = ctx.createImageData(R, R);
  const off = document.createElement('canvas');
  off.width = off.height = R;
  const octx = off.getContext('2d');
  let l = 3;
  let m = 2;
  let raf = 0;
  let visible = true;
  let scale = 1;
  // associated Legendre P_l^m(x), m >= 0
  const legendre = (L, M, x) => {
    let pmm = 1;
    const s = Math.sqrt(Math.max(0, 1 - x * x));
    for (let i = 1; i <= M; i++) pmm *= -(2 * i - 1) * s;
    if (L === M) return pmm;
    let pm1 = x * (2 * M + 1) * pmm;
    if (L === M + 1) return pm1;
    let pll = 0;
    for (let ll = M + 2; ll <= L; ll++) {
      pll = ((2 * ll - 1) * x * pm1 - (ll + M - 1) * pmm) / (ll - M);
      pmm = pm1;
      pm1 = pll;
    }
    return pll;
  };
  const Y = (theta, phi) => {
    const am = Math.abs(m);
    const p = legendre(l, am, Math.cos(theta));
    return p * (m > 0 ? Math.cos(am * phi) : m < 0 ? Math.sin(am * phi) : 1);
  };
  const normalise = () => {
    let mx = 1e-9;
    for (let i = 0; i < 40; i++) for (let j = 0; j < 80; j++) mx = Math.max(mx, Math.abs(Y((i + 0.5) / 40 * Math.PI, (j / 80) * 2 * Math.PI)));
    scale = 1 / mx;
  };
  const hex = (v) => v.split(',').map(Number);
  const sync = () => {
    l = Number(lIn?.value ?? l);
    if (mIn) {
      mIn.min = String(-l);
      mIn.max = String(l);
      m = Math.max(-l, Math.min(l, Number(mIn.value)));
      mIn.value = String(m);
    }
    root?.querySelectorAll('[data-show-l]').forEach((e) => (e.textContent = String(l)));
    root?.querySelectorAll('[data-show-m]').forEach((e) => (e.textContent = String(m)));
    if (out) out.textContent = String(l * (l + 1));
    normalise();
    kick();
  };
  const draw = (now) => {
    raf = 0;
    if (!canvas.isConnected) return;
    const c = theme();
    const pos = hex(c.a1);
    const neg = hex(c.dot);
    const dark = document.documentElement.dataset.theme === 'dark' || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    const base = dark ? [40, 40, 56] : [238, 238, 246];
    const rot = still ? 0.6 : now / 4000;
    const tilt = 0.45;
    const cr = Math.cos(rot), sr = Math.sin(rot), ct = Math.cos(tilt), st = Math.sin(tilt);
    const d = img.data;
    for (let py = 0; py < R; py++) {
      for (let px = 0; px < R; px++) {
        const u = (px + 0.5) / R * 2 - 1;
        const v = 1 - (py + 0.5) / R * 2;
        const rr = u * u + v * v;
        const k = (py * R + px) * 4;
        if (rr > 1) { d[k + 3] = 0; continue; }
        const z = Math.sqrt(1 - rr);
        // undo tilt (about x) then spin (about y)
        const y1 = v * ct + z * st;
        const z1 = -v * st + z * ct;
        const x2 = u * cr + z1 * sr;
        const z2 = -u * sr + z1 * cr;
        const val = Y(Math.acos(Math.max(-1, Math.min(1, y1))), Math.atan2(z2, x2)) * scale;
        const col = val >= 0 ? pos : neg;
        const a = Math.min(1, Math.abs(val));
        const light = 0.55 + 0.45 * (z * 0.8 + v * 0.2 + u * -0.1);
        for (let q = 0; q < 3; q++) d[k + q] = Math.round((base[q] * (1 - a) + col[q] * a) * light);
        d[k + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const size = Math.min(W, H) * 0.92;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, (W - size) / 2, (H - size) / 2, size, size);
    if (visible && !still) raf = requestAnimationFrame(draw);
  };
  function kick() { if (!raf) raf = requestAnimationFrame(draw); }
  fit(canvas);
  lIn?.addEventListener('input', sync);
  mIn?.addEventListener('input', sync);
  addEventListener('resize', () => { fit(canvas); kick(); });
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); }).observe(canvas);
  onTheme(kick);
  sync();
}
