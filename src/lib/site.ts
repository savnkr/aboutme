export const site = {
  name: 'Sawan Kumar',
  nativeName: 'सावन कुमार',
  quirk: 'I sketch my priors ✎',
  title: 'Sawan Kumar · Probabilistic ML, generative models, robotics',
  description:
    'Sawan Kumar works toward trustworthy AI for the physical world: systems that forecast, simulate and act, and know what they don’t know. Probabilistic foundations for generative models and autonomous agents, from weather and climate to robotics. PhD researcher at IIT Delhi; on the job market for 2027.',
  tagline: 'Trustworthy AI for the physical world.',
  openTo: 'Open to research opportunities and collaborations',
  email: 'sawan.kumar@am.iitd.ac.in',
  emailAlt: 'sawankr02@gmail.com',
};

export function href(path = '/'): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  if (path === '/') return `${base}/`;
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

export function minutes(text = ''): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 220));
}

export function asset(path: string): string {
  return href(path.startsWith('/') ? path : `/${path}`);
}

export const nav = [
  { href: '/research/', label: 'Research' },
  { href: '/projects/', label: 'Projects' },
  { href: '/blog/', label: 'Blog' },
  { href: '/thoughts/', label: 'Thoughts' },
  { href: '/life/', label: 'Life' },
  { href: '/resources/', label: 'Resources' },
  { href: '/cv/', label: 'CV' },
];

export const links = {
  scholar: 'https://scholar.google.com/citations?user=mPzqn-UAAAAJ&hl=en',
  github: 'https://github.com/savnkr',
  linkedin: 'https://www.linkedin.com/in/sawan-kumar-63030a150/',
  openreview: 'https://openreview.net/profile?id=~Sawan_Kumar2',
  x: 'https://x.com/sawankr02',
  iitd: 'https://home.iitd.ac.in/',
  rwf: 'https://arxiv.org/abs/2602.00987',
};

// Dated updates shown on the home page. Newest first; add one line per update.
export const news = [
  { date: '2026', text: 'Talks on <strong>Random Wavelet Features</strong> at IIT Delhi, including new results on global ERA5 climate fields.', href: '/blog/fourier-to-wavelets/' },
  { date: '2026', text: 'Selected for <strong>Y Combinator Startup School India</strong>, Bengaluru.' },
  { date: '2026', text: 'New direction: fast generative models with calibrated uncertainty, for weather, climate and robotics.', href: '/research/' },
  { date: 'May 2026', text: 'Preprint: <em>Stochastic Smoothed Particle Hydrodynamics</em> for uncertainty in particle simulations.', href: '/research/' },
  { date: 'Feb 2026', text: '<strong>Random Wavelet Features</strong> accepted at <strong>ICLR 2026</strong> (first author).', href: '/research/' },
  { date: '2026', text: 'Scalable Gaussian Process Operator published in the <em>Journal of Computational Physics</em>.', href: '/research/' },
  { date: '2025', text: 'Best oral presentation at ICCMS 2025.' },
  { date: '2025', text: 'h-adaptive probabilistic solver published in <em>CMAME</em> (joint first author).', href: '/research/' },
  { date: '2025', text: 'FuCe, a physics-informed constitutive model, published in <em>Int. J. Mech. Syst. Dyn.</em>', href: '/research/' },
  { date: '2024', text: 'Two papers on Gaussian processes for operator learning published in <em>CMAME</em>.', href: '/research/' },
];

// The through-line, told on scroll. Each chapter drives one scene of the story canvas
// (0 race car, 1 industrial robot arm, 2 Gaussian process, 3 robot that knows when to stop).
// Edit the words freely; keep four chapters in this order.
export const story = [
  {
    when: '2017–2020',
    where: 'Team Hertz Electric, DTU',
    title: 'I learned engineering by building a race car.',
    text: 'As chassis and aerodynamics lead for a Formula student electric car, I designed for loads on a track we had never driven. Every tube in that frame was a bet.',
    lesson: 'Every design is a guess about a world you haven’t seen yet.',
  },
  {
    when: '2021',
    where: 'Foundation for Smart Manufacturing, IIT Delhi',
    title: 'Then I worked with industrial robots.',
    text: 'I commissioned KUKA industrial arms in a robotic manufacturing cell. They repeated a path to a fraction of a millimetre, and had no idea when the world stopped matching their program. Move the part, and they carried on with total confidence.',
    lesson: 'Being precise is not the same as knowing when you’re wrong.',
  },
  {
    when: '2022–now',
    where: 'PhD, IIT Delhi',
    title: 'So I made uncertainty my subject.',
    text: 'With Prof. Souvik Chakraborty and Prof. Rajdip Nayek, I build models that say how sure they are: scalable Gaussian processes, operator learning with honest error bars, and non-stationary kernels with convergence guarantees, published at ICLR.',
    lesson: 'Knowing what you don’t know is a skill. It can be engineered.',
  },
  {
    when: 'Next',
    where: 'Generative models · vision · robotics',
    title: 'Now I’m bringing it back to machines.',
    text: 'The robot I commissioned in 2021 should have been able to stop and ask. I’m building toward that: one-step generative models fast enough for real-time control, with uncertainty that says when to act, when to ask, and when to defer.',
    lesson: 'The goal is machines that know what they don’t know.',
  },
];

// "What I believe" on the home page. One line of title, two of meaning.
// Research interests on the home page: one framing line and four compact areas.
// Keep them broad and honest; nothing under review, no details of unpublished work.
export const interestsLead = 'How a model should represent what it does not know, and use that when it acts:';
export const interests = [
  { title: 'Probabilistic ML and uncertainty quantification', text: 'prior modelling, Gaussian processes and Bayesian deep learning' },
  { title: 'AI for science', text: 'neural and Gaussian-process operators for physical systems' },
  { title: 'Generative models', text: 'diffusion and flow models for calibrated weather and climate forecasting' },
  { title: 'Embodied AI', text: 'robot learning with uncertainty, so robots know when to ask for help' },
];

export const values = [
  {
    title: 'Say what you don’t know.',
    text: 'In models and in people. A confident wrong answer costs more than an honest “I’m not sure”.',
  },
  {
    title: 'Build it with your hands.',
    text: 'From a race-car frame to a robot cell, I trust ideas that had to survive contact with the real world.',
  },
  {
    title: 'Look closely before you model.',
    text: 'Sketching taught me that most mistakes are mistakes of attention. A good prior begins with looking.',
  },
  {
    title: 'Make things to stay whole.',
    text: 'I write poetry to process the hard parts. Making things keeps me honest, patient and kind.',
  },
];
