// "What I'm studying beyond my papers" on the Research page.
// Keep each entry honest: what you've studied, and the question it raises for you.
export const studying = [
  {
    area: 'Current focus',
    title: 'Drifting models and kernel drifting',
    text: 'One-step generators trained by moving samples along a kernel-defined drift until model and data agree. My question: what makes the drift fast and faithful, and can the generator tell what it has not seen?',
  },
  {
    area: 'Current focus',
    title: 'Wasserstein gradient flows',
    text: 'The geometry behind Langevin dynamics, Fokker–Planck and drifting: generation as steepest descent of a divergence in the space of distributions.',
  },
  {
    area: 'Generative models',
    title: 'Score-based diffusion and SDEs',
    text: 'Forward noising, reverse-time SDEs and probability-flow ODEs. My question: can the learned score tell us where the model has never seen data?',
  },
  {
    area: 'Generative models',
    title: 'Flow matching and stochastic interpolants',
    text: 'Simulation-free training of continuous flows. Close to the GP world: both are about transporting a simple Gaussian into something structured.',
  },
  {
    area: 'Mathematics',
    title: 'Optimal transport',
    text: 'Wasserstein geometry, entropic regularisation, and why straight transport paths make generation fast.',
  },
  {
    area: 'Mathematics',
    title: 'Spectral geometry and GPs on manifolds',
    text: 'Laplacian eigenfunctions, spherical harmonics and Matérn kernels on spheres and graphs: priors that respect geometry.',
  },
  {
    area: 'Robotics',
    title: 'Diffusion policies and imitation learning',
    text: 'Generating robot actions the way we generate images. The open problem I care about: knowing when a proposed action is out of distribution.',
  },
  {
    area: 'Vision & language',
    title: 'Vision-language-action models',
    text: 'Large models that see, read and act. How do we calibrate them, and how should they ask for help?',
  },
];
