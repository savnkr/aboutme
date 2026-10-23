// Courses, books and writing that shaped how I think. One line per item.
// Keep it to other people's work; no self-promotion.
export const resources = [
  {
    topic: 'Mathematics for ML',
    blurb: 'The linear algebra, calculus and numerics underneath everything else.',
    items: [
      { title: 'Matrix Calculus for Machine Learning and Beyond', who: 'MIT, Edelman & Johnson', url: 'https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE' },
      { title: 'Mathematics of Data: From Theory to Computation', who: 'EPFL, Volkan Cevher', url: 'https://www.epfl.ch/labs/lions/teaching/ee-556-mathematics-of-data-from-theory-to-computation/slides-2024/' },
      { title: 'Numerical Algorithms', who: 'KAIST course notes', url: 'https://github.com/anar-rzayev/Numerical-Algorithms' },
      { title: 'Solving PDEs in Python (MOOC)', who: 'Université libre de Bruxelles', url: 'https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/01_Introduction/01_00_Preface.html' },
    ],
  },
  {
    topic: 'Probabilistic ML & kernels',
    blurb: 'Where uncertainty, RKHS theory and Gaussian processes come from.',
    items: [
      { title: 'Gaussian Processes for Machine Learning', who: 'Rasmussen & Williams (book)', url: 'https://gaussianprocess.org/gpml/' },
      { title: 'A Visual Exploration of Gaussian Processes', who: 'Distill', url: 'https://distill.pub/2019/visual-exploration-gaussian-processes/' },
      { title: 'Advanced Topics in ML: RKHS and kernel methods', who: 'Gatsby Unit, Arthur Gretton', url: 'https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/rkhscourse.html' },
      { title: 'Bayesian Machine Learning and Information Processing', who: 'TU Eindhoven, BIASlab', url: 'https://biaslab.github.io/teaching/bmlip/' },
      { title: 'Machine Learning Theory', who: 'CU Boulder, Stephen Becker', url: 'https://github.com/stephenbeckr/ML-theory-class' },
      { title: 'Function spaces: reproducing kernel Hilbert spaces', who: 'Fields Institute', url: 'http://www.fields.utoronto.ca/activities/21-22/function-course-II' },
    ],
  },
  {
    topic: 'Deep learning',
    blurb: 'Courses that build intuition, not just recipes.',
    items: [
      { title: 'Deep Learning (NYU DS-GA 1008)', who: 'Yann LeCun & Alfredo Canziani', url: 'https://atcold.github.io/NYU-DLSP20/' },
      { title: 'CS1470 Deep Learning', who: 'Brown University', url: 'https://brown-deep-learning.github.io/dl-website-s24/' },
      { title: 'DeepCourse', who: 'Arthur Douillard', url: 'https://arthurdouillard.com/deepcourse/' },
      { title: 'Applied Deep Learning', who: 'Maziar Raissi', url: 'https://github.com/maziarraissi/Applied-Deep-Learning' },
    ],
  },
  {
    topic: 'Generative models',
    blurb: 'Diffusion, flows and the probability that ties them together.',
    items: [
      { title: 'Introduction to Flow Matching and Diffusion Models (6.S184)', who: 'MIT, Holderrieth & Erives', url: 'https://diffusion.csail.mit.edu/' },
      { title: 'Generative Modeling by Estimating Gradients of the Data Distribution', who: 'Yang Song (blog)', url: 'https://yang-song.net/blog/2021/score/' },
      { title: 'What are Diffusion Models?', who: 'Lilian Weng (blog)', url: 'https://lilianweng.github.io/posts/2021-07-11-diffusion-models/' },
      { title: 'Diffusion models, sampling and stochastic analysis', who: 'ETH Zürich', url: 'https://metaphor.ethz.ch/x/2024/fs/401-4634-24L/' },
      { title: 'CS228 notes: learning in undirected models', who: 'Stanford, Ermon group', url: 'https://ermongroup.github.io/cs228-notes/learning/undirected/' },
      { title: 'GenAI Handbook', who: 'William Brown', url: 'https://genai-handbook.github.io/' },
    ],
  },
  {
    topic: 'Language models',
    blurb: 'Building and understanding them from scratch.',
    items: [
      { title: 'CS336: Language Modeling from Scratch', who: 'Stanford', url: 'https://stanford-cs336.github.io/spring2024/' },
      { title: 'Advanced NLP', who: 'CMU', url: 'https://cmu-l3.github.io/anlp-spring2025/' },
      { title: 'LLM Course', who: 'Maxime Labonne', url: 'https://github.com/mlabonne/llm-course' },
    ],
  },
  {
    topic: 'Robotics, vision & RL',
    blurb: 'Where models meet the physical world.',
    items: [
      { title: 'Underactuated Robotics', who: 'MIT, Russ Tedrake', url: 'https://underactuated.mit.edu/' },
      { title: 'CS 285: Deep Reinforcement Learning', who: 'UC Berkeley, Sergey Levine', url: 'https://rail.eecs.berkeley.edu/deeprlcourse/' },
      { title: 'Spinning Up in Deep RL', who: 'OpenAI', url: 'https://spinningup.openai.com/' },
      { title: 'Mathematical Foundations of Reinforcement Learning', who: 'Shiyu Zhao (book)', url: 'https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning' },
      { title: 'Computer Vision', who: 'University of Tübingen, Andreas Geiger', url: 'https://github.com/daijun10086/Tubingen_ComputerVision' },
    ],
  },
  {
    topic: 'Scientific ML',
    blurb: 'Learning with physics, PDEs and simulation.',
    items: [
      { title: 'Parallel Computing and Scientific Machine Learning', who: 'MIT, Chris Rackauckas', url: 'https://github.com/SciML/SciMLBook' },
      { title: 'Scientific ML: PDEs and finite differences', who: 'Krishna Kumar', url: 'https://kks32-courses.github.io/sciml/lectures/04-pde-fdm/04-pde-fdm.html' },
      { title: 'AI in Engineering: physics-informed ML', who: 'University of Washington', url: 'https://composites.uw.edu/AI/' },
    ],
  },
  {
    topic: 'Research engineering',
    blurb: 'Writing research code other people can trust.',
    items: [
      { title: 'Research Software Engineering with Python', who: 'The Alan Turing Institute', url: 'https://alan-turing-institute.github.io/rse-course/html/index.html' },
    ],
  },
];
