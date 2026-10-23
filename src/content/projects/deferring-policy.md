---
title: A robot policy that can refuse
summary: "A generative policy proposes a future. A probabilistic dynamics model paints the tube. A wide tube means defer."
status: In progress
year: 2026
tags: [robotics, generative, uncertainty]
figure: robot
draft: true
order: 2
impact: "Robot policies fail quietly when the world leaves their data. This one is built to notice and stop."
---

Diffusion policies and vision-language-action models imitate well. They fail quietly when contact, friction, or the dynamics leave the data they were shown. The missing piece is not another demonstration. It is a prior over the physical future that stays wide when the situation is new.

The arm in the figure is a sketch of that refusal. The pale tube is the set of futures still compatible with a probabilistic dynamics model. The ochre links are one proposed action. Where the tube is wide, the action is not evidence.

## Why this is the same program

The operator papers learn a map from a physical system and return a posterior. A robot future is another such map. The kernel, the operator, and the policy are three places to write a prior. The observation changes. The question does not.

## What is missing

A planar arm in simulation, a generative policy, and a dynamics model in the operator lineage that widens the tube under shift. Hardware is not the first step. A honest simulation, with one case where the policy should stop, is.
