---
title: Interactive posteriors with flow models
summary: "A conditional generative model of a function, set beside the Gaussian-process posterior it is trying to become."
status: In progress
year: 2026
tags: [generative, flows, uncertainty]
figure: flow
draft: true
order: 1
impact: "Generative models rarely show where they are unsure. This makes that visible, with a Gaussian process as the ground truth."
---

The Gaussian process already answers a precise question. Given a few observations, which functions are still plausible, and how wide is that set? A flow or a diffusion model is a way to write the same object when the function is an image, a field, or a trajectory, and the kernel no longer fits in memory.

This page is the argument, not a finished experiment. The drawing shows a family of curves tightening onto three observations. Early curves are the prior. Later curves are what remains after conditioning. Sample disagreement is the uncertainty. There is no trained model behind the picture yet, and no number here is a result.

## Why it matters

Generative models became the practical way to write down a distribution. They still rarely show, in a form a person can touch, where that distribution is wide. A lab working on flow matching or posterior sampling can read the Gaussian-process papers as the same question asked of a kernel, with guarantees.

## What is missing

A small conditional flow, trained on functions or on a physical field I already know how to solve, with the Gaussian process as the reference posterior on the same task. The repository will replace this drawing when that exists.
