---
title: "Forecasts that cut grid penalties"
summary: "Calibrated AI weather forecasts for India’s solar and wind plants, turned into day-ahead schedules that cut deviation charges."
impact: "Every plant pays when it misses its schedule, and the tolerance bands tighten every year to 2031. Better schedules start with knowing how uncertain tomorrow is."
role: "Founder. Work in progress: building the first backtest on real plant data and talking to pilot partners."
status: "Work in progress · pre-seed"
year: 2026
order: 0
tags: [weather, uncertainty, energy, india]
cover: /projects/grid/card.jpg
video: /projects/grid/globe.mp4
---

## The problem

Every evening, each solar and wind plant in India commits to how much power it will produce in each 15-minute block of the next day. Outside a tolerance band, the difference is charged, and from 2026 the band for solar halves from ±10% to ±5%. In Grid-India’s study of 16 plants, solar plants stayed inside the band 74–84% of the time under the old rules and only 45–58% under the new ones.

![A day the forecast got wrong: a fixed schedule with its tolerance band, actual output straying outside it, and the deviation charges adding up (simulated plant)](/aboutme/projects/grid/hook.gif)

## What we are building

Forecasts that say how sure they are, turned into the schedule each plant submits under its own state’s rules. Plants receive the schedule before each deadline, and a monthly report of what they paid and what they saved. We are building the first backtest on a real plant’s history now.

## The science underneath

The globes at the top come from my research. They reconstruct global surface pressure (ERA5) from a quarter of the grid. Weather is calm in some places and sharp in others. Wavelet features ([Random Wavelet Features](/aboutme/research/), ICLR 2026) follow that and cut the held-out error from 10.3 to 2.6 hPa, against Fourier features with the same budget and the same solver. More broadly, my work is calibrated uncertainty for physical systems: Gaussian processes, operator learning, and generative models that know where they are unsure.

## Working with us

We are talking to **wind and solar producers, asset managers and scheduling agencies** about pilots, and looking for a **co-founder** with power-market experience.
