---
title: "Convex Optimization"
date: 2025-12-07
summary: "Introduction to convex sets, functions, and optimization problems."
tags: ['math', 'optimization', 'convex']
---

# Convex Optimization

Convex optimization is a subfield of mathematical optimization that studies the problem of minimizing convex functions over convex sets.

## 1. Convex Sets

A set $ C $ is convex if the line segment between any two points in $ C $ lies in $ C $.

$$
\forall x, y \in C, \forall \theta \in [0, 1], \quad \theta x + (1 - \theta) y \in C
$$

## 2. Convex Functions

A function $ f: \mathbb{R}^n \to \mathbb{R} $ is convex if its domain is a convex set and for all $ x, y $ in the domain and $ \theta \in [0, 1] $:

$$
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
$$

## 3. Convex Optimization Problem

A standard form convex optimization problem is:

$$
\begin{aligned}
& \text{minimize} && f_0(x) \\
& \text{subject to} && f_i(x) \leq 0, \quad i = 1, \dots, m \\
& && h_i(x) = 0, \quad i = 1, \dots, p
\end{aligned}
$$

where:
- The objective function $ f_0 $ is convex.
- The inequality constraint functions $ f_i $ are convex.
- The equality constraint functions $ h_i $ are affine ($ h_i(x) = a_i^T x + b_i $).

## 4. Why Convex Optimization?

- **Global Optima**: Any local minimum of a convex function is also a global minimum.
- **Efficient Algorithms**: There are efficient algorithms (e.g., Interior Point Methods) to solve convex problems.
- **Duality**: Convex problems have a rich duality theory (Lagrange duality) that provides bounds and optimality conditions (KKT conditions).

## 5. Examples

- **Least Squares**: Minimizing $ \| Ax - b \|_2^2 $.
- **Linear Programming (LP)**: Minimizing $ c^T x $ subject to linear constraints.
- **Support Vector Machines (SVM)**: Finding the maximum margin hyperplane.

