---
title: Fitting Data
layout: default
nav_order: 2
parent: Description
---

<head>
  <title>MathJax tests</title>

  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>

  <script>
    MathJax = {
     tex: {
      inlineMath: [['$', '$']],
      displayMath: [ ['$$','$$'], ["\\(","\\)"] ],
      processEscapes: true
      }
     };
  </script>

 <script id="MathJax-script" async
     src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
  </script>
</head>

# Fitting Data
{: .no_toc}

This page outlines how to fit observational data and simulated galactic atmospheres with **ExpCGM** models.

{: .warning}
This page is still under construction.

<details closed markdown="block">
  <summary>
    Contents
  </summary>
   {: .text-delta}
- TOC
{:toc}  
</details>

## Input Parameters

All **ExpCGM** models for galactic atmospheres are based on a user-specified gravitational potential function $\varphi(\mathbf{r})$ and a user-specified shape function $\alpha(\mathbf{r})$ for the atmosphere's pressure profile. In general, they can be functions of a three dimensional vector $\mathbf{r}$. Spherically symmetric models depending only on $r = |\mathbf{r}|$ are often sufficient, and this introductory page focuses on such models.

Two parametric models are therefore the starting point for obtaining predicted observables: 

* **Gravitational Potential,** $\varphi( r ; v_\varphi , r_0 , r_{\rm s} , ... )$: The gravitational potential model needs to depend on at least two parameters: (1) the maximum circular velocity $v_\varphi$ of the confining halo's gravitational potential, and (2) a radius $r_0$ at which $\varphi = 0$. It may also depend on a scale radius $r_{\rm s}$ determining where $v_{\rm c} = v_\varphi$ along with other parameters of interest to the user.

* **Shape Function,** $\alpha( r ; \alpha_{\rm in} , \alpha_{\rm out} , r_\alpha , ... )$: The shape function model depends on at least one parameter, which can be a constant value of $\alpha$. It may also describe the shape function in terms of a limiting value $\alpha_{\rm in}$ at small radii, a limiting value $\alpha_{\rm out}$ at large radii, and a crossover radius $r_\alpha$ from one value to the other. Additional parameters may be needed to describe more complex shape functions.

Up to two additional parametric functions may be needed to represent how the thermalization fraction $f_{\rm th}$ and force modification factor $f_\varphi$ depend on radius. (See the [Essentials](Essentials) page for their definitions.)

That set of parametric functions determines the atmosphere's temperature profile
    $$T(r) = {f_{\rm th} f_\varphi} \frac {\mu m_p v_{\rm c}^2} {\alpha_{\rm eff}}$$  


### Basic Model



### NFW + Galaxy Model

## Output Models

### Thermodynamic Profiles

### Projected Observables

## Scaling Laws

