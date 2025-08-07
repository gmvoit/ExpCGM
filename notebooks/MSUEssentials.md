---
title: MSU Essentials
layout: default
nav_exclude: true
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

# MSU Essentials Notebook

*Contributed by Doruk Yaldiz and Jazzmin Partridge, edited by Mark Voit*

These Python notebook cells present an implementation of the **ExpCGM** framework that relates a galactic atmosphere's radius to its mean specific energy, based on various assumptions about the atmosphere's pressure profile and the gravitational potential confining it. To obtain a notebook file containing all of the cells, go to ...

Before executing the rest of the cells, you will want to import a few items:

```python
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
```

## Power‑Law Atmosphere in an NFW Potential
As described on the [Essentials](/ExpCGM/descriptions/Essentials) page, all **ExpCGM** atmosphere models begin with a ***shape function*** that describes the shape of a galactic atmosphere's radial pressure profile:
$$
\alpha(r) = -\frac{d\ln P}{d\ln r} \; \; .
$$
We will begin with the simplest shape function: A constant value of $\alpha$ resulting in a power-law pressure profile. Farther down the page is an example in which $\alpha$ changes with radius.

This initial model also assumes an NFW gravitational potential:
$$
\varphi_{\rm NFW} (x) = A_{\rm NFW} \, v_\varphi^2 \, \left[ 1 - \frac {\ln (1+x)}{x} \right] \; \; .
$$
Here, $x = r/r_{\rm s}$ represents radius in units of the profile's scale radius $r_{\rm s}$, and $A_{\rm NFW} = 4.625$ is a normalization constant that makes the profile's maximum circular velocity (at $x = 2.163$) equal to $v_\varphi$. Note that we have chosen to put the potential's zero point at $r = 0$.

We will now set the values of some model parameters:

```python
alpha = 1.5       # constant power-law slope for pressure profile
A_NFW = 4.625     # Normalization constant for NFW potential well
eps = 10**(-4)    # Lower limit on x=r/r_s for numerical integrations
```

### Pressure Profile and Circular Velocity Profile

In general, the dimensionless pressure-profile function $f_P$ is obtained by integrating the shape function $\alpha (x)$ over $\ln x$. However, no integration is necessary for constant $\alpha$. We can simply define the pressure-profile function to be 
$$
f_P(r) = \left(\frac{r}{r_0}\right)^{-\alpha} \; \; ,
$$
where $r_0$ is the radius at which $f_P$ is normalized to unity.

The NFW potential function is given above, but we will also be using a circular velocity profile function obtained by differentating $\varphi(x)$ by $x$ and then multiplying the result by $x$:
$$
v_c^2(x) = A_{\rm NFW}\, v_\varphi^2\, \left[ \frac{\ln(1+x)}{x} - \frac{1}{1+x} \right] \; \; .
$$

This cell defines three functions determining how those profiles depend on $x = r / r_{\rm s}$:

```python
# We choose to set f_P equal to unity at r = r_s:

def f_P(x):        
    return x**(-alpha)

# We keep the NFW profile functions dimensionless and multiply them by A_NFW and v_phi^2 as needed:

def phi(x):
    return 1 - np.log(1 + x) / x 

def vc2(x):
    return np.log(1 + x) / x - 1 / (1 + x)
```

### Dimensionless Energy and Mass Integrals

The [Essentials](/ExpCGM/descriptions/Essentials) page explains how **ExpCGM** determines a galactic atmosphere's total specific energy $\varepsilon_{\rm CGM} = E_{\rm CGM} / M_{\rm CGM}$ by way of several dimensionless integrals:

$$
I(x) = v_\varphi^2 \int_0^x \frac{\alpha(x)f_P(x)}{v_c^2(x)}x^2\,dx
\; \; \; \; \textss{(cumulative gas mass)} 
$$
$$
J_\varphi(x) = \int_0^x \frac{\alpha(x)f_P(x)\varphi(x)}{v_c^2(x)}x^2\,dx
\; \; \; \; \text{(cumulative gravitational energy)} 
$$
$$
J_{\rm th}(x) = \frac{3}{2} \int_0^x f_P(x)\,x^2\,dx
\; \; \; \; \text{(cumulative thermal energy)} 
$$

In this example, we are modeling an atmosphere supported entirely by thermal energy $(f_{\rm th} = 1)$ and so we do not need to do an integral that calculate a non-thermal energy profile.

The total specific energy in this case is
$$
\varepsilon_{\rm CGM} = \frac{E_{\rm CGM}}{M_{\rm CGM}} = v_\varphi^2\, F\left(\frac{r_{\rm CGM}}{r_0}\right)
$$
where
$$
F(x) = \frac{J_\varphi(x) + J_{\rm th}(x)}{I(x)}
$$
is a dimensionless profile tracking the atmosphere's mean specific energy within $x$.

This cell defines functions that compute the necessary integrals:
```python
def integrandI(t): # We define the integrands as seperate functions before integrating them
    return f_P(t) * t**2 / vc2(t)

def I(x):        
    resultI, _ = integrate.quad(integrandI, eps, x, limit=50)
    return alpha / A_NFW * resultI

def integrandJphi(t):
    return f_P(t) * phi(t) / vc2(t) * t**2

def Jphi(x):
    resultJphi, _ = integrate.quad(integrandJphi, eps, x, limit=50)
    return alpha * resultJphi

def integrandJth(t):
    return f_P(t) * t**2

def Jth(x):
    resultJth, _ = integrate.quad(integrandJth, eps, x, limit=50)
    return 3 / 2 * resultJth

def F(x):
    return (Jphi(x) + Jth(x)) / I(x)

```

### Plotting

We compute $F(x_{\rm CGM})$ as a function of $x_{\rm CGM}$, then invert the axes to obtain a plot of atmospheric radius vs. mean specific energy. 

The dashed line shows how the normalization $P_0 \propto 1/I(x_{\rm CGM})$ of the atmosphere’s pressure profile declines as $\varepsilon_{\rm CGM}$ rises and the atmosphere expands.


```python
# Plotting the results
x_values = np.logspace(-1.5, 2, 50)
y1_values = [F(x) for x in x_values]
y2_values = [1/I(x) for x in x_values]

gfont = {'fontname':'georgia'}
plt.rcParams['font.family'] = 'georgia' 
plt.rcParams['font.size'] = 12 

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(y1_values, x_values, color='blueviolet', label='$x_{\\text{CGM}}$')
ax1.set_xscale('linear')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\mathrm{E}_{\mathrm{CGM}} \ / \ \mathrm{M}_{\mathrm{CGM}} \ \mathrm{v}_{\varphi}$', fontsize=12, **gfont)
ax1.set_ylabel(r'$x_{\mathrm{CGM}} \ = \ r_\mathrm{CGM} \ / \ r_s$ (y1 scale)', fontsize=12, **gfont)
ax1.set_ylim(10**-1.5, 10**2)
ax1.grid(True, linestyle='--', linewidth=0.5)

ax2 = ax1.twinx()
ax2.plot(y1_values, y2_values, color='orange', linestyle='--', label='$1/I(x)$')
ax2.set_ylabel('$1\\,/\\,I(x_{\\text{CGM}})$', fontsize=12, **gfont)
ax2.set_yscale('log')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left')

plt.title('Atmospheric Radius vs Mean Specific Energy', **gfont)
plt.show()

```


    
![png](Notebook_1_files/Notebook_1_9_0.png)
    


## 2. Variable Shape Function

In this section we use a shape function $\alpha(x)$ that varies with radius. The simplified cosmological profile 
$$
\alpha(r) = 1.7 \left( \frac{2r/r_{\rm max}}{1+r/r_{\rm max}} \right)
$$
represents a cosmological atmosphere. It is designed to have $\alpha \approx 1.7$ near the radius $r_{\rm max} = 2.16\, r_s$, where $v_c^2(r)$ peaks in an NFW gravitational potential.



```python
rmax = 2.16

A_NFW = 4.625
eps = 10**(-4)

def alpha(x):
    return (3.4*x)/(rmax + x)
```

We now have to use the integral form of $f_P(x)$ since alpha is not constant. 
$$
f_P(r) = \exp \left[ -\int_1^{r/r_0} \frac{\alpha(x)}{x}dx \right]
$$


```python
def integrandf_P(t):
    return alpha(t) / t

def f_P(x):        
    resultf_P, _ = integrate.quad(integrandf_P, 1+eps, x, limit=50)
    return np.exp(-resultf_P)

```

We also redefine the energy integrals so they have $\alpha(x)$ inside them. We can also define the $v_c^2(x)$ and $\varphi(x)$ functions the same way again so each section of the notebook is self-contained.


```python
def vc2(x):
    return np.log(1+x) / x - 1 / (1+x)

def phi(x):
    return 1- np.log(1+x)/x

def integrandI(t):
    return alpha(t) * f_P(t) * t**2 / vc2(t)

def I(x):        
    resultI, _ = integrate.quad(integrandI, eps, x, limit=50)
    return 1 / A_NFW * resultI

def integrandJphi(t):
    return alpha(t) * f_P(t) * phi(t) / vc2(t) * t**2

def Jphi(x):
    resultJphi, _ = integrate.quad(integrandJphi, eps, x, limit=50)
    return resultJphi

def integrandJth(t):
    return f_P(t) * t**2

def Jth(x):
    resultJth, _ = integrate.quad(integrandJth, eps, x, limit=50)
    return 3 / 2 * resultJth

def F(x):
    return (Jphi(x) + Jth(x)) / I(x)
```


```python
# Plotting the results

x_values = np.logspace(-1.5, 4, 50)
y_values = [F(x) for x in x_values]

gfont = {'fontname':'georgia'}
plt.rcParams['font.family'] = 'georgia' 
plt.rcParams['font.size'] = 12 
plt.figure(figsize=(8, 6))
plt.plot(y_values, x_values, color='blueviolet')
plt.xlim(0, 4.7)
plt.xscale('linear')
plt.yscale('log')
plt.title('Atmospheric Radius vs Mean Specific Energy', **gfont)
plt.xlabel(r'$\mathrm{E}_{\mathrm{CGM}} \ / \ \mathrm{M}_{\mathrm{CGM}} \ \mathrm{v}_{\varphi}$', fontsize=12)
plt.ylabel(r'$x_{\mathrm{CGM}} \ = \ r_\mathrm{CGM} \ / \ r_s$', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
```


    
![png](Notebook_1_files/Notebook_1_16_0.png)
    


## 3. Including a Central Galaxy

In addition to the NFW profile, we can add a central galaxy to the potential using the Hernquist model:
$$
v_{\rm H}^2(r) = \frac{GM_*r}{(r+r_{\rm H})^2}
$$
Most central galaxies have a maximum circular velocity similar to the maximum circular velocity $v_\varphi$ of the surrounding halo. 

Therefore we assume that $\max(v_{\rm H}) = v_\varphi$, so $r_{\rm H} = GM_*/4v_\varphi^2$.

Using this, and dividing by the scale radius, we get the circular velocity profile:
$$
v_{\rm H}^2(x) =  \frac{4 v_\varphi^2 (r_{\rm H}/r_0)x}{(x + r_{\rm H}/r_0)^2}
$$
We use the scale radius $r_H=0.2 r_0$, and combine this with the NFW profile:
$$
v_c^2(x) = v_{\rm NFW}^2(x) + v_{\rm H}^2(x)
$$


```python
A_NFW = 4.625
eps = 10**(-4)
rmax = 2.163

r_H = 0.2 # The Hernquist radius where v_c is maximized in the Hernquist model, as a fraction of r_0


# Defining the NFW and Hernquist models seperately, then adding them together
# Once again we can ignore the v_phi term as it cancels out, and keep the quantities dimensionless.

def v2_NFW(x):
    return A_NFW * (np.log(1+x) / x - 1 / (1+x))

def v2_H(x):
    return 4 * r_H * x / (x + r_H)**2

def vc2(x):
    return v2_NFW(x) + v2_H(x)
```


```python
# In this version, if we let both v_c^2 and alpha go to 0, the J_phi integral misbehaves causing double values
# One solution to this is to go back to a model with constant alpha.
# Another is to give a lower bound to alpha, taking the maximum of the two functions
# This approach works for down to alpha = ~0.7, though it gives an increasingly jagged graph

def alpha(x):
    # return 3.4 * ( x/rmax /(1+x/rmax) )   # option 1
    
    return 1.5   # option 2
    
    # alpha1 = 3.4 * ( x/rmax /(1+x/rmax) )
    # alpha2 = 0.7
    # return max(alpha1,alpha2)   # option 3

def integrandf_P(t):
    return alpha(t) / t

def f_P(x):
    resultf_P, _ = integrate.quad(integrandf_P, 1+eps, x, limit=50)
    return np.exp(-resultf_P)

def integrandphi(t):
    return vc2(t) / t

def phi(x):
    resultphi, _ = integrate.quad(integrandphi, eps, x, limit=50)
    return resultphi

def integrandI(t):
    return alpha(t) * f_P(t) * t**2 / (vc2(t))

def I(x):        
    resultI, _ = integrate.quad(integrandI, eps, x, limit=50)
    return resultI

def integrandJphi(t):
    return alpha(t) * f_P(t) * phi(t) / vc2(t) * t**2

def Jphi(x):
    resultJphi, _ = integrate.quad(integrandJphi, eps, x, limit=50)
    return resultJphi

def integrandJth(t):
    return f_P(t) * t**2

def Jth(x):
    resultJth, _ = integrate.quad(integrandJth, eps, x, limit=50)
    return 3 / 2 * resultJth

def F(x):
    return (Jth(x)+Jphi(x)) / I(x)
```


    
![png](Notebook_1_files/Notebook_1_19_0.png)
    



```python
# Plotting the results

x_values = np.logspace(-1.5, 2.2, 50)
y_values = [F(x) for x in x_values]

gfont = {'fontname':'georgia'}
plt.rcParams['font.family'] = 'georgia' 
plt.rcParams['font.size'] = 12 
plt.figure(figsize=(8, 6))
plt.plot(y_values, x_values, color='blueviolet')
plt.title('Atmospheric Radius vs Mean Specific Energy', **gfont)
plt.xscale('linear')
plt.yscale('log')
plt.xlabel(r'$\mathrm{E}_{\mathrm{CGM}} \ / \ \mathrm{M}_{\mathrm{CGM}} \ \mathrm{v}_{\varphi}$', fontsize=12)
plt.ylabel(r'$x_{\mathrm{CGM}} \ = \ r_\mathrm{CGM} \ / \ r_s$', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
```

Now that we have the central galaxy as well, we can plot the rotation curve being used as a reference.


```python
x_values_2 = np.logspace(-5, 0.7, 100)
v_c_values = [np.sqrt(vc2(x))/1e3 for x in x_values_2]

gfont = {'fontname':'georgia'}
plt.rcParams['font.family'] = 'georgia' 
plt.rcParams['font.size'] = 12 
plt.figure(figsize=(8, 6))
plt.plot(x_values_2, v_c_values, color='blueviolet')
plt.xscale('linear')
plt.yscale('linear')
plt.title('Velocity Profile', **gfont)
plt.xlabel(r'$x \ = \ r \ / \ r_s$', fontsize=12)
plt.ylabel(r'$v_c$ (km/s)', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
```


    
![png](Notebook_1_files/Notebook_1_22_0.png)
    
