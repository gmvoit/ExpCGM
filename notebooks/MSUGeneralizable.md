---
title: MSU Generalizable
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

# MSU Generalizable Notebook

*Contributed by Doruk Yaldiz and Jazzmin Partridge, edited by Mark Voit*

The Python notebook cells on this page demonstrate how to extend the **ExpCGM** implementation in the [MSU Essentials Notebook](/ExpCGM/notebooks/MSUEssentials) to incorporate a user-defined pressure profile shape, a user-defined gravitational potential, and non-thermal atmospheric support energy. To copy and paste a cell into a Python notebook running on your own computer, move your cursor to the upper left corner of the cell and click on the clipboard icon that appears.

Before executing the cells that follow, import these items:  

```python
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
```

## User-Defined Pressure Profiles

To implement a pressure profile that is not a simple power law, an **ExpCGM** user needs to specify its shape by supplying a shape function $\alpha(x)$ that depends on a dimensionless radius $x$. 

We will demonstrate how to do that by implementing the simplified cosmological profile discussed on the [Pressure Profiles](/ExpCGM/extensions/PressureProfiles) page:
$$
\alpha(r) = 1.7 \left( \frac{2r/r_\mathrm{max}}{1+r/r_\mathrm{max}} \right)
$$
Its slope is very shallow $(\alpha \ll 1)$ at small radii. It steepens to $\alpha \approx 1.7$ near the radius $r_\mathrm{max} = 2.16\, r_s$ at which $v_c^2(r)$ peaks in an NFW gravitational potential. And it converges toward $\alpha = 3.4$ at large radii.

To prepare for using that shape function in conjunction with an NFW potential well, we will rewrite it as a function of $x = r / r_\mathrm{s}$:
$$
\alpha(x) = \frac {1.59 x} {1 + 0.468 x} 
$$
The following cell defines that shape function and can be replaced with a different user-defined shape function:

```python
def alpha(x):
    return (1.59*x)/(1 + 0.468*x)
```

Because $\alpha(x)$ is not constant, a numerical integration is needed to determine the dimensionless pressure profile function: 
$$
f_P(r) = \exp \left[ -\int_1^{r/r_0} \frac{\alpha(x)}{x}dx \right]
$$
Executing the next cell defines a function that integrates $\alpha (x)$ over $\ln x$ to obtain a version of $f_P(x)$ that is normalized to unity at $r = r_\mathrm{s}$:

```python
def integrandf_P(t):
    return alpha(t) / t

def f_P(x):        
    resultf_P, _ = integrate.quad(integrandf_P, 1+eps, x, limit=50)
    return np.exp(-resultf_P)

```

To check the result, this cell makes a plot showing $f_P(x)$:

```python
# Specify the domain of x and determine f_P(x)
x_values = np.logspace(-1.5, 2, 50)
y_values = [f_P(x) for x in x_values]

# Choose a font
gfont = {'fontname':'georgia'}
plt.rcParams['font.family'] = 'georgia' 
plt.rcParams['font.size'] = 12 

plt.plot(x_values, y_values, color='blueviolet')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$x = r / r_\mathrm{s}$', fontsize=12)
plt.ylabel(r'$f_P = P(r)/P(r_\mathrm{s})', fontsize=12)

plt.title('Dimensionless Pressure Profile', **gfont)
plt.show()
```
![png](MSUGeneralizable_files/f_p.png)

In the **ExpCGM** framework, a pressure profile's normalization depends on the atmosphere's mean specific energy ($\varepsilon_\mathrm{CGM}$). In this case, the normalization factor is
$$
P_0 = P(r_\mathrm{s}) = \frac {M_\mathrm{CGM} v_\varphi^2} {4 \pi r_\mathrm{s}^3} \frac {1} {I(x_\mathrm{CGM})} 
$$
As explained on the [Essentials](/ExpCGM/descriptions/Essentials) page, $I(x)$ is an integral proportional to the cumulative enclosed gas-mass profile, and $x_\mathrm{CGM}$ is a dimensionless radius that solves $\varepsilon_\mathrm{CGM} = v_\varphi^2  F(x_{\rm CGM})$.

## User-Defined Potential Wells

The default choice for a halo potential well in **ExpCGM** is an NFW potential well, described by the dimensionless functions defined in the following cell:

```python
A_NFW = 4.625      # Normalization constant for the NFW potential well
eps = 10**(-4)     # Lower limit on x=r/r_s for numerical integration

def vc2_NFW(x):
    return np.log(1+x) / x - 1 / (1+x)

def phi_NFW(x):
    return 1- np.log(1+x)/x
```

Later, we will multiply each of these functions by $v_\varphi^2$, the square of the potential's maximum circular velocity, to make them dimensional quantities. You may also choose to replace the NFW potential with a user-defined potential well.

Here, we will extend the potential well model by adding the potential well of a central galaxy with a maximum circular velocity $v_\mathrm{H} = f_\mathrm{H} v_\varphi$, where $f_\mathrm{H}$ is an adjustable model parameter. And to represent the central galaxy, we will use a Hernquist potential with a scale radius $r_\mathrm{H} = x_\mathrm{H} r_\mathrm{s}$: 

$$
\varphi_\mathrm{H} = 4 v_\mathrm{H}^2 
  \left( 1 + \frac {r_\mathrm{H}} {r + r_\mathrm{H}} \right)
$$

This cell provides the dimensionless functions that give the appropriate dimensional quantities when multiplied by $v_\varphi^2$ or $v_\varphi$:

```python
def phi_gal(x,x_H,f_H):
    return 4 * f_H**2 * (1 - x_H / (x + x_H) )

def vc2_gal(x,x_H,f_H):
    return 4 * f_H**2 * x_H * x / (x + x_H)**2

def phi(x,x_H,f_H):
    return phi_NFW(x) + phi_gal(x,x_H,f_H)

def vc2(x,x_H,f_H):
    return vc2_NFW(x) + vc2_gal(x,x_H,f_H)

def vc(x,x_H,f_H):
    return np.sqrt(vc2(x,x_H,f_H))
```

To check the result, this cell makes a plot showing $v_\mathrm{c}(x)/v_\varphi$ for $x_H = 0.1$ and $f_H = 1.0$:

```python
# Set the parameters of the Hernquist model 
x_H = 0.1
f_H = 1.0

# Specify the domain of x and determine v_c
x_values = np.logspace(-1.5, 2, 50)
vc_values = [vc(x,x_H,f_H) for x in x_values] 

# Choose a font
gfont = {'fontname':'georgia'}
plt.rcParams['font.family'] = 'georgia' 
plt.rcParams['font.size'] = 12 

# Make the plot
plt.plot(x_values, vc_values, color='blueviolet')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel(r'$x = r / r_\mathrm{s}$', fontsize=12)
plt.ylabel(r'$v_\mathrm{c} / v_\varphi$', fontsize=12)

plt.title('Dimensionless Circular Velocity Profile', **gfont)
plt.show()
```
![png](MSUGeneralizable_files/vc_dimensionless.png)

## Temperature Profile and Halo Mass

In the **ExpCGM** framework, the temperature profile of an atmosphere fully supported by thermal pressure depends only on $\alpha(x)$ and $v_c(x)$: 
$$ 
kT(x) = \frac {\mu m_p v_\mathrm{c}^2 (x)} {\alpha(x) 
$$
However, the normalization factor $v_\varphi$ of the circular velocity profile depends on the total mass within a particular radius.

To calculate the normalization factor, an **ExpCGM** user needs to specify a halo mass $M_\mathrm{halo}$, a cosmological redshift $z$, a halo concentration factor $c_\mathrm{halo} = r_\mathrm{halo} / r_\mathrm{s}$, and the density contrast factor $\Delta_\mathrm{halo}$ relating the mean mass density within $r_\mathrm{halo}$ to the universe's critical density at redshift $z$. The formula determining the normalization factor is
$$
v_\mathrm{c}(r_\mathrm{halo}) 
  = \left( \frac {\Delta_\mathrm{halo}} {2} \right)^{1/6}
    \left[ G M_\mathrm{halo} H(z) \right]^{1/3}
$$
The following cell defines a function that returns $v_\varphi$ when given $z$, $c_\mathrm{halo}$, $\Delta_\mathrm{halo}$, and $M_\mathrm{halo}$ in units of solar mass:

```python
```

## Cumulative Mass and Energy Integrals

We also redefine the energy integrals so they have $\alpha(x)$ inside them. We can also define the $v_c^2(x)$ and $\varphi(x)$ functions the same way again so each section of the notebook is self-contained.


```python

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
    


## General Potential Well

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

