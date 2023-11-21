---
title: Confinement
layout: default
nav_order: 3
parent: Description
---

# Confinement

## Binding Energy

An atmosphere's binding energy depends on how its support energy compares with the depth of the potential well confining it. 
According to the **ExpCGM** framework, the specific support energy of atmospheric gas at radius $r$ is
$$\frac {3} {2} \left( \frac {P} {f_{\rm th} \rho} \right) = \left( \frac {3 f_\varphi} {2 \alpha_{\rm eff}} \right) v_{\rm c}^2 (r)$$
when a combination of thermal and turbulent energy supports a steady-state galactic atmosphere. (See the [Essentials](Essentials) page for an explanation and definitions of these symbols.) If there are other forms of support energy, then the leading numerical coefficient (3/2) may be different. The specific gravitational binding energy of the gas at radius $r$ is
$$\epsilon_{\rm bind} (r) = \left( \frac {3 f_\varphi} {2 \alpha_{\rm eff}} \right) v_{\rm c}^2 (r) + \varphi(r) - \varphi_\infty$$
in which $\varphi(r)$ is the gravitational potential and $\varphi_\infty$ is its limit as $r \rightarrow \infty$. A gas layer at $r$ is bound to the potential if
$$\alpha_{\rm eff}(r) > \frac {3 f_\varphi} {2} \frac {v_{\rm c}^2(r)} {\varphi_\infty - \varphi(r)}$$
Otherwise, the pressure of overlying atmospheric layers external to $r$ needs to confine that layer.

To see how gravity and external pressure jointly confine atmospheric gas in the \textsc{ExpCGM} framework, consider an NFW potential well 
$$\varphi_{\rm NFW} (x) = A_{\rm NFW} v_\varphi^2 \left[ 1 - \frac {\ln (1 + x)} {x} \right]$$
with the normalization factor $A_{\rm NFW} = 4.625$. The parameter $v_\varphi$ is then the maximum value attained by the potential's circular velocity profile 
$$v_{\rm c}^2 (x) = A_{\rm NFW} v_\varphi^2 \left[ \frac {\ln (1 + x)} {x} - \frac {1} {1 + x} \right]$$
which reaches $v_\varphi^2$ at $r = 2.163 r_{\rm s}$. Circular velocity is nearly constant with radius near that peak but gradually declines at $r \gg r_{\rm s}$. Because of that decline, the value of $\varphi_\infty$ is finite.

In an NFW potential well, the condition for purely gravitational confinement (with $f_\varphi = 1$) reduces to
$$\alpha_{\rm eff} > \frac {3} {2} \left[ 1 - \frac {x} { (1 + x) \ln (1 + x) } \right]$$
Notice that $\alpha_{\rm eff} = 3/2$ is a critical value for gravitational confinement: 

* All layers having $\alpha_{\rm eff} > 3/2$ are gravitationally bound, because each layer's specific support energy $3 v_{\rm c}^2(r) / 2 \alpha_{\rm eff} (r)$ is less than the specific energy $v_{\rm c}^2(r) = G M_r / r$ required to escape the gravitational attraction of the total mass $M_r$ within $r$. (This general result does not depend on the details of $\varphi$.)

* Gravitational confinement at small radii ($r \ll r_{\rm s}$) does not depend on $\alpha_{\rm eff}$ because $v_{\rm c}^2 \ll \varphi_\infty$. (This result applies to potential wells in which $v_c^2$ goes to zero as $r \rightarrow 0$, as in the NFW potential.) 

* Atmospheric layers that have $\alpha_{\rm eff} < 3/2$ can be pressure confined by the weight of overlying layers, but the pressure profile must steepen to $\alpha_{\rm eff} > 3/2$ at larger radii in order for the entire atmosphere to be gravitationally confined.

* An atmosphere with $\alpha_{\rm eff} < 3/2$ near its outer boundary needs to be confined by external pressure forces.

If the radial gradient of the thermal support fraction $f_{\rm th}$ is insignificant, then $\alpha$ replaces $\alpha_{\rm eff}$ in this bullet list. The atmospheres of galaxy clusters appear to abide by the constraints listed above, since observations of their thermal pressure profiles (e.g., Pointecouteau et al. 2021, A&A, 651, A73) show that $\alpha \lesssim 1$ at $r \ll r_{\rm s}$ and $\alpha > 2$ at $r \gg r_{\rm s}$. 

In the **ExpCGM** framework, raising an atmosphere's mean specific energy to $\varepsilon_{\rm CGM} > \varphi_\infty$ restricts the set of pressure profiles that can be in a static equilibrium configuration. For example, the maximum value of mean specific energy in an atmosphere with $\alpha_{\rm eff} > 3/2$ cannot exceed $\varphi_\infty$, which is $4.625 v_\varphi^2$ in an NFW potential. Energy input comparable to $M_{\rm CGM} v_\varphi^2$ can exponentially expand the atmosphere, but energy input that raises $\varepsilon_{\rm CGM}$ to $\varphi_\infty$ drives the atmosphere's radius to infinity and its pressure normalization to zero. In that limit, the assumption of gravitational confinement for $\alpha_{\rm eff} > 3/2$ is somewhat artificial but is still useful for quantifying the connection between an atmosphere's specific energy $\varpesilon_{\rm CGM}$ and its pressure normalization factor $P_0$.

## Evolving Potential

The previous section used an NFW profile to represent the potential well of a virialized dark matter halo and found a limiting specific energy $\varphi_\infty \approx 4.6 v_\varphi^2$ for a gravitationally bound atmosphere. However, cosmological halos evolve with time, and the specific energy required to escape an evolving halo can be substantially larger than $4.6 v_\varphi^2$.

Specifying the gravitational potential that binds the matter near the outer radius of a cosmological halo ($R_{\rm halo}$) turns out to be a subtle business. Naively, the specific energy required to raise gas at $R_{\rm halo}$ out of the halo's potential well would seem to be $v_{\rm c}^2 (R_{\rm halo}) = G M_{\rm halo} / R_{\rm halo}$, but that part of the gravitational potential accounts only for the matter currently within $R_{\rm halo}$ and ignores matter at larger radii that has yet to fall into the halo. Therefore, we need to consider the halo's \textit{entire} accretion history in order to determine the energy input required for permanent unbinding of its atmosphere. 

In the idealized spherical collapse model (see \textbf{\textsc{ExpCGM:} Accretion}), there is a marginally bound shell containing a mass $M_\infty$ that asymptotically approaches a radius
$$R_\infty = \left( \frac {G M_\infty} {H_0^2 \Omega_\Lambda} \right)^{1/3}$$
as $t \rightarrow \infty$. All of the shells within the marginally bound shell's radius are gravitationally bound and ultimately collapse toward $R = 0$. Ideally, the zero point of the overall gravitational potential should result in zero binding energy for the marginally bound shell. However, both $M_\infty$ and $R_\infty$ may be far larger than $M_{\rm halo}$ and $R_{\rm halo}$, as they usually correspond to the mass and radius of the supercluster of galaxies to which the galaxy of interest belongs.

More pragmatically, we would like know whether gas that passes outside of $R_{\rm halo}$ early in time, when $M_{\rm halo}$ is small, still belongs to a galaxy's atmosphere later on, when both $M_{\rm halo}$ and $R_{\rm halo}$ are considerably larger. To account for that possibility, we can define $M_0$ to be a halo's mass at the present time $t_0$. The radius $R_0 (t)$ of the shell containing mass $M_0$ can be computed from the equation of motion for a ballistic shell that reaches its turnaround radius at $t_{\rm ta} = t_0 / 2$. Escaping the gravitational potential of the halo's eventual mass $M_0$ is easiest at that moment and requires a specific energy $\gtrsim G M_0 / R_0 (t_{\rm ta})$. According to the spherical collapse model, the halo's radius at time $t_0$ is $R_{\rm halo} (t_0) \approx R_0 (t_0/2) / 2$. Permanent escape from the halo's potential therefore requires a specific energy exceeding
$$\frac {G M_0} {2 R_{\rm halo}(t_0)}$$
even early in time, when the circular velocity at $R_{\rm halo} (t)$ might be considerably smaller than the value it attains later. 

For example, consider a massive galaxy that will eventually become the central galaxy of a large cluster of galaxies. Suppose that the circular velocity of that galaxy is $v_\varphi \sim 400 \, {\rm km \, s^{-1}}$ early in time, while the galaxy is still forming. Later in time that same galaxy will be centered within a galaxy cluster with a circular velocity $v_c \sim 1600 \, {\rm km \, s^{-1}}$ (corresponding to $kT_\varphi \sim 8 \, {\rm keV}$). During that galaxy's history, the circular velocity at $R_0$ reaches its minimum value $v_0 \sim 1600 \, {\rm km \, s^{-1}} / \sqrt{2}$ at the turnaround time $t_0 / 2$ of the shell containing the cluster's current mass $M_0$. Consequently, $v_0^2 \gtrsim 8 v_\varphi^2$ while the cluster's central galaxy is forming. The energy input required to unbind the central galaxy's atmosphere from the halo's eventual potential well is therefore an order of magnitude greater than what one would infer from a virialized halo's potential at early times.

## Extended Potential

To account for atmospheric confinement by matter extending beyond the halo, the **ExpCGM** framework includes an extended potential well that is unbounded. Numerical simulations of cosmological structure formation (e.g., see Diemer & Kravtsov 2014, ApJ, 789, 1) indicate that the total mass density $\rho_M$ at radii several times $R_{\rm halo}$ declines with an approximate power-law dependence on radius ranging from $r^{-1}$ to $r^{-1.5}$. The framework therefore accounts for the gravitational potential of matter external to the halo using the extended potential
$$\varphi_{\rm ext} (r) = 2 \pi G \rho_{\rm ext} R_{\rm halo}^2 \left( \frac {r} {R_{\rm halo}} - 1 \right) \left( 1 - \frac {R_{\rm halo}} {r} \right) $$
that results from assuming $\rho_M (r) = \rho_{\rm ext} (r/R_{\rm halo})^{-1}$ outside of $R_{\rm halo}$.

The extended potential's density normalization factor $\rho_{\rm ext}$ is linked to the halo's mass accretion rate. Assuming that $\rho_{\rm ext}$ corresponds to matter that is currently accreting onto the halo gives $\rho_{\rm ext} = \dot{M}_{\rm halo} / 4 \pi R_{\rm halo}^2 v_{\rm acc}$ and
$$\varphi_{\rm ext} (r) = \frac {G \dot{M}_{\rm halo}} {v_{\rm acc}} \left( \frac {r} {R_{\rm halo}} - 1 \right) \left( 1 - \frac {R_{\rm halo}} {r} \right)$$
in which $v_{\rm acc}$ is the infall speed of accreting matter. Notice that the leading factor on the right of the equation breaks up into two factors that are easier to interpret:
$$\frac {G \dot{M}_{\rm halo}} {v_{\rm acc}} = \left( \frac {G M_{\rm halo}} {R_{\rm halo}} \right) \left( \frac {R_{\rm halo}} {v_{\rm acc}} \frac {\dot{M}_{\rm halo}} {M_{\rm halo}} \right)$$
The first factor in parentheses is the specific energy required to escape from the radius $R_{\rm halo}$ encompassing a mass $M_{\rm halo}$. The second one is the fraction $f_{\rm acc}$ of a halo's mass that accretes on a timescale $R_{\rm halo} / v_{\rm acc}$. 

In the **ExpCGM** framework, $f_{\rm acc}$ is a model parameter that specifies the normalization of the extended potential:
$$\varphi_{\rm ext} (r) = f_{\rm acc} \left( \frac {G M_{\rm halo}} {R_{\rm halo}} \right) \left( \frac {r} {R_{\rm halo}} - 1 \right) \left( 1 - \frac {R_{\rm halo}} {r} \right)$$ 
It can be estimated using the approximation $v_{\rm acc} \approx (G M_{\rm halo} / 2 R_{\rm halo} )^{1/2}$ that comes from spherical collapse, giving

<p>
  $$f_{\rm acc} \approx 2 \left( \frac {\rho_{\rm cr}} {\rho_{\rm halo}} \right)^{1/2} \frac {\dot{M}_{\rm halo}} {H M_{\rm halo}}
</p>


where is $H$ is the Hubble expansion parameter, $\rho_{\rm halo}$ is the halo's mass density, and $\rho_{\rm cr} = 3 H^2 / 8 \pi G$ is the universe's critical density. Early in cosmic time, while halos are rapidly forming, $f_{\rm acc}$ can be of order unity. However, it becomes closer to $f_{\rm acc} \sim 0.1$ later in time, as structure formation slows down.

\caption{Dependence of a galactic atmosphere's scaled radius $x_{\rm CGM} = r_{\rm CGM} / r_{\rm s}$ on its scaled specific energy $\bar{\epsilon}/v_\varphi^2 = E_{\rm CGM} / M_{\rm CGM} v_\varphi^2$ in an NFW potential well extended to account for surrounding matter. Solid lines show how $x_{\rm CGM}$ depends on specific energy for a thermally supported atmosphere with $P(r) \propto r^{-3/2}$. Labels on those lines give the model parameter $f_{\rm acc}$ governing the magnitude of the external potential, and the series of lines illustrates how higher mass accretion rates reduce the equilibrium radius $r_{\rm CGM}$ corresponding to a given value of specific energy. Dashed lines show how the pressure profile normalization $P_0 \propto 1 / I(x_{\rm CGM})$ at $r_0$ declines as $E_{\rm CGM}$ increases.  Horizontal dotted lines show the chosen halo radius ($R_{\rm halo} = 10 r_{\rm s}$) and the approximate stagnation radius ($3 R_{\rm halo} = 30 r_{\rm s}$) beyond which universal expansion is carrying matter away from the halo.}
\label{fig:xCGM_ext}
\end{figure}

The figure illustrates how the relationship between an atmosphere's equilibrium radius $r_{\rm CGM}$ and its specific energy $\varepsilon_{\rm CGM}$ depends on $f_{\rm acc}$. It starts with an NFW potential well, adds the extended potential $\varphi_{\rm ext}(r)$, and assumes that thermal pressure follows $P(r) \propto r^{-3/2}$, as in the figure on the [Essentials](Essentials) page. The lines for $f_{\rm acc} = 0$ are the same as in that figure, but they shift to the right as $f_{\rm acc}$ increases. An increase in $f_{\rm acc}$ corresponds to an increase in the extended potential's ability to confine gas that has been pushed beyond $R_{\rm halo}$. Consequently, equilibrium solutions are possible for larger values of $\varepsilon_{\rm CGM}$ than in the NFW-only case.

The extended potential in **ExpCGM** is formally unbounded, with $\varphi_{\rm ext} \rightarrow \infty$ as $r \rightarrow \infty$. Any atmosphere contained within it is therefore bound to it, no matter what its specific energy. However, even an atmosphere that is formally bound becomes extremely diffuse for $E_{\rm CGM}/M_{\rm CGM} \gg G M_{\rm halo} / R_{\rm halo} \sim v_\varphi^2$. In that limit, an \textsc{ExpCGM} model atmosphere becomes unrealistic near $r_{\rm CGM}$ because the dynamical time beyond $r \sim 3 R_{\rm halo}$ starts to exceed the universe's age, meaning that the equilibrium assumption it is based on becomes unphysical. According to the spherical collapse model for structure formation, mass shells near $r \sim 3 R_{\rm halo}$ have reached their maximum radii and are starting to fall back toward the halo.\footnote{Kepler's Third Law gives $R_{\rm ta}(t) = 2^{2/3} R_{\rm ta}(t/2) \approx 2^{5/3} R_{\rm halo}(t)$ for the turnaround radius $R_{\rm ta}$ in a spherical collapse model without dark energy.} Numerical simulations corroborate that result. Atmospheric gas pushed beyond $r \sim 3 R_{\rm halo}$ is therefore entering a region where cosmological expansion dominates the dynamics and needs to be modeled in a context that accounts for the atmosphere's expansion speed.   