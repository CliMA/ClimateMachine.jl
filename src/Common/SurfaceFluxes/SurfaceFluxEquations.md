# Surface Flux Equations

This page contains an extension of the Nishizawa surface-flux scheme for use within the ClimateMachine framework. 

> Nishizawa, S., & Kitamura, Y. ( 2018). A surface flux scheme based on the Monin‐Obukhov similarity for finite volume models. _Journal of Advances in Modeling Earth Systems_, 10, 3159– 3175. [https://doi.org/10.1029/2018MS001534](https://doi.org/10.1029/2018MS001534)
> <em>Internally abbreviated as NK</em>    

> Gryanik, V., Lupkes, C., Grachev, A. & Sidorenko, D. (2020). New modified and extended stability functions for the stable boundary layer based on SHEBA and parametrizations of bulk transfer coefficients for climate models. _Journal of the Atmospheric Sciences_, 10.1175/JAS-D-19-0255.1. [https://doi.org/10.1175/JAS-D-19-0255.1](https://doi.org/10.1175/JAS-D-19-0255.1). (Early Online Release) 
> <em>Internally abbreviated as GLGS</em>

>Hill, R.J., 1989. Implications of Monin–Obukhov Similarity Theory for Scalar Quantities. _J. Atmos. Sci.,_  **46**, 2236–2244, [https://doi.org/10.1175/1520-0469(1989)046<2236:IOMSTF>2.0.CO;2](https://doi.org/10.1175/1520-0469(1989)046%3C2236:IOMSTF%3E2.0.CO;2) 

### Parameters
* $L_{MO}$ = Monin-Obukhov length
* $X_{\star}$ = Scale parameter for variable $X$
* $X_{sfc}$ = Surface value of variable $X$ 
* $\kappa$ = von-Karman coefficient (typically $0.4$)
* $\mathrm{D_T} = \nu/Pr_{T}$ is the diffusivity
*  $Pr_{T}$ the neutral limit turbulent Prandtl number (typically $\frac{1}{3}$)
* $\nu$ the kinematic viscosity
* $w$, the wall normal velocity component
* $\Delta z$ = Model reference altitude
* $\delta_{\chi_{i}}$ a measure of mixing efficiency for $i$th tracer 
* $\zeta = z/L_{MO}$ similarity variable
* $\nu/u_{\star}$ Viscous lengthscale

### Fluxes

* Momentum fluxes due to boundary shear
```math 
-\rho \tau_{j3}|_{\mathrm{sfc}} = \rho \overline{w^{\prime}u_{j}^{\prime}}|_{\mathrm{sfc}} 
``` where $u$ represents the surface parallel velocity component, for $j = 1,2$

*  Sum of specific enthalpy and moist static energy flux
```math 
\rho\mathrm{d}h_{t}|_{\mathrm{sfc}} = \rho \mathrm{(J+D)|_{sfc}} = -\rho D_{T}\nabla h_{t}|_{\mathrm{sfc}} = \rho[c_{pm}\overline{w^{\prime}\theta^{\prime}} + L_{v}\overline{w^{\prime}q_{t}^{\prime}}]|_{\mathrm{sfc}} 
```

* Subgrid flux of total water specific humidity
```math 
\rho\mathrm{d}q_{t}|_{\mathrm{sfc}}  = -\rho D_{T}\nabla q_{t}|_{\mathrm{sfc}}  = \rho L_{v} \overline{w^{\prime}q_{t}^{\prime}}|_{\mathrm{sfc}} 
```

* Subgrid flux of tracer $i$, for $i = 1..n$ tracers, n an integer
```math 
\rho\mathrm{d}\chi_{i}|_{\mathrm{sfc}}  = -\rho \delta_{\chi_{i}} D_{T}\nabla \chi_{i}|_{\mathrm{sfc}}  = \rho \overline{w^{\prime}\chi_{i}^{\prime}}|_{\mathrm{sfc}} 
```

We also have, by assumption that each variable can be represented in terms of the universal functions $\Phi_{k}$ for the $k$th variable,
* $\overline{w^{\prime}u^{\prime}}|_{\mathrm{sfc}} = -u_{\star}^2$
* $\overline{w^{\prime}\theta^{\prime}}|_{\mathrm{sfc}} = -u_{\star}\theta_{\star} = \frac{J}{\rho c_{pm}}$
* $\overline{w^{\prime}q_{t}^{\prime}}|_{\mathrm{sfc}} = -u_{\star}q_{t\star} = \frac{D}{\rho L_{v}}$
* $\overline{w^{\prime}\chi_{i}^{\prime}}|_{\mathrm{sfc}} = -u_{\star}\chi_{i\star}$


 
### Boundary conditions
Required variables for boundary condition prescription in `ClimateMachine` are

#### Momentum 
$u_{\star}$  in `bc_momentum.jl` . Usage is through the `DragLaw` function. Current use cases include prescribed values for $u_{\star}$, or prescribed exchange coefficients.

#### Energy
$\hat{n} \cdot \mathrm{d}h_{tot}$  in `bc_energy.jl` . Usage currently is through the `PrescribedEnergyFlux`

#### Moisture 
$\hat{n} \cdot \mathrm{d}q_{tot}$   in `bc_moisture.jl`  . Usage currently is through `PrescribedMoistureFlux`

#### Tracer Flux 
$\hat{n} \cdot \mathrm{d}\chi_{i}$   in `bc_moisture.jl`  . No use cases currently. Tracers do not permeate through boundaries in present implementation. 

In each of these cases $\hat{n}$ corresponds to the upward pointing unit-normal vector. 

## Equations

We start by defining the universal functions for the velocity, enthalpy and moisture terms. _NK_ express the Monin-Obukhov length in terms of the potential temperature 

```math 
L_{MO} = -\frac{u_{\star}^3 \overline{\theta}}{\kappa g \overline{w^{\prime}\theta^{\prime}}|_{sfc}} 
```

From the above definitions for the fluxes in the moist, compressible flow equations, we substitute the potential temperature flux $\overline{w^{\prime}\theta^{\prime}} = J/(\rho c_{pm})$, $\overline{w^{\prime}u^{\prime}} = \tau/\rho)$ to recover eq (1) in _GLGS_. 

```math 
L_{MO} = -\frac{(\tau/\rho)^{(3/2)} \overline{\theta}}{\kappa g \frac{J}{(\rho c_{pm})}|_{sfc}} 
```

Given the formulation of the compressible flow equations in `ClimateMachine`  we arrive at the following formulation for the Monin-Obukhov similarity length

```math 
L_{MO} = - \frac{\overline{\theta} u_{\star}^3}{\kappa g [\overline{w^{\prime}\theta^{\prime}} + \overline{w^{\prime}q_{t}^{\prime}}]} = - \frac{\overline{\theta} u_{\star}^3}{\kappa g [\frac{J}{\rho c_{pm}} + \frac{D}{\rho L_{v}}]}
```

NK use volume-averaged quantities$\langle X \rangle$ such that 
```math 
\langle X \rangle = \frac{1}{\Delta z} \int_{0}^{\Delta z} X(z) dz
```

In the `ClimateMachine` DG discretization we interpret these as the nodal variable values at a given model reference altitude $\Delta z$. This value is currently taken as the first interior grid point in the model column description, which typically is of $O(1-10)$ in LES configurations and $O(10-100)$ in the GCM configurations. An alternative to this is obtained by fixing the reference $\Delta z$ and interpolating the DG polynomials to determine variable values at the specified model height. Applying _NK_ equations (17), (18) to the `ClimateMachine` equations we arrive at 

```math 
u_{\star} = \frac{\kappa (\langle u  \rangle - u_{sfc})}{\log(\frac{z}{z_{0m}})-\Psi_{h}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0m}}{\Delta z})\Psi_{m}(\frac{\Delta z_{0m}}{L_{MO}}) +  (1 − \frac{z_{0m}}{Δz})\{ \Psi_{m} (\frac{z_{0m}}{L_{MO}}) - 1\} } 
```
```math 
h_{\star} = \frac{\frac{\kappa}{Pr_{T}}(\langle h_{tot} \rangle- h_{sfc})  }{\log(\frac{z}{z_{0h}})-\Psi_{h}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0h}}{\Delta z})\Psi_{h}(\frac{\Delta z_{0h}}{L_{MO}}) + (1 − \frac{z_{0h}}{Δz})\{ \Psi_{h} (\frac{z_{0h}}{L_{MO}}) - 1\}} 
```
```math 
q_{\star} = \frac{\frac{\kappa}{Pr_{T}} (\langle q \rangle- q_{sfc})  }{\log(\frac{z}{z_{0q}})-\Psi_{q}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0q}}{\Delta z})\Psi_{q}(\frac{\Delta z_{0q}}{L_{MO}}) +  (1 − \frac{z_{0q}}{Δz})\{ \Psi_{q} (\frac{z_{0q}}{L_{MO}}) - 1\}    } 
```
Here we follow Hill's (1989) argument for the form of the universal functions for tracers which are assumed to follow the Monin-Obukhov similarity theory. 

The three equations above combined with the M-O similarity length provide 4 equations for 4 unknowns. Once the Monin-Obukhov length is computed, we can then obtain the tracer surface values. 

```math 
\chi_{i\star} = \frac{\frac{\kappa}{Pr_{T} \delta \chi_{i}} (\langle \chi_{i} \rangle- \chi_{i,sfc})  }{\log(\frac{z}{z_{0h}})-\Psi_{\chi_{i}}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0\chi_{i}}}{\Delta z})\Psi_{\chi_{i}}(\frac{\Delta z_{0\chi_{i}}}{L_{MO}}) +  (1 − \frac{z_{0h}}{Δz})\{ \Psi_{\chi_{i}} (\frac{z_{0h}}{L_{MO}}) - 1\} } 
```


#### Additional equations

* $\frac{\kappa z}{u_{\star}}\frac{\partial u}{\partial z} = \Phi_{m}$
* $\frac{\kappa z}{h_{\star}}\frac{\partial h_{tot}}{\partial z} = \Phi_{h}$
* $\frac{\kappa z}{q_{\star}}\frac{\partial q_{tot}}{\partial z} = \Phi_{q}$
* $\frac{\kappa z}{\chi_{i\star}}\frac{\partial \chi_{i}}{\partial z} = \Phi_{\chi}$
