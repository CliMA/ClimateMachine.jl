# Surface Fluxes

```@meta
CurrentModule = ClimateMachine.SurfaceFluxes
```
![](exchange_byun1990_fig4a.svg)

Recreation of Figure 4(a) from Byun (1990)

```@example byun1990
plot(Ri_range,
     [Byun1990.compute_exchange_coefficients(param_set, Ri, scale * z_0, z_0,
                                             γ_m, γ_h, β_m, β_h, Pr_0)[2]
        for Ri in Ri_range, scale in scales],
     xlabel = "Bulk Richardson number (Ri_b)",
     ylabel = "Exchange coefficient",
     title = "Heat exchange coefficient",
     labels = scales, legendtitle=L"z/z_0")

savefig("exchange_byun1990_fig4b.svg") # hide
nothing # hide
```
![](exchange_byun1990_fig4b.svg)

Recreation of Figure 4(b) from Byun (1990)

## `Nishizawa2018`

### Plots

```@example nishizawa2018
using ClimateMachine.SurfaceFluxes.Nishizawa2018
using Plots, LaTeXStrings
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
param_set = EarthParameterSet()

FT = Float64

a = FT(4.7)
θ = FT(350)
z_0 = FT(10)
u_ave = FT(10)
flux = FT(1)
Δz = range(FT(10.0), stop=FT(100.0), length=100)
Ψ_m_tol, tol_abs, iter_max = FT(1e-3), FT(1e-3), 10

u_star = Nishizawa2018.compute_friction_velocity.(
    Ref(param_set),
    u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)

plot(u_star, Δz, title = "Friction velocity vs dz",
     xlabel = "Friction velocity", ylabel = "dz")

savefig("friction_velocity.svg") # hide
nothing # hide
```
![](friction_velocity.svg)


## Equations

This page contains an extension of the Nishizawa surface-flux scheme for use within the ClimateMachine framework. 

> Nishizawa, S., & Kitamura, Y. ( 2018). A surface flux scheme based on the Monin‐Obukhov similarity for finite volume models. _Journal of Advances in Modeling Earth Systems_, 10, 3159– 3175. [https://doi.org/10.1029/2018MS001534](https://doi.org/10.1029/2018MS001534)
> <em>Internally abbreviated as NK</em>	

> Gryanik, V., Lupkes, C., Grachev, A. & Sidorenko, D. (2020). New modified and extended stability functions for the stable boundary layer based on SHEBA and parametrizations of bulk transfer coefficients for climate models. _Journal of the Atmospheric Sciences_, 10.1175/JAS-D-19-0255.1. [https://doi.org/10.1175/JAS-D-19-0255.1](https://doi.org/10.1175/JAS-D-19-0255.1). (Early Online Release) 
> _Internally abbreviated as GLGS_

>Hill, R.J., 1989. Implications of Monin–Obukhov Similarity Theory for Scalar Quantities. _J. Atmos. Sci.,_  **46**, 2236–2244, [https://doi.org/10.1175/1520-0469(1989)046<2236:IOMSTF>2.0.CO;2](https://doi.org/10.1175/1520-0469(1989)046%3C2236:IOMSTF%3E2.0.CO;2) 

### Parameters
* $L_{MO}$ = Monin-Obukhov length
* $X__{\star}$ = Scale parameter for variable $X$
* $X__{sfc}$ = Surface value of variable $X$ 
* $\kappa$ = von-Karman coefficient (typically $0.4$)
* $\mathrm{D_T} = \nu/Pr_{T}$ is the diffusivity
*  $Pr__{T}$ the neutral limit turbulent Prandtl number (typically $\frac{1}{3}$)
* $\nu$ the kinematic viscosity
* $w$, the wall normal velocity component
* $\Delta z$ = Model reference altitude
* $\delta__{\chi_{i}}$ a measure of mixing efficiency for $i$th tracer 
* $\zeta = z/L_{MO}$ similarity variable
* $\nu/u__{\star}$ Viscous lengthscale

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

`FixedExchangeCoefficients`
`DynamicExchangeCoefficients`
`InteractiveSurfaceFluxes`

In each of these cases $\hat{n}$ corresponds to the upward pointing unit-normal vector. 

## Equations

We start by defining the universal functions for the velocity, enthalpy and moisture terms. _NK_ express the Monin-Obukhov length in terms of the potential temperature 

```math
L_{MO} = -\frac{u_{\star}^3 \overline{\theta}}{\kappa g \overline{w^{\prime}\theta^{\prime}}|_{sfc}} 
```

From the above definitions for the fluxes in the moist, compressible flow equations, we substitute the potential temperature flux $\overline{w^{\prime}\theta^{\prime}} = J/(\rho c_{p})$, $\overline{w^{\prime}u^{\prime}} = \tau/\rho)$ to recover eq (1) in _GLGS_. 

```math
L_{MO} = -\frac{(\tau/\rho)^{(3/2)} \overline{\theta}}{\kappa g \frac{J}{(\rho c_{p})}|_{sfc}} 
```


Given the formulation of the compressible flow equations in `ClimateMachine`  we arrive at the following formulation for the Monin-Obukhov similarity length

```math
L_{MO} = - \frac{\overline{\theta_v} u_{\star}^3}{\kappa g [\overline{\theta_v^{\prime}{w^{\prime}}}]}
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
\theta_{v\star} = \frac{\frac{\kappa}{Pr_{T}}(\langle \theta_{v} \rangle- \theta_{v,sfc})  }{\log(\frac{z}{z_{0h}})-\Psi_{h}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0h}}{\Delta z})\Psi_{h}(\frac{\Delta z_{0h}}{L_{MO}}) + (1 − \frac{z_{0h}}{Δz})\{ \Psi_{h} (\frac{z_{0h}}{L_{MO}}) - 1\}} 
```

Alternatively, use a new thermal scale parameter (according to the prognostic equations)
```math
h_{\star} = \frac{\frac{\kappa}{Pr_{T}}(\langle h_{tot} \rangle- h_{sfc})  }{\log(\frac{z}{z_{0h}})-\Psi_{h}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0h}}{\Delta z})\Psi_{h}(\frac{\Delta z_{0h}}{L_{MO}}) + (1 − \frac{z_{0h}}{Δz})\{ \Psi_{h} (\frac{z_{0h}}{L_{MO}}) - 1\}} 
```
```math
q_{\star} = \frac{\frac{\kappa}{Pr_{T}} (\langle q \rangle- q_{sfc})  }{\log(\frac{z}{z_{0q}})-\Psi_{q}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0q}}{\Delta z})\Psi_{q}(\frac{\Delta z_{0q}}{L_{MO}}) +  (1 − \frac{z_{0q}}{Δz})\{ \Psi_{q} (\frac{z_{0q}}{L_{MO}}) - 1\}    } 
```
Here we follow Hill's (1989) argument for the form of the universal functions for tracers which are assumed to follow the Monin-Obukhov similarity theory. 

The equations above combined with the M-O similarity length provide 4 equations for 4 unknowns. Once the Monin-Obukhov length is computed, we can then obtain the tracer surface values. 

```math
\chi_{i\star} = \frac{\frac{\kappa  \delta \chi_{i}}{Pr_{T}} (\langle \chi_{i} \rangle- \chi_{i,sfc})  }{\log(\frac{z}{z_{0h}})-\Psi_{\chi_{i}}(\frac{\Delta z}{L_{MO}}) + (\frac{z_{0\chi_{i}}}{\Delta z})\Psi_{\chi_{i}}(\frac{\Delta z_{0\chi_{i}}}{L_{MO}}) +  (1 − \frac{z_{0h}}{Δz})\{ \Psi_{\chi_{i}} (\frac{z_{0h}}{L_{MO}}) - 1\} } 
```

## References

- Businger, Joost A., et al. "Flux-profile relationships in the atmospheric surface
  layer." Journal of the atmospheric Sciences 28.2 (1971): 181-189.

- Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the Monin-Obukhov
  Similarity for Finite Volume Models." Journal of Advances in Modeling Earth Systems
  10.12 (2018): 3159-3175.

- Byun, Daewon W. "On the analytical solutions of flux-profile relationships for the
  atmospheric surface layer." Journal of Applied Meteorology 29.7 (1990): 652-657.

- Wyngaard, John C. "Modeling the planetary boundary layer-Extension to the stable case."
  Boundary-Layer Meteorology 9.4 (1975): 441-460.
