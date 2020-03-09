## Subgrid Scale Turbulence (SGS-Turb)

  

### Constant-Coefficient `SmagorinskyLilly` Model (CSMAG)

The constant coefficient [Smagorinsky-Lilly model](https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2) is an eddy-viscosity model for the representation of subgrid-scale turbulence. Eddy-viscosity in the Smagorinsky-Lilly model is given by

$$\mu_{i} = \rho\nu_{i} = 2\rho  (C_s\Delta_{i})^2\sqrt{S_{ij}S_{ij}}$$

where $\rho$ is the fluid density, $\Delta$ is the mixing-length, $\mu$ the dynamic viscosity, $\nu$ the kinematic viscosity, $C_s = 0.15-0.23$ the prescribed Smagorinsky coefficient, $\mathbf{S} = S_{ij} = \frac{1}{2}(\frac{\partial u_{i}}{\partial u_{j}} + \frac{\partial u_{j}}{\partial u_{i}})$is the rate-of-strain tensor ($u_{i}$ are velocity components) with subscripts $i=(1,2,3)$ and $j=(1,2,3)$ identifying tensor components in Cartesian indices following the Einstein summation convention.

A simple extension to the basic `SmagorinskyLilly` model includes a stratification adjustment to the mixing length ([proposed by Lilly](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.2153-3490.1962.tb00128.x)). Here, we apply the correction

$$f_{b} = \frac{g}{\theta_z} \frac{\partial \theta_v}{\partial z} \frac{1}{|\bf{S}|^2}$$


We extract the $\it{vertically}$ aligned unit vector $\hat{k}$ using the `aux`iliary variables (since CliMA solves equations on a Cartesian grid on both `FlatOrientation` (LES) and `SphericalOrientation` (GCM).

$$\hat{k} = \frac{\nabla \Phi}{|\nabla \Phi|}$$

where $\Phi$ is the geopotential. The function `vertical_unit_vector` in `orientation.jl` provides this functionality. 

$$\nu_{vert} = (\hat{k}\cdot\nu) \hat{k}$$
and 
$$\nu_{horz} = (\nu - \nu_{vert})$$ give the split vertical, tangential (horizontal) components. 

While the general construction of the turbulent model (in `turbulence.jl`) is agnostic of the problem configuration (i.e. GCM or LES), for GCM configurations, a keyword argument `diffusion_direction` is included in the `DGmodel` construction to restrict diffusion to the horizontal direction. Currently the stratification-correction $f_{b}$ is applied only to the vertical component of viscosity. CSMAG is suggested as the default SGS model in the absence of problem specific alternatives.

  
### Constant-Coefficient `Vreman` Model (CVREM)

CVREM is an eddy-viscosity model based on velocity-gradients in the flow. The approach is very similar to the CSMAG, but is less dissipative than CSMAG. We consider the same length-scale definition as in the CSMAG, while computing the eddy viscosity based on [Vreman(2004)]([[https://aip.scitation.org/doi/10.1063/1.1785131](https://aip.scitation.org/doi/10.1063/1.1785131)]). 
$$
\nu_{e} = \sqrt{\frac{B_{\beta}}{\alpha_{ij}\alpha_{ij}}}\\
\\
\alpha_{ij} = \frac{\partial u_{j}}{\partial x_{i}}\\
\\
B_{\beta} = \beta_{11}\beta_{22} + \beta_{11}\beta_{33} +\beta_{22}\beta_{33} - (\beta_{12}^2 + \beta_{23}^2 + \beta_{13}^2) \\
\\
\beta_{ij} = \Delta_m^2 \alpha_{mi} \alpha_{mj}
$$
with $\Delta_{m}$ the lengthscale along Cartesian direction $m$. 

The stratification limiter, similar to its application in the `SmagorinskyLilly` model is applied 

### Anisotropic-Minimum-Dissipation Model (AMD)

[AMD]([https://doi.org/10.1063/1.5037039](https://doi.org/10.1063/1.5037039)) is an eddy-viscosity model based on the idea of a Rayleigh-quotient minimiser (e.g. QR model). The eddy-viscosity is dependent on invariants of the rate-of-strain tensor in the case of isotropic grids (otherwise it depends on components of the rate-of-strain tensor). The viscosity consistently vanishes in Laminar flow when the subgrid energy flux is identically zero.

- [ ] Complete documentation 
 - [x] Eddy viscosity calculations.
- [ ] Coefficient computation (parameter determination for $C_{poincare}$) 

- [ ] Tracer diffusivity calculations.
