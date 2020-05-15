```@meta
EditURL = "<unknown>/src/Atmos/Model/turbulence.jl"
```

## [Turbulence Closures](@id Turbulence-Closures-docs)
In `turbulence.jl` we specify turbulence closures. Currently,
pointwise models of the eddy viscosity/eddy diffusivity type are
supported for turbulent shear and tracer diffusivity. Methods currently supported
are:\
[`ConstantViscosityWithDivergence`](@ref constant-viscosity)\
[`SmagorinskyLilly`](@ref smagorinsky-lilly)\
[`Vreman`](@ref vreman)\
[`AnisoMinDiss`](@ref aniso-min-diss)\

!!! note
    Usage: This is a quick-ref guide to using turbulence models as a subcomponent
    of `AtmosModel` \
    $\nu$ is the kinematic viscosity, $C_smag$ is the Smagorinsky Model coefficient,
    - `turbulence=ConstantViscosityWithDivergence(ν)`\
    - `turbulence=SmagorinskyLilly(C_smag)`\
    - `turbulence=Vreman(C_smag)`\
    - `turbulence=AnisoMinDiss(C_poincare)`

```julia
using DocStringExtensions
using CLIMAParameters.Atmos.SubgridScale: inv_Pr_turb
export ConstantViscosityWithDivergence, SmagorinskyLilly, Vreman, AnisoMinDiss
export turbulence_tensors
```

## Abstract Type
We define a `TurbulenceClosure` abstract type and
default functions for the generic turbulence closure
which will be overloaded with model specific functions. Minimally, overloaded functions for the
following stubs must be defined for a turbulence model.

```julia
abstract type TurbulenceClosure end


vars_state_gradient((::TurbulenceClosure, FT) = @vars()
vars_state_gradient_flux(::TurbulenceClosure, FT) = @vars()
vars_state_auxiliary(::TurbulenceClosure, FT) = @vars()

function atmos_init_aux!(
    ::TurbulenceClosure,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) end
function compute_gradient_argument!(
    ::TurbulenceClosure,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_flux!(
    ::TurbulenceClosure,
    ::Orientation,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
) end
```

The following may need to be addressed if turbulence models require
additional state variables or auxiliary variable updates (e.g. TKE
based models)

```julia
vars_state_conservative(::TurbulenceClosure, FT) = @vars()
function atmos_nodal_update_auxiliary_state!(
    ::TurbulenceClosure,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
```

## Eddy-viscosity Models
The following function provides an example of a stub for an eddy-viscosity model.
Currently, scalar and diagonal tensor viscosities and diffusivities are supported.

```@docs
ClimateMachine.Atmos.turbulence_tensors
```

Generic math functions for use within the turbulence closures such as the [principal tensor invariants](@ref tensor-invariants),
[symmetric tensors](@ref symmetric-tensors) and [tensor norms](@ref tensor-norms) have been included.

### [Pricipal Invariants](@id tensor-invariants)
```math
\textit{I}_{1} = \mathrm{tr(X)} \\
\textit{I}_{2} = (\mathrm{tr(X)}^2 - \mathrm{tr(X)^2}) / 2 \\
\textit{I}_{3} = \mathrm{det(X)} \\
```

```@docs
ClimateMachine.Atmos.principal_invariants
```

### [Symmetrize](@id symmetric-tensors)
```math
\frac{\mathrm{X} + \mathrm{X}^{T}}{2} \\
```
```@docs
ClimateMachine.Atmos.symmetrize
```

### [2-Norm](@id tensor-norms)
Given a tensor X, return the tensor dot product
```math
\sum_{i,j} S_{ij}^2
```
```@docs
ClimateMachine.Atmos.norm2
```

### [Strain-rate Magnitude](@id strain-rate-magnitude)
By definition, the strain-rate magnitude, as defined in
standard turbulence modelling is computed such that
```math
|\mathrm{S}| = \sqrt{2 \sum_{i,j} \mathrm{S}_{ij}^2}
```
where
```math
\vec{S}(\vec{u}) = \frac{1}{2}  \left(\nabla\vec{u} +  \left( \nabla\vec{u} \right)^T \right)
```
\mathrm{S} is the rate-of-strain tensor. (Symmetric component of the velocity gradient). Note that the
skew symmetric component (rate-of-rotation) is not currently computed.

```@docs
ClimateMachine.Atmos.strain_rate_magnitude
```

```julia
"""
    strain_rate_magnitude(S)
Given the rate-of-strain tensor `S`, computes its magnitude.
"""
function strain_rate_magnitude(S::SHermitianCompact{3, FT, 6}) where {FT}
    return sqrt(2 * norm2(S))
end
```

### [Constant Viscosity Model](@id constant-viscosity)
`ConstantViscosityWithDivergence` requires a user to specify the constant viscosity (kinematic)
and appropriately computes the turbulent stress tensor based on this term. Diffusivity can be
computed using the turbulent Prandtl number for the appropriate problem regime.
```math
\tau = - 2 \nu \mathrm{S}
```

```@docs
ClimateMachine.Atmos.ConstantViscosityWithDivergence
```

## [Smagorinsky-Lilly](@id smagorinsky-lilly)
The Smagorinsky turbulence model, with Lilly's correction to
stratified atmospheric flows, is included in ClimateMachine.
The input parameter to this model is the Smagorinsky coefficient.
For atmospheric flows, the coefficient `C_smag` typically takes values between
0.15 and 0.23. Flow dependent `C_smag` are currently not supported (e.g. Germano's
extension). The Smagorinsky-Lilly model does not contain explicit filtered terms.

#### Equations

```math
\nu = (C_{s} \mathrm{f}_{b} \Delta)^2 \sqrt{|\mathrm{S}|}
```
with the stratification correction term
```math
f_{b} =
   \begin{cases}
   1 & \mathrm{Ri} \leq 0 ,\\
   \max(0, 1 - \mathrm{Ri} / \mathrm{Pr}_{t})^{1/4} & \mathrm{Ri} > 0 .
   \end{cases}
```
```math
\mathrm{Ri} =  \frac{N^2}{{|S|}^2}
```
```math
N = \left( \frac{g}{\theta_v} \frac{\partial \theta_v}{\partial z}\right)^{1/2}
```
Here, $\mathrm{Ri}$ and $\mathrm{Pr}_{t}$ are the Richardson and
turbulent Prandtl numbers respectively.  $\Delta$ is the mixing length in the
relevant coordinate direction. We use the DG metric terms to determine the
local effective resolution (see `src/Mesh/Geometry.jl`), and modify the vertical lengthscale by the
stratification correction factor $\mathrm{f}_{b}$ so that $\Delta_{vert} = \Delta z f_b$.

```@docs
ClimateMachine.Atmos.SmagorinskyLilly
```

## [Vreman Model](@id vreman)
Vreman's turbulence model for anisotropic flows, which provides a
less dissipative solution (specifically in the near-wall and transitional regions)
than the Smagorinsky-Lilly method. This model
relies of first derivatives of the velocity vector (i.e., the gradient tensor).
By design, the Vreman model handles transitional as well as fully turbulent flows adequately.
The input parameter to this model is the Smagorinsky coefficient - the coefficient is modified
within the model functions to account for differences in model construction.
#### Equations
```math
\nu_{t} = 2.5 C_{s}^2 \sqrt{\frac{B_{\beta}}{u_{i,j}u_{i,j}}},
```
where ($i,j, m = (1,2,3)$)
```math
\begin{align}
B_{\beta} &= \beta_{11}\beta_{22} + \beta_{11}\beta_{33} + \beta_{22}\beta_{33} - (\beta_{13}^2 + \beta_{12}^2 + \beta_{23}^2) \\
\beta_{ij} &= \Delta_{m}^2 u_{i, m} u_{j, m} \\
u_{i,j} &= \frac{\partial u_{i}}{\partial x_{j}}.
\end{align}
```

```@docs
ClimateMachine.Atmos.Vreman
```

## [Anisotropic Minimum Dissipation](@id aniso-min-diss)
This method is based Vreugdenhil and Taylor's minimum-dissipation eddy-viscosity model.
The principles of the Rayleigh quotient minimizer are applied to the energy dissipation terms in the
conservation equations, resulting in a maximum dissipation bound, and a model for
eddy viscosity and eddy diffusivity.
```math
\nu_e = (\mathrm{C}\delta)^2  \mathrm{max}\left[0, - \frac{\hat{\partial}_k \hat{u}_{i} \hat{\partial}_k \hat{u}_{j} \mathrm{\hat{S}}_{ij}}{\hat{\partial}_p \hat{u}_{q} \hat{\partial}_p \hat{u}_{q}} \right]
```
```@docs
ClimateMachine.Atmos.AnisoMinDiss
```
