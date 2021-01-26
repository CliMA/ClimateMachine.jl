"""
    TurbulenceClosures

Functions for turbulence, sub-grid scale modeling. These include
viscosity terms, diffusivity and stress tensors.

- [`ConstantViscosity`](@ref)
- [`ViscousSponge`](@ref)
- [`SmagorinskyLilly`](@ref)
- [`Vreman`](@ref)
- [`AnisoMinDiss`](@ref)

"""
module TurbulenceClosures

# ## Turbulence Closures
# In `turbulence.jl` we specify turbulence closures. Currently,
# pointwise models of the eddy viscosity/eddy diffusivity type are
# supported for turbulent shear and tracer diffusivity. Methods currently supported
# are:\
# [`ConstantViscosity`](@ref constant-viscosity)\
# [`ViscousSponge`](@ref viscous-sponge)\
# [`SmagorinskyLilly`](@ref smagorinsky-lilly)\
# [`Vreman`](@ref vreman)\
# [`AnisoMinDiss`](@ref aniso-min-diss)\

#md # !!! note
#md #     Usage: This is a quick-ref guide to using turbulence models as a subcomponent
#md #     of `BalanceLaw` \
#md #     $\nu$ is the kinematic viscosity, $C_smag$ is the Smagorinsky Model coefficient,
#md #     `turbulence=ConstantDynamicViscosity(ρν)`\
#md #     `turbulence=ConstantKinematicViscosity(ν)`\
#md #     `turbulence=ViscousSponge(ν, z_max, z_sponge, α, γ)`\
#md #     `turbulence=SmagorinskyLilly(C_smag)`\
#md #     `turbulence=Vreman(C_smag)`\
#md #     `turbulence=AnisoMinDiss(C_poincare)`

using DocStringExtensions
using LinearAlgebra
using StaticArrays
using UnPack
import CLIMAParameters: AbstractParameterSet
using CLIMAParameters.Atmos.SubgridScale: inv_Pr_turb

using ClimateMachine

import ..Mesh.Geometry: LocalGeometry, resolutionmetric, lengthscale

using ..Orientations
using ..VariableTemplates
using ..BalanceLaws

import ..BalanceLaws:
    vars_state,
    eq_tends,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!

export TurbulenceClosureModel,
    ConstantViscosity,
    ConstantDynamicViscosity,
    ConstantKinematicViscosity,
    SmagorinskyLilly,
    Vreman,
    AnisoMinDiss,
    HyperDiffusion,
    NoHyperDiffusion,
    Biharmonic,
    DryBiharmonic,
    EquilMoistBiharmonic,
    NoViscousSponge,
    UpperAtmosSponge,
    turbulence_tensors,
    init_aux_turbulence!,
    init_aux_hyperdiffusion!,
    sponge_viscosity_modifier

# ### Abstract Type
# We define a `TurbulenceClosureModel` abstract type and
# default functions for the generic turbulence closure
# which will be overloaded with model specific functions.


"""
    Abstract type with default do-nothing behaviour for
arbitrary turbulence closure models.
"""
abstract type TurbulenceClosureModel end

vars_state(::TurbulenceClosureModel, ::AbstractStateType, FT) = @vars()

"""
    ConstantViscosity <: TurbulenceClosureModel
Abstract type for constant viscosity models
"""
abstract type ConstantViscosity <: TurbulenceClosureModel end

"""
    Abstract type for HyperDiffusion models
"""
abstract type HyperDiffusion end

"""
    Abstract type for viscous sponge layers.
Modifier for viscosity computed from existing turbulence closures.
"""
abstract type ViscousSponge end


"""
    init_aux_turbulence!
Initialise auxiliary variables for turbulence models.
Overload for specific turbulence closure type.
"""
function init_aux_turbulence!(
    ::TurbulenceClosureModel,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
) end

"""
    compute_gradient_argument!
Assign pre-gradient-transform variables specific to turbulence models.
"""
function compute_gradient_argument!(
    ::TurbulenceClosureModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
"""
    compute_gradient_flux!(::TurbulenceClosureModel, _...)
Post-gradient-transformed variables specific to turbulence models.
"""
function compute_gradient_flux!(
    ::TurbulenceClosureModel,
    ::Orientation,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
) end

# Fallback functions for hyperdiffusion model
vars_state(::HyperDiffusion, ::AbstractStateType, FT) = @vars()

function init_aux_hyperdiffusion!(
    ::HyperDiffusion,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
) end
function compute_gradient_argument!(
    ::HyperDiffusion,
    ::BalanceLaw,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function transform_post_gradient_laplacian!(
    h::HyperDiffusion,
    bl::BalanceLaw,
    hyperdiffusive::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function compute_gradient_flux!(
    h::HyperDiffusion,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

function turbulence_tensors end
function hyperviscosity_tensors end

"""
    ν, D_t, τ = turbulence_tensors(
                    ::TurbulenceClosureModel,
                    orientation::Orientation,
                    param_set::AbstractParameterSet,
                    state::Vars,
                    diffusive::Vars,
                    aux::Vars,
                    t::Real
                )
Compute the kinematic viscosity (`ν`), the diffusivity (`D_t`) and SGS momentum flux tensor (`τ`)
for a given turbulence closure. Each closure overloads this method with the appropriate calculations
for the returned quantities.

# Arguments

- `::TurbulenceClosureModel` = Struct identifier for turbulence closure model
- `orientation` = `BalanceLaw.orientation`
- `param_set` parameter set
- `state` = Array of prognostic (state) variables. See `vars_state` in `BalanceLaw`
- `diffusive` = Array of diffusive variables
- `aux` = Array of auxiliary variables
- `t` = time
"""
function turbulence_tensors(
    m::TurbulenceClosureModel,
    bl::BalanceLaw,
    state,
    diffusive,
    aux,
    t,
)
    ν, D_t, τ = turbulence_tensors(
        m,
        bl.orientation,
        bl.param_set,
        state,
        diffusive,
        aux,
        t,
    )
    ν, D_t, τ = sponge_viscosity_modifier(bl, bl.viscoussponge, ν, D_t, τ, aux)
    return (ν, D_t, τ)
end

# We also provide generic math functions for use within the turbulence closures,
# commonly used quantities such as the [principal tensor invariants](@ref tensor-invariants), handling of
# [symmetric tensors](@ref symmetric-tensors) and [tensor norms](@ref tensor-norms)are addressed.

# ### [Pricipal Invariants](@id tensor-invariants)
# ```math
# \textit{I}_{1} = \mathrm{tr(X)} \\
# \textit{I}_{2} = (\mathrm{tr(X)}^2 - \mathrm{tr(X^2)}) / 2 \\
# \textit{I}_{3} = \mathrm{det(X)} \\
# ```
"""
    principal_invariants(X)

Calculates principal invariants of a tensor `X`. Returns 3 element tuple containing the invariants.
"""
function principal_invariants(X)
    first = tr(X)
    second = (first^2 - tr(X .^ 2)) / 2
    third = det(X)
    return (first, second, third)
end

# ### [Symmetrize](@id symmetric-tensors)
# ```math
# \frac{\mathrm{X} + \mathrm{X}^{T}}{2} \\
# ```
"""
    symmetrize(X)

Given a (3,3) second rank tensor X, compute `(X + X')/2`, returning a
symmetric `SHermitianCompact` object.
"""
function symmetrize(X::StaticArray{Tuple{3, 3}})
    SHermitianCompact(SVector(
        X[1, 1],
        (X[2, 1] + X[1, 2]) / 2,
        (X[3, 1] + X[1, 3]) / 2,
        X[2, 2],
        (X[3, 2] + X[2, 3]) / 2,
        X[3, 3],
    ))
end

# ### [2-Norm](@id tensor-norms)
# Given a tensor X, return the tensor dot product
# ```math
# \sum_{i,j} S_{ij}^2
# ```
"""
    norm2(X)
Given a tensor `X`, computes X:X.
"""
function norm2(X::SMatrix{3, 3, FT}) where {FT}
    abs2(X[1, 1]) +
    abs2(X[2, 1]) +
    abs2(X[3, 1]) +
    abs2(X[1, 2]) +
    abs2(X[2, 2]) +
    abs2(X[3, 2]) +
    abs2(X[1, 3]) +
    abs2(X[2, 3]) +
    abs2(X[3, 3])
end
function norm2(X::SHermitianCompact{3, FT, 6}) where {FT}
    abs2(X[1, 1]) +
    2 * abs2(X[2, 1]) +
    2 * abs2(X[3, 1]) +
    abs2(X[2, 2]) +
    2 * abs2(X[3, 2]) +
    abs2(X[3, 3])
end

# ### [Strain-rate Magnitude](@id strain-rate-magnitude)
# By definition, the strain-rate magnitude, as defined in
# standard turbulence modelling is computed such that
# ```math
# |\mathrm{S}| = \sqrt{2 \sum_{i,j} \mathrm{S}_{ij}^2}
# ```
# where
# ```math
# \vec{S}(\vec{u}) = \frac{1}{2}  \left(\nabla\vec{u} +  \left( \nabla\vec{u} \right)^T \right)
# ```
# \mathrm{S} is the rate-of-strain tensor. (Symmetric component of the velocity gradient). Note that the
# skew symmetric component (rate-of-rotation) is not currently computed.
"""
    strain_rate_magnitude(S)
Given the rate-of-strain tensor `S`, computes its magnitude.
"""
function strain_rate_magnitude(S::SHermitianCompact{3, FT, 6}) where {FT}
    return sqrt(2 * norm2(S))
end

"""
    WithDivergence
A divergence type which includes the divergence term in the momentum flux tensor
"""
struct WithDivergence end
export WithDivergence
"""
    WithoutDivergence
A divergence type which does not include the divergence term in the momentum flux tensor
"""
struct WithoutDivergence end
export WithoutDivergence

# ### [Constant Viscosity Model](@id constant-viscosity)
# `ConstantViscosity` requires a user to specify the constant viscosity (dynamic or kinematic)
# and appropriately computes the turbulent stress tensor based on this term. Diffusivity can be
# computed using the turbulent Prandtl number for the appropriate problem regime.
# ```math
# \tau =
#     \begin{cases}
#     - 2 \nu \mathrm{S} & \mathrm{WithoutDivergence},\\
#     - 2 \nu \mathrm{S} + \frac{2}{3} \nu \mathrm{tr(S)} I_3 & \mathrm{WithDivergence}.
#     \end{cases}
# ```


"""
    ConstantDynamicViscosity <: ConstantViscosity

Turbulence with constant dynamic viscosity (`ρν`).
Divergence terms are included in the momentum flux tensor if divergence_type is WithDivergence.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct ConstantDynamicViscosity{FT, DT} <: ConstantViscosity
    "Dynamic Viscosity [kg/m/s]"
    ρν::FT
    divergence_type::DT
    function ConstantDynamicViscosity(
        ρν::FT,
        divergence_type::Union{WithDivergence, WithoutDivergence} = WithoutDivergence(),
    ) where {FT}
        return new{FT, typeof(divergence_type)}(ρν, divergence_type)
    end
end

"""
    ConstantKinematicViscosity <: ConstantViscosity

Turbulence with constant kinematic viscosity (`ν`).
Divergence terms are included in the momentum flux tensor if divergence_type is WithDivergence.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct ConstantKinematicViscosity{FT, DT} <: ConstantViscosity
    "Kinematic Viscosity [m2/s]"
    ν::FT
    divergence_type::DT
    function ConstantKinematicViscosity(
        ν::FT,
        divergence_type::Union{WithDivergence, WithoutDivergence} = WithoutDivergence(),
    ) where {FT}
        return new{FT, typeof(divergence_type)}(ν, divergence_type)
    end
end

vars_state(::ConstantViscosity, ::Gradient, FT) = @vars()
vars_state(::ConstantViscosity, ::GradientFlux, FT) =
    @vars(S::SHermitianCompact{3, FT, 6})

function compute_gradient_flux!(
    ::ConstantViscosity,
    ::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    diffusive.turbulence.S = symmetrize(∇transform.u)
end

compute_stress(div_type::WithoutDivergence, ν, S) = (-2 * ν) * S
compute_stress(div_type::WithDivergence, ν, S) =
    (-2 * ν) * S + (2 * ν / 3) * tr(S) * I

function turbulence_tensors(
    m::ConstantDynamicViscosity,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)

    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    S = diffusive.turbulence.S
    ν = m.ρν / state.ρ
    D_t = ν * _inv_Pr_turb
    τ = compute_stress(m.divergence_type, ν, S)
    return ν, D_t, τ
end

function turbulence_tensors(
    m::ConstantKinematicViscosity,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)

    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    S = diffusive.turbulence.S
    ν = m.ν
    D_t = ν * _inv_Pr_turb
    τ = compute_stress(m.divergence_type, ν, S)
    return ν, D_t, τ
end

# ### [Smagorinsky-Lilly](@id smagorinsky-lilly)
# The Smagorinsky turbulence model, with Lilly's correction to
# stratified atmospheric flows, is included in ClimateMachine.
# The input parameter to this model is the Smagorinsky coefficient.
# For atmospheric flows, the coefficient `C_smag` typically takes values between
# 0.15 and 0.23. Flow dependent `C_smag` are currently not supported (e.g. Germano's
# extension). The Smagorinsky-Lilly model does not contain explicit filtered terms.
# #### Equations
# ```math
# \nu = (C_{s} \mathrm{f}_{b} \Delta)^2 \sqrt{|\mathrm{S}|}
# ```
# with the stratification correction term
# ```math
# f_{b} =
#    \begin{cases}
#    1 & \mathrm{Ri} \leq 0 ,\\
#    \max(0, 1 - \mathrm{Ri} / \mathrm{Pr}_{t})^{1/4} & \mathrm{Ri} > 0 .
#    \end{cases}
# ```
# ```math
# \mathrm{Ri} =  \frac{N^2}{{|S|}^2}
# ```
# ```math
# N = \left( \frac{g}{\theta_v} \frac{\partial \theta_v}{\partial z}\right)^{1/2}
# ```
# Here, $\mathrm{Ri}$ and $\mathrm{Pr}_{t}$ are the Richardson and
# turbulent Prandtl numbers respectively.  $\Delta$ is the mixing length in the
# relevant coordinate direction. We use the DG metric terms to determine the
# local effective resolution (see `src/Mesh/Geometry.jl`), and modify the vertical lengthscale by the
# stratification correction factor $\mathrm{f}_{b}$ so that $\Delta_{vert} = \Delta z f_b$.

"""
    SmagorinskyLilly <: TurbulenceClosureModel

# Fields

$(DocStringExtensions.FIELDS)

# Smagorinsky Model Reference

See [Smagorinsky1963](@cite)

# Lilly Model Reference

See [Lilly1962](@cite)

# Brunt-Väisälä Frequency Reference

See [Durran1982](@cite)

"""
struct SmagorinskyLilly{FT} <: TurbulenceClosureModel
    "Smagorinsky Coefficient [dimensionless]"
    C_smag::FT
end

vars_state(::SmagorinskyLilly, ::Auxiliary, FT) = @vars(Δ::FT)
vars_state(::SmagorinskyLilly, ::Gradient, FT) = @vars(θ_v::FT)
vars_state(::SmagorinskyLilly, ::GradientFlux, FT) =
    @vars(S::SHermitianCompact{3, FT, 6}, N²::FT)


function init_aux_turbulence!(
    ::SmagorinskyLilly,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.turbulence.Δ = lengthscale(geom)
end

function compute_gradient_argument!(
    m::SmagorinskyLilly,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.θ_v = aux.moisture.θ_v
end

function compute_gradient_flux!(
    ::SmagorinskyLilly,
    orientation::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    diffusive.turbulence.S = symmetrize(∇transform.u)
    ∇Φ = ∇gravitational_potential(orientation, aux)
    diffusive.turbulence.N² =
        dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end

function turbulence_tensors(
    m::SmagorinskyLilly,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)

    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    S = diffusive.turbulence.S
    normS = strain_rate_magnitude(S)
    k̂ = vertical_unit_vector(orientation, param_set, aux)

    # squared buoyancy correction
    Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
    f_b² = sqrt(clamp(FT(1) - Richardson * _inv_Pr_turb, FT(0), FT(1)))
    ν₀ = normS * (m.C_smag * aux.turbulence.Δ)^2 + FT(1e-5)
    ν = SVector{3, FT}(ν₀, ν₀, ν₀)
    ν_v = k̂ .* dot(ν, k̂)
    ν_h = ν₀ .- ν_v
    ν = SDiagonal(ν_h + ν_v .* f_b²)
    D_t = diag(ν) * _inv_Pr_turb
    τ = -2 * ν * S
    return ν, D_t, τ
end

# ### [Vreman Model](@id vreman)
# Vreman's turbulence model for anisotropic flows, which provides a
# less dissipative solution (specifically in the near-wall and transitional regions)
# than the Smagorinsky-Lilly method. This model
# relies of first derivatives of the velocity vector (i.e., the gradient tensor).
# By design, the Vreman model handles transitional as well as fully turbulent flows adequately.
# The input parameter to this model is the Smagorinsky coefficient - the coefficient is modified
# within the model functions to account for differences in model construction.
# #### Equations
# ```math
# \nu_{t} = 2.5 C_{s}^2 \sqrt{\frac{B_{\beta}}{u_{i,j}u_{i,j}}},
# ```
# where ($i,j, m = (1,2,3)$)
# ```math
# \begin{align}
# B_{\beta} &= \beta_{11}\beta_{22} + \beta_{11}\beta_{33} + \beta_{22}\beta_{33} - (\beta_{13}^2 + \beta_{12}^2 + \beta_{23}^2) \\
# \beta_{ij} &= \Delta_{m}^2 u_{i, m} u_{j, m} \\
# u_{i,j} &= \frac{\partial u_{i}}{\partial x_{j}}.
# \end{align}
# ```

"""
    Vreman{FT} <: TurbulenceClosureModel

Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Smagorinsky coefficient
is specified and used to compute the equivalent Vreman coefficient.

1) ν_e = √(Bᵦ/(αᵢⱼαᵢⱼ)) where αᵢⱼ = ∂uⱼ∂uᵢ with uᵢ the resolved scale velocity component.
2) βij = Δ²αₘᵢαₘⱼ
3) Bᵦ = β₁₁β₂₂ + β₂₂β₃₃ + β₁₁β₃₃ - β₁₂² - β₁₃² - β₂₃²
βᵢⱼ is symmetric, positive-definite.
If Δᵢ = Δ, then β = Δ²αᵀα


# Fields

$(DocStringExtensions.FIELDS)

# Reference

 - [Vreman2004](@cite)
"""
struct Vreman{FT} <: TurbulenceClosureModel
    "Smagorinsky Coefficient [dimensionless]"
    C_smag::FT
end
vars_state(::Vreman, ::Auxiliary, FT) = @vars(Δ::FT)
vars_state(::Vreman, ::Gradient, FT) = @vars(θ_v::FT)
vars_state(::Vreman, ::GradientFlux, FT) =
    @vars(∇u::SMatrix{3, 3, FT, 9}, N²::FT)

function init_aux_turbulence!(
    ::Vreman,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.turbulence.Δ = lengthscale(geom)
end
function compute_gradient_argument!(
    m::Vreman,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.θ_v = aux.moisture.θ_v
end
function compute_gradient_flux!(
    ::Vreman,
    orientation::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.turbulence.∇u = ∇transform.u
    ∇Φ = ∇gravitational_potential(orientation, aux)
    diffusive.turbulence.N² =
        dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end

function turbulence_tensors(
    m::Vreman,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    α = diffusive.turbulence.∇u
    S = symmetrize(α)
    k̂ = vertical_unit_vector(orientation, param_set, aux)

    normS = strain_rate_magnitude(S)
    Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
    f_b² = sqrt(clamp(1 - Richardson * _inv_Pr_turb, 0, 1))

    β = (aux.turbulence.Δ)^2 * (α' * α)
    Bβ = principal_invariants(β)[2]

    ν₀ = m.C_smag^2 * FT(2.5) * sqrt(abs(Bβ / (norm2(α) + eps(FT))))

    ν = SVector{3, FT}(ν₀, ν₀, ν₀)
    ν_v = k̂ .* dot(ν, k̂)
    ν_h = ν₀ .- ν_v
    ν = SDiagonal(ν_h + ν_v .* f_b²)
    D_t = diag(ν) * _inv_Pr_turb
    τ = -2 * ν * S
    return ν, D_t, τ
end

# ### [Anisotropic Minimum Dissipation](@id aniso-min-diss)
# This method is based Vreugdenhil and Taylor's minimum-dissipation eddy-viscosity model.
# The principles of the Rayleigh quotient minimizer are applied to the energy dissipation terms in the
# conservation equations, resulting in a maximum dissipation bound, and a model for
# eddy viscosity and eddy diffusivity.
# ```math
# \nu_e = (\mathrm{C}\delta)^2  \mathrm{max}\left[0, - \frac{\hat{\partial}_k \hat{u}_{i} \hat{\partial}_k \hat{u}_{j} \mathrm{\hat{S}}_{ij}}{\hat{\partial}_p \hat{u}_{q}} \right]
# ```
"""
    AnisoMinDiss{FT} <: TurbulenceClosureModel

Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Poincare coefficient
is specified and used to compute the equivalent AnisoMinDiss coefficient (computed as the solution to the
eigenvalue problem for the Laplacian operator).

# Fields
$(DocStringExtensions.FIELDS)

# Reference

See [Vreugdenhil2018](@cite)

"""
struct AnisoMinDiss{FT} <: TurbulenceClosureModel
    C_poincare::FT
end
vars_state(::AnisoMinDiss, ::Auxiliary, FT) = @vars(Δ::FT)
vars_state(::AnisoMinDiss, ::Gradient, FT) = @vars(θ_v::FT)
vars_state(::AnisoMinDiss, ::GradientFlux, FT) =
    @vars(∇u::SMatrix{3, 3, FT, 9}, N²::FT)
function init_aux_turbulence!(
    ::AnisoMinDiss,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.turbulence.Δ = lengthscale(geom)
end
function compute_gradient_argument!(
    m::AnisoMinDiss,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.θ_v = aux.moisture.θ_v
end
function compute_gradient_flux!(
    ::AnisoMinDiss,
    orientation::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Φ = ∇gravitational_potential(orientation, aux)
    diffusive.turbulence.∇u = ∇transform.u
    diffusive.turbulence.N² =
        dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end
function turbulence_tensors(
    m::AnisoMinDiss,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    k̂ = vertical_unit_vector(orientation, param_set, aux)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)

    ∇u = diffusive.turbulence.∇u
    S = symmetrize(∇u)
    normS = strain_rate_magnitude(S)

    δ = aux.turbulence.Δ
    Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
    f_b² = sqrt(clamp(1 - Richardson * _inv_Pr_turb, 0, 1))

    δ_vec = SVector(δ, δ, δ)
    δ_m = δ_vec ./ transpose(δ_vec)
    ∇û = ∇u .* δ_m
    Ŝ = symmetrize(∇û)
    ν₀ =
        (m.C_poincare .* δ_vec) .^ 2 * max(
            FT(1e-5),
            -dot(transpose(∇û) * (∇û), Ŝ) / (dot(∇û, ∇û) .+ eps(normS)),
        )

    ν_v = k̂ .* dot(ν₀, k̂)
    ν_h = ν₀ .- ν_v
    ν = SDiagonal(ν_h + ν_v .* f_b²)
    D_t = diag(ν) * _inv_Pr_turb
    τ = -2 * ν * S
    return ν, D_t, τ
end

"""
  NoHyperDiffusion <: HyperDiffusion
Defines a default hyperdiffusion model with zero hyperdiffusive fluxes.
"""
struct NoHyperDiffusion <: HyperDiffusion end

hyperviscosity_tensors(m::HyperDiffusion, bl::BalanceLaw, args...) =
    hyperviscosity_tensors(m, bl.orientation, bl.param_set, args...)


abstract type HyperdiffMoistureModel <: BalanceLaw end
struct HyperdiffDryModel <: HyperdiffMoistureModel end
struct HyperdiffEquilMoist{FT} <: HyperdiffMoistureModel
    τ_timescale_q_tot::FT
end
struct HyperdiffNonEquilMoist <: HyperdiffMoistureModel end


vars_state(::HyperdiffEquilMoist, ::Gradient, FT) = @vars(q_tot::FT)
vars_state(::HyperdiffEquilMoist, ::GradientLaplacian, FT) = @vars(q_tot::FT)
vars_state(::HyperdiffEquilMoist, ::Hyperdiffusive, FT) =
    @vars(ν∇³q_tot::SVector{3, FT})

"""
  Biharmonic{FT} <: HyperDiffusion

Assumes compressible flow.
Horizontal hyperdiffusion methods for application in GCM and LES settings
Timescales are prescribed by the user while the diffusion coefficient is
computed as a function of the grid lengthscale.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Biharmonic{FT, M} <: HyperDiffusion
    τ_timescale::FT
    moisture::M
end

DryBiharmonic(τ_timescale::FT) where {FT} =
    Biharmonic(τ_timescale, HyperdiffDryModel())

EquilMoistBiharmonic(τ_timescale::FT) where {FT} =
    Biharmonic(τ_timescale, HyperdiffEquilMoist(τ_timescale))
EquilMoistBiharmonic(τ_timescale::FT, τ_timescale_q_tot::FT) where {FT} =
    Biharmonic(τ_timescale, HyperdiffEquilMoist(τ_timescale_q_tot))

vars_state(m::Biharmonic, st::Auxiliary, FT) = @vars(Δ::FT)
vars_state(m::Biharmonic, st::Gradient, FT) = @vars(
    u_h::SVector{3, FT},
    h_tot::FT,
    moisture::vars_state(m.moisture, st, FT)
)
vars_state(m::Biharmonic, st::GradientLaplacian, FT) = @vars(
    u_h::SVector{3, FT},
    h_tot::FT,
    moisture::vars_state(m.moisture, st, FT)
)
vars_state(m::Biharmonic, st::Hyperdiffusive, FT) = @vars(
    ν∇³u_h::SMatrix{3, 3, FT, 9},
    ν∇³h_tot::SVector{3, FT},
    moisture::vars_state(m.moisture, st, FT)
)

function init_aux_hyperdiffusion!(
    ::Biharmonic,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.hyperdiffusion.Δ = lengthscale(geom)
end

function compute_gradient_argument!(
    h::Biharmonic,
    bl::BalanceLaw,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    u = state.ρu * ρinv
    k̂ = vertical_unit_vector(bl, aux)
    u_h = (SDiagonal(1, 1, 1) - k̂ * k̂') * u
    transform.hyperdiffusion.u_h = u_h
    transform.hyperdiffusion.h_tot = transform.h_tot
    compute_gradient_argument!(h.moisture, transform, state)
end
compute_gradient_argument!(::HyperdiffDryModel, _...) = nothing
function compute_gradient_argument!(m::HyperdiffEquilMoist, transform, state)
    ρinv = 1 / state.ρ
    transform.hyperdiffusion.moisture.q_tot = state.moisture.ρq_tot * ρinv
end

function transform_post_gradient_laplacian!(
    h::Biharmonic,
    bl::BalanceLaw,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    _inv_Pr_turb = eltype(state)(inv_Pr_turb(bl.param_set))
    ∇Δu_h = hypertransform.hyperdiffusion.u_h
    ∇Δh_tot = hypertransform.hyperdiffusion.h_tot
    # Unpack
    τ_timescale = h.τ_timescale
    # Compute hyperviscosity coefficient
    ν₄ = (aux.hyperdiffusion.Δ / 2)^4 / 2 / τ_timescale
    hyperdiffusive.hyperdiffusion.ν∇³u_h = ν₄ * ∇Δu_h
    hyperdiffusive.hyperdiffusion.ν∇³h_tot = ν₄ * ∇Δh_tot
    transform_post_gradient_laplacian!(
        h.moisture,
        h,
        bl,
        hyperdiffusive,
        hypertransform,
        state,
        aux,
        t,
    )
end

transform_post_gradient_laplacian!(::HyperdiffDryModel, _...) = nothing
function transform_post_gradient_laplacian!(
    m::HyperdiffEquilMoist,
    h::Biharmonic,
    bl::BalanceLaw,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Δq_tot = hypertransform.hyperdiffusion.moisture.q_tot
    # Unpack
    τ_timescale_q_tot = m.τ_timescale_q_tot
    # Compute hyperviscosity coefficient
    ν₄_q_tot = (aux.hyperdiffusion.Δ / 2)^4 / 2 / τ_timescale_q_tot
    hyperdiffusive.hyperdiffusion.moisture.ν∇³q_tot = ν₄_q_tot * ∇Δq_tot
end

# ### [Viscous Sponge](@id viscous-sponge)
# `ViscousSponge` requires a user to specify a constant viscosity (kinematic),
# a sponge start height, the domain height, a sponge strength, and a sponge
# exponent.
# Given viscosity, diffusivity and stresses from arbitrary turbulence models,
# the viscous sponge enhances diffusive terms within a user-specified layer,
# typically used at the top of the domain to absorb waves. A smooth onset is
# ensured through a weight function that increases weight height from the sponge
# onset height.
# ```
"""
    NoViscousSponge
No modifiers applied to viscosity/diffusivity in sponge layer
# Fields
#
$(DocStringExtensions.FIELDS)
"""
struct NoViscousSponge <: ViscousSponge end
function sponge_viscosity_modifier(
    bl::BalanceLaw,
    m::NoViscousSponge,
    ν,
    D_t,
    τ,
    aux,
)
    return (ν, D_t, τ)
end

"""
    Upper domain viscous relaxation
Applies modifier to viscosity and diffusivity terms
in a user-specified upper domain sponge region
# Fields
#
$(DocStringExtensions.FIELDS)
"""
struct UpperAtmosSponge{FT} <: ViscousSponge
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Sponge exponent"
    γ::FT
end

function sponge_viscosity_modifier(
    bl::BalanceLaw,
    m::UpperAtmosSponge,
    ν,
    D_t,
    τ,
    aux::Vars,
)
    z = altitude(bl.orientation, bl.param_set, aux)
    if z >= m.sponge
        r = (z - m.z_sponge) / (m.z_max - m.z_sponge)
        β_sponge = m.α_max * sinpi(r / 2)^m.γ
        ν += β_sponge * ν
        D_t += β_sponge * D_t
        τ += β_sponge * τ
    end
    return (ν, D_t, τ)
end

export HyperdiffEnthalpyFlux
struct HyperdiffEnthalpyFlux{PV} <: TendencyDef{Flux{SecondOrder}, PV} end

export HyperdiffViscousFlux
struct HyperdiffViscousFlux{PV} <: TendencyDef{Flux{SecondOrder}, PV} end

export hyperdiff_enthalpy_and_momentum_flux

"""
    hyperdiff_enthalpy_and_momentum_flux(
        ::PrognosticVariable,
        ::HyperDiffusion,
        ::AbstractTendencyType,
    )

A tuple of the hyperdiffusive enthalpy
and viscous flux types based on the
diffusive model.
"""
function hyperdiff_enthalpy_and_momentum_flux end

# empty tuple by default
hyperdiff_enthalpy_and_momentum_flux(
    pv::PV,
    ::HyperDiffusion,
    ::Flux{SecondOrder},
) where {PV} = ()

# Enthalpy and viscous for Biharmonic model
hyperdiff_enthalpy_and_momentum_flux(
    pv::PV,
    ::Biharmonic,
    ::Flux{SecondOrder},
) where {PV} = (HyperdiffEnthalpyFlux{PV}(), HyperdiffViscousFlux{PV}())

export hyperdiff_momentum_flux
"""
    hyperdiff_momentum_flux(
        ::PrognosticVariable,
        ::HyperDiffusion,
        ::AbstractTendencyType,
    )

A tuple of the hyperdiffusive viscous
flux types based on the diffusive model.
"""
function hyperdiff_momentum_flux end

# empty tuple by default
hyperdiff_momentum_flux(
    pv::PV,
    ::HyperDiffusion,
    ::Flux{SecondOrder},
) where {PV} = ()

# Viscous for Biharmonic model
hyperdiff_momentum_flux(pv::PV, ::Biharmonic, ::Flux{SecondOrder}) where {PV} =
    (HyperdiffViscousFlux{PV}(),)

end #module TurbulenceClosures.jl
