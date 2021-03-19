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

using DocStringExtensions
using LinearAlgebra
using StaticArrays
using UnPack
import CLIMAParameters: AbstractParameterSet
using CLIMAParameters.Atmos.SubgridScale: inv_Pr_turb

using ClimateMachine

import ..Mesh.Geometry: LocalGeometry, lengthscale, lengthscale_horizontal

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
    DryBiharmonic,
    EquilMoistBiharmonic,
    NoViscousSponge,
    UpperAtmosSponge,
    turbulence_tensors,
    init_aux_turbulence!,
    init_aux_hyperdiffusion!,
    sponge_viscosity_modifier,
    HyperdiffEnthalpyFlux,
    HyperdiffViscousFlux,
    WithoutDivergence,
    WithDivergence

"""
    TurbulenceClosureModel

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
    HyperDiffusion

Abstract type for HyperDiffusion models
"""
abstract type HyperDiffusion end

"""
    ViscousSponge

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

Compute the kinematic viscosity (`ν`), the
diffusivity (`D_t`) and SGS momentum flux
tensor (`τ`) for a given turbulence closure.
Each closure overloads this method with the
appropriate calculations for the returned
quantities.

# Arguments

- `::TurbulenceClosureModel` = Struct identifier
   for turbulence closure model
- `orientation` = `BalanceLaw.orientation`
- `param_set` parameter set
- `state` = Array of prognostic (state) variables.
   See `vars_state` in `BalanceLaw`
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
    param_set = parameter_set(bl)
    ν, D_t, τ = turbulence_tensors(
        m,
        bl.orientation,
        param_set,
        state,
        diffusive,
        aux,
        t,
    )
    ν, D_t, τ = sponge_viscosity_modifier(bl, bl.viscoussponge, ν, D_t, τ, aux)
    return (ν, D_t, τ)
end

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

"""
    strain_rate_magnitude(S)

Given the rate-of-strain tensor `S`, computes its magnitude.
"""
function strain_rate_magnitude(S::SHermitianCompact{3, FT, 6}) where {FT}
    return sqrt(2 * norm2(S))
end

"""
    WithDivergence

A divergence type which includes the
divergence term in the momentum flux tensor
"""
struct WithDivergence end

"""
    WithoutDivergence

A divergence type which does not include the
divergence term in the momentum flux tensor
"""
struct WithoutDivergence end

"""
    ConstantDynamicViscosity <: ConstantViscosity

Turbulence with constant dynamic viscosity (`ρν`).
Divergence terms are included in the momentum flux
tensor if divergence_type is WithDivergence.

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
Divergence terms are included in the momentum flux
tensor if divergence_type is WithDivergence.

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

"""
    AnisoMinDiss{FT} <: TurbulenceClosureModel

Filter width Δ is the local grid resolution
calculated from the mesh metric tensor. A
Poincare coefficient is specified and used
to compute the equivalent AnisoMinDiss
coefficient (computed as the solution to the
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

Defines a default hyperdiffusion model with
zero hyperdiffusive fluxes.
"""
struct NoHyperDiffusion <: HyperDiffusion end

"""
    EquilMoistBiharmonic{FT} <: HyperDiffusion

Assumes equilibrium thermodynamics in compressible
flow. Horizontal hyperdiffusion methods for
application in GCM and LES settings Timescales are
prescribed by the user while the diffusion coefficient
is computed as a function of the grid lengthscale.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct EquilMoistBiharmonic{FT} <: HyperDiffusion
    τ_timescale::FT
    τ_timescale_q_tot::FT
end

EquilMoistBiharmonic(τ_timescale::FT) where {FT} =
    EquilMoistBiharmonic(τ_timescale, τ_timescale)

vars_state(::EquilMoistBiharmonic, ::Auxiliary, FT) = @vars(Δ::FT)
vars_state(::EquilMoistBiharmonic, ::Gradient, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT, q_tot::FT)
vars_state(::EquilMoistBiharmonic, ::GradientLaplacian, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT, q_tot::FT)
vars_state(::EquilMoistBiharmonic, ::Hyperdiffusive, FT) = @vars(
    ν∇³u_h::SMatrix{3, 3, FT, 9},
    ν∇³h_tot::SVector{3, FT},
    ν∇³q_tot::SVector{3, FT}
)

function init_aux_hyperdiffusion!(
    ::EquilMoistBiharmonic,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.hyperdiffusion.Δ = lengthscale_horizontal(geom)
end

function compute_gradient_argument!(
    h::EquilMoistBiharmonic,
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
    transform.hyperdiffusion.h_tot = transform.energy.h_tot
    transform.hyperdiffusion.q_tot = state.moisture.ρq_tot * ρinv
end

function transform_post_gradient_laplacian!(
    h::EquilMoistBiharmonic,
    bl::BalanceLaw,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    param_set = parameter_set(bl)
    _inv_Pr_turb = eltype(state)(inv_Pr_turb(param_set))
    ∇Δu_h = hypertransform.hyperdiffusion.u_h
    ∇Δh_tot = hypertransform.hyperdiffusion.h_tot
    ∇Δq_tot = hypertransform.hyperdiffusion.q_tot
    # Unpack
    τ_timescale = h.τ_timescale
    τ_timescale_q_tot = h.τ_timescale_q_tot
    # Compute hyperviscosity coefficient
    ν₄ = (aux.hyperdiffusion.Δ / 2)^4 / 2 / τ_timescale
    ν₄_q_tot = (aux.hyperdiffusion.Δ / 2)^4 / 2 / τ_timescale_q_tot
    hyperdiffusive.hyperdiffusion.ν∇³u_h = ν₄ * ∇Δu_h
    hyperdiffusive.hyperdiffusion.ν∇³h_tot = ν₄ * ∇Δh_tot
    hyperdiffusive.hyperdiffusion.ν∇³q_tot = ν₄_q_tot * ∇Δq_tot
end

"""
    DryBiharmonic{FT} <: HyperDiffusion

Assumes dry compressible flow.
Horizontal hyperdiffusion methods for application
in GCM and LES settings Timescales are prescribed
by the user while the diffusion coefficient is
computed as a function of the grid lengthscale.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DryBiharmonic{FT} <: HyperDiffusion
    τ_timescale::FT
end
vars_state(::DryBiharmonic, ::Auxiliary, FT) = @vars(Δ::FT)
vars_state(::DryBiharmonic, ::Gradient, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT)
vars_state(::DryBiharmonic, ::GradientLaplacian, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT)
vars_state(::DryBiharmonic, ::Hyperdiffusive, FT) =
    @vars(ν∇³u_h::SMatrix{3, 3, FT, 9}, ν∇³h_tot::SVector{3, FT})

function init_aux_hyperdiffusion!(
    ::DryBiharmonic,
    ::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.hyperdiffusion.Δ = lengthscale_horizontal(geom)
end

function compute_gradient_argument!(
    h::DryBiharmonic,
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
    transform.hyperdiffusion.h_tot = transform.energy.h_tot
end

function transform_post_gradient_laplacian!(
    h::DryBiharmonic,
    bl::BalanceLaw,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    param_set = parameter_set(bl)
    _inv_Pr_turb = eltype(state)(inv_Pr_turb(param_set))
    ∇Δu_h = hypertransform.hyperdiffusion.u_h
    ∇Δh_tot = hypertransform.hyperdiffusion.h_tot
    # Unpack
    τ_timescale = h.τ_timescale
    # Compute hyperviscosity coefficient
    ν₄ = (aux.hyperdiffusion.Δ / 2)^4 / 2 / τ_timescale
    hyperdiffusive.hyperdiffusion.ν∇³u_h = ν₄ * ∇Δu_h
    hyperdiffusive.hyperdiffusion.ν∇³h_tot = ν₄ * ∇Δh_tot
end

"""
    NoViscousSponge

No modifiers applied to viscosity/diffusivity in sponge layer

# Fields

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
    UpperAtmosSponge{FT} <: ViscousSponge

Upper domain viscous relaxation.

Applies modifier to viscosity and diffusivity terms
in a user-specified upper domain sponge region

# Fields
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
    param_set = parameter_set(bl)
    z = altitude(bl.orientation, param_set, aux)
    if z >= m.sponge
        r = (z - m.z_sponge) / (m.z_max - m.z_sponge)
        β_sponge = m.α_max * sinpi(r / 2)^m.γ
        ν += β_sponge * ν
        D_t += β_sponge * D_t
        τ += β_sponge * τ
    end
    return (ν, D_t, τ)
end

const Biharmonic = Union{EquilMoistBiharmonic, DryBiharmonic}

struct HyperdiffEnthalpyFlux <: TendencyDef{Flux{SecondOrder}} end
struct HyperdiffViscousFlux <: TendencyDef{Flux{SecondOrder}} end

# empty by default
eq_tends(pv::PV, ::HyperDiffusion, ::AbstractTendencyType) where {PV} = ()

# Enthalpy and viscous for Biharmonic model
eq_tends(::AbstractEnergy, ::Biharmonic, ::Flux{SecondOrder}) =
    (HyperdiffEnthalpyFlux(), HyperdiffViscousFlux())

# Viscous for Biharmonic model
eq_tends(::AbstractMomentum, ::Biharmonic, ::Flux{SecondOrder}) =
    (HyperdiffViscousFlux(),)

end #module TurbulenceClosures.jl
