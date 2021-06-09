using ClimateMachine.VariableTemplates: Vars, Grad, @vars
using ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws:
    BalanceLaw,
    vars_state,
    state_to_entropy_variables!,
    entropy_variables_to_state!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    state_to_entropy,
    boundary_conditions,
    boundary_state!,
    wavespeed,
    flux_first_order!,
    source!,
    drag_source!
using StaticArrays
using UnPack
using LinearAlgebra: dot, I
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
import ClimateMachine.DGMethods.NumericalFluxes:
    EntropyConservative,
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    logave,
    numerical_flux_first_order!
using ClimateMachine.Orientations:
    Orientation, FlatOrientation, SphericalOrientation
using ClimateMachine.Atmos: NoReferenceState
using ClimateMachine.Grids

using CLIMAParameters: AbstractEarthParameterSet
using CLIMAParameters.Planet: grav, R_d, cp_d, cv_d, planet_radius, MSLP, Omega

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

@inline gamma(ps::EarthParameterSet) = cp_d(ps) / cv_d(ps)

abstract type AbstractDryAtmosProblem end

struct DryAtmosModel{D, O, P, RS, S, DS} <: BalanceLaw
    orientation::O
    problem::P
    ref_state::RS
    sources::S
    drag_source::DS
end
function DryAtmosModel{D}(orientation,
                          problem::AbstractDryAtmosProblem;
                          ref_state=NoReferenceState(),
                          sources=(),
                          drag_source=NoDrag()) where {D}
    O = typeof(orientation)
    P = typeof(problem)
    RS = typeof(ref_state)
    S = typeof(sources)
    DS = typeof(drag_source)
    DryAtmosModel{D, O, P, RS, S, DS}(orientation, problem,
                                      ref_state, sources, drag_source)
end

boundary_conditions(::DryAtmosModel) = (1, 2)
# XXX: Hack for Impenetrable.
#      This is NOT entropy stable / conservative!!!!
function boundary_state!(
    ::NumericalFluxFirstOrder,
    bctype,
    ::DryAtmosModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    _...,
)
    state⁺.ρ = state⁻.ρ
    state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
    state⁺.ρθ = state⁻.ρθ
    aux⁺.Φ = aux⁻.Φ
end

function init_state_prognostic!(
    m::DryAtmosModel,
    args...,
)
  init_state_prognostic!(m, m.problem, args...)
end

function nodal_init_state_auxiliary!(
    m::DryAtmosModel,
    state_auxiliary,
    tmp,
    geom,
)
  init_state_auxiliary!(m, m.orientation, state_auxiliary, geom)
  init_state_auxiliary!(m, m.ref_state, state_auxiliary, geom)
  init_state_auxiliary!(m, m.problem, state_auxiliary, geom)
end

function altitude(::DryAtmosModel{dim},
                  ::FlatOrientation,
                  geom) where {dim}
  @inbounds geom.coord[dim]
end

function altitude(::DryAtmosModel,
                  ::SphericalOrientation,
                  geom)
  FT = eltype(geom)
  _planet_radius::FT = planet_radius(param_set)
  norm(geom.coord) - _planet_radius
end

"""
    init_state_auxiliary!(
        m::DryAtmosModel,
        aux::Vars,
        geom::LocalGeometry
        )

Initialize geopotential for the `DryAtmosModel`.
"""
function init_state_auxiliary!(
    ::DryAtmosModel{dim},
    ::FlatOrientation,
    state_auxiliary,
    geom,
) where {dim}
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    #_grav = FT(0)
    @inbounds r = geom.coord[dim]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = dim == 2 ? 
          SVector{3, FT}(0, _grav, 0) :
          SVector{3, FT}(0, 0, _grav)
end
function init_state_auxiliary!(
    ::DryAtmosModel,
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    r = norm(geom.coord)
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = _grav * geom.coord / r 
end

function init_state_auxiliary!(
    ::DryAtmosModel,
    ::NoReferenceState,
    state_auxiliary,
    geom,
)
end

function init_state_auxiliary!(
    ::DryAtmosModel,
    ::AbstractDryAtmosProblem,
    state_auxiliary,
    geom,
)
end

struct DryReferenceState{TP}
  temperature_profile::TP
end
vars_state(::DryAtmosModel, ::DryReferenceState, ::Auxiliary, FT) =
  @vars(T::FT, p::FT, ρ::FT, ρθ::FT)
vars_state(::DryAtmosModel, ::NoReferenceState, ::Auxiliary, FT) =
  @vars()

function init_state_auxiliary!(
    m::DryAtmosModel,
    ref_state::DryReferenceState,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    z = altitude(m, m.orientation, geom)
    T, p = ref_state.temperature_profile(param_set, z)

    _R_d::FT = R_d(param_set)
    ρ = p / (_R_d * T)
    Φ = state_auxiliary.Φ
    ρu = SVector{3, FT}(0, 0, 0)


    state_auxiliary.ref_state.T = T
    state_auxiliary.ref_state.p = p
    state_auxiliary.ref_state.ρ = ρ
    state_auxiliary.ref_state.ρθ = thetadensity(p)
end

@inline function flux_first_order!(
    m::DryAtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    ρinv = 1 / ρ
    ρu = state.ρu
    ρθ = state.ρθ
    u = ρinv * ρu
    Φ = aux.Φ
    
    p = pressure(ρθ)

    flux.ρ = ρ * u
    flux.ρu = p * I + ρ * u .* u'
    flux.ρθ = u * ρθ
end

function wavespeed(::DryAtmosModel,
                   nM,
                   state::Vars,
                   aux::Vars,
                   t::Real,
                   direction)
  ρ = state.ρ
  ρu = state.ρu
  ρθ = state.ρθ
  Φ = aux.Φ
  p = pressure(ρθ)

  u = ρu / ρ
  uN = abs(dot(nM, u))
  return uN + soundspeed(ρ, p)
end

"""
    pressure(ρθ)

"""
function pressure(ρθ)
    FT = eltype(ρθ)
    γ = FT(gamma(param_set))
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    (_R_d * ρθ) ^ γ / _MSLP ^ (γ - 1)
    #return ρθ ^ γ
end

function thetadensity(p)
    FT = eltype(p)
    γ = FT(gamma(param_set))
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    ρθ  = (p * _MSLP ^ (γ - 1)) ^ (1 / γ) / _R_d
end

"""
    soundspeed(ρ, p)

Compute the speed of sound from the density `ρ` and pressure `p`.
"""
function soundspeed(ρ, p)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    sqrt(γ * p / ρ)
end

"""
    vars_state(::DryAtmosModel, ::Prognostic, FT)

The prognostic state variables for the `DryAtmosModel` are density `ρ`, momentum `ρu`,
and total energy `ρe`
"""
function vars_state(::DryAtmosModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρθ::FT
    end
end

"""
    vars_state(::DryAtmosModel, ::Auxiliary, FT)

The auxiliary variables for the `DryAtmosModel` is gravitational potential
`Φ`
"""
function vars_state(m::DryAtmosModel, st::Auxiliary, FT)
    @vars begin
        Φ::FT
        ∇Φ::SVector{3, FT} # TODO: only needed for the linear model
        ref_state::vars_state(m, m.ref_state, st, FT)
        problem::vars_state(m, m.problem, st, FT)
    end
end
vars_state(::DryAtmosModel, ::AbstractDryAtmosProblem, ::Auxiliary, FT) = @vars()

"""
    vars_state(::DryAtmosModel, ::Entropy, FT)

The entropy variables for the `DryAtmosModel` correspond to the state
variables density `ρ`, momentum `ρu`, and total energy `ρe` as well as the
auxiliary variable gravitational potential `Φ`
"""
function vars_state(::DryAtmosModel, ::Entropy, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρθ::FT
    end
end

"""
    state_to_entropy_variables!(
        ::DryAtmosModel,
        entropy::Vars,
        state::Vars,
        aux::Vars,
    )

See [`BalanceLaws.state_to_entropy_variables!`](@ref)
"""
function state_to_entropy_variables!(
    ::DryAtmosModel,
    entropy::Vars,
    state::Vars,
    aux::Vars,
)
    ρ, ρu, ρθ, Φ = state.ρ, state.ρu, state.ρθ, aux.Φ

    FT = eltype(state)
    γ = FT(gamma(param_set))

    p = pressure(ρθ)
    u = ρu / ρ

    entropy.ρ = -dot(u, u)
    entropy.ρu = u
    entropy.ρθ = γ / (γ - 1) * p / ρθ
end

"""
    entropy_variables_to_state!(
        ::DryAtmosModel,
        state::Vars,
        aux::Vars,
        entropy::Vars,
    )

See [`BalanceLaws.entropy_variables_to_state!`](@ref)
"""
function entropy_variables_to_state!(
    ::DryAtmosModel,
    state::Vars,
    aux::Vars,
    entropy::Vars,
)
  error("non invertible")
end

function state_to_entropy(::DryAtmosModel, state::Vars, aux::Vars)
    FT = eltype(state)
    ρ, ρu, ρθ, Φ = state.ρ, state.ρu, state.ρθ, aux.Φ
    p = pressure(ρθ)
    γ = FT(gamma(param_set))
    η = p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ
    return η
end

function γave(a, b)
    FT = eltype(a)
    γ = FT(gamma(param_set))
    f = (b - a) / (b + a)
    v = f ^ 2
    if v < 1e-4
      F = @evalpoly(v,
                    one(f), (γ - 2) / 3,
                    -(γ + 1) * (γ - 2) * (γ - 3) / 45,
                    (γ + 1) * (γ - 2) * (γ - 3) * (2γ * (γ - 2) - 9) / 945)
      ave(a, b) * F
    else
      (γ - 1) / γ * (a ^ γ - b ^ γ) / (a ^ (γ - 1) - b ^ (γ - 1))
    end
    #return (γ - 1) / γ * (a ^ γ - b ^ γ) / (a ^ (γ - 1) - b ^ (γ - 1)  + 1e-14)
end

function numerical_volume_conservative_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    ρ_1, ρu_1, ρθ_1 = state_1.ρ, state_1.ρu, state_1.ρθ
    ρ_2, ρu_2, ρθ_2 = state_2.ρ, state_2.ρu, state_2.ρθ

    u_1 = ρu_1 / ρ_1
    u_2 = ρu_2 / ρ_2
    p_1 = pressure(ρθ_1)
    p_2 = pressure(ρθ_2)

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    p_avg = ave(p_1, p_2)
    ρθ_avg = γave(ρθ_1, ρθ_2)
    #ρθ_avg = ave(ρθ_1, ρθ_2)

    Fρ = u_avg * ρ_avg
    Fρu = u_avg * Fρ' + p_avg * I
    Fρθ = u_avg * ρθ_avg

    F.ρ += Fρ
    F.ρu += Fρu
    F.ρθ += Fρθ
end

function numerical_volume_fluctuation_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(D)
    Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
    ρ_1, ρu_1, ρθ_1 = state_1.ρ, state_1.ρu, state_1.ρθ
    ρ_2, ρu_2, ρθ_2 = state_2.ρ, state_2.ρu, state_2.ρθ

    ρ_avg = ave(ρ_1, ρ_2)
    #α = b_avg * ρ_log / 2b_1
    α = ρ_avg / 2

    D.ρu -= α * (Φ_1 - Φ_2) * I
end

struct Coriolis end

function source!(
    m::DryAtmosModel,
    ::Coriolis,
    source,
    state_prognostic,
    state_auxiliary,
)
    FT = eltype(state_prognostic)
    _Omega::FT = Omega(param_set)
    # note: this assumes a SphericalOrientation
    source.ρu -= SVector(0, 0, 2 * _Omega) × state_prognostic.ρu
end

function source!(
    m::DryAtmosModel,
    source,
    state_prognostic,
    state_auxiliary,
)
  ntuple(Val(length(m.sources))) do s
    Base.@_inline_meta
    source!(m, m.sources[s], source, state_prognostic, state_auxiliary)
  end
end


struct EntropyConservativeWithPenalty <: NumericalFluxFirstOrder end
function numerical_flux_first_order!(
    numerical_flux::EntropyConservativeWithPenalty,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)

    numerical_flux_first_order!(
        EntropyConservative(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn = parent(fluxᵀn)

    wavespeed⁻ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    max_wavespeed = max.(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed .*
        (parent(state_prognostic⁻) - parent(state_prognostic⁺))

    fluxᵀn .+= penalty / 2
end

#Base.@kwdef struct MatrixFlux{FT} <: NumericalFluxFirstOrder
#  Mcut::FT = 0
#  low_mach::Bool = false
#  kinetic_energy_preserving::Bool = false
#end
#
#function numerical_flux_first_order!(
#    numerical_flux::MatrixFlux,
#    balance_law::BalanceLaw,
#    fluxᵀn::Vars{S},
#    normal_vector::SVector,
#    state_prognostic⁻::Vars{S},
#    state_auxiliary⁻::Vars{A},
#    state_prognostic⁺::Vars{S},
#    state_auxiliary⁺::Vars{A},
#    t,
#    direction,
#) where {S, A}
#
#    FT = eltype(fluxᵀn)
#    numerical_flux_first_order!(
#        EntropyConservative(),
#        balance_law,
#        fluxᵀn,
#        normal_vector,
#        state_prognostic⁻,
#        state_auxiliary⁻,
#        state_prognostic⁺,
#        state_auxiliary⁺,
#        t,
#        direction,
#    )
#    fluxᵀn = parent(fluxᵀn)
#    
#    γ = FT(gamma(param_set))
#    
#    low_mach = numerical_flux.low_mach
#    Mcut = numerical_flux.Mcut
#    kinetic_energy_preserving = numerical_flux.kinetic_energy_preserving
#
#    ω = FT(π) / 3
#    δ = FT(π) / 5
#    random_unit_vector = SVector(sin(ω) * cos(δ), cos(ω) * cos(δ), sin(δ))
#    # tangent space basis
#    τ1 = random_unit_vector × normal_vector
#    τ2 = τ1 × normal_vector
#
#    ρ⁻ = state_prognostic⁻.ρ
#    ρu⁻ = state_prognostic⁻.ρu
#    ρe⁻ = state_prognostic⁻.ρe
#   
#    Φ⁻ = state_auxiliary⁻.Φ
#    u⁻ = ρu⁻ / ρ⁻
#    p⁻ = pressure(ρ⁻, ρu⁻, ρe⁻, Φ⁻)
#    β⁻ = ρ⁻ / 2p⁻
#    
#    Φ⁺ = state_auxiliary⁺.Φ
#    ρ⁺ = state_prognostic⁺.ρ
#    ρu⁺ = state_prognostic⁺.ρu
#    ρe⁺ = state_prognostic⁺.ρe
#    
#    u⁺ = ρu⁺ / ρ⁺
#    p⁺ = pressure(ρ⁺, ρu⁺, ρe⁺, Φ⁺)
#    β⁺ = ρ⁺ / 2p⁺
#
#    ρ_log = logave(ρ⁻, ρ⁺)
#    β_log = logave(β⁻, β⁺)
#    Φ_avg = ave(Φ⁻, Φ⁺)
#    u_avg = ave.(u⁻, u⁺)
#    p_avg = ave(ρ⁻, ρ⁺) / 2ave(β⁻, β⁺)
#    u²_bar = 2 * sum(u_avg .^ 2) - sum(ave(u⁻ .^ 2, u⁺ .^ 2))
#
#    h_bar = γ / (2 * β_log * (γ - 1)) + u²_bar / 2 + Φ_avg
#    c_bar = sqrt(γ * p_avg / ρ_log)
#
#    umc = u_avg - c_bar * normal_vector
#    upc = u_avg + c_bar * normal_vector
#    u_avgᵀn = u_avg' * normal_vector
#    R = hcat(
#      SVector(1, umc[1], umc[2], umc[3], h_bar - c_bar * u_avgᵀn),
#      SVector(1, u_avg[1], u_avg[2], u_avg[3], u²_bar / 2 + Φ_avg),
#      SVector(0, τ1[1], τ1[2], τ1[3], τ1' * u_avg),
#      SVector(0, τ2[1], τ2[2], τ2[3], τ2' * u_avg),
#      SVector(1, upc[1], upc[2], upc[3], h_bar + c_bar * u_avgᵀn),
#    )
#
#    if low_mach
#      M = abs(u_avg' * normal_vector) / c_bar
#      c_bar *= max(min(M, FT(1)), Mcut)
#    end
#
#    if kinetic_energy_preserving
#      λl = abs(u_avgᵀn) + c_bar
#      λr = λl
#    else
#      λl = abs(u_avgᵀn - c_bar)
#      λr = abs(u_avgᵀn + c_bar)
#    end
#
#    Λ = SDiagonal(
#      λl,
#      abs(u_avgᵀn),
#      abs(u_avgᵀn),
#      abs(u_avgᵀn),
#      λr,
#    )
#    #Ξ = sqrt(abs((p⁺ - p⁻) / (p⁺ + p⁻)))
#    #Λ = Ξ * abs(u_avgᵀn + c_bar) * I + (1 - Ξ) * ΛM
#
#    T = SDiagonal(
#      ρ_log / 2γ,
#      ρ_log * (γ - 1) / γ,
#      p_avg,
#      p_avg,
#      ρ_log / 2γ,
#    )
#
#    entropy⁻ = similar(parent(state_prognostic⁻), Size(6))
#    state_to_entropy_variables!(
#      balance_law,
#      Vars{vars_state(balance_law, Entropy(), FT)}(entropy⁻),
#      state_prognostic⁻,
#      state_auxiliary⁻,
#    )
#    
#    entropy⁺ = similar(parent(state_prognostic⁺), Size(6))
#    state_to_entropy_variables!(
#      balance_law,
#      Vars{vars_state(balance_law, Entropy(), FT)}(entropy⁺),
#      state_prognostic⁺,
#      state_auxiliary⁺,
#    )
#
#    Δentropy = parent(entropy⁺) - parent(entropy⁻)
#
#    fluxᵀn .-= R * Λ * T * R' * Δentropy[SOneTo(5)] / 2
#end

function vertical_unit_vector(m::DryAtmosModel, aux::Vars)
  FT = eltype(aux)
  aux.∇Φ / FT(grav(param_set))
end

norm_u(state::Vars, k̂::AbstractVector, ::VerticalDirection) =
    abs(dot(state.ρu, k̂)) / state.ρ
norm_u(state::Vars, k̂::AbstractVector, ::HorizontalDirection) =
    norm((state.ρu .- dot(state.ρu, k̂) * k̂) / state.ρ)
norm_u(state::Vars, k̂::AbstractVector, ::Direction) = norm(state.ρu / state.ρ)

function advective_courant(
    m::DryAtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    k̂ = vertical_unit_vector(m, aux)
    normu = norm_u(state, k̂, direction)
    return Δt * normu / Δx
end

function nondiffusive_courant(
    m::DryAtmosModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ
    Φ = aux.Φ
    p = pressure(ρθ)

    k̂ = vertical_unit_vector(m, aux)
    normu = norm_u(state, k̂, direction)
    ss = soundspeed(ρ, p)
    return Δt * (normu + ss) / Δx
end

function drag_source!(m::DryAtmosModel, args...)
  drag_source!(m, m.drag_source, args...)
end
struct NoDrag end
drag_source!(m::DryAtmosModel, ::NoDrag, args...) = nothing


include("linear.jl")
include("vorticitymodel.jl")
