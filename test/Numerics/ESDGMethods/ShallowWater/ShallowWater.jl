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
    source!
using StaticArrays
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
import ClimateMachine.DGMethods.NumericalFluxes:
    EntropyConservative,
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    numerical_flux_first_order!
using ClimateMachine.Grids

abstract type AbstractShallowWaterProblem end

struct ShallowWaterModel{P, FT} <: BalanceLaw
    problem::P
    g::FT
end

boundary_conditions(::ShallowWaterModel) = (0, 1)

function boundary_state!(
    ::NumericalFluxFirstOrder,
    bctype,
    ::ShallowWaterModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    _...,
)
  nh = @SVector [n[1], n[2]]
  state⁺.ρu -= 2 * dot(state⁻.ρu, nh) .* nh
end

function init_state_prognostic!(
    m::ShallowWaterModel,
    args...,
)
  init_state_prognostic!(m, m.problem, args...)
end

function nodal_init_state_auxiliary!(
    m::ShallowWaterModel,
    state_auxiliary,
    tmp,
    geom,
)
  init_state_auxiliary!(m, m.problem, state_auxiliary, geom)
end


function init_state_auxiliary!(
    ::ShallowWaterModel,
    ::AbstractShallowWaterProblem,
    state_auxiliary,
    geom,
)
end

function wavespeed(m::ShallowWaterModel,
                   nM,
                   state::Vars,
                   aux::Vars,
                   t::Real,
                   direction)
  ρ = state.ρ
  ρu = state.ρu

  u = ρu / ρ
  v = @SVector [u[1], u[2], -0]
  uN = abs(dot(nM, v))
  return uN + sqrt(m.g * ρ)
end

function vars_state(::ShallowWaterModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{2, FT}
        ρθ::FT
    end
end

function vars_state(m::ShallowWaterModel, st::Auxiliary, FT)
    @vars begin
        problem::vars_state(m, m.problem, st, FT)
    end
end
vars_state(::ShallowWaterModel, ::AbstractShallowWaterProblem, ::Auxiliary, FT) = @vars()

function vars_state(::ShallowWaterModel, ::Entropy, FT)
    @vars begin
        ρ::FT
        ρu::SVector{2, FT}
    end
end

#"""
#    state_to_entropy_variables!(
#        ::DryAtmosModel,
#        entropy::Vars,
#        state::Vars,
#        aux::Vars,
#    )
#
#See [`BalanceLaws.state_to_entropy_variables!`](@ref)
#"""
#function state_to_entropy_variables!(
#    ::DryAtmosModel,
#    entropy::Vars,
#    state::Vars,
#    aux::Vars,
#)
#    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ
#
#    FT = eltype(state)
#    γ = FT(gamma(param_set))
#
#    p = pressure(ρ, ρu, ρe, Φ)
#    s = log(p / ρ^γ)
#    b = ρ / 2p
#    u = ρu / ρ
#
#    entropy.ρ = (γ - s) / (γ - 1) - (dot(u, u) - 2Φ) * b
#    entropy.ρu = 2b * u
#    entropy.ρe = -2b
#    entropy.Φ = 2ρ * b
#end
#
#function entropy_variables_to_state!(
#    ::DryAtmosModel,
#    state::Vars,
#    aux::Vars,
#    entropy::Vars,
#)
#    FT = eltype(state)
#    β = entropy
#    γ = FT(gamma(param_set))
#
#    b = -β.ρe / 2
#    ρ = β.Φ / (2b)
#    ρu = ρ * β.ρu / (2b)
#
#    p = ρ / (2b)
#    s = log(p / ρ^γ)
#    Φ = dot(ρu, ρu) / (2 * ρ^2) - ((γ - s) / (γ - 1) - β.ρ) / (2b)
#
#    ρe = p / (γ - 1) + dot(ρu, ρu) / (2ρ) + ρ * Φ
#
#    state.ρ = ρ
#    state.ρu = ρu
#    state.ρe = ρe
#    aux.Φ = Φ
#end
#
function state_to_entropy(m::ShallowWaterModel, state::Vars, aux::Vars)
    FT = eltype(state)
    ρ, ρu = state.ρ, state.ρu
    g = m.g
    η = ρu' * ρu / 2ρ + g * ρ ^ 2 / 2
    return η
end

function numerical_volume_conservative_flux_first_order!(
    ::EntropyConservative,
    m::ShallowWaterModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    g = m.g
    FT = eltype(F)
    ρ_1, ρu_1, ρθ_1 = state_1.ρ, state_1.ρu, state_1.ρθ
    ρ_2, ρu_2, ρθ_2 = state_2.ρ, state_2.ρu, state_2.ρθ
    
    u_1 = ρu_1 / ρ_1
    u_2 = ρu_2 / ρ_2
    
    θ_1 = ρθ_1 / ρ_1
    θ_2 = ρθ_2 / ρ_2

    ρ_avg = ave(ρ_1, ρ_2)
    ρ²_avg = ave(ρ_1^2, ρ_2^2)
    u_avg = ave(u_1, u_2)
    ρu_avg = ave(ρu_1, ρu_2)
    ρθ_avg = ave(ρθ_1, ρθ_2)
    θ_avg = ave(θ_1, θ_2)

    Iʰ = @SMatrix [ 1 -0
                   -0  1
                   -0 -0]


    ρv_avg = @SVector [ρu_avg[1], ρu_avg[2], -0]
    v_avg = @SVector [u_avg[1], u_avg[2], -0]

    Fρ = ρv_avg
    Fρu = ρv_avg * u_avg' + g * (ρ_avg ^ 2 - ρ²_avg / 2) * Iʰ
    
    # not sure what's better for tracer
    #Fρθ = v_avg * ρθ_avg
    Fρθ = ρv_avg * θ_avg

    F.ρ += Fρ
    F.ρu += Fρu
    F.ρθ += Fρθ
end

function numerical_volume_fluctuation_flux_first_order!(
    ::EntropyConservative,
    ::ShallowWaterModel,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
end

source!(::ShallowWaterModel, _...) = nothing

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
