# A mini balance law for computing hyperdiffusion using the DG kernels.
#

using CLIMAParameters
using StaticArrays
using UnPack

using ..BalanceLaws
using ..MPIStateArrays
using ..Orientations
using ..VariableTemplates

import ..BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    eq_tends,
    flux,
    source,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!,
    init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!


# A mini balance law that is used to take the gradient of u and v
# to obtain hyperdiffusion.
struct DryBiharmonicModel{PS, O, FT} <: BalanceLaw
    param_set::PS
    orientation::O
    τ_timescale::FT
end

vars_state(::DryBiharmonicModel, ::Prognostic, FT) =
    @vars(hyper_e::FT, hyper_u::SVector{3, FT})
function vars_state(h::DryBiharmonicModel, st::Auxiliary, FT)
    @vars begin
        Δ::FT
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        temperature::FT
        orientation::vars_state(h.orientation, st, FT)
    end
end
vars_state(::DryBiharmonicModel, ::Gradient, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT)
vars_state(::DryBiharmonicModel, ::GradientLaplacian, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT)
vars_state(::DryBiharmonicModel, ::Hyperdiffusive, FT) =
    @vars(ν∇³u_h::SMatrix{3, 3, FT, 9}, ν∇³h_tot::SVector{3, FT})

# required for each balance law 
function init_state_auxiliary!(
    h::DryBiharmonicModel,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_aux!(h, h.orientation, state_auxiliary, grid, direction)
    init_state_auxiliary!(
        h,
        nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
        direction,
    )
end
function init_state_prognostic!(
    h::DryBiharmonicModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
) end

function nodal_init_state_auxiliary!(
    h::DryBiharmonicModel,
    state_auxiliary::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    state_auxiliary.Δ = lengthscale(geom)
end

function compute_gradient_argument!(
    h::DryBiharmonicModel,
    transformstate::Vars,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
)
    ρinv = 1 / state_auxiliary.ρ
    u = state_auxiliary.ρu * ρinv
    k̂ = vertical_unit_vector(h.orientation, h.param_set, state_auxiliary)
    u_h = (SDiagonal(1, 1, 1) - k̂ * k̂') * u
    transformstate.u_h = u_h
    ts = recover_thermo_state
    e_tot = state_auxiliary.ρe * (1 / state_auxiliary.ρ)
    FT = typeof(h.τ_timescale)
    _R_m = gas_constant_air(h.param_set, FT)
    T = state_auxiliary.temperature
    transformstate.h_tot = total_specific_enthalpy(e_tot, _R_m, T)
end

function compute_gradient_flux!(
    h::DryBiharmonicModel,
    state_gradient_flux::Vars,
    ∇transformstate::Grad,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
) end

function transform_post_gradient_laplacian!(
    h::DryBiharmonicModel,
    Qhypervisc_div::Vars,
    ∇Δtransformstate::Grad,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
)
    ∇Δu_h = ∇Δtransformstate.u_h
    ∇Δh_tot = ∇Δtransformstate.h_tot

    τ_timescale = h.τ_timescale

    # Compute hyperviscosity coefficient
    ν₄ = (state_auxiliary.Δ / 2)^4 / 2 / τ_timescale
    Qhypervisc_div.ν∇³u_h = ν₄ * ∇Δu_h
    Qhypervisc_div.ν∇³h_tot = ν₄ * ∇Δh_tot
end

function flux_first_order!(
    h::DryBiharmonicModel,
    flux::Grad,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
    direction,
) end

function flux_second_order!(
    h::DryBiharmonicModel,
    flux::Grad,
    state_prognostic::Vars,
    state_gradient_flux::Vars,
    hyperdiffusive::Vars,
    state_auxiliary::Vars,
    t::Real,
)
    flux.hyper_u = state_auxiliary.ρ * hyperdiffusive.ν∇³u_h
    flux.hyper_e = hyperdiffusive.ν∇³u_h * state_auxiliary.ρu
    flux.hyper_e += hyperdiffusive.ν∇³h_tot * state_auxiliary.ρ
end

source!(::DryBiharmonicModel, _...) = nothing
boundary_conditions(::DryBiharmonicModel) = ntuple(i -> nothing, 6)
boundary_state!(nf, ::Nothing, ::DryBiharmonicModel, _...) = nothing
