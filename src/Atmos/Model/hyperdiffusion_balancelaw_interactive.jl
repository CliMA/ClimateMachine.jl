# A mini balance law for computing hyperdiffusion using the DG kernels.
#

using CLIMAParameters
using StaticArrays
using UnPack

using ..BalanceLaws
using ..MPIStateArrays
using ..Orientations
using ..VariableTemplates

using ..DGMethods: init_ode_state

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
struct DryBiharmonicModelIn{PS, O, FT} <: BalanceLaw
    param_set::PS
    orientation::O
    τ_timescale::FT
end

function vars_state(::DryBiharmonicModelIn, ::Prognostic, FT)
    #@vars(hyper_e::FT, hyper_u::SVector{3, FT})
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end    
function vars_state(h::DryBiharmonicModelIn, st::Auxiliary, FT)
    @vars begin
        Δ::FT
        temperature::FT
        orientation::vars_state(h.orientation, st, FT)
    end
end
vars_state(::DryBiharmonicModelIn, ::Gradient, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT)
vars_state(::DryBiharmonicModelIn, ::GradientLaplacian, FT) =
    @vars(u_h::SVector{3, FT}, h_tot::FT)
vars_state(::DryBiharmonicModelIn, ::Hyperdiffusive, FT) =
    @vars(ν∇³u_h::SMatrix{3, 3, FT, 9}, ν∇³h_tot::SVector{3, FT})

# required for each balance law 
function init_state_auxiliary!(
    h::DryBiharmonicModelIn,
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
    h::DryBiharmonicModelIn,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
) end

function nodal_init_state_auxiliary!(
    h::DryBiharmonicModelIn,
    state_auxiliary::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    state_auxiliary.Δ = lengthscale(geom)
end

function compute_gradient_argument!(
    h::DryBiharmonicModelIn,
    transformstate::Vars,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
)
    ρinv = 1 / state_prognostic.ρ
    u = state_prognostic.ρu * ρinv
    k̂ = vertical_unit_vector(h.orientation, h.param_set, state_auxiliary)
    u_h = (SDiagonal(1, 1, 1) - k̂ * k̂') * u
    transformstate.u_h = u_h
    ts = recover_thermo_state
    e_tot = state_prognostic.ρe * (1 / state_prognostic.ρ)
    FT = typeof(h.τ_timescale)
    _R_m = gas_constant_air(h.param_set, FT)
    T = state_auxiliary.temperature
    transformstate.h_tot = total_specific_enthalpy(e_tot, _R_m, T)
end

function compute_gradient_flux!(
    h::DryBiharmonicModelIn,
    state_gradient_flux::Vars,
    ∇transformstate::Grad,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
) end

function transform_post_gradient_laplacian!(
    h::DryBiharmonicModelIn,
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
    h::DryBiharmonicModelIn,
    flux::Grad,
    state_prognostic::Vars,
    state_auxiliary::Vars,
    t::Real,
    direction,
) end

function flux_second_order!(
    h::DryBiharmonicModelIn,
    flux::Grad,
    state_prognostic::Vars,
    state_gradient_flux::Vars,
    hyperdiffusive::Vars,
    state_auxiliary::Vars,
    t::Real,
)
    flux.ρu= state_prognostic.ρ * hyperdiffusive.ν∇³u_h
    flux.ρe = hyperdiffusive.ν∇³u_h * state_prognostic.ρu
    flux.ρe += hyperdiffusive.ν∇³h_tot * state_prognostic.ρ
end

source!(::DryBiharmonicModelIn, _...) = nothing
boundary_conditions(::DryBiharmonicModelIn) = ntuple(i -> nothing, 6)
boundary_state!(nf, ::Nothing, ::DryBiharmonicModelIn, _...) = nothing

# This sets up the mini DG Model 
HD_DGModel_init(::HyperDiffusion, atmos, grid, params) = nothing
function HD_DGModel_init(atmos, grid, params)
    # set up the hyperdiffusion mini balance law
    hyper_state = params.hyper_state
    FT = Float64
    if isfinite(params.timescale)
        hyper_state.bl = DryBiharmonicModelIn(
            atmos.param_set,
            atmos.orientation,
            params.timescale,
        )
        hyper_state.dg = DGModel(
            hyper_state.bl,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            diffusion_direction = HorizontalDirection(),
        )
        hyper_state.state = init_ode_state(hyper_state.dg, FT(0)) # initiate state 
        # initiate state atrray
        hyper_state.dQ = similar(
            hyper_state.state;
            vars = @vars(ρ::FT, ρu::SVector{3, FT}, ρe::FT),
            nstate = 5,
        )
    end
end


abstract type BLIGroupParams end
"""
BLIGroup

Holds a set of BLI parameters that will be fed into the driver (equiv to dgngrp)
"""
mutable struct BLIGroup{BLIP <: Union{Nothing, BLIGroupParams}}
    name::String
    params::BLIP

    BLIGroup(
        name,
        params = nothing,
    ) = new{typeof(params)}(
        name,
        params,
    )
end

"""
HyperdiffusionBLIState

Holds / collects DGModel, BL info, state array, and aux array for BLI 
"""
mutable struct HyperdiffusionBLIState
    bl::Union{Nothing, DryBiharmonicModelIn}
    dg::Union{Nothing, DGModel}
    state::Union{Nothing, MPIStateArray}
    dQ::Union{Nothing, MPIStateArray}

    HyperdiffusionBLIState() = new(nothing, nothing, nothing, nothing)
end

"""
BLIParams

Holds params for BLI defined in the experiment file using the setup_BLI function
"""
struct BLIParams{FT} <: BLIGroupParams
    timescale::FT
    hyper_state::HyperdiffusionBLIState

    BLIParams(timescale::FT) where {FT} =
        new{FT}(timescale, HyperdiffusionBLIState())
end


"""
setup_BLI

Function that will be used in driver to setup the BLIGroup
"""
function setup_BLI(
    ::AtmosGCMConfigType;
    timescale = Inf,
) where {FT}
    return BLIGroup(
        "BLI_params",
        BLIParams(timescale),
    )
end