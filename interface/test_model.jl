#include("../boilerplate.jl")

import ClimateMachine.BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_conditions,
    boundary_state!
import ClimateMachine.DGMethods: DGModel
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

using ClimateMachine.BalanceLaws
include("abstractions.jl")
"""
    TestEquations <: BalanceLaw
A `BalanceLaw` for general testing.

# Usage
    
"""
abstract type AbstractEquations <: BalanceLaw end

abstract type AbstractEquations3D <: AbstractEquations end

struct TestEquations{D,FT} <: AbstractEquations3D
    domain::D
    advection::Union{BalanceLaw, Nothing}
    turbulence::Union{BalanceLaw, Nothing}
    hyperdiffusion::Union{BalanceLaw, Nothing}
    coriolis::Union{BalanceLaw, Nothing}
    forcing::Union{BalanceLaw, Nothing}
    boundary_conditions::Union{BalanceLaw, Nothing}
    params::Union{FT, Nothing}
    function TestEquations{FT}(
        domain::D;
        advection = nothing,
        turbulence = nothing,
        hyperdiffusion = nothing,
        coriolis = nothing,
        forcing = nothing,
        boundary_conditions = nothing,
        params = nothing
    ) where {FT <: AbstractFloat, D}
        return new{D, FT}(
            domain,
            advection,
            turbulence,
            hyperdiffusion,
            coriolis,
            forcing,
            boundary_conditions,
            params,
        )
    end
end

include("hyperdiffusion_model.jl")


function vars_state(m::TestEquations, st::Prognostic, FT)
    @vars begin
        ρ::FT
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end
function vars_state(m::TestEquations, aux::Auxiliary, FT)
    @vars begin
        coord::SVector{3, FT}
        hyperdiffusion::vars_state(m.hyperdiffusion, aux, FT)
    end
end
function vars_state(m::TestEquations, grad::Gradient, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, grad, FT)
    end
end
function vars_state(m::TestEquations, grad::GradientFlux, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, grad, FT)
    end
end 
function vars_state(m::TestEquations, st::GradientLaplacian, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end
function vars_state(m::TestEquations, st::Hyperdiffusive, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end

function nodal_init_state_auxiliary!(
    m::TestEquations,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
    nodal_init_state_auxiliary!(
        m.hyperdiffusion,
        aux,
        tmp,
        geom,
    )
end

function init_state_prognostic!(
    m::TestEquations,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    init_state_prognostic!(
        m.hyperdiffusion,
        state::Vars,
        aux::Vars,
        localgeo,
        t::Real,
    )
end

function compute_gradient_argument!(
    m::TestEquations,
    grad::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_argument!(m.hyperdiffusion, grad, state, aux, t)
end

function compute_gradient_flux!(
    model::TestEquations,
    gradflux::Vars,
    grad::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing# don't need anything for hyperdiffusion here
end

function transform_post_gradient_laplacian!(
    m::TestEquations,
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform_post_gradient_laplacian!(
    m.hyperdiffusion,
    auxHDG, 
    gradvars,
    state,
    aux,
    t,
    )
end

@inline function flux_first_order!(
    model::TestEquations,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    return nothing
end


function flux_second_order!(
    model::TestEquations,
    flux::Grad,
    state::Vars,
    gradflux::Vars,
    auxMISC::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(model.hyperdiffusion, flux, state, gradflux, auxMISC, aux, t)
end

@inline function source!(
    model::TestEquations,
    source::Vars,
    state::Vars,
    gradflux::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    #coriolis_force!(model, model.coriolis, source, state, aux, t)
    #forcing_term!(model, model.forcing, source, state, aux, t)
    #linear_drag!(model, model.turbulence, source, state, aux, t)

    return nothing
end




boundary_conditions(::TestEquations) = ()
boundary_state!(nf, ::TestEquations, _...) = nothing


"""
# DGModel constructor - move somewhere general
"""
function DGModel(model::SpatialModel{BL}) where {BL <: AbstractEquations3D}
    
    numerical_flux_first_order = model.numerics.flux # should be a function

    rhs = DGModel(
        model.balance_law,
        model.grid,
        numerical_flux_first_order,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction=HorizontalDirection(),
    )

    return rhs
end