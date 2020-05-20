module Atmos

export AtmosModel,
    AtmosAcousticLinearModel, AtmosAcousticGravityLinearModel, RemainderModel

using CLIMAParameters
using CLIMAParameters.Planet: grav, cp_d
using CLIMAParameters.Atmos.SubgridScale: C_smag
using DocStringExtensions
using LinearAlgebra, StaticArrays
using ..ConfigTypes
using ..VariableTemplates
using ..MoistThermodynamics
import ..MoistThermodynamics: internal_energy
using ..MPIStateArrays: MPIStateArray
using ..Mesh.Grids:
    VerticalDirection, HorizontalDirection, min_node_distance, EveryDirection

import ClimateMachine.DGmethods:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_gradient_laplacian,
    vars_state_gradient_flux,
    vars_hyperdiffusive,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!,
    init_state_auxiliary!,
    init_state_conservative!,
    update_auxiliary_state!,
    LocalGeometry,
    lengthscale,
    resolutionmetric,
    DGModel,
    nodal_update_auxiliary_state!,
    number_state_conservative,
    num_integrals,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!
import ..DGmethods.NumericalFluxes:
    boundary_state!,
    boundary_flux_second_order!,
    normal_boundary_flux_second_order!,
    NumericalFluxFirstOrder,
    NumericalFluxGradient,
    NumericalFluxSecondOrder,
    CentralNumericalFluxHigherOrder,
    CentralNumericalFluxDivergence

import ..Courant: advective_courant, nondiffusive_courant, diffusive_courant

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modeling.
Users may over-ride prescribed default values for each field.

# Usage

    AtmosModel(
        param_set,
        orientation,
        ref_state,
        turbulence,
        hyperdiffusion,
        moisture,
        radiation,
        source,
        tracers,
        boundarycondition,
        init_state_conservative
    )


# Fields
$(DocStringExtensions.FIELDS)
"""
struct AtmosModel{FT, PS, O, RS, T, HD, M, P, R, S, TR, BC, IS, DC} <:
       BalanceLaw
    "Parameter Set (type to dispatch on, e.g., planet parameters. See CLIMAParameters.jl package)"
    param_set::PS
    "Orientation ([`ClimateMachine.FlatOrientation`](@ref)(LES in a box) or [`ClimateMachine.SphericalOrientation`](GCM))"
    orientation::O
    "Reference State (For initial conditions, or for linearisation when using implicit solvers)"
    ref_state::RS
    "Turbulence Closure (Equations for dynamics of under-resolved turbulent flows)"
    turbulence::T
    "Hyperdiffusion Model (Equations for dynamics of high-order spatial wave attenuation)"
    hyperdiffusion::HD
    "Moisture Model (Equations for dynamics of moist variables)"
    moisture::M
    "Precipitation Model (Equations for dynamics of precipitating species)"
    precipitation::P
    "Radiation Model (Equations for radiative fluxes)"
    radiation::R
    "Source Terms (Problem specific source terms)"
    source::S
    "Tracer Terms (Equations for dynamics of active and passive tracers)"
    tracers::TR
    "Boundary condition specification"
    boundarycondition::BC
    "Initial Condition (Function to assign initial values of state variables)"
    init_state_conservative::IS
    "Data Configuration (Helper field for experiment configuration)"
    data_config::DC
end


"""
    function AtmosModel{FT}()
Constructor for `AtmosModel` (where `AtmosModel <: BalanceLaw`)

"""
function AtmosModel{FT}(
    ::Type{AtmosLESConfigType},
    param_set::AbstractParameterSet;
    orientation::O = FlatOrientation(),
    ref_state::RS = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence::T = SmagorinskyLilly{FT}(0.21),
    hyperdiffusion::HD = NoHyperDiffusion(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis(), GeostrophicForcing{FT}(7.62e-5, 0, 0)),
    tracers::TR = NoTracers(),
    boundarycondition::BC = AtmosBC(),
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
) where {FT <: AbstractFloat, O, RS, T, HD, M, P, R, S, TR, BC, IS, DC}
    @assert param_set ≠ nothing
    @assert init_state_conservative ≠ nothing

    atmos = (
        param_set,
        orientation,
        ref_state,
        turbulence,
        hyperdiffusion,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        boundarycondition,
        init_state_conservative,
        data_config,
    )

    return AtmosModel{FT, typeof.(atmos)...}(atmos...)
end
function AtmosModel{FT}(
    ::Type{AtmosGCMConfigType},
    param_set::AbstractParameterSet;
    orientation::O = SphericalOrientation(),
    ref_state::RS = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence::T = SmagorinskyLilly{FT}(C_smag(param_set)),
    hyperdiffusion::HD = NoHyperDiffusion(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis()),
    tracers::TR = NoTracers(),
    boundarycondition::BC = AtmosBC(),
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
) where {FT <: AbstractFloat, O, RS, T, HD, M, P, R, S, TR, BC, IS, DC}
    @assert param_set ≠ nothing
    @assert init_state_conservative ≠ nothing
    atmos = (
        param_set,
        orientation,
        ref_state,
        turbulence,
        hyperdiffusion,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        boundarycondition,
        init_state_conservative,
        data_config,
    )

    return AtmosModel{FT, typeof.(atmos)...}(atmos...)
end


"""
    vars_state_conservative(m::AtmosModel, FT)
Conserved state variables (Prognostic Variables)
"""
function vars_state_conservative(m::AtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        turbulence::vars_state_conservative(m.turbulence, FT)
        hyperdiffusion::vars_state_conservative(m.hyperdiffusion, FT)
        moisture::vars_state_conservative(m.moisture, FT)
        radiation::vars_state_conservative(m.radiation, FT)
        tracers::vars_state_conservative(m.tracers, FT)
    end
end

"""
    vars_state_gradient(m::AtmosModel, FT)
Pre-transform gradient variables
"""
function vars_state_gradient(m::AtmosModel, FT)
    @vars begin
        u::SVector{3, FT}
        h_tot::FT
        turbulence::vars_state_gradient(m.turbulence, FT)
        hyperdiffusion::vars_state_gradient(m.hyperdiffusion, FT)
        moisture::vars_state_gradient(m.moisture, FT)
        tracers::vars_state_gradient(m.tracers, FT)
    end
end
"""
    vars_state_gradient_flux(m::AtmosModel, FT)
Post-transform gradient variables
"""
function vars_state_gradient_flux(m::AtmosModel, FT)
    @vars begin
        ∇h_tot::SVector{3, FT}
        turbulence::vars_state_gradient_flux(m.turbulence, FT)
        hyperdiffusion::vars_state_gradient_flux(m.hyperdiffusion, FT)
        moisture::vars_state_gradient_flux(m.moisture, FT)
        tracers::vars_state_gradient_flux(m.tracers, FT)
    end
end

"""
    vars_gradient_laplacian(m::AtmosModel, FT)
Pre-transform hyperdiffusive variables
"""
function vars_gradient_laplacian(m::AtmosModel, FT)
    @vars begin
        hyperdiffusion::vars_gradient_laplacian(m.hyperdiffusion, FT)
    end
end

"""
    vars_hyperdiffusive(m::AtmosModel, FT)
Post-transform hyperdiffusive variables
"""
function vars_hyperdiffusive(m::AtmosModel, FT)
    @vars begin
        hyperdiffusion::vars_hyperdiffusive(m.hyperdiffusion, FT)
    end
end

"""
    vars_state_auxiliary(m::AtmosModel, FT)
Auxiliary variables, such as vertical (stack)
integrals, coordinates, orientation information,
reference states, subcomponent auxiliary vars,
debug variables
"""
function vars_state_auxiliary(m::AtmosModel, FT)
    @vars begin
        ∫dz::vars_integrals(m, FT)
        ∫dnz::vars_reverse_integrals(m, FT)
        coord::SVector{3, FT}
        orientation::vars_state_auxiliary(m.orientation, FT)
        ref_state::vars_state_auxiliary(m.ref_state, FT)
        turbulence::vars_state_auxiliary(m.turbulence, FT)
        hyperdiffusion::vars_state_auxiliary(m.hyperdiffusion, FT)
        moisture::vars_state_auxiliary(m.moisture, FT)
        tracers::vars_state_auxiliary(m.tracers, FT)
        radiation::vars_state_auxiliary(m.radiation, FT)
    end
end
"""
    vars_integrals(m::AtmosModel, FT)
"""
function vars_integrals(m::AtmosModel, FT)
    @vars begin
        radiation::vars_integrals(m.radiation, FT)
    end
end
"""
    vars_reverse_integrals(m::AtmosModel, FT)
"""
function vars_reverse_integrals(m::AtmosModel, FT)
    @vars begin
        radiation::vars_reverse_integrals(m.radiation, FT)
    end
end

include("orientation.jl")
include("ref_state.jl")
include("turbulence.jl")
include("hyperdiffusion.jl")
include("moisture.jl")
include("precipitation.jl")
include("radiation.jl")
include("source.jl")
include("tracers.jl")
include("boundaryconditions.jl")
include("linear.jl")
include("remainder.jl")
include("courant.jl")

@doc """
    flux_first_order!(
        m::AtmosModel,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes and assembles non-diffusive fluxes in the model
equations.
""" flux_first_order!
@inline function flux_first_order!(
    m::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    ρinv = 1 / ρ
    ρu = state.ρu
    u = ρinv * ρu

    # advective terms
    flux.ρ = ρ * u
    flux.ρu = ρ * u .* u'
    flux.ρe = u * state.ρe

    # pressure terms
    p = pressure(m, m.moisture, state, aux)
    if m.ref_state isa HydrostaticState
        flux.ρu += (p - aux.ref_state.p) * I
    else
        flux.ρu += p * I
    end
    flux.ρe += u * p
    flux_radiation!(m.radiation, m, flux, state, aux, t)
    flux_moisture!(m.moisture, m, flux, state, aux, t)
    flux_tracers!(m.tracers, m, flux, state, aux, t)
end

function compute_gradient_argument!(
    atmos::AtmosModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.u = ρinv * state.ρu
    transform.h_tot = total_specific_enthalpy(atmos, atmos.moisture, state, aux)

    compute_gradient_argument!(atmos.moisture, transform, state, aux, t)
    compute_gradient_argument!(atmos.turbulence, transform, state, aux, t)
    compute_gradient_argument!(atmos.hyperdiffusion, transform, state, aux, t)
    compute_gradient_argument!(atmos.tracers, transform, state, aux, t)
end

function compute_gradient_flux!(
    atmos::AtmosModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.∇h_tot = ∇transform.h_tot

    # diffusion terms required for SGS turbulence computations
    compute_gradient_flux!(
        atmos.turbulence,
        atmos.orientation,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    # diffusivity of moisture components
    compute_gradient_flux!(atmos.moisture, diffusive, ∇transform, state, aux, t)
    compute_gradient_flux!(atmos.tracers, diffusive, ∇transform, state, aux, t)
end

function transform_post_gradient_laplacian!(
    atmos::AtmosModel,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform_post_gradient_laplacian!(
        atmos.hyperdiffusion,
        hyperdiffusive,
        hypertransform,
        state,
        aux,
        t,
    )
end

@doc """
    flux_second_order!(
        atmos::AtmosModel,
        flux::Grad,
        state::Vars,
        diffusive::Vars,
        hyperdiffusive::Vars,
        aux::Vars,
        t::Real
    )
Diffusive fluxes in AtmosModel. Viscosity, diffusivity are calculated
in the turbulence subcomponent and accessed within the diffusive flux
function. Contributions from subcomponents are then assembled (pointwise).
""" flux_second_order!
@inline function flux_second_order!(
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    d_h_tot = -D_t .* diffusive.∇h_tot
    flux_second_order!(atmos, flux, state, τ, d_h_tot)
    flux_second_order!(atmos.moisture, flux, state, diffusive, aux, t, D_t)
    flux_second_order!(
        atmos.hyperdiffusion,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )
    flux_second_order!(atmos.tracers, flux, state, diffusive, aux, t, D_t)
end

#TODO: Consider whether to not pass ρ and ρu (not state), foc BCs reasons
@inline function flux_second_order!(
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    τ,
    d_h_tot,
)
    flux.ρu += τ * state.ρ
    flux.ρe += τ * state.ρu
    flux.ρe += d_h_tot * state.ρ
end

@inline function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
    ρinv = 1 / state.ρ
    u = ρinv * state.ρu
    return abs(dot(nM, u)) + soundspeed(m, m.moisture, state, aux)
end


function update_auxiliary_state!(
    dg::DGModel,
    m::AtmosModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    state_auxiliary = dg.state_auxiliary

    if num_integrals(m, FT) > 0
        indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
        reverse_indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
    end

    nodal_update_auxiliary_state!(
        atmos_nodal_update_auxiliary_state!,
        dg,
        m,
        Q,
        t,
        elems,
    )

    return true
end

function atmos_nodal_update_auxiliary_state!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    atmos_nodal_update_auxiliary_state!(m.moisture, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.radiation, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.turbulence, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.tracers, m, state, aux, t)
end

function integral_load_auxiliary_state!(
    m::AtmosModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    integral_load_auxiliary_state!(m.radiation, integ, state, aux)
end

function integral_set_auxiliary_state!(m::AtmosModel, aux::Vars, integ::Vars)
    integral_set_auxiliary_state!(m.radiation, aux, integ)
end

function reverse_integral_load_auxiliary_state!(
    m::AtmosModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    reverse_integral_load_auxiliary_state!(m.radiation, integ, state, aux)
end

function reverse_integral_set_auxiliary_state!(
    m::AtmosModel,
    aux::Vars,
    integ::Vars,
)
    reverse_integral_set_auxiliary_state!(m.radiation, aux, integ)
end


@doc """
    init_state_auxiliary!(
        m::AtmosModel,
        aux::Vars,
        geom::LocalGeometry
        )
Initialise auxiliary variables for each AtmosModel subcomponent.
Store Cartesian coordinate information in `aux.coord`.
""" init_state_auxiliary!
function init_state_auxiliary!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
    aux.coord = geom.coord
    atmos_init_aux!(m.orientation, m, aux, geom)
    atmos_init_aux!(m.ref_state, m, aux, geom)
    atmos_init_aux!(m.turbulence, m, aux, geom)
    atmos_init_aux!(m.hyperdiffusion, m, aux, geom)
    atmos_init_aux!(m.tracers, m, aux, geom)
end

@doc """
    source!(
        m::AtmosModel,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction::Direction
    )
Computes (and assembles) source terms `S(Y)` in:
```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
""" source!
function source!(
    m::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    atmos_source!(m.source, m, source, state, diffusive, aux, t, direction)
end

@doc """
    init_state_conservative!(
        m::AtmosModel,
        state::Vars,
        aux::Vars,
        coords,
        t,
        args...)
Initialise state variables.
`args...` provides an option to include configuration data
(current use cases include problem constants, spline-interpolants)
""" init_state_conservative!
function init_state_conservative!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    m.init_state_conservative(m, state, aux, coords, t, args...)
end
end # module
