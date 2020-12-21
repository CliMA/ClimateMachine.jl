module Atmos

export AtmosModel,
    AtmosAcousticLinearModel,
    AtmosAcousticGravityLinearModel,
    HLLCNumericalFlux,
    RoeNumericalFlux,
    RoeNumericalFluxMoist

using UnPack
using CLIMAParameters
using CLIMAParameters.Planet: grav, cp_d, R_v, LH_v0, e_int_v0
using CLIMAParameters.Atmos.SubgridScale: C_smag
using DocStringExtensions
using LinearAlgebra, StaticArrays
using ..ConfigTypes
using ..Orientations
import ..Orientations:
    vertical_unit_vector,
    altitude,
    latitude,
    longitude,
    projection_normal,
    gravitational_potential,
    ∇gravitational_potential,
    projection_tangential

using ..VariableTemplates
using ..Thermodynamics
using ..TemperatureProfiles

using ..TurbulenceClosures
import ..TurbulenceClosures: turbulence_tensors
using ..TurbulenceConvection

import ..Thermodynamics: internal_energy
using ..MPIStateArrays: MPIStateArray
using ..Mesh.Grids:
    VerticalDirection,
    HorizontalDirection,
    min_node_distance,
    EveryDirection,
    Direction

using ..BalanceLaws
using ClimateMachine.Problems

import ..BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    eq_tends,
    flux,
    precompute,
    source,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!,
    prognostic_to_primitive!,
    primitive_to_prognostic!,
    init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.DGMethods:
    LocalGeometry, lengthscale, resolutionmetric, DGModel

import ..DGMethods.NumericalFluxes:
    boundary_state!,
    boundary_flux_second_order!,
    normal_boundary_flux_second_order!,
    NumericalFluxFirstOrder,
    NumericalFluxGradient,
    NumericalFluxSecondOrder,
    CentralNumericalFluxHigherOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxFirstOrder,
    numerical_flux_first_order!,
    NumericalFluxFirstOrder
using ..DGMethods.NumericalFluxes:
    RoeNumericalFlux,
    HLLCNumericalFlux,
    RusanovNumericalFlux,
    RoeNumericalFluxMoist

import ..Courant: advective_courant, nondiffusive_courant, diffusive_courant


"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modeling. Users may over-ride prescribed
default values for each field.

# Usage

    AtmosModel(
        param_set,
        problem,
        orientation,
        ref_state,
        turbulence,
        hyperdiffusion,
        spongelayer,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        data_config,
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AtmosModel{FT, PS, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, LF, DC} <:
       BalanceLaw
    "Parameter Set (type to dispatch on, e.g., planet parameters. See CLIMAParameters.jl package)"
    param_set::PS
    "Problem (initial and boundary conditions)"
    problem::PR
    "An orientation model"
    orientation::O
    "Reference State (For initial conditions, or for linearisation when using implicit solvers)"
    ref_state::RS
    "Turbulence Closure (Equations for dynamics of under-resolved turbulent flows)"
    turbulence::T
    "Turbulence Convection Closure (e.g., EDMF)"
    turbconv::TC
    "Hyperdiffusion Model (Equations for dynamics of high-order spatial wave attenuation)"
    hyperdiffusion::HD
    "Viscous sponge layers"
    viscoussponge::VS
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
    "Large-scale forcing (Forcing information from GCMs, reanalyses, or observations)"
    lsforcing::LF
    "Data Configuration (Helper field for experiment configuration)"
    data_config::DC
end

"""
    AtmosModel{FT}()

Constructor for `AtmosModel` (where `AtmosModel <: BalanceLaw`) for LES
and single stack configurations.
"""
function AtmosModel{FT}(
    ::Union{Type{AtmosLESConfigType}, Type{SingleStackConfigType}},
    param_set::AbstractParameterSet;
    init_state_prognostic::ISP = nothing,
    problem::PR = AtmosProblem(init_state_prognostic = init_state_prognostic),
    orientation::O = FlatOrientation(),
    ref_state::RS = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence::T = SmagorinskyLilly{FT}(0.21),
    turbconv::TC = NoTurbConv(),
    hyperdiffusion::HD = NoHyperDiffusion(),
    viscoussponge::VS = NoViscousSponge(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (
        Gravity(),
        Coriolis(),
        GeostrophicForcing(FT, 7.62e-5, 0, 0),
        turbconv_sources(turbconv)...,
    ),
    tracers::TR = NoTracers(),
    lsforcing::LF = NoLSForcing(),
    data_config::DC = nothing,
) where {
    FT <: AbstractFloat,
    ISP,
    PR,
    O,
    RS,
    T,
    TC,
    HD,
    VS,
    M,
    P,
    R,
    S,
    TR,
    LF,
    DC,
}
    @assert !any(isa.(source, Tuple))

    atmos = (
        param_set,
        problem,
        orientation,
        ref_state,
        turbulence,
        turbconv,
        hyperdiffusion,
        viscoussponge,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        lsforcing,
        data_config,
    )

    return AtmosModel{FT, typeof.(atmos)...}(atmos...)
end

"""
    AtmosModel{FT}()

Constructor for `AtmosModel` (where `AtmosModel <: BalanceLaw`) for GCM
configurations.
"""
function AtmosModel{FT}(
    ::Type{AtmosGCMConfigType},
    param_set::AbstractParameterSet;
    init_state_prognostic::ISP = nothing,
    problem::PR = AtmosProblem(init_state_prognostic = init_state_prognostic),
    orientation::O = SphericalOrientation(),
    ref_state::RS = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence::T = SmagorinskyLilly{FT}(C_smag(param_set)),
    turbconv::TC = NoTurbConv(),
    hyperdiffusion::HD = NoHyperDiffusion(),
    viscoussponge::VS = NoViscousSponge(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis(), turbconv_sources(turbconv)...),
    tracers::TR = NoTracers(),
    lsforcing::LF = NoLSForcing(),
    data_config::DC = nothing,
) where {
    FT <: AbstractFloat,
    ISP,
    PR,
    O,
    RS,
    T,
    TC,
    HD,
    VS,
    M,
    P,
    R,
    S,
    TR,
    LF,
    DC,
}

    @assert !any(isa.(source, Tuple))

    atmos = (
        param_set,
        problem,
        orientation,
        ref_state,
        turbulence,
        turbconv,
        hyperdiffusion,
        viscoussponge,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        lsforcing,
        data_config,
    )

    return AtmosModel{FT, typeof.(atmos)...}(atmos...)
end

"""
    vars_state(m::AtmosModel, ::Prognostic, FT)

Conserved state variables (prognostic variables).

!!! warning

    The order of the fields for `AtmosModel` needs to match the one for
    `AtmosLinearModel` since a shared state is used
"""
function vars_state(m::AtmosModel, st::Prognostic, FT)
    @vars begin
        # start of inclusion in `AtmosLinearModel`
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        turbulence::vars_state(m.turbulence, st, FT)
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        moisture::vars_state(m.moisture, st, FT)
        # end of inclusion in `AtmosLinearModel`
        precipitation::vars_state(m.precipitation, st, FT)
        turbconv::vars_state(m.turbconv, st, FT)
        radiation::vars_state(m.radiation, st, FT)
        tracers::vars_state(m.tracers, st, FT)
        lsforcing::vars_state(m.lsforcing, st, FT)
    end
end

function vars_state(m::AtmosModel, st::Primitive, FT)
    @vars begin
        ρ::FT
        u::SVector{3, FT}
        p::FT
        moisture::vars_state(m.moisture, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::Gradient, FT)

Pre-transform gradient variables.
"""
function vars_state(m::AtmosModel, st::Gradient, FT)
    @vars begin
        u::SVector{3, FT}
        h_tot::FT
        turbulence::vars_state(m.turbulence, st, FT)
        turbconv::vars_state(m.turbconv, st, FT)
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        moisture::vars_state(m.moisture, st, FT)
        lsforcing::vars_state(m.lsforcing, st, FT)
        precipitation::vars_state(m.precipitation, st, FT)
        tracers::vars_state(m.tracers, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::GradientFlux, FT)

Post-transform gradient variables.
"""
function vars_state(m::AtmosModel, st::GradientFlux, FT)
    @vars begin
        ∇h_tot::SVector{3, FT}
        turbulence::vars_state(m.turbulence, st, FT)
        turbconv::vars_state(m.turbconv, st, FT)
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        moisture::vars_state(m.moisture, st, FT)
        lsforcing::vars_state(m.lsforcing, st, FT)
        precipitation::vars_state(m.precipitation, st, FT)
        tracers::vars_state(m.tracers, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::GradientLaplacian, FT)

Pre-transform hyperdiffusive variables.
"""
function vars_state(m::AtmosModel, st::GradientLaplacian, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::Hyperdiffusive, FT)

Post-transform hyperdiffusive variables.
"""
function vars_state(m::AtmosModel, st::Hyperdiffusive, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::Auxiliary, FT)

Auxiliary variables, such as vertical (stack) integrals, coordinates,
orientation information, reference states, subcomponent auxiliary vars,
debug variables.
"""
function vars_state(m::AtmosModel, st::Auxiliary, FT)
    @vars begin
        ∫dz::vars_state(m, UpwardIntegrals(), FT)
        ∫dnz::vars_state(m, DownwardIntegrals(), FT)
        coord::SVector{3, FT}
        orientation::vars_state(m.orientation, st, FT)
        ref_state::vars_state(m.ref_state, st, FT)
        turbulence::vars_state(m.turbulence, st, FT)
        turbconv::vars_state(m.turbconv, st, FT)
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        moisture::vars_state(m.moisture, st, FT)
        precipitation::vars_state(m.precipitation, st, FT)
        tracers::vars_state(m.tracers, st, FT)
        radiation::vars_state(m.radiation, st, FT)
        lsforcing::vars_state(m.lsforcing, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::UpwardIntegrals, FT)
"""
function vars_state(m::AtmosModel, st::UpwardIntegrals, FT)
    @vars begin
        radiation::vars_state(m.radiation, st, FT)
        turbconv::vars_state(m.turbconv, st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::DownwardIntegrals, FT)
"""
function vars_state(m::AtmosModel, st::DownwardIntegrals, FT)
    @vars begin
        radiation::vars_state(m.radiation, st, FT)
    end
end

####
#### Forward orientation methods
####
projection_normal(bl, aux, u⃗) =
    projection_normal(bl.orientation, bl.param_set, aux, u⃗)
projection_tangential(bl, aux, u⃗) =
    projection_tangential(bl.orientation, bl.param_set, aux, u⃗)
latitude(bl, aux) = latitude(bl.orientation, aux)
longitude(bl, aux) = longitude(bl.orientation, aux)
altitude(bl, aux) = altitude(bl.orientation, bl.param_set, aux)
vertical_unit_vector(bl, aux) =
    vertical_unit_vector(bl.orientation, bl.param_set, aux)
gravitational_potential(bl, aux) = gravitational_potential(bl.orientation, aux)
∇gravitational_potential(bl, aux) =
    ∇gravitational_potential(bl.orientation, aux)

turbulence_tensors(atmos::AtmosModel, args...) =
    turbulence_tensors(atmos.turbulence, atmos, args...)

export AbstractSource
abstract type AbstractSource end

include("declare_prognostic_vars.jl") # declare prognostic variables
include("multiphysics_types.jl")      # types for multi-physics tendencies
include("tendencies_mass.jl")         # specify mass tendencies
include("tendencies_momentum.jl")     # specify momentum tendencies
include("tendencies_energy.jl")       # specify energy tendencies
include("tendencies_moisture.jl")     # specify moisture tendencies
include("tendencies_precipitation.jl")# specify precipitation tendencies
include("tendencies_tracers.jl")      # specify tracer tendencies

include("problem.jl")
include("ref_state.jl")
include("moisture.jl")
include("precipitation.jl")
include("thermo_states.jl")
include("radiation.jl")
include("source.jl")
include("tracers.jl")
include("lsforcing.jl")
include("linear.jl")
include("courant.jl")
include("filters.jl")
include("prog_prim_conversion.jl")   # prognostic<->primitive conversion

include("atmos_tendencies.jl")        # specify atmos tendencies
include("get_prognostic_vars.jl")     # get tuple of prognostic variables

precompute(atmos::AtmosModel, args, tt::Flux{FirstOrder}) =
    (ts = recover_thermo_state(atmos, args.state, args.aux),)

"""
    flux_first_order!(
        m::AtmosModel,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes and assembles non-diffusive fluxes in the model
equations.
"""
@inline function flux_first_order!(
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)

    flux_pad = SVector(1, 1, 1)
    ts = recover_thermo_state(m, state, aux)
    tend = Flux{FirstOrder}()
    args = (m, state, aux, t, ts, direction)
    flux.ρ = Σfluxes(eq_tends(Mass(), m, tend), args...) .* flux_pad
    flux.ρu = Σfluxes(eq_tends(Momentum(), m, tend), args...) .* flux_pad
    flux.ρe = Σfluxes(eq_tends(Energy(), m, tend), args...) .* flux_pad

    flux_first_order!(m.moisture, m, flux, state, aux, t, ts, direction)
    flux_first_order!(m.precipitation, m, flux, state, aux, t, ts, direction)
    flux_first_order!(m.tracers, m, flux, state, aux, t, ts, direction)
    flux_first_order!(m.turbconv, m, flux, state, aux, t, ts, direction)

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
    ts = recover_thermo_state(atmos, state, aux)
    e_tot = state.ρe * (1 / state.ρ)
    transform.h_tot = total_specific_enthalpy(ts, e_tot)

    compute_gradient_argument!(atmos.moisture, transform, state, aux, t)
    compute_gradient_argument!(atmos.precipitation, transform, state, aux, t)
    compute_gradient_argument!(atmos.turbulence, transform, state, aux, t)
    compute_gradient_argument!(
        atmos.hyperdiffusion,
        atmos,
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(atmos.tracers, transform, state, aux, t)
    compute_gradient_argument!(atmos.lsforcing, transform, state, aux, t)
    compute_gradient_argument!(atmos.turbconv, atmos, transform, state, aux, t)
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
    compute_gradient_flux!(
        atmos.lsforcing,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        atmos.precipitation,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(atmos.tracers, diffusive, ∇transform, state, aux, t)
    compute_gradient_flux!(
        atmos.turbconv,
        atmos,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
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
        atmos,
        hyperdiffusive,
        hypertransform,
        state,
        aux,
        t,
    )
end

precompute(atmos::AtmosModel, args, ::Flux{SecondOrder}) =
    (ts = recover_thermo_state(atmos, args.state, args.aux),)

"""
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
"""
@inline function flux_second_order!(
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_pad = SVector(1, 1, 1)
    tend = Flux{SecondOrder}()

    _args = (
        state = state,
        aux = aux,
        t = t,
        diffusive = diffusive,
        hyperdiffusive = hyperdiffusive,
    )

    args = merge(_args, (precomputed = precompute(atmos, _args, tend),))

    flux.ρ = Σfluxes(eq_tends(Mass(), atmos, tend), atmos, args) .* flux_pad
    flux.ρu =
        Σfluxes(eq_tends(Momentum(), atmos, tend), atmos, args) .* flux_pad
    flux.ρe = Σfluxes(eq_tends(Energy(), atmos, tend), atmos, args) .* flux_pad

    flux_second_order!(atmos.moisture, flux, atmos, args)
    flux_second_order!(atmos.precipitation, flux, atmos, args)
    flux_second_order!(atmos.hyperdiffusion, flux, args)
    flux_second_order!(atmos.tracers, flux, atmos, args)
    flux_second_order!(atmos.turbconv, flux, atmos, args)
end

@inline function wavespeed(
    m::AtmosModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρinv = 1 / state.ρ
    u = ρinv * state.ρu
    uN = abs(dot(nM, u))
    ts = recover_thermo_state(m, state, aux)
    ss = soundspeed_air(ts)

    FT = typeof(state.ρ)
    ws = fill(uN + ss, MVector{number_states(m, Prognostic()), FT})
    vars_ws = Vars{vars_state(m, Prognostic(), FT)}(ws)

    wavespeed_tracers!(m.tracers, vars_ws, nM, state, aux, t)

    return ws
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

    if number_states(m, UpwardIntegrals()) > 0
        indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
        reverse_indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
    end

    update_auxiliary_state!(nodal_update_auxiliary_state!, dg, m, Q, t, elems)

    # TODO: Remove this hook. This hook was added for implementing
    # the first draft of EDMF, and should be removed so that we can
    # rely on a single vertical element traversal. This hook allows
    # us to compute globally vertical quantities specific to EDMF
    # until we're able to remove them or somehow incorporate them
    # into a higher level hierarchy.
    update_auxiliary_state!(dg, m.turbconv, m, Q, t, elems)

    return true
end

function nodal_update_auxiliary_state!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    atmos_nodal_update_auxiliary_state!(m.moisture, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.precipitation, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.radiation, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.tracers, m, state, aux, t)
    turbconv_nodal_update_auxiliary_state!(m.turbconv, m, state, aux, t)
end

function integral_load_auxiliary_state!(
    m::AtmosModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    integral_load_auxiliary_state!(m.radiation, integ, state, aux)
    integral_load_auxiliary_state!(m.turbconv, m, integ, state, aux)
end

function integral_set_auxiliary_state!(m::AtmosModel, aux::Vars, integ::Vars)
    integral_set_auxiliary_state!(m.radiation, aux, integ)
    integral_set_auxiliary_state!(m.turbconv, m, aux, integ)
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

function atmos_nodal_init_state_auxiliary!(
    m::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
    init_aux_turbulence!(m.turbulence, m, aux, geom)
    atmos_init_aux!(m.ref_state, m, aux, tmp, geom)
    init_aux_hyperdiffusion!(m.hyperdiffusion, m, aux, geom)
    atmos_init_aux!(m.tracers, m, aux, geom)
    init_aux_turbconv!(m.turbconv, m, aux, geom)
    m.problem.init_state_auxiliary(m.problem, m, aux, geom)
end

"""
    init_state_auxiliary!(
        m::AtmosModel,
        aux::Vars,
        grid,
        direction
    )

Initialise auxiliary variables for each AtmosModel subcomponent.
Store Cartesian coordinate information in `aux.coord`.
"""
function init_state_auxiliary!(
    m::AtmosModel,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_aux!(m, m.orientation, state_auxiliary, grid, direction)

    init_state_auxiliary!(
        m,
        (m, aux, tmp, geom) ->
            atmos_init_ref_state_pressure!(m.ref_state, m, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )

    ∇p = ∇reference_pressure(m.ref_state, state_auxiliary, grid)

    init_state_auxiliary!(
        m,
        atmos_nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
        direction;
        state_temporary = ∇p,
    )
end

precompute(atmos::AtmosModel, args, tt::Source) =
    (ts = recover_thermo_state(atmos, args.state, args.aux),)

"""
    source!(
        m::AtmosModel,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction::Direction,
    )

Computes (and assembles) source terms `S(Y)` in:
```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function source!(
    m::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρu_pad = SVector(1, 1, 1)
    tend = Source()

    _args = (
        state = state,
        aux = aux,
        t = t,
        direction = direction,
        diffusive = diffusive,
    )

    args = merge(_args, (precomputed = precompute(m, _args, tend),))

    source.ρ = Σsources(eq_tends(Mass(), m, tend), m, args)
    source.ρu = Σsources(eq_tends(Momentum(), m, tend), m, args) .* ρu_pad
    source.ρe = Σsources(eq_tends(Energy(), m, tend), m, args)
    source!(m.moisture, source, m, args)
    source!(m.precipitation, source, m, args)

    atmos_source!(m.source, m, source, state, diffusive, aux, t, direction)
end

"""
    init_state_prognostic!(
        m::AtmosModel,
        state::Vars,
        aux::Vars,
        localgeo,
        t,
        args...,
    )

Initialise state variables. `args...` provides an option to include
configuration data (current use cases include problem constants,
spline-interpolants).
"""
function init_state_prognostic!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
    args...,
)
    m.problem.init_state_prognostic(
        m.problem,
        m,
        state,
        aux,
        localgeo,
        t,
        args...,
    )
end

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFlux,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert balance_law.moisture isa DryModel

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
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

    FT = eltype(fluxᵀn)
    param_set = balance_law.param_set
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ts⁻ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁻,
        state_auxiliary⁻,
    )

    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    p⁻ = air_pressure(ts⁻)
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe

    # TODO: state_auxiliary⁺ is not up-to-date
    # with state_prognostic⁺ on the boundaries
    ts⁺ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁺,
        state_auxiliary⁺,
    )

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    ρ̃ = sqrt(ρ⁻ * ρ⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    c̃ = sqrt(roe_average(ρ⁻, ρ⁺, c⁻^2, c⁺^2))

    ũᵀn = ũ' * normal_vector

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δuᵀn = Δu' * normal_vector

    w1 = abs(ũᵀn - c̃) * (Δp - ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w2 = abs(ũᵀn + c̃) * (Δp + ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w3 = abs(ũᵀn) * (Δρ - Δp / c̃^2)
    w4 = abs(ũᵀn) * ρ̃

    fluxᵀn.ρ -= (w1 + w2 + w3) / 2
    fluxᵀn.ρu -=
        (
            w1 * (ũ - c̃ * normal_vector) +
            w2 * (ũ + c̃ * normal_vector) +
            w3 * ũ +
            w4 * (Δu - Δuᵀn * normal_vector)
        ) / 2
    fluxᵀn.ρe -=
        (
            w1 * (h̃ - c̃ * ũᵀn) +
            w2 * (h̃ + c̃ * ũᵀn) +
            w3 * (ũ' * ũ / 2 + Φ - _T_0 * _cv_d) +
            w4 * (ũ' * Δu - ũᵀn * Δuᵀn)
        ) / 2

    if !(balance_law.tracers isa NoTracers)
        ρχ⁻ = state_prognostic⁻.tracers.ρχ
        χ⁻ = ρχ⁻ / ρ⁻

        ρχ⁺ = state_prognostic⁺.tracers.ρχ
        χ⁺ = ρχ⁺ / ρ⁺

        χ̃ = roe_average(ρ⁻, ρ⁺, χ⁻, χ⁺)
        Δρχ = ρχ⁺ - ρχ⁻

        wt = abs(ũᵀn) * (Δρχ - χ̃ * Δp / c̃^2)

        fluxᵀn.tracers.ρχ -= ((w1 + w2) * χ̃ + wt) / 2
    end
end

"""
    NumericalFluxFirstOrder()
        ::HLLCNumericalFlux,
        balance_law::AtmosModel,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

An implementation of the numerical flux based on the HLLC method for
the AtmosModel. For more information on this particular implementation,
see Chapter 10.4 in the provided reference below.

## References

 - [Toro2013](@cite)

"""
function numerical_flux_first_order!(
    ::HLLCNumericalFlux,
    balance_law::AtmosModel,
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
    num_state_prognostic = number_states(balance_law, Prognostic())
    param_set = balance_law.param_set

    # Extract the first-order fluxes from the AtmosModel (underlying BalanceLaw)
    # and compute normals on the positive + and negative - sides of the
    # interior facets
    flux⁻ = similar(parent(fluxᵀn), Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    fluxᵀn⁻ = flux⁻' * normal_vector

    flux⁺ = similar(flux⁻)
    fill!(flux⁺, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn⁺ = flux⁺' * normal_vector

    # Extract relevant fields and thermodynamic variables defined on
    # the positive + and negative - sides of the interior facets
    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ts⁻ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁻,
        state_auxiliary⁻,
    )

    u⁻ = ρu⁻ / ρ⁻
    c⁻ = soundspeed_air(ts⁻)

    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = air_pressure(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe
    ts⁺ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁺,
        state_auxiliary⁺,
    )

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    # Wave speeds estimates S⁻ and S⁺
    S⁻ = min(uᵀn⁻ - c⁻, uᵀn⁺ - c⁺)
    S⁺ = max(uᵀn⁻ + c⁻, uᵀn⁺ + c⁺)

    # Compute the middle wave speed S⁰ in the contact/star region
    S⁰ =
        (p⁺ - p⁻ + ρ⁻ * uᵀn⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * uᵀn⁺ * (S⁺ - uᵀn⁺)) /
        (ρ⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * (S⁺ - uᵀn⁺))

    p⁰ =
        (
            p⁺ +
            p⁻ +
            ρ⁻ * (S⁻ - uᵀn⁻) * (S⁰ - uᵀn⁻) +
            ρ⁺ * (S⁺ - uᵀn⁺) * (S⁰ - uᵀn⁺)
        ) / 2

    # Compute p * D = p * (0, n₁, n₂, n₃, S⁰)
    pD = @MVector zeros(FT, num_state_prognostic)
    if balance_law.ref_state isa HydrostaticState
        # pressure should be continuous but it doesn't hurt to average
        ref_p⁻ = state_auxiliary⁻.ref_state.p
        ref_p⁺ = state_auxiliary⁺.ref_state.p
        ref_p⁰ = (ref_p⁻ + ref_p⁺) / 2

        momentum_p = p⁰ - ref_p⁰
    else
        momentum_p = p⁰
    end

    pD[2] = momentum_p * normal_vector[1]
    pD[3] = momentum_p * normal_vector[2]
    pD[4] = momentum_p * normal_vector[3]
    pD[5] = p⁰ * S⁰

    # Computes both +/- sides of intermediate flux term flux⁰
    flux⁰⁻ =
        (S⁰ * (S⁻ * parent(state_prognostic⁻) - fluxᵀn⁻) + S⁻ * pD) / (S⁻ - S⁰)
    flux⁰⁺ =
        (S⁰ * (S⁺ * parent(state_prognostic⁺) - fluxᵀn⁺) + S⁺ * pD) / (S⁺ - S⁰)

    if 0 <= S⁻
        parent(fluxᵀn) .= fluxᵀn⁻
    elseif S⁻ < 0 <= S⁰
        parent(fluxᵀn) .= flux⁰⁻
    elseif S⁰ < 0 <= S⁺
        parent(fluxᵀn) .= flux⁰⁺
    else # 0 > S⁺
        parent(fluxᵀn) .= fluxᵀn⁺
    end
end

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFluxMoist,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    balance_law.moisture isa EquilMoist ||
        error("Must use a EquilMoist model for RoeNumericalFluxMoist")
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
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

    FT = eltype(fluxᵀn)
    param_set = balance_law.param_set
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    γ::FT = cp_d(param_set) / cv_d(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ρq_tot⁻ = state_prognostic⁻.moisture.ρq_tot

    u⁻ = ρu⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    qt⁻ = ρq_tot⁻ / ρ⁻
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe
    ρq_tot⁺ = state_prognostic⁺.moisture.ρq_tot

    u⁺ = ρu⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    qt⁺ = ρq_tot⁺ / ρ⁺
    c⁺ = soundspeed_air(ts⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    e_tot = roe_average(ρ⁻, ρ⁺, e⁻, e⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    qt = roe_average(ρ⁻, ρ⁺, qt⁻, qt⁺)
    ρ = sqrt(ρ⁻ * ρ⁺)
    e_int⁻ = internal_energy(ts⁻)
    e_int⁺ = internal_energy(ts⁺)
    e_int = roe_average(ρ⁻, ρ⁺, e_int⁻, e_int⁺)
    ts = PhaseEquil(
        param_set,
        e_int,
        ρ,
        qt,
        balance_law.moisture.maxiter,
        balance_law.moisture.tolerance,
    )
    c̃ = sqrt((γ - 1) * (h̃ - (ũ[1]^2 + ũ[2]^2 + ũ[3]^2) / 2))
    (R_m, _cp_m, _cv_m, gamma) = gas_constants(ts)
    # chosen by fair dice roll
    # guaranteed to be random
    ω = FT(π) / 3
    δ = FT(π) / 5
    random_unit_vector = SVector(sin(ω) * cos(δ), cos(ω) * cos(δ), sin(δ))
    # tangent space basis
    τ1 = random_unit_vector × normal_vector
    τ2 = τ1 × normal_vector
    ũᵀn⁻ = u⁻' * normal_vector
    ũᵀn⁺ = u⁺' * normal_vector
    ũᵀn = ũ' * normal_vector
    ũc̃⁻ = ũ + c̃ * normal_vector
    ũc̃⁺ = ũ - c̃ * normal_vector
    e_kin_pot = h̃ - _e_int_v0 * qt - _cp_m * c̃^2 / R_m
    if (numerical_flux.LM == true)
        Mach⁺ = sqrt(u⁺' * u⁺) / c⁺
        Mach⁻ = sqrt(u⁻' * u⁻) / c⁻
        Mach = (Mach⁺ + Mach⁻) / 2
        c̃_LM = c̃ * min(Mach * sqrt(4 + (1 - Mach^2)^2) / (1 + Mach^2), 1)
    else
        c̃_LM = c̃
    end
    #Standard Roe
    Λ = SDiagonal(
        abs(ũᵀn - c̃_LM),
        abs(ũᵀn),
        abs(ũᵀn),
        abs(ũᵀn),
        abs(ũᵀn + c̃_LM),
        abs(ũᵀn),
    )
    #Harten Hyman
    if (numerical_flux.HH == true)
        Λ = SDiagonal(
            max(
                abs(ũᵀn - c̃_LM),
                max(
                    0,
                    ũᵀn - c̃_LM - (u⁻' * normal_vector - c⁻),
                    u⁺' * normal_vector - c⁺ - (ũᵀn - c̃_LM),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn + c̃_LM),
                max(
                    0,
                    ũᵀn + c̃_LM - (u⁻' * normal_vector + c⁻),
                    u⁺' * normal_vector + c⁺ - (ũᵀn + c̃_LM),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
        )
    end
    if (numerical_flux.LV == true)
        #Pseudo LeVeque Fix
        δ_L_1 = max(0, ũᵀn - ũᵀn⁻)
        δ_L_2 = max(0, ũᵀn - c̃_LM - (ũᵀn⁻ - c⁻))
        δ_L_3 = max(0, ũᵀn + c̃_LM - (ũᵀn⁻ + c⁻))
        δ_R_1 = max(0, ũᵀn⁺ - ũᵀn)
        δ_R_2 = max(0, ũᵀn⁺ - c⁺ - (ũᵀn - c̃_LM))
        δ_R_3 = max(0, ũᵀn⁺ + c⁺ - (ũᵀn + c̃_LM))
        if (ũᵀn < δ_L_1 && ũᵀn > -δ_R_1)
            qa1 = ((δ_L_1 - δ_R_1) * ũᵀn + 2 * δ_L_1 * δ_R_1) / (δ_L_1 + δ_R_1)
        else
            qa1 = abs(ũᵀn)
        end
        if (ũᵀn - c̃ < δ_L_2 && ũᵀn - c̃_LM > -δ_R_2)
            qa2 =
                ((δ_L_2 - δ_R_2) * (ũᵀn - c̃_LM) + 2 * δ_L_2 * δ_R_2) /
                (δ_L_2 + δ_R_2)
        else
            qa2 = abs(ũᵀn - c̃_LM)
        end
        if (ũᵀn + c̃_LM < δ_L_3 && ũᵀn + c̃ > -δ_R_3)
            qa3 =
                ((δ_L_3 - δ_R_3) * (ũᵀn + c̃_LM) + 2 * δ_R_3 * δ_R_3) /
                (δ_L_3 + δ_R_3)
        else
            qa3 = abs(ũᵀn + c̃_LM)
        end
        Λ = SDiagonal(qa2, qa1, qa1, qa1, qa3, qa1)
    end
    if (numerical_flux.LVPP == true)
        #PosPreserving with LeVeque
        b_L = min(ũᵀn - c̃_LM, ũᵀn⁻ - c⁻)
        b_R = max(ũᵀn + c̃_LM, ũᵀn⁺ + c⁺)
        b⁻ = min(0, b_L)
        b⁺ = max(0, b_R)
        δ_L_1 = max(0, ũᵀn - b⁻)
        δ_L_2 = max(0, ũᵀn - c̃_LM - b⁻)
        δ_L_3 = max(0, ũᵀn + c̃_LM - b⁻)
        δ_R_1 = max(0, b⁺ - ũᵀn)
        δ_R_2 = max(0, b⁺ - (ũᵀn - c̃_LM))
        δ_R_3 = max(0, b⁺ - (ũᵀn + c̃_LM))
        if (ũᵀn < δ_L_1 && ũᵀn > -δ_R_1)
            qa1 = ((δ_L_1 - δ_R_1) * ũᵀn + 2 * δ_L_1 * δ_R_1) / (δ_L_1 + δ_R_1)
        else
            qa1 = abs(ũᵀn)
        end
        if (ũᵀn - c̃_LM < δ_L_2 && ũᵀn - c̃_LM > -δ_R_2)
            qa2 =
                ((δ_L_2 - δ_R_2) * (ũᵀn - c̃) + 2 * δ_L_2 * δ_R_2) /
                (δ_L_2 + δ_R_2)
        else
            qa2 = abs(ũᵀn - c̃_LM)
        end
        if (ũᵀn + c̃_LM < δ_L_3 && ũᵀn + c̃_LM > -δ_R_3)
            qa3 =
                ((δ_L_3 - δ_R_3) * (ũᵀn + c̃_LM) + 2 * δ_R_3 * δ_R_3) /
                (δ_L_3 + δ_R_3)
        else
            qa3 = abs(ũᵀn + c̃_LM)
        end
        Λ = SDiagonal(qa2, qa1, qa1, qa1, qa3, qa1)
    end

    M = hcat(
        SVector(1, ũc̃⁺[1], ũc̃⁺[2], ũc̃⁺[3], h̃ - c̃ * ũᵀn, qt),
        SVector(0, τ1[1], τ1[2], τ1[3], τ1' * ũ, 0),
        SVector(0, τ2[1], τ2[2], τ2[3], τ2' * ũ, 0),
        SVector(1, ũ[1], ũ[2], ũ[3], ũ' * ũ / 2 + Φ - _T_0 * _cv_m, 0),
        SVector(1, ũc̃⁻[1], ũc̃⁻[2], ũc̃⁻[3], h̃ + c̃ * ũᵀn, qt),
        SVector(0, 0, 0, 0, _e_int_v0, 1),
    )
    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻
    Δρq_tot = ρq_tot⁺ - ρq_tot⁻
    Δstate = SVector(Δρ, Δρu[1], Δρu[2], Δρu[3], Δρe, Δρq_tot)
    parent(fluxᵀn) .-= M * Λ * (M \ Δstate) / 2
end

end # module
