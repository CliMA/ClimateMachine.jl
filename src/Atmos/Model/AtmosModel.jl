module Atmos

export AtmosModel, AtmosAcousticLinearModel, AtmosAcousticGravityLinearModel

using CLIMAParameters
using CLIMAParameters.Planet: grav, cp_d
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

import ..Thermodynamics: internal_energy, total_specific_enthalpy
using ..MPIStateArrays: MPIStateArray
using ..Mesh.Grids:
    VerticalDirection,
    HorizontalDirection,
    min_node_distance,
    EveryDirection,
    Direction

using ClimateMachine.BalanceLaws
using ClimateMachine.Problems

import ClimateMachine.BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
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
    numerical_flux_first_order!
using ..DGMethods.NumericalFluxes: RoeNumericalFlux

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
        radiation,
        source,
        tracers,
        data_config,
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AtmosModel{FT, PS, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, DC} <:
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
        GeostrophicForcing{FT}(7.62e-5, 0, 0),
        turbconv_sources(turbconv)...,
    ),
    tracers::TR = NoTracers(),
    data_config::DC = nothing,
) where {FT <: AbstractFloat, ISP, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, DC}

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
    data_config::DC = nothing,
) where {FT <: AbstractFloat, ISP, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, DC}

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
        data_config,
    )

    return AtmosModel{FT, typeof.(atmos)...}(atmos...)
end

"""
    vars_state(m::AtmosModel, ::Prognostic, FT)

Conserved state variables (prognostic variables).
"""
function vars_state(m::AtmosModel, st::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        turbulence::vars_state(m.turbulence, st, FT)
        turbconv::vars_state(m.turbconv, st, FT)
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        moisture::vars_state(m.moisture, st, FT)
        radiation::vars_state(m.radiation, st, FT)
        tracers::vars_state(m.tracers, st, FT)
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
        tracers::vars_state(m.tracers, st, FT)
        radiation::vars_state(m.radiation, st, FT)
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


include("problem.jl")
include("ref_state.jl")
include("moisture.jl")
include("precipitation.jl")
include("radiation.jl")
include("source.jl")
include("tracers.jl")
include("linear.jl")
include("courant.jl")
include("filters.jl")

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
    direction,
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
    flux_first_order!(m.turbconv, m, flux, state, aux, t)
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
    compute_gradient_argument!(
        atmos.hyperdiffusion,
        atmos,
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(atmos.tracers, transform, state, aux, t)
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
    sponge_viscosity_modifier!(atmos, atmos.viscoussponge, ν, D_t, aux)
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
    flux_second_order!(atmos.turbconv, atmos, flux, state, diffusive, aux, t)
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
    ss = soundspeed(m, m.moisture, state, aux)

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
    atmos_nodal_update_auxiliary_state!(m.radiation, m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(m.tracers, m, state, aux, t)
    turbulence_nodal_update_auxiliary_state!(m.turbulence, m, state, aux, t)
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

@doc """
    init_state_auxiliary!(
        m::AtmosModel,
        aux::Vars,
        geom::LocalGeometry,
    )

Initialise auxiliary variables for each AtmosModel subcomponent.
Store Cartesian coordinate information in `aux.coord`.
""" init_state_auxiliary!
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

@doc """
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
    init_state_prognostic!(
        m::AtmosModel,
        state::Vars,
        aux::Vars,
        coords,
        t,
        args...,
    )

Initialise state variables. `args...` provides an option to include
configuration data (current use cases include problem constants,
spline-interpolants).
""" init_state_prognostic!
function init_state_prognostic!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    m.problem.init_state_prognostic(
        m.problem,
        m,
        state,
        aux,
        coords,
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
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
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
        state_conservative⁻,
        state_auxiliary⁻,
        state_conservative⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    FT = eltype(fluxᵀn)
    param_set = balance_law.param_set
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_conservative⁻.ρ
    ρu⁻ = state_conservative⁻.ρu
    ρe⁻ = state_conservative⁻.ρe
    ts⁻ = thermo_state(
        balance_law,
        balance_law.moisture,
        state_conservative⁻,
        state_auxiliary⁻,
    )

    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    p⁻ = pressure(
        balance_law,
        balance_law.moisture,
        state_conservative⁻,
        state_auxiliary⁻,
    )
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_conservative⁺.ρ
    ρu⁺ = state_conservative⁺.ρu
    ρe⁺ = state_conservative⁺.ρe
    ts⁺ = thermo_state(
        balance_law,
        balance_law.moisture,
        state_conservative⁺,
        state_auxiliary⁺,
    )

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    p⁺ = pressure(
        balance_law,
        balance_law.moisture,
        state_conservative⁺,
        state_auxiliary⁺,
    )
    c⁺ = soundspeed_air(ts⁺)

    ρ̃ = sqrt(ρ⁻ * ρ⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    c̃ = sqrt(roe_average(ρ⁻, ρ⁺, c⁻^2, c⁺^2))

    ũᵀn = ũ' * normal_vector
    ũc̃⁻ = ũ - c̃ * normal_vector
    ũc̃⁺ = ũ + c̃ * normal_vector

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
        (w1 * ũc̃⁻ + w2 * ũc̃⁺ + w3 * ũ + w4 * (Δu - Δuᵀn * normal_vector)) /
        2
    fluxᵀn.ρe -=
        (
            w1 * (h̃ - c̃ * ũᵀn) +
            w2 * (h̃ + c̃ * ũᵀn) +
            w3 * (ũ' * ũ / 2 + Φ - _T_0 * _cv_d) +
            w4 * (ũ' * Δu - ũᵀn * Δuᵀn)
        ) / 2

    if !(balance_law.tracers isa NoTracers)
        ρχ⁻ = state_conservative⁻.tracers.ρχ
        χ⁻ = ρχ⁻ / ρ⁻

        ρχ⁺ = state_conservative⁺.tracers.ρχ
        χ⁺ = ρχ⁺ / ρ⁺

        χ̃ = roe_average(ρ⁻, ρ⁺, χ⁻, χ⁺)
        Δρχ = ρχ⁺ - ρχ⁻

        wt = abs(ũᵀn) * (Δρχ - χ̃ * Δp / c̃^2)

        fluxᵀn.tracers.ρχ -= ((w1 + w2) * χ̃ + wt) / 2
    end
end
end # module
