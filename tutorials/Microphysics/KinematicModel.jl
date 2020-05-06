# The set-up was designed for the
# 8th International Cloud Modelling Workshop
# (ICMW, Muhlbauer et al., 2013, case 1, doi:10.1175/BAMS-D-12-00188.1)
#
# See chapter 2 in Arabas et al 2015 for setup details:
#@Article{gmd-8-1677-2015,
#AUTHOR = {Arabas, S. and Jaruga, A. and Pawlowska, H. and Grabowski, W. W.},
#TITLE = {libcloudph++ 1.0: a single-moment bulk, double-moment bulk,
#         and particle-based warm-rain microphysics library in C++},
#JOURNAL = {Geoscientific Model Development},
#VOLUME = {8},
#YEAR = {2015},
#NUMBER = {6},
#PAGES = {1677--1707},
#URL = {https://www.geosci-model-dev.net/8/1677/2015/},
#DOI = {10.5194/gmd-8-1677-2015}
#}

using Dates
using DocStringExtensions
using LinearAlgebra
using Logging
using MPI
using Printf
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.Grids
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MoistThermodynamics:
    gas_constants,
    PhaseEquil,
    PhasePartition_equil,
    PhasePartition,
    internal_energy,
    q_vap_saturation,
    air_temperature
using ClimateMachine.Microphysics
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, cv_v, T_0, e_int_v0, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.DGmethods:
    BalanceLaw,
    DGModel,
    LocalGeometry,
    vars_state_conservative,
    vars_state_auxiliary,
    vars_state_gradient,
    vars_state_gradient_flux,
    init_state_conservative!,
    init_state_auxiliary!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    wavespeed,
    boundary_state!,
    source!

struct KinematicModelConfig{FT}
    xmax::FT
    ymax::FT
    zmax::FT
    wmax::FT
    θ_0::FT
    p_0::FT
    p_1000::FT
    qt_0::FT
    z_0::FT
end

struct KinematicModel{FT, PS, O, M, P, S, BC, IS, DC} <: BalanceLaw
    param_set::PS
    orientation::O
    moisture::M
    precipitation::P
    source::S
    boundarycondition::BC
    init_state_conservative::IS
    data_config::DC
end

function KinematicModel{FT}(
    ::Type{AtmosLESConfigType},
    param_set::AbstractParameterSet;
    orientation::O = FlatOrientation(),
    moisture::M = nothing,
    precipitation::P = nothing,
    source::S = nothing,
    boundarycondition::BC = nothing,
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
) where {FT <: AbstractFloat, O, M, P, S, BC, IS, DC}

    @assert param_set ≠ nothing
    @assert init_state_conservative ≠ nothing

    atmos = (
        param_set,
        orientation,
        moisture,
        precipitation,
        source,
        boundarycondition,
        init_state_conservative,
        data_config,
    )

    return KinematicModel{FT, typeof.(atmos)...}(atmos...)
end

vars_state_gradient(m::KinematicModel, FT) = @vars()

vars_state_gradient_flux(m::KinematicModel, FT) = @vars()

function init_state_auxiliary!(
    m::KinematicModel,
    aux::Vars,
    geom::LocalGeometry,
)

    FT = eltype(aux)
    x, y, z = geom.coord
    dc = m.data_config

    _R_d::FT = R_d(m.param_set)
    _cp_d::FT = cp_d(m.param_set)
    _grav::FT = grav(m.param_set)

    # TODO - should R_d and cp_d here be R_m and cp_m?
    R_m, cp_m, cv_m, γ = gas_constants(m.param_set, PhasePartition(dc.qt_0))

    # Pressure profile assuming hydrostatic and constant θ and qt profiles.
    # It is done this way to be consistent with Arabas paper.
    # It's not necessarily the best way to initialize with our model variables.
    p =
        dc.p_1000 *
        (
            (dc.p_0 / dc.p_1000)^(_R_d / _cp_d) -
            _R_d / _cp_d * _grav / dc.θ_0 / R_m * (z - dc.z_0)
        )^(_cp_d / _R_d)
    aux.p = p
    aux.z = z
end

function init_state_conservative!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    m.init_state_conservative(m, state, aux, coords, t, args...)
end

function update_auxiliary_state!(
    dg::DGModel,
    m::KinematicModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(
        kinematic_model_nodal_update_auxiliary_state!,
        dg,
        m,
        Q,
        t,
        elems,
    )
    return true
end

function boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
) end

@inline function flux_second_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) end

@inline function flux_second_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    τ,
    d_h_tot,
) end

function config_kinematic_eddy(
    FT,
    N,
    resolution,
    xmax,
    ymax,
    zmax,
    wmax,
    θ_0,
    p_0,
    p_1000,
    qt_0,
    z_0,
)
    # Choose explicit solver
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    kmc = KinematicModelConfig(
        FT(xmax),
        FT(ymax),
        FT(zmax),
        FT(wmax),
        FT(θ_0),
        FT(p_0),
        FT(p_1000),
        FT(qt_0),
        FT(z_0),
    )

    # Set up the model
    model = KinematicModel{FT}(
        AtmosLESConfigType,
        param_set;
        boundarycondition = nothing,
        init_state_conservative = init_kinematic_eddy!,
        data_config = kmc,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "KinematicModel",
        N,
        resolution,
        FT(xmax),
        FT(ymax),
        FT(zmax),
        param_set,
        init_kinematic_eddy!;
        solver_type = ode_solver,
        model = model,
    )

    return config
end
