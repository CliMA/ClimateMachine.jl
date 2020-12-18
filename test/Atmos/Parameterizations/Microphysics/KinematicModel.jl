# The set-up was designed for the
# 8th International Cloud Modelling Workshop
# ([Muhlbauer2013](@cite))
#
# See chapter 2 in [Arabas2015](@cite) for setup details:

using Dates
using DocStringExtensions
using LinearAlgebra
using Logging
using MPI
using Printf
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init(diagnostics = "default")
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.Grids
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Thermodynamics:
    gas_constants,
    PhaseEquil,
    PhasePartition_equil,
    PhasePartition,
    internal_energy,
    q_vap_saturation,
    relative_humidity,
    PhaseEquil_ρTq,
    PhaseNonEquil_ρTq,
    air_temperature,
    latent_heat_fusion,
    Liquid,
    Ice,
    PhaseEquil_ρpq,
    air_pressure,
    supersaturation,
    vapor_specific_humidity

using ClimateMachine.Microphysics
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
using CLIMAParameters.Planet:
    R_d, cp_d, cv_d, cv_v, cv_l, cv_i, T_0, T_freeze, e_int_v0, e_int_i0, grav

using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end
struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liquid::L
    ice::I
    rain::R
    snow::S
end
struct EarthParameterSet{M} <: AbstractEarthParameterSet
    microphys_param_set::M
end

const microphys_param_set = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)
const param_set = EarthParameterSet(microphys_param_set)
const liquid_param_set = param_set.microphys_param_set.liquid
const ice_param_set = param_set.microphys_param_set.ice
const rain_param_set = param_set.microphys_param_set.rain
const snow_param_set = param_set.microphys_param_set.snow

using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, Hyperdiffusive,
    Flux, FirstOrder, SecondOrder, Source

using ClimateMachine.Atmos:
    Mass, Momentum, Energy, Moisture, TotalMoisture, Advect, KinematicModelPressure

import ClimateMachine.Atmos:
    eq_tends,
    atmos_nodal_init_state_auxiliary!

import ClimateMachine.BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    nodal_update_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    source!

import ClimateMachine.DGMethods: DGModel
using ClimateMachine.Mesh.Geometry: LocalGeometry

# Override Atmos eq_tends:
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Mass} =
    ()
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Momentum} =
    ()
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Moisture} =
    (Advect{PV}(),)
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Energy} =
    (Advect{PV}(), KinematicModelPressure{PV}())

eq_tends(pv::PV, m::AtmosModel, ::Flux{SecondOrder}) where {PV <: Momentum} =
    ()
eq_tends(pv::PV, m::AtmosModel, ::Flux{SecondOrder}) where {PV <: Mass} =
    ()
eq_tends(pv::PV, m::AtmosModel, ::Flux{SecondOrder}) where {PV <: Energy} =
    ()
eq_tends(pv::PV, m::AtmosModel, ::Flux{SecondOrder}) where {PV <: Moisture} =
    ()
eq_tends(pv::PV, m::AtmosModel, ::Flux{SecondOrder}) where {PV} =
    ()

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
    periodicity_x::Bool
    periodicity_y::Bool
    periodicity_z::Bool
    idx_bc_left::Int
    idx_bc_right::Int
    idx_bc_front::Int
    idx_bc_back::Int
    idx_bc_bottom::Int
    idx_bc_top::Int
end

function atmos_nodal_init_state_auxiliary!(
    m::AtmosModel,
    aux::Vars,
    tmp::Vars,
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

    T::FT = dc.θ_0 * (p / dc.p_1000)^(R_m / cp_m)
    ρ::FT = p / R_m / T

    ts = PhaseEquil_ρpq(
        m.param_set,
        ρ,
        p,
        dc.qt_0,
        false
    )

    @inbounds begin
        aux.ref_state.p = air_pressure(ts)
        aux.x_coord = x
        aux.z_coord = z
    end
end

function init_state_prognostic!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
    args...,
)
    m.problem.init_state_prognostic(m, state, aux, localgeo, t, args...)
end
boundary_conditions(::AtmosModel) = (1, 2, 3, 4, 5, 6)

function boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    bctype,
    m::AtmosModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
) end

function config_kinematic_eddy(
    ::Type{FT},
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
    periodicity_x,
    periodicity_y,
    periodicity_z,
    idx_bc_left,
    idx_bc_right,
    idx_bc_front,
    idx_bc_back,
    idx_bc_bottom,
    idx_bc_top,
) where {FT}
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
        Bool(periodicity_x),
        Bool(periodicity_y),
        Bool(periodicity_z),
        Int(idx_bc_left),
        Int(idx_bc_right),
        Int(idx_bc_front),
        Int(idx_bc_back),
        Int(idx_bc_bottom),
        Int(idx_bc_top),
    )

    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(),
            AtmosBC(),
        ),
        init_state_prognostic=init_kinematic_eddy!,
    )

    # Set up the model
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        data_config = kmc,
        source = ()
        problem = problem
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "KinematicModel",
        N,
        resolution,
        FT(xmax),
        FT(ymax),
        FT(zmax),
        param_set,
        init_kinematic_eddy!,
        boundary = (
            (Int(idx_bc_left), Int(idx_bc_right)),
            (Int(idx_bc_front), Int(idx_bc_back)),
            (Int(idx_bc_bottom), Int(idx_bc_top)),
        ),
        periodicity = (
            Bool(periodicity_x),
            Bool(periodicity_y),
            Bool(periodicity_z),
        ),
        xmin = FT(0),
        ymin = FT(0),
        zmin = FT(0),
        solver_type = ode_solver,
        model = model,
    )

    return config
end
