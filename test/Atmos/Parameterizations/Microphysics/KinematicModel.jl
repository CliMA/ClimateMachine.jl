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
using Thermodynamics:
    gas_constants,
    PhaseEquil,
    PhaseEquil_ρeq,
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
    supersaturation,
    vapor_specific_humidity,
    liquid_ice_pottemp,
    dry_pottemp

using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CloudMicrophysics.Microphysics_0M
using CloudMicrophysics.Microphysics_1M
import CloudMicrophysics
const CM1M = CloudMicrophysics.Microphysics_1M

using CLIMAParameters
using CLIMAParameters.Planet:
    R_d, cp_d, cv_d, cv_v, cv_l, cv_i, T_0, T_freeze, e_int_v0, e_int_i0, grav

using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, Hyperdiffusive

import ClimateMachine.BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    nodal_init_state_auxiliary!,
    nodal_update_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    wavespeed,
    parameter_set,
    boundary_conditions,
    boundary_state!,
    source!

import ClimateMachine.DGMethods: DGModel
using ClimateMachine.Mesh.Geometry: LocalGeometry

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

struct KinematicModel{FT, PS, O, M, P, S, BC, IS, DC} <: BalanceLaw
    param_set::PS
    orientation::O
    moisture::M
    precipitation::P
    source::S
    boundarycondition::BC
    init_state_prognostic::IS
    data_config::DC
end

parameter_set(m::KinematicModel) = m.param_set

function KinematicModel{FT}(
    ::Type{AtmosLESConfigType},
    param_set::AbstractParameterSet;
    orientation::O = FlatOrientation(),
    moisture::M = nothing,
    precipitation::P = nothing,
    source::S = nothing,
    boundarycondition::BC = nothing,
    init_state_prognostic::IS = nothing,
    data_config::DC = nothing,
) where {FT <: AbstractFloat, O, M, P, S, BC, IS, DC}

    @assert param_set ≠ nothing
    @assert init_state_prognostic ≠ nothing

    atmos = (
        param_set,
        orientation,
        moisture,
        precipitation,
        source,
        boundarycondition,
        init_state_prognostic,
        data_config,
    )

    return KinematicModel{FT, typeof.(atmos)...}(atmos...)
end

vars_state(m::KinematicModel, ::Gradient, FT) = @vars()

vars_state(m::KinematicModel, ::GradientFlux, FT) = @vars()

function nodal_init_state_auxiliary!(
    m::KinematicModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    x, y, z = geom.coord
    dc = m.data_config
    param_set = parameter_set(m)

    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    # TODO - should R_d and cp_d here be R_m and cp_m?
    R_m, cp_m, cv_m, γ = gas_constants(param_set, PhasePartition(dc.qt_0))

    # Pressure profile assuming hydrostatic and constant θ and qt profiles.
    # It is done this way to be consistent with Arabas paper.
    # It's not necessarily the best way to initialize with our model variables.
    p =
        dc.p_1000 *
        (
            (dc.p_0 / dc.p_1000)^(_R_d / _cp_d) -
            _R_d / _cp_d * _grav / dc.θ_0 / R_m * (z - dc.z_0)
        )^(_cp_d / _R_d)

    @inbounds begin
        aux.p = p
        aux.x_coord = x
        aux.z_coord = z
    end
end

function init_state_prognostic!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
    args...,
)
    m.init_state_prognostic(m, state, aux, localgeo, t, args...)
end
boundary_conditions(::KinematicModel) = (1, 2, 3, 4, 5, 6)

function boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    bctype,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
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
    periodicity_x,
    periodicity_y,
    periodicity_z,
    idx_bc_left,
    idx_bc_right,
    idx_bc_front,
    idx_bc_back,
    idx_bc_bottom,
    idx_bc_top,
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

    # Set up the model
    model = KinematicModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = init_kinematic_eddy!,
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
        model = model,
    )

    return config, ode_solver
end
