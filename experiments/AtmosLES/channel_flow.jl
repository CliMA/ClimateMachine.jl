using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Orientations

using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day, T_0
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.Atmos: source!, atmos_source!, altitude
import ClimateMachine.Atmos: flux_second_order!, thermo_state

"""
  Pressure driven channel flow
"""
function init_channelflow!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)
    T0::FT = 300
    T = FT(300)
    state.ρ = FT(1)
    state.ρu = SVector{3, FT}(0, 0, 0)
    if FT(1 / 2) < z < FT(3 / 2)
        state.ρu += SVector{3, FT}(rand(), rand(), rand()) ./ 10
    end
    state.ρe = state.ρ * total_energy(bl.param_set, FT(0), FT(0), T)
    if z > FT(1.8) || z < FT(0.2)
        state.tracers.ρχ = SVector{2, FT}(0.1, 1)
    else
        state.tracers.ρχ = SVector{2, FT}(0, 0)
    end
end

"""
    Pressure Gradient Forcing
"""
struct PressureGradient{FT} <: Source
    "Re_τ Friction Reynolds Number [1]"
    Re_τ::FT
    "ν Fluid Kinematic Viscosity [m²/s]"
    ν::FT
end
function atmos_source!(
    s::PressureGradient,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    Re_τ = s.Re_τ
    ν = s.ν
    u_τ = Re_τ * ν
    ρ = state.ρ
    ∂P∂x = -ρ / 2 * u_τ^2
    source.ρu -= 1 / ρ * SVector{3, eltype(state)}(∂P∂x, 0, 0)
    source.ρe -= state.ρu[1] / ρ * ∂P∂x
    return nothing
end

function config_channelflow(FT, N, resolution, xmax, ymax, zmax)
    C_smag = FT(0.23)
    δ_χ = SVector{2, FT}(5, 10)
    T_wall = FT(300)
    Re_τ = FT(590)
    ν = FT(1e-5)
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag),
        source = (PressureGradient{FT}(Re_τ, ν)),
        boundarycondition = (
            AtmosBC(
                momentum = Impenetrable(NoSlip()),
                energy = PrescribedTemperature((state, aux, t) -> T_wall),
            ),
            AtmosBC(
                momentum = Impenetrable(NoSlip()),
                energy = PrescribedTemperature((state, aux, t) -> T_wall),
            ),
        ),
        moisture = DryModel(),
        tracers = NTracers{2, FT}(δ_χ),
        init_state_prognostic = init_channelflow!,
    )
    config = ClimateMachine.AtmosLESConfiguration(
        "IsothermalChannelFlow",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_channelflow!,
        solver_type = ode_solver,
        model = model,
        grid_stretching = SVector{3}(
            DoubleSidedSingleExponentialStretching(FT(0.5)),
            NoStretching(),
            NoStretching(),
        ),
    )

    return config
end

function config_diagnostics(driver_config)
    interval = "0.1smins"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    # Assumes half channel height is == 1 
    # Re = u_τδ/ ν
    xmax = FT(4π)
    ymax = FT(4π / 3)
    zmax = FT(2)

    Npts = 32

    Δh₁ = FT(xmax / 4Npts)
    Δh₂ = FT(ymax / 2Npts)
    Δh₃ = FT(zmax / 3Npts)
    resolution = (Δh₁, Δh₂, Δh₃)
    t0 = FT(0)
    timeend = FT(0.0001)

    driver_config = config_channelflow(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = FT(1.5),
    )
    dgn_config = config_diagnostics(driver_config)
    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

main()
