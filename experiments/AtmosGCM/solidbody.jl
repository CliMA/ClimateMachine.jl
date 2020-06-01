#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.ColumnwiseLUSolver: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.MoistThermodynamics: total_energy
using ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: R_d, day, grav, cp_d, cv_d, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_solidbody!(bl, state, aux, coords, t)
    FT = eltype(state)
    _R_d::FT = R_d(bl.param_set)

    # Set initial state to reference state
    z = altitude(bl.orientation, bl.param_set, aux)
    temp_profile = DecayingTemperatureProfile{FT}(bl.param_set)
    T, p = temp_profile(bl.param_set, z)
    ρ = p / (_R_d * T)
    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    
    state.ρ = ρ 
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)

    nothing
end

function config_solidbody(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref = IsothermalProfile(param_set, FT(290))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "SolidBody"
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state, 
        turbulence = ConstantViscosityWithDivergence(FT(0)), 
        hyperdiffusion = NoHyperDiffusion(),
        moisture = DryModel(),
        source = (Gravity(), Coriolis()),
        init_state_conservative = init_solidbody!,
    )

    domain_height = FT(45e3)
    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_solidbody!;
        model = model,
    )

    return config
end

function config_diagnostics(FT, driver_config)
    interval = "100000steps" # chosen to allow a single diagnostics collection

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(5), FT(5), FT(1000)) # in (deg, deg, m)
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    dgngrp = setup_dump_state_and_aux_diagnostics(
        interval,
        driver_config.name,
        interpol = interpol,
        project = true,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 5                           # discontinuous Galerkin polynomial order
    n_horz = 3                               # horizontal element number
    n_vert = 3                               # vertical element number
    timestart = FT(0)                        # start time (s)
    timeend = FT(2*60*60)                     # end time (s)

    # Set up driver configuration
    driver_config = config_solidbody(FT, poly_order, (n_horz, n_vert))

    # Set up experiment
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = 0.005,
        CFL_direction = HorizontalDirection(),
        diffdir = EveryDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        @views begin
          solver_config.Q.data[:, 1, :] .-= solver_config.dg.state_auxiliary.data[:, 8, :]
          solver_config.Q.data[:, 5, :] .-= solver_config.dg.state_auxiliary.data[:, 11, :]
          Filters.apply!(
            solver_config.Q,
            1:size(solver_config.Q, 2),
            solver_config.dg.grid,
            filter,
          )
          solver_config.Q.data[:, 1, :] .+= solver_config.dg.state_auxiliary.data[:, 8, :]
          solver_config.Q.data[:, 5, :] .+= solver_config.dg.state_auxiliary.data[:, 11, :]
        end
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

main()
