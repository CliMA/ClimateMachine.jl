#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics: air_density, total_energyy

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet:
    day, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_solid_body_rotation!(problem, bl, state, aux, coords, t)
    FT = eltype(state)

    # ATTENTION!: Below section is to change initial condition from reference state
    ## The initial state is chosen to be in hydrostatic balance, but differs from
    ## the reference state.
    #temp_profile = DecayingTemperatureProfile{FT}(param_set, FT(300), FT(210), FT(9e3))
    #
    ## Useful variables
    #z = altitude(bl.orientation, bl.param_set, aux)
    #T₀, p = temperature_profile(bl.param_set, z)
    #ρ = air_density(bl.param_set, T₀, p)
    #e_pot = gravitational_potential(bl.orientation, aux)
    #e_kin = FT(0)

    # Assign state variables
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    #state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T₀)
    state.ρe = aux.ref_state.ρe 

    nothing
end

function config_solid_body_rotation(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "SolidBodyRotation"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = init_solid_body_rotation!,
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        #hyperdiffusion = DryBiharmonic(FT(8 * 3600)),
        moisture = DryModel(),
        source = (Gravity(), Coriolis()),
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_solid_body_rotation!;
        model = model,
    )

    return config
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 3                           # discontinuous Galerkin polynomial order
    n_horz = 12                              # horizontal element number
    n_vert = 6                               # vertical element number
    n_days::FT = 1
    timestart::FT = 0                        # start time (s)
    timeend::FT = n_days * day(param_set)    # end time (s)

    # Set up driver configuration
    driver_config = config_solid_body_rotation(FT, poly_order, (n_horz, n_vert))

    # Set up experiment
    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = true,
        discrete_splitting = false,
    )

    CFL = FT(0.1) # target acoustic CFL number

    # time step is computed such that the horizontal acoustic Courant number is CFL
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = HorizontalDirection(),
        diffdir = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 20
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

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

function config_diagnostics(FT, driver_config)
    interval = "40000steps" # chosen to allow a single diagnostics collection

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(1), FT(1), FT(1000)) # in (deg, deg, m)
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    dgngrp = setup_atmos_default_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

main()
