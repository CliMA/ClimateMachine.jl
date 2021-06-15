#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.Mesh.Topologies
using ClimateMachine.TemperatureProfiles
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics: air_density, total_energy
import ClimateMachine.DGMethods.FVReconstructions: FVLinear

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: day, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_solid_body_rotation!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    # initial velocity profile (we need to transform the vector into the Cartesian
    # coordinate system)
    u_0::FT = 0
    u_sphere = SVector{3, FT}(u_0, 0, 0)
    u_init = sphr_to_cart_vec(bl.orientation, u_sphere, aux)
    e_kin::FT = 0.5 * sum(abs2.(u_init))

    # Assign state variables
    state.ρ = aux.ref_state.ρ
    state.ρu = u_init
    state.energy.ρe = aux.ref_state.ρe + state.ρ * e_kin

    nothing
end

function config_solid_body_rotation(
    FT,
    poly_order,
    fv_reconstruction,
    resolution,
    ref_state,
)

    # Set up the atmosphere model
    exp_name = "SolidBodyRotation"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        #hyperdiffusion = DryBiharmonic(FT(8 * 3600)),
        moisture = DryModel(),
    )
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        physics;
        init_state_prognostic = init_solid_body_rotation!,
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
        numerical_flux_first_order = RoeNumericalFlux(),
        fv_reconstruction = HBFVReconstruction(model, fv_reconstruction),
        #grid_stretching = (SingleExponentialStretching(FT(2.0)),),
    )

    return config
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = (5, 0)                     # discontinuous Galerkin polynomial order
    n_horz = 8                              # horizontal element number
    n_vert = 20                               # vertical element number
    timestart::FT = 0                        # start time (s)
    timeend::FT = 3600    # end time (s)
    fv_reconstruction = FVLinear()

    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))
    ref_state = HydrostaticState(temp_profile_ref; subtract_off = false)

    # Set up driver configuration
    driver_config = config_solid_body_rotation(
        FT,
        poly_order,
        fv_reconstruction,
        (n_horz, n_vert),
        ref_state,
    )

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )

    CFL = FT(0.5) # target acoustic CFL number

    # time step is computed such that the horizontal acoustic Courant number is CFL
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = EveryDirection(),
        diffdir = HorizontalDirection(),
    )

    # initialize using a different ref state (mega-hack)
    temp_profile_init =
        DecayingTemperatureProfile{FT}(param_set, FT(280), FT(230), FT(9e3))
    init_ref_state = HydrostaticState(temp_profile_init; subtract_off = false)

    init_driver_config = config_solid_body_rotation(
        FT,
        poly_order,
        fv_reconstruction,
        (n_horz, n_vert),
        init_ref_state,
    )
    init_solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        init_driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = EveryDirection(),
        diffdir = HorizontalDirection(),
    )

    # initialization
    solver_config.Q .= init_solver_config.Q

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 20
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            # driver_config.bl,
            solver_config.dg.grid,
            filter,
            # filter perturbations from the initial state
            state_auxiliary = init_solver_config.dg.state_auxiliary,
            direction = HorizontalDirection(),
        )
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = false,
    )

    relative_error =
        norm(solver_config.Q .- init_solver_config.Q) /
        norm(init_solver_config.Q)
    @info "Relative error = $relative_error"
    @test relative_error < 1e-9
end

function config_diagnostics(FT, driver_config)
    interval = "0.5shours" # chosen to allow diagnostics every 30 simulated minutes

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info

    # Setup diagnostic grid(s)

    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]

    lats = collect(range(boundaries[1, 1], boundaries[2, 1], step = FT(2)))

    lons = collect(range(boundaries[1, 2], boundaries[2, 2], step = FT(2)))

    lvls = collect(range(
        boundaries[1, 3],
        boundaries[2, 3],
        step = FT(1000), # in m
    ))

    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config.grid.topology,
        driver_config,
        boundaries,
        [lats, lons, lvls];
        nr_toler = FT(1e-7),
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
