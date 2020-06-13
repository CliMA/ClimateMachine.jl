#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics: total_energy, air_pressure 
using ClimateMachine.VariableTemplates

using Distributions: Uniform
using LinearAlgebra
using StaticArrays

using CLIMAParameters
using CLIMAParameters.Planet:
    R_d, day, grav, cp_d, cv_d, planet_radius, Omega, kappa_d, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5

function init_sk_nonhydrostatic_gravity_wave!(bl, state, aux, coords, t)
    FT = eltype(state)

    # grid
    x = coords[1]
    z = coords[3]
    
    # parameters
    _grav::FT = grav(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _c_v::FT = cv_d(bl.param_set)
    _c_p::FT = cp_d(bl.param_set)
    _kappa::FT = kappa_d(bl.param_set)
    _p_eq::FT = MSLP(bl.param_set)

    L::FT = 300e3
    H::FT = 10e3
    N::FT = 0.01
    u_0::FT = 20.0
    T_eq::FT = 300
    Δθ::FT = 0.01
    a::FT = 5e3

    # initial velocity profile (we need to transform the vector into the Cartesian
    # coordinate system)
    u_init = SVector{3, FT}(u_0, 0, 0)

    # background temperature, pressure, and density
    θ_b::FT = T_eq * exp(N^2 / _grav * z)
    T_b::FT = θ_b * (p / _p_eq)^_kappa
    π_b::FT = 1 - _grav / _c_p / θ_b * z
    ρ_b::FT = _p_eq / _R_d / θ_b * π_b^(_c_v / _R_d)
    p_b::FT = air_pressure(bl.param_set, T_b, ρ_b)

    # temperature perturbation
    θ′::FT = Δθ * sin(π * z / H) / (1 + ((x - 0.5*L)/a)^2)
    T′::FT = θ′ * (p_b / _p_eq)^_kappa

    # temperature
    T::FT = T_b + T′

    # potential & kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * sum(abs2.(u_init))

    state.ρ = ρ
    state.ρu = ρ * u_init
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)
    aux.θ₀ = θ_b
    aux.θ′ = θ′

    nothing
end

function config_sk_nonhydrostatic_gravity_wave(FT, poly_order, resolution, x_max, y_max, z_max)
    # Set up a reference state for linearization of equations
    temp_profile_ref = DecayingTemperatureProfile{FT}(param_set, FT(300), FT(100), FT(27.5e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "SkNonhydrostaticGravityWave"

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = SmagorinskyLilly(FT(0.0)),
        moisture = DryModel(),
        source = (Gravity(),),
        init_state_conservative = init_sk_nonhydrostatic_gravity_wave!,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        exp_name,
        poly_order,
        resolution,
        x_max, 
        y_max, 
        z_max,
        param_set,
        init_sk_nonhydrostatic_gravity_wave!;
        model = model,
        periodictiy = (true, false, false),
    )

    return config
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 4                           # discontinuous Galerkin polynomial order
    Δx::FT = 100
    Δy::FT = 250
    Δv::FT = 100
    x_max::FT = 300e3
    y_max:FT = 1e3
    z_max::FT = 10e3 
    resolution = (Δx, Δy, Δv)
    timestart = FT(0)                        # start time (s)
    timeend = FT(3600)                       # end time (s)

    # Set up driver configuration
    driver_config =
        config_sk_nonhydrostatic_gravity_wave(FT, poly_order, resolution)

    # Set up experiment
    CFL = FT(0.4)
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        CFL_direction = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    #filterorder = 1024
    #filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    #cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
    #    Filters.apply!(
    #        solver_config.Q,
    #        AtmosFilterPerturbations(driver_config.bl),
    #        solver_config.dg.grid,
    #        filter,
    #        state_auxiliary = solver_config.dg.state_auxiliary,
    #    )
    #    nothing
    #end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        #user_callbacks = (cbfilter,),
        user_callbacks = (),
        check_euclidean_distance = true,
    )
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

main()
