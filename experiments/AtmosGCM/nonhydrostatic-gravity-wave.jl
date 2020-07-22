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
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics: total_energy, air_density
using ClimateMachine.VariableTemplates

using Distributions: Uniform
using LinearAlgebra
using StaticArrays

using CLIMAParameters
using CLIMAParameters.Planet:
    R_d, day, grav, cp_d, planet_radius, Omega, kappa_d, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 0.0
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371e6 / 125.0
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5


function init_nonhydrostatic_gravity_wave!(bl, state, aux, coords, t)
    FT = eltype(state)

    # grid
    φ = latitude(bl, aux)
    λ = longitude(bl, aux)
    z = altitude(bl, aux)

    # parameters
    _grav::FT = grav(bl.param_set)
    _cp::FT = cp_d(bl.param_set)
    _Ω::FT = Omega(bl.param_set)
    _a::FT = planet_radius(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _kappa::FT = kappa_d(bl.param_set)
    _p_eq::FT = MSLP(bl.param_set)

    N::FT = 0.01
    u_0::FT = 0.0
    G::FT = _grav^2 / N^2 / _cp
    T_eq::FT = 300
    Δθ::FT = 0.0
    d::FT = 5e3
    λ_c::FT = 2 * π / 3
    φ_c::FT = 0
    L_z::FT = 20e3

    # initial velocity profile (we need to transform the vector into the Cartesian
    # coordinate system)
    u_sphere = SVector{3, FT}(u_0 * cos(φ), 0, 0)
    u_init = sphr_to_cart_vec(bl.orientation, u_sphere, aux)

    # background temperature
    T_s::FT =
        G +
        (T_eq - G) *
        exp(-u_0 * N^2 / 4 / _grav^2 * (u_0 + 2 * _Ω * _a) * (cos(2 * φ) - 1))
    T_b::FT = G * (1 - exp(N^2 / _grav * z)) + T_s * exp(N^2 / _grav * z)

    # pressure
    p_s::FT =
        _p_eq *
        exp(u_0 / 4 / G / _R_d * (u_0 + 2 * _Ω * _a) * (cos(2 * φ) - 1)) *
        (T_s / T_eq)^(1 / _kappa)
    p::FT = p_s * (G / T_s * exp(-N^2 / _grav * z) + 1 - G / T_s)^(1 / _kappa)

    # background potential temperature
    θ_b::FT = T_b * (_p_eq / p)^_kappa

    # potential temperature perturbation
    r::FT = _a * acos(sin(φ_c) * sin(φ) + cos(φ_c) * cos(φ) * cos(λ - λ_c))
    s::FT = d^2 / (d^2 + r^2)
    θ′::FT = Δθ * s * sin(2 * π * z / L_z)

    # temperature perturbation
    T′::FT = θ′ * (p / _p_eq)^_kappa

    # temperature
    T::FT = T_b + T′

    # density
    ρ = air_density(bl.param_set, T_b, p)

    # potential & kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * sum(abs2.(u_init))

    state.ρ = ρ
    state.ρu = ρ * u_init
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)

    nothing
end

function config_nonhydrostatic_gravity_wave(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(300), FT(100), FT(27.5e3))
    ref_state = HydrostaticState(temp_profile_ref)

    domain_height::FT = 10e3               # distance between surface and top of atmosphere (m)

    # Set up the atmosphere model
    exp_name = "NonhydrostaticGravityWave"

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = (Gravity(),),
        init_state_prognostic = init_nonhydrostatic_gravity_wave!,
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_nonhydrostatic_gravity_wave!;
        model = model,
    )

    return config
end

function config_diagnostics(FT, driver_config)
    interval = "40000steps" # chosen to allow a single diagnostics collection

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(10), FT(10), FT(100)) # in (deg, deg, m)
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

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 4                           # discontinuous Galerkin polynomial order
    n_horz = 12                               # horizontal element number
    n_vert = 5                               # vertical element number
    timestart = FT(0)                        # start time (s)
    timeend = FT(3600)                       # end time (s)

    # Set up driver configuration
    driver_config =
        config_nonhydrostatic_gravity_wave(FT, poly_order, (n_horz, n_vert))

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

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

main()
