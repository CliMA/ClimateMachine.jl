#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics:
    air_temperature, internal_energy, air_pressure
using ClimateMachine.VariableTemplates

using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMAParameters
using CLIMAParameters.Planet: R_d, day, grav, cp_d, cv_d, planet_radius, Omega, kappa_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
CLIMAParameters.planet.Omega(::EarthParameterSet) = 0.0
CLIMAParameters.planet.planet_radius(::EarthParameterSet) = 6.371 * 10^6 / 125

struct dcmip31DataConfig{FT}
    T_ref::FT
end

function init_dcmip31!(bl, state, aux, coords, t)
    FT = eltype(state)

    ϕ = latitude(bl, aux)
    λ = longitude(bl, aux)
    z = altitude(bl, aux)

    # initial velocity profile
    u_0 = FT(20)
    u = u_0 * cos(ϕ)
    v = FT(0)
    w = FT(0)

    # surface temperature
    _grav = FT(grav(bl.param_set))
    _N = FT(0.01)
    _cp = FT(cp_d(bl.param_set))
    _Ω = FT(Omega(bl.param_set))
    _a = FT(planet_radius(bl.param_set))
    G = _grav^2 / (_N^2 * _cp)
    T_eq = FT(300)
    T_s = G  + (T_eq - G) * exp( -(u_0 * _N^2) / (4 * _grav^2) * (u_0 + 2 * _Ω *_a)* (cos(2ϕ) - 1))

    # background temperature
    T_b = G * (1 - exp((_N^2 / _grav)*z)) + T_s * exp((_N^2 / _grav)*z)

    # surface pressure
    p_eq = FT(100000)  # Pa
    _R_d = FT(R_d(bl.param_set))
    _kappa = FT(kappa_d(bl.param_set))
    p_s = p_eq * exp( u_0 / (4 * G * _R_d) * (u_0 + 2 * _Ω *_a)* (cos(2ϕ) - 1)) * (T_s / T_eq)^(1/_kappa)

    # unperturbed pressure field
    p = p_s * (G / T_s * exp(-_N^2/_grav * z) + 1 - G/T_s)^(1/_kappa)

    # Background potential temperature
    θ_b = T_s * ( p_eq / p_s )^(_kappa) * exp(_N^2/_grav * z)

    # density
    ρ = p / (_R_d * T_b)

    # potential temperature perturbation
    Δθ = FT(1.0)
    d = FT(5000)
    λ_c = 2*FT(π)/3
    ϕ_c = 0
    r = _a * acos(sin(ϕ_c)*sin(ϕ) + cos(ϕ_c)*cos(ϕ)*cos(λ - λ_c))
    s = d^2 / (d^2 + r^2)
    L_z = FT(20000)
    θ' = Δθ * s * sin(2*FT(π) z / L_z)

    # Temperature perturbation
    T' = θ' * (p / p_eq)^(_kappa)

    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin = FT(0.5)*u^2

    state.ρ = ρ
    state.ρu = ρ * SVector{3, FT}(u, v, w)
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T_b + T')

    nothing
end

function config_dcmip31(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref = DecayingTemperatureProfile{FT}(param_set)
    ref_state = HydrostaticState(temp_profile_ref)

    domain_height::FT = 10e3               # distance between surface and top of atmosphere (m)

    # Set up the atmosphere model
    exp_name = "DCMIP Case 3-1"

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0.0)),
        moisture = DryModel(),
        source = (Gravity(),),
        init_state_conservative = init_dcmip31!,
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_dcmip31!;
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
    resolution = (FT(10), FT(10), FT(1000)) # in (deg, deg, m)
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
    poly_order = 5                           # discontinuous Galerkin polynomial order
    n_horz = 5                               # horizontal element number
    n_vert = 5                               # vertical element number
    timestart = FT(0)                        # start time (s)
    timeend = FT(3600)                       # end time (s)

    # Set up driver configuration
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))

    # Set up experiment
    CFL = FT(0.2)
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        init_on_cpu = true,
        CFL_direction = HorizontalDirection(),
        diffdir = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 10
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            1:size(solver_config.Q, 2),
            solver_config.dg.grid,
            filter,
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
