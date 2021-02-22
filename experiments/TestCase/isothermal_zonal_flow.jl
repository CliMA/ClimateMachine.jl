#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.Orientations
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics: total_energy, air_density
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet:
    R_d, day, grav, cp_d, planet_radius, Omega, kappa_d, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 0.0
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371e6 / 125.0
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5

function init_isothermal_zonal_flow!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    φ = latitude(bl.orientation, aux)
    z = altitude(bl.orientation, param_set, aux)

    _grav::FT = grav(bl.param_set)
    _a::FT = planet_radius(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _MSLP::FT = MSLP(bl.param_set)

    u₀ = FT(20)
    T₀ = FT(300)

    shallow_atmos = false
    if shallow_atmos
        f1 = z
        f2 = FT(0)
        shear = FT(1)
    else
        f1 = z
        f2 = z / _a + z^2 / (2 * _a^2)
        shear = 1 + z / _a
    end

    u_sphere = SVector{3, FT}(u₀ * shear * cos(φ), 0, 0)
    u_init = sphr_to_cart_vec(bl.orientation, u_sphere, aux)

    prefac = u₀^2 / (_R_d * T₀)
    fac1 = prefac * f2 * cos(φ)^2
    fac2 = prefac * sin(φ)^2 / 2
    fac3 = _grav * f1 / (_R_d * T₀)
    exparg = fac1 - fac2 - fac3
    p = _MSLP * exp(exparg)

    ρ = air_density(bl.param_set, T₀, p)

    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin = u_init' * u_init / 2

    state.ρ = ρ
    state.ρu = ρ * u_init
    state.energy.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T₀)
end

function config_isothermal_zonal_flow(
    FT,
    poly_order,
    cutoff_order,
    resolution,
    ref_state,
)
    # Set up a reference state for linearization of equations

    domain_height = FT(10e3)

    # Set up the atmosphere model
    exp_name = "IsothermalZonalFlow"

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = init_isothermal_zonal_flow!,
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        moisture = DryModel(),
        source = (Gravity(),),
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_isothermal_zonal_flow!;
        model = model,
        numerical_flux_first_order = RoeNumericalFlux(),
        Ncutoff = cutoff_order,
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
    poly_order = 5                           # discontinuous Galerkin polynomial order
    cutoff_order = 4
    n_horz = 10                              # horizontal element number
    n_vert = 5                               # vertical element number
    timestart = FT(0)                        # start time (s)
    timeend = FT(3600)                       # end time (s)

    # set up the reference state for linearization of equations
    temp_profile_ref = IsothermalProfile(param_set, FT(300))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up driver configuration
    driver_config = config_isothermal_zonal_flow(
        FT,
        poly_order,
        cutoff_order,
        (n_horz, n_vert),
        ref_state,
    )

    # Set up experiment
    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = false,
        discrete_splitting = true,
    )
    CFL = FT(0.4)
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        init_on_cpu = true,
        ode_solver_type = ode_solver_type,
        CFL_direction = HorizontalDirection(),
    )

    # save the initial condition for testing
    Q0 = copy(solver_config.Q)

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

    relative_error = norm(solver_config.Q .- Q0) / norm(Q0)
    @info "Relative error = $relative_error"
    @test relative_error < 1e-7

end

main()
