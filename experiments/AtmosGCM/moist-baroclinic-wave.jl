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
using ClimateMachine.Thermodynamics: total_energy, air_density
using ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: R_d, day, grav, planet_radius, Omega
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_moist_baroclinic_wave!(bl, state, aux, coords, t)
    FT = eltype(state)

    φ = latitude(bl, aux)
    λ = longitude(bl, aux)
    z = altitude(bl, aux)

    # parameters 
    _grav::FT = grav(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _Ω::FT = Omega(bl.param_set)
    _a::FT = planet_radius(bl.param_set)
    
    T_E::FT = 310
    T_P::FT = 240
    T_0::FT = 0.5 * (T_E + T_P)
    Γ::FT = 0.005
    b::FT = 2
    K::FT = 3
    p_0::FT = 1e5
    z_p::FT = 15e3
    λ_p::FT = π/9
    φ_p::FT = 2*π/9
    R_p::FT = _a/10
    u_p::FT = 1
    q_0::FT = 0.018
    φ_w::FT = 2*π/9
    p_w::FT = 0.34e5
    p_t::FT = 0.1e5
    q_t::FT = 1e-12 

    # convenience functions for temperature and pressure
    I_T::FT = cos(φ)^K - K/(K+2) * cos(φ)^(K+2) 
    τ_z_1::FT = exp(Γ*z/T_0)
    τ_z_2::FT = 1 - 2*(z*_grav/b/_R_d/T_0)^2
    τ_z_3::FT = exp(-(z*_grav/b/_R_d/T_0)^2)
    τ_1::FT = 1/T_0 * τ_z_1 + (T_0-T_P)/T_0/T_P * τ_z_2 * τ_z_3 
    τ_2::FT = 0.5 * (K+2) * (T_E-T_P)/T_E/T_P * τ_z_2 * τ_z_3 
    τ_int_1::FT = 1/Γ * (τ_z_1-1) + z * (T_0-T_P)/T_0/T_P * τ_z_3
    τ_int_2::FT = 0.5 * (K+2) * (T_E-T_P)/T_E/T_P * z * τ_z_3

    # temperature, pressure, specific humidity, density
    T::FT = 1 / (τ_1 - τ_2 * I_T) 
    p::FT = p_0 * exp(-_grav/_R_d * (τ_int_1 - τ_int_2 * I_T))
    η::FT = p/p_0
    ρ = air_density(bl.param_set, T, p)

    # zonal velocity
    U::FT = _grav*K/_a * τ_int_2 * (cos(φ)^(K-1) - cos(φ)^(K+1)) * T
    u::FT = -_Ω*_a*cos(φ) + sqrt((_Ω*_a*cos(φ))^2 + _a*cos(φ)*U)

    # potential & kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * u^2

    state.ρ = ρ
    state.ρu = ρ * SVector{3, FT}(-sin(λ)*u, cos(λ)*u, 0) 
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)
    
    nothing
end

function config_moist_baroclinic_wave(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    ref_state = NoReferenceState()

    domain_height::FT = 44e3 # distance between surface and top of atmosphere (m)

    # Set up the atmosphere model
    exp_name = "MoistBaroclinicWave"

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0.0)),
        moisture = DryModel(),
        source = (Gravity(), Coriolis(),),
        init_state_conservative = init_moist_baroclinic_wave!,
    )

    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_moist_baroclinic_wave!;
        solver_type = ode_solver,
        model = model,
    )

    return config
end

function config_diagnostics(FT, driver_config)
    interval = "100steps" # chosen to allow a single diagnostics collection

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(5), FT(5), FT(500)) # in (deg, deg, m)
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
    n_vert = 10                              # vertical element number
    n_days::FT = 2 / 86400 
    timestart::FT = 0                        # start time (s)
    timeend::FT = n_days * day(param_set)    # end time (s)

    # Set up driver configuration
    driver_config = config_moist_baroclinic_wave(FT, poly_order, (n_horz, n_vert))

    # Set up experiment
    CFL::FT = 0.8
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        CFL_direction = EveryDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        @views begin
          Filters.apply!(
              solver_config.Q,
              1:size(solver_config.Q, 2),
              solver_config.dg.grid,
              filter,
          )
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
