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

    #φ = latitude(bl, aux)
    #λ = longitude(bl, aux)
    #z = altitude(bl, aux)

    ## parameters 
    #_grav::FT = grav(bl.param_set)
    #_R_d::FT = R_d(bl.param_set)
    #_Ω::FT = Omega(bl.param_set)
    #_a::FT = planet_radius(bl.param_set)
    #
    #T_E::FT = 310
    #T_P::FT = 240
    #T_0::FT = 0.5 * (T_E + T_P)
    #Γ::FT = 0.005
    #b::FT = 2
    #K::FT = 3
    #p_0::FT = 1e5
    #M_v::FT = 0.608
    #z_p::FT = 15e3
    #λ_p::FT = π/9
    #φ_p::FT = 2*π/9
    #R_p::FT = _a/10
    #u_p::FT = 1
    #q_0::FT = 0.018
    #φ_w::FT = 2*π/9
    #p_w::FT = 0.34e5
    #p_t::FT = 0.1e5
    #q_t::FT = 1e-12 

    ## convenience functions for temperature and pressure
    #I_T::FT = cos(φ)^K - K/(K+2) * cos(φ)^(K+2) 
    #τ_z_1::FT = exp(Γ*z/T_0)
    #τ_z_2::FT = 1 - 2*(z*_grav/b/_R_d/T_0)^2
    #τ_z_3::FT = exp(-(z*_grav/b/_R_d/T_0)^2)
    #τ_1::FT = 1/T_0 * τ_z_1 + (T_0-T_P)/T_0/T_P * τ_z_2 * τ_z_3 
    #τ_2::FT = 0.5 * (K+2) * (T_E-T_P)/T_E/T_P * τ_z_2 * τ_z_3 
    #τ_int_1::FT = 1/Γ * (τ_z_1-1) + z * (T_0-T_P)/T_0/T_P * τ_z_2 
    #τ_int_2::FT = 0.5 * (K+2) * (T_E-T_P)/T_E/T_P * z * τ_z_3

    ## temperature, pressure, specific humidity, density
    #T_v::FT = 1 / (τ_1 - τ_2 * I_T) 
    #p::FT = p_0 * exp(-_grav/_R_d * (τ_int_1 - τ_int_2 * I_T))
    #η::FT = p/p_0
    ##q::FT = q_0 * exp(-(φ/φ_w)^4) * exp(-((η-1)*p_0/p_w)^2)
    ##if η > p_t/p_s
    ##  q = q_t
    ##end
    #q::FT = 0
    #T::FT = T_v / (1 + M_v * q)
    #ρ = air_density(bl.param_set, T, p)

    ## zonal velocity
    #U::FT = _grav*K/_a * τ_int_2 * (cos(φ)^(K-1) - cos(φ)^(K+1)) * T_v
    #u_ref::FT = -_Ω*_a*cos(φ) + sqrt((_Ω*_a*cos(φ))^2 + _a*cos(φ)*U)

    ## perturbations to zonal velocity
    #Z_p::FT = 1 - 3*(z/z_p)^2 + 2*(z/z_p)^3 
    #if z <= z_p
    #  Z_p = FT(0)
    #end
    #R::FT = _a*acos( sin(φ)*sin(φ_p) + cos(φ)*cos(φ_p)*cos(λ-λ_p) ) 
    #u′::FT = u_p * Z_p * exp(-(R/R_p)^2)
    #if R < R_p
    #  u′ = FT(0)
    #end
    #u = u_ref + u′
    #v::FT = 0
    #w::FT = 0
    #trafo = SMatrix{3, 3, FT, 9}(0, 0, 0, 0, 0, 0, -sin(λ), cos(λ), 0)
    #u_sphere = SVector{3, FT}(w, v, u)
    #u_cart = trafo * u_sphere
    #
    ## potential & kinetic energy
    #e_pot = gravitational_potential(bl.orientation, aux)
    #e_kin::FT = 0.5 * sum(abs2.(u_cart))

    #state.ρ = ρ
    #state.ρu = ρ * u_cart 
    #state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)
    
    ρ = aux.ref_state.ρ
    state.ρ = ρ
    state.ρu = ρ * SVector{3, FT}(0, 0, 0)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, aux.ref_state.T)

    nothing
end

function config_moist_baroclinic_wave(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref = IsothermalProfile(param_set, FT(300.0))
    ref_state = HydrostaticState(temp_profile_ref)

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
    n_vert = 5                              # vertical element number
    n_days::FT = 30 
    timestart::FT = 0                        # start time (s)
    #timeend::FT = n_days * day(param_set)   # end time (s)
    timeend::FT = 10000 

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
