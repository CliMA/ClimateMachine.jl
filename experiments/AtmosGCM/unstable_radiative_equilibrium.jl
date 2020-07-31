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
using ClimateMachine.Thermodynamics:
    air_pressure, air_density, air_temperature, total_energy, internal_energy, PhasePartition
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet:
    MSLP, R_d, day, cp_d, cv_d, grav, Omega, planet_radius

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_unstable_rad_equilibrium!(bl, state, aux, coords, t)
    FT = eltype(state)

    # parameters
    _grav::FT = grav(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _Ω::FT = Omega(bl.param_set)
    _a::FT = planet_radius(bl.param_set)
    _p_0::FT = MSLP(bl.param_set)

    k::FT = 3
    T_E::FT = 310
    T_P::FT = 240
    T_0::FT = 0.5 * (T_E + T_P)
    Γ::FT = 0.005
    A::FT = 1 / Γ
    B::FT = (T_0 - T_P) / T_0 / T_P
    C::FT = 0.5 * (k + 2) * (T_E - T_P) / T_E / T_P
    b::FT = 2
    H::FT = _R_d * T_0 / _grav

    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 1 # initial perturbation amplitude

    # Moist model params
    M_v::FT = 0.608
    p_w::FT = 34e3             ## Pressure width parameter for specific humidity
    η_crit::FT = 10 * _p_0 / p_w ## Critical pressure coordinate
    q_0::FT = 0.018            ## Maximum specific humidity (default: 0.018)
    q_t::FT = 1e-12            ## Specific humidity above artificial tropopause
    φ_w::FT = 2π / 9           ## Specific humidity latitude wind parameter

    # grid
    φ = latitude(bl.orientation, aux)
    λ = longitude(bl.orientation, aux)
    z = altitude(bl.orientation, bl.param_set, aux)
    r::FT = z + _a
    γ::FT = 1 # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

    # convenience functions for temperature and pressure
    τ_z_1::FT = exp(Γ * z / T_0)
    τ_z_2::FT = 1 - 2 * (z / b / H)^2
    τ_z_3::FT = exp(-(z / b / H)^2)
    τ_1::FT = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2::FT = C * τ_z_2 * τ_z_3
    τ_int_1::FT = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2::FT = C * z * τ_z_3
    I_T::FT =
        (cos(φ) * (1 + γ * z / _a))^k -
        k / (k + 2) * (cos(φ) * (1 + γ * z / _a))^(k + 2)

    # base state virtual temperature, pressure, specific humidity, density
    T_v::FT = (τ_1 - τ_2 * I_T)^(-1)
    p::FT = _p_0 * exp(-_grav / _R_d * (τ_int_1 - τ_int_2 * I_T))

    # deterministic velocity perturbation
    F_z::FT = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
    if z > z_t
        F_z = FT(0)
    end
    d::FT = _a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
    c3::FT = cos(π * d / 2 / d_0)^3
    s1::FT = sin(π * d / 2 / d_0)
    if 0 < d < d_0 && d != FT(_a * π)
        u′::FT =
            -16 * V_p / 3 / sqrt(3) *
            F_z *
            c3 *
            s1 *
            (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
            sin(d / _a)
        v′::FT =
            16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
            sin(d / _a)
    else
        u′ = FT(0)
        v′ = FT(0)
    end
    w′::FT = 0
    u_sphere = SVector{3, FT}(u′, v′, w′)
    u_cart = sphr_to_cart_vec(bl.orientation, u_sphere, aux)

    ## Compute moisture profile
    ## Pressure coordinate η
    ## η_crit = p_t / p_w ; p_t = 10000 hPa, p_w = 340 hPa
    η = p / _p_0
    if η > η_crit
        q_tot = q_0 * exp(-(φ / φ_w)^4) * exp(-((η - 1) * _p_0 / p_w)^2)
    else
        q_tot = q_t
    end
    phase_partition = PhasePartition(q_tot)

    ## temperature & density
    T::FT = T_v / (1 + M_v * q_tot)
    ρ::FT = air_density(bl.param_set, T, p, phase_partition)

    ## potential & kinetic energy
    e_pot::FT = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * u_cart' * u_cart
    e_tot::FT = total_energy(bl.param_set, e_kin, e_pot, T, phase_partition)

    ## Assign state variables
    state.ρ = ρ
    state.ρu = ρ * u_cart
    state.ρe = ρ * e_tot
    state.moisture.ρq_tot = ρ * q_tot

    nothing
end

function unstable_rad_equilibrium!(
    bl,
    source,
    state,
    diffusive,
    aux,
    t::Real,
    direction,
)
    FT = eltype(state)

    # Extract the state
    ρ = state.ρ
    u = state.ρu / ρ
    e = state.ρe / ρ

    coord = aux.coord
    e_int = internal_energy(bl.moisture, bl.orientation, state, aux)
    T = air_temperature(bl.param_set, e_int)
    _R_d = FT(R_d(bl.param_set))
    _day = FT(day(bl.param_set))
    _grav = FT(grav(bl.param_set))

    _cv_d = FT(cv_d(bl.param_set))
    _p0 = FT(MSLP(bl.param_set))

    # unstable_rad_equilibrium parameters
    T_ref                    = FT(260) # South pole surface temperature
	delh                     = FT(90) #Equator-pole surface temperature contrast
    k_a                      = FT(1 / (50 * _day))
    k_s                      = FT(1 / (7 * _day))
	Cdrag                    = FT(0.0e-5) #
	t_strat                  = FT(200.0)
	σ_b                      = FT(0.85)
	scale_height_ratio       = FT(3.5)
	_p0_ref                  = FT(100000)
	phi0                     = FT(0.0)  #Offset of temperature maximum from equator
    lat_tropic               = FT(32.092) # latitude within/outside surface temps are flat

    φ = latitude(bl.orientation, aux)
    p = air_pressure(bl.param_set, T, ρ)

    # Surface Drag
    u_hor = projection_tangential(bl, aux, state.ρu) / ρ
    vg_norm = sqrt(u_hor' * u_hor)

    σ = p / _p0
    σ_ref = p / _p0_ref
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)

    source.ρu -= Cdrag * height_factor * vg_norm * projection_tangential(bl, aux, state.ρu)

    # Diabatic Forcing: relax to gray equilibrium (cf. Goody and Yung, pp. 392 ff.)

    # prescribe surface temperatures
    latitude_dependence = cos(φ)*cos(φ) + 2. * sin(phi0*π / 180.0) * ( 1 + sin(φ) )
    t_sfc = T_ref + delh * latitude_dependence

    # make surface temperature flat inside/outside lat_tropic
    match = T_ref + delh * cos(lat_tropic*π/180.0)^2

    flatt = false
    flatxt = false

    if (flatt & (abs(φ*180.0 / π) < lat_tropic) )
        t_sfc = match
    elseif (flatxt & (abs(φ*180.0 / π) > lat_tropic) )
        t_sfc = match
    end

    #t_sfc = temp  * ( _p0 / p )^(kappa)

    # latitude-dependent optical thickness
    optical_thickness = (t_sfc / t_strat)^4 - 1.0

    # Newtonian damping coefficient
    k_T  = k_a + (k_s - k_a) * height_factor * cos( φ - phi0 * π / 180.0 )^8

    # equilibrium temperature profile for relaxation
    T_equil =  t_strat * (1.0 + optical_thickness * σ_ref^scale_height_ratio)^( 1.0 / 4.0 )

    # relax towards background state
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end

function config_unstable_rad_equilibrium(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(275), FT(75), FT(45e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "UnstableRE"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        hyperdiffusion = EquilMoistBiharmonic(FT(8 * 3600)),
        #hyperdiffusion = DryBiharmonic(FT(8 * 3600)),
        #moisture = DryModel(),
        moisture = EquilMoist{FT}(),
        source = (Gravity(), Coriolis(), unstable_rad_equilibrium!),
        init_state_prognostic = init_unstable_rad_equilibrium!,
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_unstable_rad_equilibrium!;
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
    n_days::FT = 5
    timestart::FT = 0                        # start time (s)
    timeend::FT = n_days * day(param_set)    # end time (s)

    # Set up driver configuration
    driver_config = config_unstable_rad_equilibrium(FT, poly_order, (n_horz, n_vert))

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

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
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
