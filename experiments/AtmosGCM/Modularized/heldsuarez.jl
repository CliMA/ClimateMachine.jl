#!/usr/bin/env julia --project
using ClimateMachine
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--number-of-tracers"
    help = "Number of dummy tracers"
    metavar = "<number>"
    arg_type = Int
    default = 0
end

parsed_args = ClimateMachine.init(parse_clargs = true, custom_clargs = s)
const number_of_tracers = parsed_args["number-of-tracers"]

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
    air_pressure, air_density, air_temperature, total_energy, internal_energy
using ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet:
    MSLP, R_d, day, cp_d, cv_d, grav, Omega, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_heldsuarez!(bl, state, aux, coords, t)
    FT = eltype(state)

    # parameters 
    _a::FT = planet_radius(bl.param_set)

    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 10

    # grid
    φ = latitude(bl.orientation, aux)
    λ = longitude(bl.orientation, aux)
    z = altitude(bl.orientation, bl.param_set, aux)

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

    ## potential & kinetic energy
    e_kin::FT = 0.5 * u_cart' * u_cart

    ## Assign state variables
    state.ρ = aux.ref_state.ρ
    state.ρu = state.ρ * u_cart
    state.ρe = aux.ref_state.ρe + state.ρ * e_kin
    if number_of_tracers > 0
        state.tracers.ρχ = @SVector [FT(ii) for ii in 1:number_of_tracers]
    end

    nothing
end

function held_suarez_forcing!(
    bl,
    source,
    state,
    diffusive,
    aux,
    t::Real,
    direction,
)
    FT = eltype(state)

    # Parameters
    T_ref = FT(255)

    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe

    coord = aux.coord
    e_int = internal_energy(bl.moisture, bl.orientation, state, aux)
    T = air_temperature(bl.param_set, e_int)
    _R_d = FT(R_d(bl.param_set))
    _day = FT(day(bl.param_set))
    _grav = FT(grav(bl.param_set))
    _cp_d = FT(cp_d(bl.param_set))
    _cv_d = FT(cv_d(bl.param_set))
    _p0 = FT(MSLP(bl.param_set))

    # Held-Suarez parameters
    k_a = FT(1 / (40 * _day))
    k_f = FT(1 / _day)
    k_s = FT(1 / (4 * _day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)

    # Held-Suarez forcing
    φ = latitude(bl.orientation, aux)
    p = air_pressure(bl.param_set, T, ρ)

    #TODO: replace _p0 with dynamic surfce pressure in Δσ calculations to account
    #for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor

    # Apply Held-Suarez forcing
    source.ρu -= k_v * projection_tangential(bl, aux, ρu)
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end

function config_heldsuarez(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(275), FT(75), FT(45e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "HeldSuarez"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)

    if number_of_tracers > 0
        δ_χ = @SVector [FT(ii) for ii in 1:number_of_tracers]
        tracers = NTracers{number_of_tracers, FT}(δ_χ)
    else
        tracers = NoTracers()
    end

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        hyperdiffusion = DryBiharmonic(FT(8 * 3600)),
        moisture = DryModel(),
        source = (Gravity(), Coriolis(), held_suarez_forcing!),
        init_state_prognostic = init_heldsuarez!,
        tracers = tracers,
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_heldsuarez!;
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
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))

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
