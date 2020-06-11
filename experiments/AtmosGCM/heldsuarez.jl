#!/usr/bin/env julia --project
using ClimateMachine
using ArgParse
ClimateMachine.cli()

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics:
    air_temperature, internal_energy, air_pressure, air_density, total_energy
using ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, kappa_d, day, grav, cv_d, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct HeldSuarezDataConfig{FT}
    T_ref::FT
end

function init_heldsuarez!(bl, state, aux, coords, t)
    FT = eltype(state)

    # grid
    φ = latitude(bl, aux)
    λ = longitude(bl, aux)
    z = altitude(bl, aux)

    # parameters
    _a::FT = planet_radius(bl.param_set)

    ΔT::FT = 1.0
    d::FT = 1e6
    λ_c::FT = 2*π/3
    φ_c_p::FT = π/4
    φ_c_m::FT = -π/4
    L_z::FT = 10e3
    
    # temperature and pressure
    T::FT = aux.ref_state.T
    p::FT = aux.ref_state.p

    # temperature perturbation
    r::FT = _a * acos( sin(φ_c_p) * sin(φ) + cos(φ_c_p) * cos(φ) * cos(λ - λ_c) )
    s::FT = d^2 / (d^2 + r^2)
    T′::FT = ΔT * s * sin( 2 * π * z / L_z)
    if z > FT(L_z/2)
      T′ = FT(0)
    end

    # perturbed temperature profile
    T = T + T′

    # density
    ρ = air_density(bl.param_set, T, p)

    # potential & kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0 

    state.ρ = ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)

    nothing
end

function config_heldsuarez(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref = DecayingTemperatureProfile{FT}(param_set, FT(285), FT(190), FT(9.5e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up a Rayleigh sponge to dampen flow at the top of the domain
    domain_height::FT = 20e3 # distance between surface and top of atmosphere (m)

    # Set up the atmosphere model
    exp_name = "HeldSuarez"
    T_ref::FT = 255        # reference temperature for Held-Suarez forcing (K)
    τ_hyper::FT = 4 * 3600 # hyperdiffusion time scale in (s)

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        hyperdiffusion = StandardHyperDiffusion(τ_hyper),
#        hyperdiffusion = NoHyperDiffusion(),
        moisture = DryModel(),
        source = (Gravity(), Coriolis(), held_suarez_forcing!),
        init_state_conservative = init_heldsuarez!,
        data_config = HeldSuarezDataConfig(T_ref),
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
    T_ref = bl.data_config.T_ref
    _kappa = FT(kappa_d(bl.param_set))
    _day = FT(day(bl.param_set))
    _cv_d = FT(cv_d(bl.param_set))
    _p_0 = FT(MSLP(bl.param_set)) 
    
    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe

    e_int = internal_energy(bl.moisture, bl.orientation, state, aux)
    T = air_temperature(bl.param_set, e_int)

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
    σ = p / _p_0
    exner_p = σ^_kappa
    
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor

    # Apply Held-Suarez forcing
    source.ρu -= k_v * projection_tangential(bl.orientation, bl.param_set, aux, ρu)
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end

function config_diagnostics(FT, driver_config)
    interval = "40000steps" # chosen to allow a single diagnostics collection

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

    pdgngrp = setup_atmos_refstate_perturbations(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([dgngrp, pdgngrp])
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 5                           # discontinuous Galerkin polynomial order
    n_horz = 5                               # horizontal element number
    n_vert = 20                              # vertical element number
    n_days = 5                               # experiment day number
    timestart = FT(0)                        # start time (s)
    timeend = FT(n_days * day(param_set))    # end time (s)

    # Set up driver configuration
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))

    ode_solver_type = ClimateMachine.IMEXSolverType(
        #splitting_type = HEVISplitting(),
        #implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
    )
    CFL = FT(0.2)

    # Set up experiment
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
