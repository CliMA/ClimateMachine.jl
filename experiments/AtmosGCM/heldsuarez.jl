#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.ColumnwiseLUSolver: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.MoistThermodynamics:
    air_temperature, internal_energy, air_pressure
using ClimateMachine.VariableTemplates

using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMAParameters
using CLIMAParameters.Planet: R_d, day, grav, cp_d, cv_d, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct HeldSuarezDataConfig{FT}
    T_ref::FT
end

function init_heldsuarez!(bl, state, aux, coords, t)
    FT = eltype(state)

    # Set initial state to reference state with random perturbation
    rnd = FT(1.0 + rand(Uniform(-1e-3, 1e-3)))
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = rnd * aux.ref_state.ρe

    nothing
end

function config_heldsuarez(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref = DecayingTemperatureProfile{FT}(param_set)
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up a Rayleigh sponge to dampen flow at the top of the domain
    domain_height::FT = 30e3               # distance between surface and top of atmosphere (m)
    z_sponge::FT = 12e3                    # height at which sponge begins (m)
    α_relax::FT = 1 / 60 / 15              # sponge relaxation rate (1/s)
    exp_sponge = 2                         # sponge exponent for squared-sinusoid profile
    u_relax = SVector(FT(0), FT(0), FT(0)) # relaxation velocity (m/s)
    sponge = RayleighSponge{FT}(
        domain_height,
        z_sponge,
        α_relax,
        u_relax,
        exp_sponge,
    )

    # Set up the atmosphere model
    exp_name = "HeldSuarez"
    T_ref::FT = 255        # reference temperature for Held-Suarez forcing (K)
    τ_hyper::FT = 4 * 3600 # hyperdiffusion time scale in (s)
    c_smag::FT = 0.21      # Smagorinsky coefficient
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = SmagorinskyLilly(c_smag),
        hyperdiffusion = StandardHyperDiffusion(τ_hyper),
        moisture = DryModel(),
        source = (Gravity(), Coriolis(), held_suarez_forcing!, sponge),
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
    _p0 = 1.01325e5
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

function config_diagnostics(FT, driver_config)
    interval = "100000steps" # chosen to allow a single diagnostics collection

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

    dgngrp = setup_dump_state_and_aux_diagnostics(
        interval,
        driver_config.name,
        interpol = interpol,
        project = true,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    # Driver configuration parameters
    FT = Float32                             # floating type precision
    poly_order = 5                           # discontinuous Galerkin polynomial order
    n_horz = 5                               # horizontal element number
    n_vert = 5                               # vertical element number
    n_days = 120                             # experiment day number
    timestart = FT(0)                        # start time (s)
    timeend = FT(n_days * day(param_set))    # end time (s)

    # Set up driver configuration
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))

    # Set up experiment
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = 0.2,
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
