using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.Diagnostics
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: R_d, day, grav, cp_d, cv_d, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct HeldSuarezDataConfig{FT}
    T_ref::FT
end

function init_heldsuarez!(bl, state, aux, coords, t)
    FT = eltype(state)

    # Parameters need to set initial state
    T_ref = bl.data_config.T_ref
    temp_profile = IsothermalProfile(T_ref)

    # Calculate the initial state variables
    T, p = temp_profile(bl.orientation, aux)
    thermo_state = PhaseDry_given_pT(p, T, bl.param_set)
    ρ = air_density(thermo_state)
    e_int = internal_energy(thermo_state)
    e_pot = gravitational_potential(bl.orientation, aux)

    # Set initial state with random perturbation
    rnd = FT(1.0 + rand(Uniform(-1e-3, 1e-3)))
    state.ρ = ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = rnd * state.ρ * (e_int + e_pot)

    nothing
end

function config_heldsuarez(FT, poly_order, resolution)
    exp_name = "HeldSuarez"

    # Parameters
    T_ref::FT = 255
    Rh_ref::FT = 0
    domain_height::FT = 30e3
    turb_visc::FT = 0 # no visc. here

    # Set up a reference state for linearization
    temp_profile_ref = IsothermalProfile(T_ref)
    ref_state = HydrostaticState(temp_profile_ref, Rh_ref)

    # Rayleigh sponge to dampen flow at the top of the domain
    z_sponge = FT(15e3) # height at which sponge begins
    α_relax = FT(1 / 60 / 15) # sponge relaxation rate in (1/seconds)
    u_relax = SVector(FT(0), FT(0), FT(0)) # relaxation velocity
    exp_sponge = 2 # sponge exponent for squared-sinusoid profile
    sponge = RayleighSponge{FT}(
        domain_height,
        z_sponge,
        α_relax,
        u_relax,
        exp_sponge,
    )

    # Set up the atmosphere model
    model = AtmosModel{FT}(
        AtmosGCMConfigType;
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(turb_visc),
        moisture = DryModel(),
        source = (Gravity(), Coriolis(), held_suarez_forcing!, sponge),
        init_state = init_heldsuarez!,
        data_config = HeldSuarezDataConfig(T_ref),
        param_set = param_set,
    )

    config = CLIMA.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        init_heldsuarez!;
        model = model,
    )

    return config
end

function held_suarez_forcing!(bl, source, state, diffusive, aux, t::Real)
    FT = eltype(state)

    # Parameters
    T_ref = bl.data_config.T_ref

    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe

    coord = aux.coord
    e_int = internal_energy(bl.moisture, bl.orientation, state, aux)
    T = air_temperature(e_int, bl.param_set)
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
    λ = longitude(bl.orientation, aux)
    φ = latitude(bl.orientation, aux)
    z = altitude(bl.orientation, aux)
    scale_height = _R_d * T_ref / _grav
    σ = exp(-z / scale_height)

    # TODO: use
    #  p = air_pressure(T, ρ)
    #  σ = p/p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor

    # Apply Held-Suarez forcing
    source.ρu -= k_v * projection_tangential(bl.orientation, aux, ρu)
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end

function config_diagnostics(FT, driver_config)
    interval = 1000 # in time steps

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(10), FT(10), FT(1000))
    interpol =
        CLIMA.InterpolationConfiguration(driver_config, boundaries, resolution)

    dgngrp = setup_dump_state_and_aux_diagnostics(
        interval,
        driver_config.name,
        interpol = interpol,
        project = true,
    )
    return CLIMA.DiagnosticsConfiguration([dgngrp])
end

function main()
    CLIMA.init()

    # Driver configuration parameters
    FT = Float32                        # floating type precision
    poly_order = 5                      # discontinuous Galerkin polynomial order
    n_horz = 5                          # horizontal element number
    n_vert = 5                          # vertical element number
    days = 100                          # experiment day number
    timestart = FT(0)                   # start time (seconds)
    timeend = FT(days * 24 * 60 * 60)   # end time (seconds)

    # Set up driver configuration
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))

    # Set up ODE solver configuration
    ode_solver_type = CLIMA.IMEXSolverType(
        linear_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
    )

    # Set up experiment
    solver_config = CLIMA.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        ode_solver_type = ode_solver_type,
        Courant_number = 0.14,
        init_on_cpu = true,
        CFL_direction = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    # TODO: This callback needs to live somewhere else
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
    result = CLIMA.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

main()
