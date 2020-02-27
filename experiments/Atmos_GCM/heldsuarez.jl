using LinearAlgebra
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates


struct HeldSuarezDataConfig{FT}
  p_ground::FT
  T_initial::FT
  domain_height::FT
end

function init_heldsuarez!(bl, state, aux, coords, t)
    dc = bl.data_config

    FT = eltype(state)

    # TODO: change to altitude function
    r = norm(coords, 2)
    h = r - FT(planet_radius)
    # h = altitude(bl.orientation, aux)

    scale_height = R_d * dc.T_initial / grav
    p            = dc.p_ground * exp(-h / scale_height)

    state.ρ  = air_density(dc.T_initial, p)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (internal_energy(dc.T_initial) + aux.orientation.Φ)

    return nothing
end

function config_heldsuarez(FT, N, resolution)
    p_ground::FT = MSLP
    T_initial::FT = 255
    domain_height::FT = 30e3

    ref_state = HydrostaticState(IsothermalProfile(T_initial), FT(0))
    model = AtmosModel{FT}(AtmosGCMConfiguration;
                           ref_state  = ref_state,
                           turbulence = ConstantViscosityWithDivergence(FT(0)),
                           moisture   = DryModel(),
                           source     = (Gravity(), Coriolis(), held_suarez_forcing!),
                           init_state = init_heldsuarez!,
                           data_config = HeldSuarezDataConfig(p_ground, T_initial, domain_height))
    config = CLIMA.Atmos_GCM_Configuration("HeldSuarez", N, resolution,
                                           domain_height,
                                           init_heldsuarez!;
                                           model = model)

    return config
end

function held_suarez_forcing!(bl, source, state, diffusive, aux, t::Real)
    dc = bl.data_config

    FT = eltype(state)

    ρ     = state.ρ
    ρu    = state.ρu
    ρe    = state.ρe
    coord = aux.coord
    Φ     = aux.orientation.Φ
    e     = ρe / ρ
    u     = ρu / ρ
    e_int = e - u' * u / 2 - Φ
    T     = air_temperature(e_int)

    # Held-Suarez constants
    k_a       = FT(1 / (40 * day))
    k_f       = FT(1 / day)
    k_s       = FT(1 / (4 * day)) # TODO: day is actually seconds per day; should it be named better?
    ΔT_y      = FT(60)
    Δθ_z      = FT(10)
    T_equator = FT(315)
    T_min     = FT(200)

    σ_b          = FT(7 / 10)
    r            = norm(coord, 2)
    @inbounds λ  = atan(coord[2], coord[1])
    @inbounds φ  = asin(coord[3] / r)
    h            = r - FT(planet_radius)
    scale_height = R_d * FT(dc.T_initial) / grav
    σ            = exp(-h / scale_height)

    # TODO: use
    #  p = air_pressure(T, ρ)
    #  σ = p/p0
    exner_p       = σ ^ (R_d / cp_d)
    Δσ            = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil       = (T_equator - ΔT_y * sin(φ) ^ 2 - Δθ_z * log(σ) * cos(φ) ^ 2 ) * exner_p
    T_equil       = max(T_min, T_equil)
    k_T           = k_a + (k_s - k_a) * height_factor * cos(φ) ^ 4
    k_v           = k_f * height_factor

    source.ρu += -k_v * ρu
    source.ρe += -k_T * ρ * cv_d * (T - T_equil)
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 5

    # Domain resolution
    nelem_horz = 6
    nelem_vert = 8
    resolution = (nelem_horz, nelem_vert)

    t0 = FT(0)
    timeend = FT(60) # 400day

    driver_config = config_heldsuarez(FT, N, resolution)
    ode_solver_type = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config,
                                       ode_solver_type=ode_solver_type)

    # Set up the filter callback
    filterorder = 14
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(solver_config.Q, 1:size(solver_config.Q, 2),
                       solver_config.dg.grid, filter)
        nothing
    end

    result = CLIMA.invoke!(solver_config;
                           user_callbacks=(cbfilter,),
                          check_euclidean_distance=true)
end

main()
