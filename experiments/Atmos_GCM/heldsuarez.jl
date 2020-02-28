using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

# Import directional keywords (CLIMA.Mesh.Grids)
import CLIMA.Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection


const pressure_ground = MSLP
const T_init = 255.0 # unit: Kelvin 
const domain_height = 30e3 # unit: meters


function init_heldsuarez!(bl, state, aux, coords, t)
    FT            = eltype(state)
    pressure_sfc  = FT(MSLP)
    temp_init     = FT(255.0)
    radius        = FT(planet_radius)

    # Initialize the state variables of the model
    height = norm(coords, 2) - radius #TODO: altitude(bl.orientation, aux)
    scale_height = R_d * temp_init / grav
    pressure = pressure_sfc * exp(-height / scale_height)

    rnd      = FT(1.0 + rand(Uniform(-1e-6, 1e-6)))
    state.ρ  = rnd * air_density(temp_init, pressure)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (internal_energy(temp_init) + aux.orientation.Φ)

    nothing
end


function config_heldsuarez(FT, poly_order, resolution)
    name          = "HeldSuarez"
    domain_height = FT(30e3)
    turb_visc     = FT(0.0)
    
    # Reference state
    Rh_ref            = FT(0.0)
    Γ                 = FT(0.7 * grav / cp_d)
    T_sfc             = FT(300.0)
    T_min             = FT(200.0)
    temp_profile_ref  = LinearTemperatureProfile(T_min, T_sfc, Γ)
    ref_state         = HydrostaticState(temp_profile_ref, Rh_ref)

    # Rayleigh sponge 
    zsponge = FT(15e3) # begin of sponge
    τ_relax = FT(60*60) # sponge relaxation time 
    u_relax = SVector(FT(0), FT(0), FT(0)) # relaxation velocity
    rayleigh_sponge = RayleighSponge{FT}(domain_height, zsponge, 1/τ_relax, u_relax, 2)

    # Configure the model setup
    model = AtmosModel{FT}(
      AtmosGCMConfiguration;
      
      ref_state  = ref_state,
      turbulence = ConstantViscosityWithDivergence(turb_visc),
      moisture   = DryModel(),
      source     = (Gravity(), Coriolis(), held_suarez_forcing!, rayleigh_sponge),
      init_state = init_heldsuarez!
    )

    config = CLIMA.Atmos_GCM_Configuration(
      name, 
      poly_order, 
      resolution,
      domain_height,
      init_heldsuarez!;
      
      model = model
    )

    return config
end


function held_suarez_forcing!(bl, source, state, diffusive, aux, t::Real)
    global T_init

    FT = eltype(state)

    τ_ramp = FT(2 * 86400)
    ρ      = state.ρ
    ρu     = state.ρu
    ρe     = state.ρe
    coord  = aux.coord
    Φ      = aux.orientation.Φ
    e      = ρe / ρ
    u      = ρu / ρ
    e_int  = e - u' * u / 2 - Φ
    T      = air_temperature(e_int)

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
    scale_height = R_d * FT(T_init) / grav
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

    # Ramp up the forcing over time
    # rampup = FT((1 - exp(-2 * t / τ_ramp)) / (1 + exp(-2 * t / τ_ramp)))

    # TODO: bottom drag should only be applied in tangential direction
    source.ρu -= k_v * projection_tangential(bl.orientation, aux, ρu)
    source.ρe -= k_T * ρ * cv_d * (T - T_equil)
end

function main()
    CLIMA.init()

    # Driver configuration parameters
    FT            = Float32           # floating type precision
    poly_order    = 5                 # discontinuous Galerkin polynomial order
    n_horz        = 5                 # horizontal element number  
    n_vert        = 5                 # vertical element number
    days          = 3650              # experiment day number
    timestart     = FT(0)             # start time (seconds)
    timeend       = FT(days*24*60*60) # end time (seconds)
    
    # Set up driver configuration
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))
    
    # Set up ODE solver configuration
    #ode_solver_type = CLIMA.ExplicitSolverType(
    #  solver_method=LSRK144NiegemannDiehlBusch
    #)
    ode_solver_type = CLIMA.IMEXSolverType(
      linear_solver = ManyColumnLU,
      solver_method = ARK2GiraldoKellyConstantinescu
    )

    # Set up experiment
    solver_config = CLIMA.setup_solver(
      timestart, 
      timeend, 
      driver_config,
      ode_solver_type=ode_solver_type,
      Courant_number=0.05,
      forcecpu=true
    )

    # Set up user-defined callbacks
    # TODO: This callback needs to live somewhere else 
    filterorder = 14
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)#, HorizontalDirection())
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
          solver_config.Q, 
          1:size(solver_config.Q, 2),
          solver_config.dg.grid, 
          filter
        )
        nothing
    end

    # Run the model
    result = CLIMA.invoke!(
      solver_config;
      user_callbacks = (cbfilter,),
      check_euclidean_distance = true
    )
end

main()
