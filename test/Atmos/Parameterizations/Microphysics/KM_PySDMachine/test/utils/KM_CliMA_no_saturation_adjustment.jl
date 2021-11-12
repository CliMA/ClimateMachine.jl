using StaticArrays
using Statistics
using Test

using ..PySDMCallbacks
using ..PySDMCall


__init__()

function vars_state(m::KinematicModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
    end
end

function vars_state(m::KinematicModel, ::Auxiliary, FT)
    @vars begin
        # defined in init_state_auxiliary
        p::FT
        x_coord::FT
        z_coord::FT
        # defined in update_aux
        u::FT
        w::FT
        q_tot::FT # total water specific humidity
        q_vap::FT # water vapor specific humidity
        q_liq::FT # cloud water specific humidity
        q_ice::FT # cloud ice   specific humidity
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        theta_liq_ice::FT
        theta_dry::FT
        S_liq::FT
        RH::FT
    end
end

function init_kinematic_eddy!(eddy_model, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    FT = eltype(state)

    _grav::FT = grav(param_set)

    dc = eddy_model.data_config
    @inbounds begin
        # density
        q_pt_0 = PhasePartition(dc.qt_0)
        R_m, cp_m, cv_m, γ = gas_constants(param_set, q_pt_0)
        T::FT = dc.θ_0 * (aux.p / dc.p_1000)^(R_m / cp_m)
        ρ::FT = aux.p / R_m / T
        state.ρ = ρ




        # moisture
        state.ρq_tot = ρ * dc.qt_0

        # velocity (derivative of streamfunction)
        ρu::FT =
            dc.wmax * dc.xmax / dc.zmax *
            cos(FT(π) * z / dc.zmax) *
            cos(2 * FT(π) * x / dc.xmax)
        ρw::FT =
            2 * dc.wmax * sin(FT(π) * z / dc.zmax) * sin(2 * π * x / dc.xmax)
        state.ρu = SVector(ρu, FT(0), ρw)
        u::FT = ρu / ρ
        w::FT = ρw / ρ

        # energy
        e_kin::FT = 1 // 2 * (u^2 + w^2)
        e_pot::FT = _grav * z
        e_int::FT = internal_energy(param_set, T, q_pt_0)
        e_tot::FT = e_kin + e_pot + e_int
        state.ρe = ρ * e_tot
    end
    return nothing
end

function nodal_update_auxiliary_state!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _grav::FT = grav(param_set)
    @inbounds begin
        aux.u = state.ρu[1] / state.ρ
        aux.w = state.ρu[3] / state.ρ

        aux.q_tot = state.ρq_tot / state.ρ

        aux.e_tot = state.ρe / state.ρ
        aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
        aux.e_pot = _grav * aux.z_coord
        aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot


        # no saturation adjustment
        q = PhasePartition(aux.q_tot, 0.0, 0.0)
        aux.T = air_temperature(param_set, aux.e_int, q)

        aux.theta_liq_ice = liquid_ice_pottemp(param_set, aux.T, state.ρ, q)
        aux.theta_dry = dry_pottemp(param_set, aux.T, state.ρ)
        aux.q_vap = vapor_specific_humidity(q) #changes in space
        aux.q_liq = q.liq
        aux.q_ice = q.ice
    end
end

function boundary_state!(
    ::RusanovNumericalFlux,
    bctype,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
) end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    @inbounds u = state.ρu / state.ρ
    return abs(dot(nM, u))
end

@inline function flux_first_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    @inbounds begin
        # advect moisture ...
        flux.ρq_tot = SVector(
            state.ρu[1] * state.ρq_tot / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_tot / state.ρ,
        )
        # ... energy ...
        flux.ρe = SVector(
            state.ρu[1] / state.ρ * (state.ρe + aux.p),
            FT(0),
            state.ρu[3] / state.ρ * (state.ρe + aux.p),
        )
        # ... and don't advect momentum (kinematic setup)
    end
end

source!(::KinematicModel, _...) = nothing

function set_up_machine(
    domain_extents_3d,
    domain_resolution_3d,
    t_end,
    dt,
    qt_0,
)
    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4 # 1 2 regular cells
    # Domain resolution and size
    Δx = domain_resolution_3d[1]
    Δy = domain_resolution_3d[2]
    Δz = domain_resolution_3d[3]
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = domain_extents_3d[1]
    ymax = domain_extents_3d[2]
    zmax = domain_extents_3d[3]
    # initial configuration
    wmax = FT(0.6)  # max velocity of the eddy  [m/s]
    θ_0 = FT(289) # init. theta value (const) [K]
    p_0 = FT(101500) # surface pressure [Pa]
    p_1000 = FT(100000) # reference pressure in theta definition [Pa]
    z_0 = FT(0) # surface height

    # time stepping
    t_ini = FT(0)

    # periodicity and boundary numbers
    periodicity_x = true
    periodicity_y = true
    periodicity_z = false
    idx_bc_left = 0
    idx_bc_right = 0
    idx_bc_front = 0
    idx_bc_back = 0
    idx_bc_bottom = 1
    idx_bc_top = 2

    driver_config, ode_solver_type = config_kinematic_eddy(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        wmax,
        θ_0,
        p_0,
        p_1000,
        qt_0,
        z_0,
        periodicity_x,
        periodicity_y,
        periodicity_z,
        idx_bc_left,
        idx_bc_right,
        idx_bc_front,
        idx_bc_back,
        idx_bc_bottom,
        idx_bc_top,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t_ini,
        t_end,
        driver_config;
        ode_solver_type = ode_solver_type,
        ode_dt = dt,
        init_on_cpu = true,
        #Courant_number = CFL,
    )

    return driver_config, solver_config
end
