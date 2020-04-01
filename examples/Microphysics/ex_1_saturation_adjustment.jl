include("KinematicModel.jl")

function vars_state(m::KinematicModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
    end
end

function vars_aux(m::KinematicModel, FT)
    @vars begin
        # defined in init_aux
        p::FT
        z::FT
        # defined in update_aux
        u::FT
        w::FT
        q_tot::FT
        q_vap::FT
        q_liq::FT
        q_ice::FT
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        S::FT
        RH::FT
    end
end

function init_kinematic_eddy!(eddy_model, state, aux, (x, y, z), t)
    FT = eltype(state)
    dc = eddy_model.data_config

    # density
    q_pt_0 = PhasePartition(dc.qt_0)
    R_m, cp_m, cv_m, γ = gas_constants(q_pt_0)
    T::FT = dc.θ_0 * (aux.p / dc.p_1000)^(R_m / cp_m)
    ρ::FT = aux.p / R_m / T
    state.ρ = ρ

    # moisture
    state.ρq_tot = ρ * dc.qt_0

    # velocity (derivative of streamfunction)
    ρu::FT =
        dc.wmax * dc.xmax / dc.zmax *
        cos(π * z / dc.zmax) *
        cos(2 * π * x / dc.xmax)
    ρw::FT = 2 * dc.wmax * sin(π * z / dc.zmax) * sin(2 * π * x / dc.xmax)
    state.ρu = SVector(ρu, FT(0), ρw)
    u::FT = ρu / ρ
    w::FT = ρw / ρ

    # energy
    e_kin::FT = 1 // 2 * (u^2 + w^2)
    e_pot::FT = grav * z
    e_int::FT = internal_energy(T, q_pt_0)
    e_tot::FT = e_kin + e_pot + e_int
    state.ρe = ρ * e_tot

    return nothing
end

function kinematic_model_nodal_update_aux!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)

    aux.u = state.ρu[1] / state.ρ
    aux.w = state.ρu[3] / state.ρ

    aux.q_tot = state.ρq_tot / state.ρ

    aux.e_tot = state.ρe / state.ρ
    aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
    aux.e_pot = grav * aux.z
    aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot

    # saturation adjustment happens here
    ts = PhaseEquil(aux.e_int, state.ρ, aux.q_tot)
    pp = PhasePartition(ts)

    aux.T = ts.T
    aux.q_vap = aux.q_tot - pp.liq - pp.ice
    aux.q_liq = pp.liq
    aux.q_ice = pp.ice

    q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
    aux.S =
        max(0, aux.q_vap / q_vap_saturation(aux.T, state.ρ, q) - FT(1)) *
        FT(100)
    aux.RH = aux.q_vap / q_vap_saturation(aux.T, state.ρ, q) * FT(100)
end

function boundary_state!(
    ::Rusanov,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
) end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    u = state.ρu / state.ρ
    return abs(dot(nM, u))
end

@inline function flux_nondiffusive!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)

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

function source!(
    m::KinematicModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) end

function main()
    CLIMA.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δx = FT(20)
    Δy = FT(1)
    Δz = FT(20)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = 1500
    ymax = 10
    zmax = 1500
    # initial configuration
    wmax = FT(0.6)  # max velocity of the eddy  [m/s]
    θ_0 = FT(289) # init. theta value (const) [K]
    p_0 = FT(101500) # surface pressure [Pa]
    p_1000 = FT(100000) # reference pressure in theta definition [Pa]
    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]
    z_0 = FT(0) # surface height

    # time stepping
    t_ini = FT(0)
    t_end = FT(60 * 30)
    dt = 40
    output_freq = 9

    driver_config = config_kinematic_eddy(
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
    )
    solver_config = CLIMA.SolverConfiguration(
        t_ini,
        t_end,
        driver_config;
        ode_dt = dt,
        init_on_cpu = true,
        #Courant_number = CFL,
    )

    mpicomm = MPI.COMM_WORLD

    # output for paraview
    model = driver_config.bl
    step = [0]
    cbvtk =
        GenericCallbacks.EveryXSimulationSteps(output_freq) do (init = false)
            mkpath("vtk/")
            outprefix = @sprintf(
                "vtk/new_ex_1_mpirank%04d_step%04d",
                MPI.Comm_rank(mpicomm),
                step[1]
            )
            @info "doing VTK output" outprefix
            writevtk(
                outprefix,
                solver_config.Q,
                solver_config.dg,
                flattenednames(vars_state(model, FT)),
                solver_config.dg.auxstate,
                flattenednames(vars_aux(model, FT)),
            )
            step[1] += 1
            nothing
        end

    # get aux variables indices for testing
    q_tot_ind = varsindex(vars_aux(model, FT), :q_tot)
    q_vap_ind = varsindex(vars_aux(model, FT), :q_vap)
    q_liq_ind = varsindex(vars_aux(model, FT), :q_liq)
    q_ice_ind = varsindex(vars_aux(model, FT), :q_ice)
    S_ind = varsindex(vars_aux(model, FT), :S)

    # call solve! function for time-integrator
    result = CLIMA.invoke!(
        solver_config;
        user_callbacks = (cbvtk,),
        check_euclidean_distance = true,
    )

    # no supersaturation
    max_S = maximum(abs.(solver_config.dg.auxstate[:, S_ind, :]))
    @test isequal(max_S, FT(0))

    # qt is conserved
    max_q_tot = maximum(abs.(solver_config.dg.auxstate[:, q_tot_ind, :]))
    min_q_tot = minimum(abs.(solver_config.dg.auxstate[:, q_tot_ind, :]))
    @test isapprox(max_q_tot, qt_0; rtol = 1e-3)
    @test isapprox(min_q_tot, qt_0; rtol = 1e-3)

    # q_vap + q_liq = q_tot
    max_water_diff = maximum(abs.(
        solver_config.dg.auxstate[:, q_tot_ind, :] .-
        solver_config.dg.auxstate[:, q_vap_ind, :] .-
        solver_config.dg.auxstate[:, q_liq_ind, :],
    ))
    @test isequal(max_water_diff, FT(0))

    # no ice
    max_q_ice = maximum(abs.(solver_config.dg.auxstate[:, q_ice_ind, :]))
    @test isequal(max_q_ice, FT(0))

    # q_liq ∈ reference range
    max_q_liq = max(solver_config.dg.auxstate[:, q_liq_ind, :]...)
    min_q_liq = min(solver_config.dg.auxstate[:, q_liq_ind, :]...)
    @test max_q_liq < FT(1e-3)
    @test isequal(min_q_liq, FT(0))
end

main()
