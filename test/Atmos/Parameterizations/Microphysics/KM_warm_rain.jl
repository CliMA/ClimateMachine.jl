include("KinematicModel.jl")

function vars_state(m::KinematicModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
        ρq_liq::FT
        ρq_ice::FT
        ρq_rai::FT
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
        q_tot::FT
        q_vap::FT
        q_liq::FT
        q_ice::FT
        q_rai::FT
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        S_liq::FT
        RH::FT
        rain_w::FT
        # more diagnostics
        src_cloud_liq::FT
        src_cloud_ice::FT
        src_acnv::FT
        src_accr::FT
        src_rain_evap::FT
        flag_rain::FT
        flag_cloud_liq::FT
        flag_cloud_ice::FT
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
        state.ρq_liq = ρ * q_pt_0.liq
        state.ρq_ice = ρ * q_pt_0.ice
        state.ρq_rai = ρ * FT(0)

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
        # velocity
        aux.u = state.ρu[1] / state.ρ
        aux.w = state.ρu[3] / state.ρ
        # water
        aux.q_tot = state.ρq_tot / state.ρ
        aux.q_liq = state.ρq_liq / state.ρ
        aux.q_ice = state.ρq_ice / state.ρ
        aux.q_rai = state.ρq_rai / state.ρ
        q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
        aux.q_vap = vapor_specific_humidity(q)
        # energy
        aux.e_tot = state.ρe / state.ρ
        aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
        aux.e_pot = _grav * aux.z_coord
        aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot
        # supersaturation
        q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
        aux.T = air_temperature(param_set, aux.e_int, q)
        ts_neq = PhaseNonEquil_ρTq(param_set, state.ρ, aux.T, q)
        aux.S_liq = max(0, supersaturation(ts_neq, Liquid()))
        aux.RH = relative_humidity(ts_neq) * FT(100)

        aux.rain_w =
            terminal_velocity(param_set, CM1M.RainType(), state.ρ, aux.q_rai)

        # more diagnostics
        ts_eq = PhaseEquil_ρTq(param_set, state.ρ, aux.T, aux.q_tot)
        q_eq = PhasePartition(ts_eq)

        aux.src_cloud_liq =
            conv_q_vap_to_q_liq_ice(param_set, CM1M.LiquidType(), q_eq, q)
        aux.src_cloud_ice =
            conv_q_vap_to_q_liq_ice(param_set, CM1M.IceType(), q_eq, q)
        aux.src_acnv = conv_q_liq_to_q_rai(param_set, aux.q_liq)
        aux.src_accr = accretion(
            param_set,
            CM1M.LiquidType(),
            CM1M.RainType(),
            aux.q_liq,
            aux.q_rai,
            state.ρ,
        )
        aux.src_rain_evap = evaporation_sublimation(
            param_set,
            CM1M.RainType(),
            q,
            aux.q_rai,
            state.ρ,
            aux.T,
        )
        aux.flag_cloud_liq = FT(0)
        aux.flag_cloud_ice = FT(0)
        aux.flag_rain = FT(0)
        if (aux.q_liq >= FT(0))
            aux.flag_cloud_liq = FT(1)
        end
        if (aux.q_ice >= FT(0))
            aux.flag_cloud_ice = FT(1)
        end
        if (aux.q_rai >= FT(0))
            aux.flag_rain = FT(1)
        end
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
)
    FT = eltype(state⁻)

    #state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)

    #state⁺.ρq_rai = -state⁻.ρq_rai
    @inbounds state⁺.ρq_rai = FT(0)

end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    FT = eltype(state)

    @inbounds begin
        u = state.ρu / state.ρ
        q_rai::FT = state.ρq_rai / state.ρ

        rain_w = terminal_velocity(param_set, CM1M.RainType(), state.ρ, q_rai)
        nu = nM[1] * u[1] + nM[3] * max(u[3], rain_w, u[3] - rain_w)
    end
    return abs(nu)
end

@inline function flux_first_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    FT = eltype(state)

    @inbounds begin
        q_rai::FT = state.ρq_rai / state.ρ
        rain_w = terminal_velocity(param_set, CM1M.RainType(), state.ρ, q_rai)

        # advect moisture ...
        flux.ρq_tot = SVector(
            state.ρu[1] * state.ρq_tot / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_tot / state.ρ,
        )
        flux.ρq_liq = SVector(
            state.ρu[1] * state.ρq_liq / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_liq / state.ρ,
        )
        flux.ρq_ice = SVector(
            state.ρu[1] * state.ρq_ice / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_ice / state.ρ,
        )
        flux.ρq_rai = SVector(
            state.ρu[1] * state.ρq_rai / state.ρ,
            FT(0),
            (state.ρu[3] / state.ρ - rain_w) * state.ρq_rai,
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

function source!(
    m::KinematicModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    _grav::FT = grav(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _cv_v::FT = cv_v(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    @inbounds begin
        e_tot = state.ρe / state.ρ
        q_tot = state.ρq_tot / state.ρ
        q_liq = state.ρq_liq / state.ρ
        q_ice = state.ρq_ice / state.ρ
        q_rai = state.ρq_rai / state.ρ
        u = state.ρu[1] / state.ρ
        w = state.ρu[3] / state.ρ
        e_int = e_tot - 1 // 2 * (u^2 + w^2) - _grav * aux.z_coord

        q = PhasePartition(q_tot, q_liq, q_ice)
        T = air_temperature(param_set, e_int, q)
        # equilibrium state at current T
        ts_eq = PhaseEquil_ρTq(param_set, state.ρ, T, q_tot)
        q_eq = PhasePartition(ts_eq)

        # zero out the source terms
        source.ρq_tot = FT(0)
        source.ρq_liq = FT(0)
        source.ρq_ice = FT(0)
        source.ρq_rai = FT(0)
        source.ρe = FT(0)

        # cloud water and ice condensation/evaporation
        source.ρq_liq +=
            state.ρ *
            conv_q_vap_to_q_liq_ice(param_set, CM1M.LiquidType(), q_eq, q)
        source.ρq_ice +=
            state.ρ *
            conv_q_vap_to_q_liq_ice(param_set, CM1M.IceType(), q_eq, q)

        # tendencies from rain
        src_q_rai_acnv = conv_q_liq_to_q_rai(param_set, q_liq)
        src_q_rai_accr = accretion(
            param_set,
            CM1M.LiquidType(),
            CM1M.RainType(),
            q_liq,
            q_rai,
            state.ρ,
        )
        src_q_rai_evap = evaporation_sublimation(
            param_set,
            CM1M.RainType(),
            q,
            q_rai,
            state.ρ,
            T,
        )

        src_q_rai_tot = src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap

        source.ρq_liq -= state.ρ * (src_q_rai_acnv + src_q_rai_accr)
        source.ρq_rai += state.ρ * src_q_rai_tot
        source.ρq_tot -= state.ρ * src_q_rai_tot
        source.ρe -=
            state.ρ * src_q_rai_tot * (_e_int_v0 - (_cv_v - _cv_d) * (T - _T_0))
    end
end

function main()
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
    t_end = FT(30 * 60)
    dt = FT(5)
    #CFL = FT(1.75)
    filter_freq = 1
    output_freq = 72
    interval = "9steps"

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

    model = driver_config.bl

    mpicomm = MPI.COMM_WORLD

    # get state variables indices for filtering
    ρq_liq_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_liq)
    ρq_ice_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_ice)
    ρq_rai_ind = varsindex(vars_state(model, Prognostic(), FT), :ρq_rai)
    # get aux variables indices for testing
    q_tot_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_tot)
    q_vap_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_vap)
    q_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_liq)
    q_ice_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_ice)
    q_rai_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_rai)
    S_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :S_liq)
    rain_w_ind = varsindex(vars_state(model, Auxiliary(), FT), :rain_w)

    # filter out negative values
    cb_tmar_filter =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:ρq_liq, :ρq_ice, :ρq_rai),
                solver_config.dg.grid,
                TMARFilter(),
            )
            nothing
        end

    # output for paraview

    # initialize base output prefix directory from rank 0
    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    # vtk output
    vtkstep = [0]
    cb_vtk =
        GenericCallbacks.EveryXSimulationSteps(output_freq) do (init = false)
            out_dirname = @sprintf(
                "microphysics_test_3_mpirank%04d_step%04d",
                MPI.Comm_rank(mpicomm),
                vtkstep[1]
            )
            out_path_prefix = joinpath(vtkdir, out_dirname)
            @info "doing VTK output" out_path_prefix
            writevtk(
                out_path_prefix,
                solver_config.Q,
                solver_config.dg,
                flattenednames(vars_state(model, Prognostic(), FT)),
                solver_config.dg.state_auxiliary,
                flattenednames(vars_state(model, Auxiliary(), FT)),
            )
            vtkstep[1] += 1
            nothing
        end

    # output for netcdf
    boundaries = [
        FT(0) FT(0) FT(0)
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    dgngrps = [
        setup_dump_state_diagnostics(
            AtmosLESConfigType(),
            interval,
            driver_config.name,
            interpol = interpol,
        ),
        setup_dump_aux_diagnostics(
            AtmosLESConfigType(),
            interval,
            driver_config.name,
            interpol = interpol,
        ),
    ]
    dgn_config = ClimateMachine.DiagnosticsConfiguration(dgngrps)

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cb_tmar_filter, cb_vtk),
        check_euclidean_distance = true,
    )

    # supersaturation in the model
    max_S_liq = maximum(abs.(solver_config.dg.state_auxiliary[:, S_liq_ind, :]))
    @test max_S_liq < FT(0.25)
    @test max_S_liq > FT(0)

    # qt < reference number
    max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    @test max_q_tot < FT(0.0077)

    # no ice
    max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    @test isequal(max_q_ice, FT(0))

    # q_liq ∈ reference range
    max_q_liq = maximum(solver_config.dg.state_auxiliary[:, q_liq_ind, :])
    min_q_liq = minimum(solver_config.dg.state_auxiliary[:, q_liq_ind, :])
    @test max_q_liq < FT(1e-3)
    @test abs(min_q_liq) < FT(1e-5)

    # q_rai ∈ reference range
    max_q_rai = maximum(solver_config.dg.state_auxiliary[:, q_rai_ind, :])
    min_q_rai = minimum(solver_config.dg.state_auxiliary[:, q_rai_ind, :])
    @test max_q_rai < FT(3e-5)
    @test abs(min_q_rai) < FT(7e-8)

    # terminal velocity ∈ reference range
    max_rain_w = maximum(solver_config.dg.state_auxiliary[:, rain_w_ind, :])
    min_rain_w = minimum(solver_config.dg.state_auxiliary[:, rain_w_ind, :])
    @test max_rain_w < FT(4)
    @test isequal(min_rain_w, FT(0))
end

main()
