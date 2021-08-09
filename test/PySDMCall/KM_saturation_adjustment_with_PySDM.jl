
include("../Atmos/Parameterizations/Microphysics/KinematicModel.jl")
include("../../src/PySDMCall/PySDMCallback.jl")
include("../../src/PySDMCall/PySDMCall.jl")

using .PySDMCallbacks
using .PySDMCall

using StaticArrays

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
        

        # R_m 288.31131120772176
        

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

        # saturation adjustment happens here
        ts = PhaseEquil(param_set, aux.e_int, state.ρ, aux.q_tot)
        q = PhasePartition(ts)

        aux.T = ts.T
        aux.theta_liq_ice = liquid_ice_pottemp(param_set, ts.T, state.ρ, q)
        aux.theta_dry = dry_pottemp(param_set, ts.T, state.ρ)
        aux.q_vap = vapor_specific_humidity(q) # zmienne w przestrzeni 
        aux.q_liq = q.liq
        aux.q_ice = q.ice

        # TODO: add super_saturation method in moist thermo
        #aux.S = max(0, aux.q_vap / q_vap_saturation(ts) - FT(1)) * FT(100)
        aux.S_liq = max(0, supersaturation(ts, Liquid()))
        aux.RH = relative_humidity(ts)
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

function main()
    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4 # 1 2 regular cells
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
    dt = 10 # was 40
    output_freq = 9
    interval = "90steps"

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
        ode_dt = dt,
        init_on_cpu = true,
        #Courant_number = CFL,
    )

    mpicomm = MPI.COMM_WORLD

    # output for paraview
    # initialize base prefix directory from rank 0
    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    model = driver_config.bl
    vtkstep = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(output_freq) do
        out_dirname = @sprintf(
            "new_ex_1_mpirank%04d_step%04d",
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

    
    # configuring PySDM
    # TODO kernel here !!!!!!!! done?
    krnl = PySDMKernels()

    spectra = PySDMSpectra()

    rho_STP = 1.2252141358659048
    micrometre = 1e-6
    centimetre = 0.01
    spectrum_per_mass_of_dry_air = spectra.Lognormal(
                                                norm_factor=60 / centimetre ^ 3 / rho_STP,
                                                m_mode=0.04 * micrometre,
                                                s_geom=1.4
                                            )

    pysdmconf = PySDMConf(
                        (xmax, zmax), 
                        (Δx, Δz), 
                        t_end, 
                        solver_config.dt, 
                        25, 
                        1, 
                        krnl.Geometric(collection_efficiency=1), 
                        spectrum_per_mass_of_dry_air
                    ) #???? 

    testcb = GenericCallbacks.EveryXSimulationSteps(PySDMCallback("PySDMCallback", 
                                                                  solver_config.dg, 
                                                                  interpol, 
                                                                  mpicomm, 
                                                                  pysdmconf
                                                                  ), 1)

    #testcb = GenericCallbacks.EveryXSimulationSteps(PySDMCallback("PySDMCallback", 
    #                                                              solver_config.dg, 
    #                                                              interpol, 
    #                                                              mpicomm, 
    #                                                              solver_config.dt, 
    #                                                              Δx,
    #                                                              t_end,
    #                                                              nothing), 1)

    # get aux variables indices for testing
    q_tot_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_tot)
    q_vap_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_vap)
    q_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_liq)
    q_ice_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_ice)
    S_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :S_liq)

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbvtk, testcb,),
        check_euclidean_distance = true,
    )

    # no supersaturation
    max_S_liq = maximum(abs.(solver_config.dg.state_auxiliary[:, S_liq_ind, :]))
    @test isequal(max_S_liq, FT(0))

    # qt is conserved
    max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    min_q_tot = minimum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    @test isapprox(max_q_tot, qt_0; rtol = 1e-3)
    @test isapprox(min_q_tot, qt_0; rtol = 1e-3)

    # q_vap + q_liq = q_tot
    max_water_diff = maximum(abs.(
        solver_config.dg.state_auxiliary[:, q_tot_ind, :] .-
        solver_config.dg.state_auxiliary[:, q_vap_ind, :] .-
        solver_config.dg.state_auxiliary[:, q_liq_ind, :],
    ))
    @test isequal(max_water_diff, FT(0))

    # no ice
    max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    @test isequal(max_q_ice, FT(0))

    # q_liq ∈ reference range
    max_q_liq = maximum(solver_config.dg.state_auxiliary[:, q_liq_ind, :])
    min_q_liq = minimum(solver_config.dg.state_auxiliary[:, q_liq_ind, :])
    @test max_q_liq < FT(1e-3)
    @test isequal(min_q_liq, FT(0))
end

main()
