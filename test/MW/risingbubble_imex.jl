using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.Orientations
using ClimateMachine.TurbulenceClosures
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: GeneralizedMinimalResidual
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps, EveryXSimulationTime
using ClimateMachine.MPIStateArrays: euclidean_distance
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Atmos
using ClimateMachine.Atmos: vars_state
using ClimateMachine.VariableTemplates: @vars, Vars, flattenednames
using ClimateMachine.BalanceLaws: Prognostic, Auxiliary

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = false

function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    FT = Float64
    polynomialorder = 4
    numelems = (10, 10, 10)
    dims = 3
    split_explicit_implicit = true
    let
        split = split_explicit_implicit ? "(Nonlinear, Linear)" :
            "(Full, Linear)"
        @info @sprintf """Configuration
                          ArrayType = %s
                          FT    = %s
                          dims      = %d
                          splitting = %s
                          """ ArrayType "$FT" dims split
    end

    setup = RisingThermalBubbleSetup{FT}()

    expected = 7.3759283720505702e+08
   
    #for i in 1:10
      result = run(
          mpicomm,
          ArrayType,
          polynomialorder,
          numelems,
          setup,
          split_explicit_implicit,
          FT,
          dims,
      )
      #@test result == expected
    #end
end

function run(
    mpicomm,
    ArrayType,
    polynomialorder,
    numelems,
    setup,
    split_explicit_implicit,
    FT,
    dims,
)
    brickrange = ntuple(dims) do dim
        range(
            FT(0);
            length = numelems[dim] + 1,
            stop = setup.domain[dim],
        )
    end

    topology = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = (true, true, false),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = FlatOrientation(),
        #orientation = NoOrientation(),
        ref_state = HydrostaticState(DryAdiabaticProfile{FT}(param_set,  FT(setup.θ₀), typemin(FT)), FT(0)),
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = Gravity(),
        #source = (),
        init_state_prognostic = setup,
    )



    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        #CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
    )

    #linear_model = AtmosAcousticLinearModel(model)
    #schur_complement = AtmosAcousticLinearSchurComplement()

    linear_model = AtmosAcousticGravityLinearModel(model)
    schur_complement = AtmosAcousticGravityLinearSchurComplement()

    #nonlinear_model = RemainderModel(model, (linear_model,))
    #schur_complement = nothing
    dg_linear = DGModel(
        linear_model,
        grid,
        #RusanovNumericalFlux(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        state_auxiliary = dg.state_auxiliary,
        schur_complement = schur_complement
    )

    #dg_nonlinear = DGModel(
    #    nonlinear_model,
    #    grid,
    #    RusanovNumericalFlux(),
    #    #CentralNumericalFluxFirstOrder(),
    #    CentralNumericalFluxSecondOrder(),
    #    CentralNumericalFluxGradient();
    #    state_auxiliary = dg.state_auxiliary,
    #)

    dg_nonlinear = remainder_DGModel(dg, (dg_linear,);
            numerical_flux_first_order = RusanovNumericalFlux(),
            numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
            numerical_flux_gradient = CentralNumericalFluxGradient(),
    )

    timeend = FT(350)

    # determine the time step
    dx = min_node_distance(grid)
    wavespeed = soundspeed_air(model.param_set, setup.θ₀)
    cfl = 10
    dt = cfl * dx / wavespeed

    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))

    if !isnothing(schur_complement)
      linearsolver = GeneralizedMinimalResidual(dg_linear.states_schur_complement.state; M = 50, rtol = 1e-3)
    else
      linearsolver = GeneralizedMinimalResidual(Q; M = 20, rtol = 1e-3)
    end
    odesolver = ARK2GiraldoKellyConstantinescu(
        split_explicit_implicit ? dg_nonlinear : dg,
        dg_linear,
        LinearBackwardEulerSolver(linearsolver; isadjustable = true),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = split_explicit_implicit,
        variant = NaiveVariant(),
    )

    eng0 = norm(Q)
    dims == 2 && (numelems = (numelems..., 0))
    @info @sprintf """Starting
                      numelems  = (%d, %d, %d)
                      dt        = %.16e
                      norm(Q₀)  = %.16e
                      """ numelems... dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end

    #cbcfl = EveryXSimulationTime(1, odesolver) do
    #    dt = ODESolvers.getdt(odesolver)
    #    simtime = gettime(odesolver)
    #    cfl_adv = DGmethods.courant(
    #        Atmos.advective_courant,
    #        dg,
    #        model,
    #        Q,
    #        dt,
    #        simtime,
    #        EveryDirection(),
    #    )
    #    cfl_nondiff = DGmethods.courant(
    #        Atmos.nondiffusive_courant,
    #        dg,
    #        model,
    #        Q,
    #        fast_dt,
    #        simtime,
    #        EveryDirection(),
    #    )
    #    @info @sprintf(
    #        """Courant number
    #        simtime = %.16e
    #        acoustic courant = %.16e
    #        advective courant = %.16e""",
    #        simtime,
    #        cfl_nondiff,
    #        cfl_adv
    #    )
    #end

    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_risingbubble" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

        # setup the output callback
        #outputtime = 50
        outputtime = dt
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; timeend = timeend, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, timeend)
    engf = norm(Q)
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf / eng0 engf - eng0 errf errf / engfe
    errf
    engf
end

Base.@kwdef struct RisingThermalBubbleSetup{FT}
    domain::SVector{3, FT} = (1000, 1000, 1000)
    x_c::FT = 500
    z_c::FT = 350
    R::FT = 250
    δθ_c::FT = 0.5
    θ₀::FT = 300
end

function (setup::RisingThermalBubbleSetup)(bl, state, aux, (x, y, z), t)
    FT = eltype(state)

    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)

    r = sqrt((x - setup.x_c)^2 + (y - setup.x_c)^2 + (z - setup.z_c)^2)

    δθ =  r <= setup.R ? setup.δθ_c / 2 * (1 + cospi(r / setup.R)) : FT(0)
    #θ = setup.θ₀ + δθ
    θ = setup.θ₀

    π_exner = FT(1) - _grav / (c_p * θ) * z
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)

    q_tot = FT(0)
    ts = LiquidIcePotTempSHumEquil(bl.param_set, θ, ρ, q_tot)

    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    #state.ρ = ρ
    #state.ρu = SVector(FT(0), FT(0), FT(0))
    #state.ρe = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector(FT(0), FT(0), FT(0))
    state.ρe = aux.ref_state.ρe
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "rtb",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end

end

main()
