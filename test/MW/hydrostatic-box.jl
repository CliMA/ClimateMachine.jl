using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: BrickTopology
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps, EveryXSimulationTime
using ClimateMachine.MPIStateArrays: euclidean_distance
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Atmos
using ClimateMachine.Atmos: vars_state_conservative, vars_state_auxiliary
using ClimateMachine.VariableTemplates: @vars, Vars, flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = false

function main()
    ClimateMachine.cli()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    FT = Float64
    polynomialorder = 4
    numelems = (10, 10, 3)
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

    setup = HydrostaticBoxSetup{FT}()
   
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

    topology = BrickTopology(
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

    refstate = HydrostaticState(IsothermalProfile(param_set, FT(300)))
    initrefstate = HydrostaticState(IsothermalProfile(param_set, FT(300)))
    #refstate = HydrostaticState(IsothermalProfile(param_set, FT))
    #initrefstate = HydrostaticState(IsothermalProfile(param_set, FT))

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = FlatOrientation(),
        ref_state = refstate,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = setup,
    )
    initmodel = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = FlatOrientation(),
        ref_state = initrefstate,
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = setup,
    )
    
    dg = DGModel(
        model,
        grid,
        #RusanovNumericalFlux(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
    )
    
    initdg = DGModel(
        initmodel,
        grid,
        #RusanovNumericalFlux(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
    )

    linearmodel = AtmosAcousticGravityLinearModel(model)
    nonlinearmodel = RemainderModel(model, (linearmodel,))

    lineardg = DGModel(
        linearmodel,
        grid,
        #RusanovNumericalFlux(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        state_auxiliary = dg.state_auxiliary
    )

    nonlineardg = DGModel(
        nonlinearmodel,
        grid,
        RusanovNumericalFlux(),
        #CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        state_auxiliary = dg.state_auxiliary,
    )

    timeend = FT(100)

    # determine the time step
    dx = min_node_distance(grid)
    wavespeed = soundspeed_air(model.param_set, FT(300))
    cfl = 0.1
    dt = cfl * dx / wavespeed

    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(initdg, FT(0))

    linearsolver = GeneralizedMinimalResidual(Q; M = 20, rtol = 1e-5)
    odesolver = ARK2GiraldoKellyConstantinescu(
        split_explicit_implicit ? nonlineardg : dg,
        lineardg,
        LinearBackwardEulerSolver(linearsolver; isadjustable = true),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = split_explicit_implicit,
        variant = NaiveVariant(),
    )
    
    #odesolver = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

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

    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_hydrostaticbox" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

        # setup the output callback
        outputtime = 50
        #outputtime = dt
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    #solve!(Q, odesolver; timeend = timeend, callbacks = callbacks)
    solve!(Q, odesolver; timeend = 1000dt, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, FT(0))
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

Base.@kwdef struct HydrostaticBoxSetup{FT}
    domain::SVector{3, FT} = (1000, 1000, 50000)
end

function (setup::HydrostaticBoxSetup)(bl, state, aux, (x, y, z), t)
    FT = eltype(state)
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
    testname = "hydrostatic_box",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
    auxnames = flattenednames(vars_state_auxiliary(model, eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

        @info "Done writing VTK: $pvtuprefix"
    end

end

main()
