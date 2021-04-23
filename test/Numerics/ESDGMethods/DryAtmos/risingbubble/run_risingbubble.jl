using MPI
using ClimateMachine
using Logging
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid, min_node_distance
using ClimateMachine.Thermodynamics
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.VariableTemplates: flattenednames
using ClimateMachine.ODESolvers
using StaticArrays: @SVector
using LazyArrays
using JLD2

using DoubleFloats
using GaussQuadrature
GaussQuadrature.maxiterations[Double64] = 40

using ClimateMachine.TemperatureProfiles: DryAdiabaticProfile

include("risingbubble.jl")
include("../../diagnostics.jl")

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    
    #FT = Double64
    FT = Float64
    problem = RisingBubble{FT}()

    mpicomm = MPI.COMM_WORLD
    N = 4
    K = (10, 10)
    timeend = 1000

    for relaxation in (true,)
      for surfaceflux in (EntropyConservative, MatrixFlux)
        result = run(
            mpicomm,
            N,
            K,
            timeend,
            ArrayType,
            FT,
            problem,
            surfaceflux,
            relaxation
        )
      end
    end
end

function run(
    mpicomm,
    N,
    K,
    timeend,
    ArrayType,
    FT,
    problem,
    surfaceflux,
    relaxation
)

    dim = 2
    brickrange = (
        range(FT(0), stop = problem.xmax, length = K[1] + 1),
        range(FT(0), stop = problem.zmax, length = K[2] + 1),
    )
    boundary = ((0, 0), (1, 2))
    periodicity = (true, false)
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )
    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    T_surface = FT(problem.θref)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = DryReferenceState(T_profile)

    model = DryAtmosModel{dim}(FlatOrientation(),
                               problem;
                               ref_state=ref_state)

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = surfaceflux(),
    )

    # determine the time step
    dx = min_node_distance(grid)
    cfl = FT(1.5)
    dt = cfl * dx / 330

    Q = init_ode_state(esdg, FT(0))

    η = similar(Q, vars = @vars(η::FT), nstate=1)

    ∫η0 = entropy_integral(esdg, η, Q)

    η_int = function(dg, Q1)
      entropy_integral(dg, η, Q1)
    end
    η_prod = function(dg, Q1, Q2)
      entropy_product(dg, η, Q1, Q2)
    end

    if relaxation
      odesolver = RLSRK144NiegemannDiehlBusch(esdg, η_int, η_prod, Q; dt = dt, t0 = 0)
    else
      odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)
    end

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem         = (%d, %d)
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      ∫η              = %.16e
                      """ "$ArrayType" "$FT" N K... dt eng0 ∫η0


    dη_timeseries = NTuple{2, FT}[]
    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXSimulationSteps(100) do (s = false)
        if s
            starttime[] = now()
        else
            ∫η = entropy_integral(esdg, η, Q)
            dη = (∫η - ∫η0) / abs(∫η0)
            time = gettime(odesolver)
            push!(dη_timeseries, (time, dη))
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime            = %.16e
                              runtime            = %s
                              norm(Q)            = %.16e
                              ∫η                 = %.16e
                              (∫η - ∫η0) / |∫η0| = %.16e 
                              """ gettime(odesolver) runtime energy ∫η dη
        end
    end
    callbacks = (cbinfo,)

    relax = relaxation ? "lsrk" : "rlsrk"
    outdir = joinpath("esdg_output",
                      "risingbubble",
                      "$relax",
                      "$surfaceflux",
                      "$N",
                      "$(K[1])x$(K[2])")
    outputtime = timeend / 10

    output_vtk = false
    if output_vtk
        # create vtk dir
        Nelem = Ne[1]
        vtkdir = joinpath(outdir, vtk)
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N)
        end
        callbacks = (callbacks..., cbvtk)
    end

    stepsdir = joinpath(outdir, "steps")
    mkpath(stepsdir)
    cbstep = EveryXSimulationSteps(floor(outputtime / dt)) do
      step = getsteps(odesolver)
      time = gettime(odesolver)
      let
        state_prognostic = Array(Q)
        state_auxiliary = Array(esdg.state_auxiliary)
        vgeo = Array(grid.vgeo)

        @save(joinpath(stepsdir, "rtb_step_$(lpad(step, 8, '0')).jld2"),
              model,
              problem,
              step,
              time,
              N,
              K,
              surfaceflux,
              state_prognostic,
              state_auxiliary,
              vgeo)
      end
    end
    callbacks = (callbacks..., cbstep)

    solve!(Q, odesolver; callbacks = callbacks, timeend = timeend)

    @save(joinpath(outdir, "rtb_entropy_residual.jld2"), dη_timeseries)

    # final statistics
    engf = norm(Q)
    ∫ηf = entropy_integral(esdg, η, Q)
    dηf = (∫ηf - ∫η0) / abs(∫η0)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    ∫η                      = %.16e
    (∫η - ∫η0) / |∫η0|      = %.16e 
    """ engf engf / eng0 engf - eng0 ∫ηf dηf
    engf
end

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, N, testname = "RTB")
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

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames;
             number_sample_points = 2 * (N + 1))

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
