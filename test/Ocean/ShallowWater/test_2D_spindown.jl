#!/usr/bin/env julia --project
using Test
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.Ocean.ShallowWater
using ClimateMachine.Ocean.OceanProblems

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function run_hydrostatic_spindown(; refDat = ())
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D,
        periodicity = (true, true),
        boundary = ((0, 0), (0, 0)),
    )
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    problem = SimpleBox{FT}(Lˣ, Lʸ, H)

    model_2D = ShallowWaterModel{FT}(
        param_set,
        problem,
        ShallowWater.ConstantViscosity{FT}(5e3),
        nothing;
        c = FT(1),
        fₒ = FT(0),
        β = FT(0),
    )

    dt_fast = 300
    nout = ceil(Int64, tout / dt_fast)
    dt_fast = tout / nout

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

    lsrk_2D = LSRK54CarpenterKennedy(dg_2D, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = lsrk_2D

    step = [0, 0]
    cbvector = make_callbacks(
        vtkpath,
        step,
        nout,
        mpicomm,
        odesolver,
        dg_2D,
        model_2D,
        Q_2D,
    )

    eng0 = norm(Q_2D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    solve!(Q_2D, odesolver; timeend = timeend, callbacks = cbvector)

    Qe_2D = init_ode_state(dg_2D, timeend, init_on_cpu = true)

    error_2D = euclidean_distance(Q_2D, Qe_2D) / norm(Qe_2D)

    println("2D error = ", error_2D)
    @test isapprox(error_2D, FT(0.0); atol = 0.005)

    ## Check results against reference
    ClimateMachine.StateCheck.scprintref(cbvector[end])
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[end], refDat)
    end

    return nothing
end

function make_callbacks(
    vtkpath,
    step,
    nout,
    mpicomm,
    odesolver,
    dg_fast,
    model_fast,
    Q_fast,
)
    if isdir(vtkpath)
        rm(vtkpath, recursive = true)
    end
    mkpath(vtkpath)
    mkpath(vtkpath * "/fast")

    function do_output(span, step, model, dg, Q)
        outprefix = @sprintf(
            "%s/%s/mpirank%04d_step%04d",
            vtkpath,
            span,
            MPI.Comm_rank(mpicomm),
            step
        )
        @info "doing VTK output" outprefix
        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)
    end

    do_output("fast", step[2], model_fast, dg_fast, Q_fast)
    cbvtk_fast = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("fast", step[2], model_fast, dg_fast, Q_fast)
        step[2] += 1
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q_fast)
            @info @sprintf(
                """Update
                simtime = %8.2f / %8.2f
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                timeend,
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [(Q_fast, "2D state")],
        nout;
        prec = 12,
    )

    # don't write vtk during CI testing
    # return (cbvtk_fast, cbinfo, cbcs_dg)
    return (cbinfo, cbcs_dg)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_shallow_spindown"

const timeend = FT(24 * 3600) # s
const tout = FT(2 * 3600) # s
# const timeend = 1200 # s
# const tout = 600 # s

const N = 4
const Nˣ = 5
const Nʸ = 5
const Lˣ = 1e6  # m
const Lʸ = 1e6  # m
const H = 400  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)

@testset "$(@__FILE__)" begin
    include("../refvals/2D_hydrostatic_spindown_refvals.jl")

    run_hydrostatic_spindown(refDat = refVals.explicit) # error = 0.00011327920483879001
end
