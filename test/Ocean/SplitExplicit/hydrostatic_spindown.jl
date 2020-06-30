#!/usr/bin/env julia --project
using Test
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.Ocean.HydrostaticBoussinesq
using ClimateMachine.Ocean.ShallowWater

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.BalanceLaws: vars_state_conservative, vars_state_auxiliary
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

import ClimateMachine.Ocean.ShallowWater: shallow_init_state!, shallow_init_aux!

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct GyreInABox{T} <: ShallowWaterProblem
    τₒ::T
    fₒ::T # value includes τₒ, g, and ρ
    β::T
    Lˣ::T
    Lʸ::T
    H::T
end

function shallow_init_state!(
    m::ShallowWaterModel,
    p::GyreInABox,
    Q,
    A,
    coords,
    t,
)
    @inbounds x = coords[1]

    Lˣ = p.Lˣ
    H = p.H

    kˣ = 2π / Lˣ
    νʰ = m.turbulence.ν

    M = @SMatrix [-νʰ * kˣ^2 grav(m.param_set) * H * kˣ; -kˣ 0]
    A = exp(M * t) * @SVector [1, 1]

    U = A[1] * sin(kˣ * x)

    Q.U = @SVector [U, -0]
    Q.η = A[2] * cos(kˣ * x)

    return nothing
end

function shallow_init_aux!(p::GyreInABox, aux, geom)
    @inbounds y = geom.coord[2]

    Lʸ = p.Lʸ
    τₒ = p.τₒ
    fₒ = p.fₒ
    β = p.β

    aux.τ = @SVector [-τₒ * cos(π * y / Lʸ), 0]
    aux.f = fₒ + β * (y - Lʸ / 2)

    return nothing
end

function run_hydrostatic_spindown(; coupled = true, refDat = ())
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

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob_3D = SimpleBox{FT}(Lˣ, Lʸ, H)
    prob_2D = GyreInABox{FT}(-0, -0, -0, Lˣ, Lʸ, H)

    model_3D = HydrostaticBoussinesqModel{FT}(
        param_set,
        prob_3D;
        cʰ = FT(1),
        αᵀ = FT(0),
        κʰ = FT(0),
        κᶻ = FT(0),
        fₒ = FT(0),
        β = FT(0),
    )

    model_2D = ShallowWaterModel(
        param_set,
        prob_2D,
        ShallowWater.ConstantViscosity{FT}(model_3D.νʰ),
        nothing,
        FT(1),
    )

    dt_fast = 300
    dt_slow = 300
    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout

    vert_filter = CutoffFilter(grid_3D, polynomialorder(grid_3D) - 1)
    exp_filter = ExponentialFilter(grid_3D, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        modeldata = modeldata,
    )

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg_3D, FT(0); init_on_cpu = true)
    Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

    lsrk_3D = LSRK54CarpenterKennedy(dg_3D, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_2D = LSRK54CarpenterKennedy(dg_2D, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = SplitExplicitSolver(lsrk_3D, lsrk_2D; coupled = coupled)

    step = [0, 0]
    cbvector = make_callbacks(
        vtkpath,
        step,
        nout,
        mpicomm,
        odesolver,
        dg_3D,
        model_3D,
        Q_3D,
        dg_2D,
        model_2D,
        Q_2D,
    )

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    # slow fast state tuple
    Qvec = (slow = Q_3D, fast = Q_2D)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbvector)

    Qe_3D = init_ode_state(dg_3D, timeend, init_on_cpu = true)
    Qe_2D = init_ode_state(dg_2D, timeend, init_on_cpu = true)

    error_3D = euclidean_distance(Q_3D, Qe_3D) / norm(Qe_3D)
    error_2D = euclidean_distance(Q_2D, Qe_2D) / norm(Qe_2D)

    println("3D error = ", error_3D)
    println("2D error = ", error_2D)
    @test isapprox(error_3D, FT(0.0); atol = 0.005)
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
    dg_slow,
    model_slow,
    Q_slow,
    dg_fast,
    model_fast,
    Q_fast,
)
    if isdir(vtkpath)
        rm(vtkpath, recursive = true)
    end
    mkpath(vtkpath)
    mkpath(vtkpath * "/slow")
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
        statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
        auxnames = flattenednames(vars_state_auxiliary(model, eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)
    end

    do_output("slow", step[1], model_slow, dg_slow, Q_slow)
    cbvtk_slow = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("slow", step[1], model_slow, dg_slow, Q_slow)
        step[1] += 1
        nothing
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
            energy = norm(Q_slow)
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
        [
            (Q_slow, "3D state"),
            (dg_slow.state_auxiliary, "3D aux"),
            (Q_fast, "2D state"),
        ],
        nout;
        prec = 12,
    )

    return (cbvtk_slow, cbvtk_fast, cbinfo, cbcs_dg)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_split"

const timeend = 24 * 3600 # s
const tout = 2 * 3600 # s
# const timeend = 1200 # s
# const tout = 600 # s

const N = 4
const Nˣ = 5
const Nʸ = 5
const Nᶻ = 8
const Lˣ = 1e6  # m
const Lʸ = 1e6  # m
const H = 400  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

#const cʰ = sqrt(grav * H)
const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

@testset "$(@__FILE__)" begin
    include("../refvals/hydrostatic_spindown_refvals.jl")

    run_hydrostatic_spindown(coupled = false, refDat = refVals.uncoupled)
end
