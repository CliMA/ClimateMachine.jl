using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.VariableTemplates: flattenednames
using CLIMA.SplitExplicit
using CLIMA.HydrostaticBoussinesq
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.PlanetParameters: grav
import CLIMA.SplitExplicit:
    ocean_init_aux!,
    ocean_init_state!,
    ocean_boundary_state!,
    CoastlineFreeSlip,
    CoastlineNoSlip,
    OceanFloorFreeSlip,
    OceanFloorNoSlip,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing
import CLIMA.DGmethods:
    update_aux!, update_aux_diffusive!, vars_state, vars_aux, VerticalDirection
using GPUifyLoops

const ArrayType = CLIMA.array_type()

struct SimpleBox{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::SimpleBox,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineNoSlip(), x...)
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::SimpleBox, A, geom)
    @inbounds A.y = geom.coord[2]

    return nothing
end

function main()
    CLIMA.init()
    mpicomm = MPI.COMM_WORLD

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    brickrange_2D = (xrange, yrange)
    topl_2D =
        BrickTopology(mpicomm, brickrange_2D, periodicity = (false, false))
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob = SimpleBox{FT}(Lˣ, Lʸ, H, τₒ)
    model = OceanModel{FT}(prob, cʰ = cʰ)
    barotropicmodel =
        BarotropicModel(model, CLIMA.SplitExplicit.SurfaceStress())

    @show typeof(barotropicmodel.source)

    minΔx = Lˣ / Nˣ / (N + 1)
    CFL_gravity = minΔx / model.cʰ
    dt_fast = 120 # 1 // 2 * minimum([CFL_gravity])
    nout = ceil(Int64, tout / dt_fast)
    dt_fast = tout / nout

    @info @sprintf(
        """Update
           Gravity CFL   = %.1f
           Timestep      = %.1f""",
        CFL_gravity,
        dt_fast
    )

    barotropic_dg = DGModel(
        barotropicmodel,
        grid_2D,
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )

    Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    lsrk_barotropic =
        LSRK144NiegemannDiehlBusch(barotropic_dg, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = lsrk_barotropic

    step = [0, 0]
    cbvector = make_callbacks(
        vtkpath,
        step,
        nout,
        mpicomm,
        odesolver,
        barotropic_dg,
        barotropicmodel,
        Q_2D,
    )

    eng0 = norm(Q_2D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    solve!(Q_2D, odesolver; timeend = timeend, callbacks = cbvector)

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
        statenames = flattenednames(vars_state(model, eltype(Q)))
        auxnames = flattenednames(vars_aux(model, eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.auxstate, auxnames)
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

    return (cbvtk_fast, cbinfo)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_barotropic"

const timeend = 30 * 24 * 3600 # s
const tout = 6 * 3600 # s

const N = 4
const Nˣ = 20
const Nʸ = 20

const Lˣ = 4e6  # m
const Lʸ = 4e6  # m
const H = 1000  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)

const cʰ = sqrt(grav * H)
const cᶻ = 0

const τₒ = 1e-1  # (m/s)^2

main()
