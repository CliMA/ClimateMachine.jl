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
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.SplitExplicit:
    CoastlineFreeSlip,
    CoastlineNoSlip,
    OceanFloorFreeSlip,
    OceanFloorNoSlip,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing
using CLIMA.DGmethods: vars_state, vars_aux
import CLIMA.SplitExplicit: ocean_init_aux!, ocean_init_state!

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct SimpleBox{T, BC} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    λʳ::T
    θᴱ::T
    boundary_condition::BC
end

function ocean_init_state!(p::SimpleBox, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [-0, -0]
    Q.η = -0
    Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    return nothing
end

function ocean_init_aux!(m::OceanModel, p::SimpleBox, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::SimpleBox, A, geom)
    @inbounds A.y = geom.coord[2]

    return nothing
end

function main(BC, solver)
    CLIMA.init()
    ArrayType = CLIMA.array_type()
    mpicomm = MPI.COMM_WORLD

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D;
        periodicity = (false, false),
        boundary = ((1, 1), (1, 1)),
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
        periodicity = (false, false, false),
        boundary = ((1, 1), (1, 1), (2, 3)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob = SimpleBox{FT, typeof(BC)}(Lˣ, Lʸ, H, τₒ, λʳ, θᴱ, BC)

    model = OceanModel{FT}(param_set, prob, cʰ = cʰ)

    horizontalmodel = HorizontalModel(model)

    barotropicmodel =
        BarotropicModel(model, CLIMA.SplitExplicit.IntegratedTendency())

    minΔx = Lˣ / Nˣ / (N + 1)
    CFL_gravity = minΔx / model.cʰ
    dt_fast = 2 # 1 // 2 * minimum([CFL_gravity])

    minΔz = H / Nᶻ / (N + 1)
    CFL_viscous = minΔz^2 / model.νᶻ
    CFL_diffusive = minΔz^2 / model.κᶻ
    dt_slow = 1 // 2 * minimum([CFL_diffusive, CFL_viscous])

    dt_slow = 120
    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout

    @info @sprintf(
        """Update
           Gravity CFL   = %.1f
           Timestep      = %.1f""",
        CFL_gravity,
        dt_fast
    )

    @info @sprintf(
        """Update
       Viscous CFL   = %.1f
       Diffusive CFL = %.1f
       Timestep      = %.1f""",
        CFL_viscous,
        CFL_diffusive,
        dt_slow
    )

    dg = OceanDGModel(
        model,
        grid_3D,
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )

    horizontal_dg = DGModel(
        horizontalmodel,
        grid_3D,
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient();
        auxstate = dg.auxstate,
        diffstate = dg.diffstate,
    )

    barotropic_dg = DGModel(
        barotropicmodel,
        grid_2D,
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg, FT(0); init_on_cpu = true)
    Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    lsrk_ocean = LSRK144NiegemannDiehlBusch(dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_horizontal =
        LSRK144NiegemannDiehlBusch(horizontal_dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_barotropic =
        LSRK144NiegemannDiehlBusch(barotropic_dg, Q_2D, dt = dt_fast, t0 = 0)

    odesolver =
        solver(lsrk_ocean, lsrk_barotropic; sAlt_solver = lsrk_horizontal)

    step = [0, 0]
    cbvector = make_callbacks(
        vtkpath,
        step,
        nout,
        mpicomm,
        odesolver,
        dg,
        model,
        Q_3D,
        barotropic_dg,
        barotropicmodel,
        Q_2D,
    )

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    # slow fast state tuple
    Qvec = (slow = Q_3D, fast = Q_2D)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbvector)

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
        statenames = flattenednames(vars_state(model, eltype(Q)))
        auxnames = flattenednames(vars_aux(model, eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.auxstate, auxnames)
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

    return (cbvtk_slow, cbvtk_fast, cbinfo)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_test_integrals"

const timeend = 360   # s
const tout = 120 # s

const N = 4
const Nˣ = 20
const Nʸ = 20
const Nᶻ = 20
const Lˣ = 4e6  # m
const Lʸ = 4e6  # m
const H = 1000  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

const cʰ = sqrt(grav(param_set) * H)
const cᶻ = 0

const τₒ = 1e-1  # (m/s)^2
const λʳ = 10 // 86400 # m / s
const θᴱ = 10    # K

#### boundary conditions

### standard no slip
BC_noslip = (CoastlineNoSlip(), OceanFloorNoSlip(), OceanSurfaceStressForcing())

### standard free slip
BC_freeslip =
    (CoastlineFreeSlip(), OceanFloorFreeSlip(), OceanSurfaceStressForcing())

### no wind no slip
BC_noslip_nowind =
    (CoastlineNoSlip(), OceanFloorNoSlip(), OceanSurfaceNoStressForcing())

### no wind free slip
BC_freeslip =
    (CoastlineFreeSlip(), OceanFloorFreeSlip(), OceanSurfaceNoStressForcing())

BC_noslip_noforcing =
    (CoastlineNoSlip(), OceanFloorNoSlip(), OceanSurfaceStressNoForcing())

solver_multirate = MultistateMultirateRungeKutta
solver_singlerate = MultistateRungeKutta

main(BC_noslip_noforcing, solver_singlerate)
