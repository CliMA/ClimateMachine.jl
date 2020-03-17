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
    fₒ::T
    β::T
    λʳ::T
    θᴱ::T
end

@inline function ocean_boundary_state!(
    m::Union{OceanModel, HorizontalModel},
    p::SimpleBox,
    bctype,
    x...,
)
    if bctype == 1
        ocean_boundary_state!(m, CoastlineNoSlip(), x...)
    elseif bctype == 2
        ocean_boundary_state!(m, OceanFloorNoSlip(), x...)
    elseif bctype == 3
        ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
    end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::SimpleBox,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineNoSlip(), x...)
end

function ocean_init_state!(P::SimpleBox, Q, A, coords, t)
    @inbounds z = coords[3]
    @inbounds H = P.H

    Q.u = @SVector [0, 0]
    Q.η = 0
    Q.θ = 9 + 8z / H
end

# A is Filled afer the state
function ocean_init_aux!(m::OceanModel, P::SimpleBox, A, geom)
    FT = eltype(A)
    @inbounds y = geom.coord[2]

    Lʸ = P.Lʸ
    τₒ = P.τₒ
    fₒ = P.fₒ
    β = P.β
    θᴱ = P.θᴱ

    A.τ = -τₒ * cos(y * π / Lʸ)
    A.f = fₒ + β * y
    A.θʳ = θᴱ * (1 - y / Lʸ)

    A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
    A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::SimpleBox, A, geom)
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

    minΔx = Lˣ / Nˣ / (N + 1)
    CFL_gravity = minΔx / cʰ
    dt_fast = 1 // 2 * minimum([CFL_gravity])

    minΔz = H / Nᶻ / (N + 1)
    CFL_viscous = minΔz^2 / νᶻ
    CFL_diffusive = minΔz^2 / κᶻ
    dt_slow = 1 // 2 * minimum([CFL_diffusive, CFL_viscous])

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

    prob = SimpleBox{FT}(Lˣ, Lʸ, H, τₒ, fₒ, β, λʳ, θᴱ)

    model = OceanModel{FT}(prob, cʰ = cʰ)

    horizontalmodel = HorizontalModel(model)

    barotropicmodel = BarotropicModel(model)

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
        direction = VerticalDirection(),
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
    update_aux!(dg, model, Q_3D, FT(0))
    update_aux_diffusive!(dg, model, Q_3D, FT(0))

    Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    lsrk_ocean = LSRK144NiegemannDiehlBusch(dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_horizontal =
        LSRK144NiegemannDiehlBusch(horizontal_dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_barotropic =
        LSRK144NiegemannDiehlBusch(barotropic_dg, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = MultistateMultirateRungeKutta(
        lsrk_ocean,
        lsrk_horizontal,
        lsrk_barotropic,
    )

    step = [0, 0]
    cbvector =
        make_callbacks(vtkpath, step, nout, mpicomm, odesolver, dg, model, Q_3D)

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    # slow fast state tuple
    Qvec = (slow = Q_3D, fast = Q_2D)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbvector)

    return nothing
end

function make_callbacks(vtkpath, step, nout, mpicomm, odesolver, dg, model, Q)
    if isdir(vtkpath)
        rm(vtkpath, recursive = true)
    end
    mkpath(vtkpath)
    mkpath(vtkpath * "/weekly")
    mkpath(vtkpath * "/monthly")

    function do_output(span, step)
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

    do_output("weekly", step[1])
    cbvtkw = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("weekly", step[1])
        step[1] += 1
        nothing
    end

    do_output("monthly", step[2])
    cbvtkm = GenericCallbacks.EveryXSimulationSteps(5 * nout) do (init = false)
        do_output("monthly", step[2])
        step[2] += 1
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    return (cbvtkw, cbvtkm, cbinfo)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_ekman_spiral_IMEX"

const timeend = 3 * 30 * 86400   # s
const tout = 6 * 24 * 60 * 60 # s

const N = 4
const Nˣ = 20
const Nʸ = 20
const Nᶻ = 20
const Lˣ = 4e6  # m
const Lʸ = 4e6  # m
const H = 400  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

const cʰ = sqrt(grav * H)
const cᶻ = 0

const τₒ = 1e-1  # (m/s)^2
const fₒ = 1e-4  # Hz
const β = 1e-11 # Hz / m
const θᴱ = 25    # K

const αᵀ = 2e-4  # (m/s)^2 / K
const νʰ = 5e3   # m^2 / s
const νᶻ = 5e-3  # m^2 / s
const κʰ = 1e3   # m^2 / s
const κᶻ = 1e-4  # m^2 / s
const λʳ = 4 // 86400 # m / s

main()
