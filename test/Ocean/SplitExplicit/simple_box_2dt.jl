#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates: flattenednames
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.GenericCallbacks
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.Ocean.SplitExplicit01:
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
import ClimateMachine.DGMethods:
    update_auxiliary_state!, update_auxiliary_state_gradient!, VerticalDirection
# using GPUifyLoops

const ArrayType = ClimateMachine.array_type()

struct SimpleBox{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    λʳ::T
    θᴱ::T
end

@inline function ocean_boundary_state!(
    m::OceanModel,
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
    m::Continuity3dModel,
    p::SimpleBox,
    bctype,
    x...,
)
    #if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
    #end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::SimpleBox,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineNoSlip(), x...)
end

function ocean_init_state!(p::SimpleBox, Q, A, localgeo, t)
    coords = localgeo.coord
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

    # not sure if this is needed but getting weird intialization stuff
    A.w = -0
    A.pkin = -0
    A.wz0 = -0
    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::SimpleBox, A, geom)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.U_c = @SVector [-0, -0]
    A.η_c = -0
    A.U_s = @SVector [-0, -0]
    A.η_s = -0
    A.Δu = @SVector [-0, -0]
    A.η_diag = -0
    A.Δη = -0

    return nothing
end

function main()
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

    prob = SimpleBox{FT}(Lˣ, Lʸ, H, τₒ, λʳ, θᴱ)
    gravity::FT = grav(param_set)

    #- set model time-step:
    dt_fast = 240
    dt_slow = 5400
    # dt_fast = 300
    # dt_slow = 300
    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow

    model = OceanModel{FT}(
        prob,
        grav = gravity,
        cʰ = cʰ,
        add_fast_substeps = add_fast_substeps,
    )
    # model = OceanModel{FT}(prob, cʰ = cʰ, fₒ = FT(0), β = FT(0) )
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(1e3), νᶻ = FT(1e-3) )
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(0), fₒ = FT(0), β = FT(0) )

    barotropicmodel = BarotropicModel(model)

    minΔx = min_node_distance(grid_3D, HorizontalDirection())
    minΔz = min_node_distance(grid_3D, VerticalDirection())
    #- 2 horiz directions
    gravity_max_dT = 1 / (2 * sqrt(gravity * H) / minΔx)
    # dt_fast = minimum([gravity_max_dT])

    #- 2 horiz directions + harmonic visc or diffusion: 2^2 factor in CFL:
    viscous_max_dT = 1 / (2 * model.νʰ / minΔx^2 + model.νᶻ / minΔz^2) / 4
    diffusive_max_dT = 1 / (2 * model.κʰ / minΔx^2 + model.κᶻ / minΔz^2) / 4
    # dt_slow = minimum([diffusive_max_dT, viscous_max_dT])

    @info @sprintf(
        """Update
           Gravity Max-dT = %.1f
           Timestep       = %.1f""",
        gravity_max_dT,
        dt_fast
    )

    @info @sprintf(
        """Update
       Viscous   Max-dT = %.1f
       Diffusive Max-dT = %.1f
       Timestep      = %.1f""",
        viscous_max_dT,
        diffusive_max_dT,
        dt_slow
    )

    dg = OceanDGModel(
        model,
        grid_3D,
        # CentralNumericalFluxFirstOrder(),
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    barotropic_dg = DGModel(
        barotropicmodel,
        grid_2D,
        # CentralNumericalFluxFirstOrder(),
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg, FT(0); init_on_cpu = true)
    # update_auxiliary_state!(dg, model, Q_3D, FT(0))
    # update_auxiliary_state_gradient!(dg, model, Q_3D, FT(0))

    Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    lsrk_ocean = LSRK54CarpenterKennedy(dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_barotropic =
        LSRK54CarpenterKennedy(barotropic_dg, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = SplitExplicitLSRK2nSolver(lsrk_ocean, lsrk_barotropic)

    #-- Set up State Check call back for config state arrays, called every ntFreq time steps
    ntFreq = 1
    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [
            (Q_3D, "oce Q_3D"),
            (dg.state_auxiliary, "oce aux"),
            # (dg.diffstate,"oce diff",),
            # (lsrk_ocean.dQ,"oce_dQ",),
            # (dg.modeldata.tendency_dg.state_auxiliary,"tend Int aux",),
            # (dg.modeldata.conti3d_Q,"conti3d_Q",),
            (Q_2D, "baro Q_2D"),
            (barotropic_dg.state_auxiliary, "baro aux"),
        ],
        ntFreq;
        prec = 12,
    )
    # (barotropic_dg.diffstate,"baro diff",),
    # (lsrk_barotropic.dQ,"baro_dQ",)
    #--

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
    # solve!(Qvec, odesolver; timeend = timeend, callbacks = cbvector)
    cbv = (cbvector..., cbcs_dg)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbv)

    ## Enable the code block below to print table for use in reference value code
    ## reference value code sits in a file named $(@__FILE__)_refvals.jl. It is hand
    ## edited using code generated by block below when reference values are updated.
    regenRefVals = false
    if regenRefVals
        ## Print state statistics in format for use as reference values
        println(
            "# SC ========== Test number ",
            1,
            " reference values and precision match template. =======",
        )
        println("# SC ========== $(@__FILE__) test reference values ======================================")
        ClimateMachine.StateCheck.scprintref(cbcs_dg)
        println("# SC ====================================================================================")
    end

    ## Check results against reference if present
    checkRefVals = true
    if checkRefVals
        include("../refvals/simple_box_2dt_refvals.jl")
        refDat = (refVals[1], refPrecs[1])
        checkPass = ClimateMachine.StateCheck.scdocheck(cbcs_dg, refDat)
        checkPass ? checkRep = "Pass" : checkRep = "Fail"
        @info @sprintf("""Compare vs RefVals: %s""", checkRep)
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
        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)

        mycomm = Q.mpicomm
        ## Generate the pvtu file for these vtk files
        if MPI.Comm_rank(mpicomm) == 0 && MPI.Comm_size(mpicomm) > 1
            ## name of the pvtu file
            pvtuprefix = @sprintf("%s/%s/step%04d", vtkpath, span, step)
            ## name of each of the ranks vtk files
            prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                @sprintf("mpirank%04d_step%04d", i - 1, step)
            end
            writepvtu(
                pvtuprefix,
                prefixes,
                (statenames..., auxnames...),
                eltype(Q),
            )
            @info "Done writing VTK: $pvtuprefix"
        end

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
vtkpath = "vtk_split"

const timeend = 5 * 24 * 3600 # s
const tout = 24 * 3600 # s
#const timeend = 6 * 3600 # s
#const tout = 6 * 3600 # s

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

#const cʰ = sqrt(gravity * H)
const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

#- inverse ratio of additional fast time steps (for weighted average)
#  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
# e.g., = 1 --> 100% more ; = 2 --> 50% more ; = 3 --> 33% more ...
add_fast_substeps = 2

#- number of Implicit vertical-diffusion sub-time-steps within one model full time-step
# default = 0 : disable implicit vertical diffusion
numImplSteps = 0

#const τₒ = 2e-1  # (Pa = N/m^2)
# since we are using old BC (with factor of 2), take only half:
const τₒ = 1e-1
const λʳ = 10 // 86400 # m/s
const θᴱ = 10    # deg.C

main()
