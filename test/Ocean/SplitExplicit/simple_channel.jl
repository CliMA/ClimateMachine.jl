#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.BalanceLaws
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
using ClimateMachine.Checkpoint

import ClimateMachine.Ocean.SplitExplicit01:
    velocity_flux, temperature_flux, sponge_relaxation

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
    OceanFloorLinearDrag,
    OceanFloorNoSlip,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing
import ClimateMachine.DGMethods:
    update_auxiliary_state!, update_auxiliary_state_gradient!, VerticalDirection
# using GPUifyLoops

const ArrayType = ClimateMachine.array_type()

struct EddyingChannel{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    h::T # relaxation e-folding length
    efl::T # for sponge relaxation
    σʳ::T # relaxation time for sponge relaxation
    τₒ::T
    λʳ::T
    θᴱ::T
    λᴰ::T # bottom drag coefficient
end

@inline function ocean_boundary_state!(
    m::OceanModel,
    p::EddyingChannel,
    bctype,
    x...,
)
    if bctype == 1
        ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
    elseif bctype == 2
        ocean_boundary_state!(m, OceanFloorLinearDrag(), x...)
    elseif bctype == 3
        ocean_boundary_state!(m, OceanSurfaceNoStressNoForcing(), x...)
    end
end

@inline function ocean_boundary_state!(
    m::Continuity3dModel,
    p::EddyingChannel,
    bctype,
    x...,
)
    #if bctype == 1
    ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
    #end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::EddyingChannel,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
end

function ocean_init_aux!(m::OceanModel, p::EddyingChannel, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]
    @inbounds A.z = geom.coord[3]

    # not sure if this is needed but getting weird intialization stuff
    A.w = -0
    A.pkin = -0
    A.wz0 = -0
    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::EddyingChannel, A, geom)
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

function main(; restart = 0)
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
        brickrange_2D,
        periodicity = (true, false),
        boundary = ((0, 0), (1, 1)),
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
        periodicity = (true, false, false),
        boundary = ((0, 0), (1, 1), (2, 3)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob = EddyingChannel{FT}(Lˣ, Lʸ, H, h, efl, σʳ, τₒ, λʳ, θᴱ, λᴰ)
    gravity::FT = grav(param_set)

    #- set model time-step:    
    @show minΔx = min_node_distance(grid_3D, HorizontalDirection())
    @show minΔz = min_node_distance(grid_3D, VerticalDirection())

    @show dt_fast = floor(Int, 1 / (2 * sqrt(gravity * H) / minΔx)) # / 4
    @show dt_slow = floor(Int, minΔx / 15) # / 4
    # minimum([floor(Int, minΔx / 16), floor(Int, minΔz / 16)])

    @show nout = ceil(Int64, tout / dt_slow)
    @show dt_slow = tout / nout
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow

    model_3D = OceanModel{FT}(
        prob,
        grav = gravity,
        cʰ = FT(cʰ),
        νʰ = FT(νʰ),
        νᶻ = FT(νᶻ),
        κʰ = FT(κʰ),
        κᶻ = FT(κᶻ),
        κᶜ = FT(κᶜ),
        fₒ = FT(fₒ),
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
    )

    model_2D = BarotropicModel(model_3D)

    #- 2 horiz directions
    gravity_max_dT = 1 / (2 * sqrt(gravity * H) / minΔx)
    # dt_fast = minimum([gravity_max_dT])

    @info @sprintf(
        """Update
           Gravity Max-dT = %.1f
           Timestep       = %.1f""",
        gravity_max_dT,
        dt_fast
    )

    #- 2 horiz directions + harmonic visc or diffusion: 2^2 factor in CFL:
    viscous_max_dT = 1 / (2 * model_3D.νʰ / minΔx^2 + model_3D.νᶻ / minΔz^2) / 4
    diffusive_max_dT =
        1 / (2 * model_3D.κʰ / minΔx^2 + model_3D.κᶻ / minΔz^2) / 4
    # dt_slow = minimum([diffusive_max_dT, viscous_max_dT])

    @info @sprintf(
        """Update
       Viscous   Max-dT = %.1f
       Diffusive Max-dT = %.1f
       Timestep      = %.1f""",
        viscous_max_dT,
        diffusive_max_dT,
        dt_slow
    )

    if restart > 0
        direction = EveryDirection()
        Q_3D, A_3D, t0 =
            read_checkpoint(vtkpath, "baroclinic", ArrayType, mpicomm, restart)
        Q_2D, A_2D, _ =
            read_checkpoint(vtkpath, "barotropic", ArrayType, mpicomm, restart)

        A_3D = restart_auxiliary_state(model_3D, grid_3D, A_3D, direction)
        A_2D = restart_auxiliary_state(model_2D, grid_2D, A_2D, direction)

        dg_3D = OceanDGModel(
            model_3D,
            grid_3D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient();
            state_auxiliary = A_3D,
        )
        dg_2D = DGModel(
            model_2D,
            grid_2D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            state_auxiliary = A_2D,
        )

        Q_3D = restart_ode_state(dg_3D, Q_3D; init_on_cpu = true)
        Q_2D = restart_ode_state(dg_2D, Q_2D; init_on_cpu = true)

        lsrk_3D = LSRK54CarpenterKennedy(dg_3D, Q_3D, dt = dt_slow, t0 = t0)
        lsrk_2D = LSRK54CarpenterKennedy(dg_2D, Q_2D, dt = dt_fast, t0 = t0)

        timeendlocal = timeend + t0
    else
        dg_3D = OceanDGModel(
            model_3D,
            grid_3D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
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

        timeendlocal = timeend
    end

    odesolver = SplitExplicitLSRK2nSolver(lsrk_3D, lsrk_2D)

    #-- Set up State Check call back for config state arrays, called every ntFreq time steps
    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [
            (Q_3D, "oce Q_3D"),
            (dg_3D.state_auxiliary, "oce aux"),
            (Q_2D, "baro Q_2D"),
            (dg_2D.state_auxiliary, "baro aux"),
        ],
        nout;
        prec = 12,
    )
    #--

    step = [restart, restart, restart + 1, restart + 1]
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
    cbv = (cbvector..., cbcs_dg)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbv)

    ## Enable the code block below to print table for use in reference value code
    ## reference value code sits in a file named $(@__FILE__)_refvals.jl. It is hand
    ## edited using code generated by block below when reference values are updated.
    ## Print state statistics in format for use as reference values
    println(
        "# SC ========== Test number ",
        1,
        " reference values and precision match template. =======",
    )
    println("# SC ========== $(@__FILE__) test reference values ======================================")
    ClimateMachine.StateCheck.scprintref(cbcs_dg)
    println("# SC ====================================================================================")

    ## Check results against reference if present
    checkRefVals = false
    if checkRefVals
        include("../refvals/simple_box_ivd_refvals.jl")
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
    # if isdir(vtkpath)
    # rm(vtkpath, recursive = true)
    # end
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
            writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))
            @info "Done writing VTK: $pvtuprefix"
        end

    end

    do_output("slow", step[1], model_slow, dg_slow, Q_slow)
    step[1] += 1
    cbvtk_slow = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("slow", step[1], model_slow, dg_slow, Q_slow)
        step[1] += 1
        nothing
    end

    do_output("fast", step[2], model_fast, dg_fast, Q_fast)
    step[2] += 1
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

    cb_checkpoint = GenericCallbacks.EveryXSimulationSteps(nout) do
        write_checkpoint(
            Q_slow,
            dg_slow.state_auxiliary,
            odesolver,
            vtkpath,
            "baroclinic",
            mpicomm,
            step[3],
        )

        write_checkpoint(
            Q_fast,
            dg_fast.state_auxiliary,
            odesolver,
            vtkpath,
            "barotropic",
            mpicomm,
            step[4],
        )

        # rm_checkpoint(vtkpath, "baroclinic", mpicomm, step[3] - 1)

        # rm_checkpoint(vtkpath, "barotropic", mpicomm, step[4] - 1)

        step[3] += 1
        step[4] += 1
        nothing
    end

    return (cbvtk_slow, cbvtk_fast, cbinfo, cb_checkpoint)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_channel_simplestrat_small_tall_notempadvec"

const timeend = 60 * 24 * 3600 # s
const tout = 24 * 3600 # s

const N = 4
const Nˣ = 4 # 48 / 12
const Nʸ = 4 # 48 / 12 
const Nᶻ = 60
const Lˣ = 1e6 / 12  # m
const Lʸ = 1e6 / 12  # m
const H = 3000  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

#const cʰ = sqrt(gravity * H)
const cʰ = 1  # typical of ocean internal-wave speed 
const cᶻ = 0

const νʰ = 100.0
const νᶻ = 0.02
const κʰ = 100.0
const κᶻ = 0.02
const κᶜ = 0.1

const fₒ = -1e-4

#- inverse ratio of additional fast time steps (for weighted average)
#  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
# e.g., = 1 --> 100% more ; = 2 --> 50% more ; = 3 --> 33% more ...
add_fast_substeps = 2

#- number of Implicit vertical-diffusion sub-time-steps within one model full time-step
# default = 0 : disable implicit vertical diffusion
numImplSteps = 5

# fixed factor of 2 in BC
const τₒ = 2e-1  # (Pa = N/m^2)
const σʳ = 1 // (7 * 86400) # 1/s
const λʳ = 10 // 86400 # m/s
const θᴱ = 10    # deg.C
const λᴰ = 1e-3   # drag coefficent, m/s
const h = 1000 # m
const efl = 50e3 # e folding length 

init_temp(θᴱ, Lʸ, h, H, y, z) =
    0.5 * θᴱ * (1 - cos(π * y / Lʸ)) * (exp(z / h) - exp(-H / h)) /
    (1 - exp(-H / h))
init_temp(θᴱ, h, H, z) = θᴱ * (exp(z / h) - exp(-H / h)) / (1 - exp(-H / h))

function ocean_init_state!(p::EddyingChannel, Q, A, coords, t)
    Q.u = @SVector [-0, -0]
    Q.η = -0
    # Q.θ = init_temp(p.θᴱ, p.Lʸ, p.h, p.H, A.y, A.z)
    Q.θ = init_temp(p.θᴱ, p.h, p.H, A.z)

    return nothing
end

@inline function velocity_flux(p::EddyingChannel, y, ρ)
    Σ = (p.Lʸ^2 / 32)
    term1 = exp(-(y - p.Lʸ / 2)^2 / Σ)
    term2 = exp(-(p.Lʸ / 2)^2 / Σ)
    # return -(p.τₒ / ρ) * (term1 - term2)
    return eltype(ρ)(0)
end

@inline function temperature_flux(p::EddyingChannel, y, θ)
    θʳ = p.θᴱ * y / p.Lʸ

    # return p.λʳ * (θʳ - θ)
    return eltype(θ)(0)
end

@inline function sponge_relaxation(m::OceanModel, p::EddyingChannel, S, Q, A)
    θʳ = init_temp(p.θᴱ, p.Lʸ, p.h, p.H, A.y, A.z)
    # θʳ = init_temp(p.θᴱ, p.h, p.H, A.z)
    # S.θ = p.σʳ * (θʳ - Q.θ) * exp((A.y - p.Lʸ) / p.efl)
    S.θ = 0

    return nothing
end

main(restart = 0)
