using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.ESDGMethods
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.DGMethods.NumericalFluxes:
    normal_boundary_flux_second_order!

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("DryAtmos.jl")

abstract type AbstractProblem end
struct PeriodicIsothermalTest <: AbstractProblem end

Base.@kwdef struct IsothermalPeriodicExample{FT}
    η::FT = 1 // 10000
end

function (setup::IsothermalPeriodicExample)(
    bl::DryAtmosphereModel,
    state,
    aux,
    coords,
    t,
)
    FT = eltype(state)
    η = setup.η
    @inbounds x = coords[1]
    aux.Φ = sin(2π * x)
    state.ρ = exp(-aux.Φ)
    state.ρu = FT(0)
    p = state.ρ + η * exp(-100 * (x - FT(1 // 2))^2)
    state.ρe = totalenergy(state.ρ, state.ρu, p, aux.Φ)
    nothing
end

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 5

    timeend = 1
    FT = Float64
    result = run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        ArrayType,
        FT,
    )
end

function run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    ArrayType,
    FT,
)

    setup = IsothermalPeriodicExample{FT}()

    model = DryAtmosModel()

    dim = 3
    brickrange = ntuple(j -> range(FT(0); length = Ne + 1, stop = 1), dim)
    periodicity = ntuple(j -> true, dim)

    topl = StackedBrickTopology(mpicomm, brickrange; periodicity = periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem = PeriodicIsothermalTest()
    esdg = ESDGModel(
        model,
        problem,
        grid,
        # Need to construct state aux with geopotential
    )

    # determine the time step
    Ne = 2^(numelem_vert - 1) * numelem_horz
    dt = 1 / (Ne * polynomialorder^2)^2

    Q = init_ode_state(esdg, FT(0))
    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

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

    solve!(Q, odesolver; callbacks = callbacks)

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end

main()
