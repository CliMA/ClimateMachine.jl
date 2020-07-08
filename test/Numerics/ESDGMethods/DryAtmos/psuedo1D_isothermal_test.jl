using MPI
using ClimateMachine
using Logging
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.Mesh.Topologies: BrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.NumericalFluxes: normal_boundary_flux_second_order!
import ClimateMachine.BalanceLaws: init_state_conservative!
import ClimateMachine.ODESolvers: LSRK144NiegemannDiehlBusch, solve!, gettime
using StaticArrays: @SVector
using LazyArrays

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("DryAtmos.jl")

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "IsothermalPeriodicExample",
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

struct PeriodicOrientation <: Orientation end

function init_state_auxiliary!(
    ::DryAtmosModel{dim, PeriodicOrientation},
    state_auxiliary,
    geom,
) where {dim}
    FT = eltype(state_auxiliary)
    @inbounds state_auxiliary.Φ = sin(2π * geom.coord[1])
end

Base.@kwdef struct IsothermalPeriodicExample{FT}
    η::FT = 1 // 10000
end

function init_state_conservative!(
    m::DryAtmosModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    FT = eltype(state)
    η::FT = 1 // 10000
    @inbounds x = coords[1]
    state.ρ = exp(-aux.Φ)
    state.ρu = @SVector [FT(0), FT(0), FT(0)]
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

    dim = 2
    Ne = [10, 10, 10]
    brickrange = ntuple(j -> range(FT(0); length = Ne[j] + 1, stop = 1), dim)
    periodicity = ntuple(j -> true, dim)

    topl = BrickTopology(mpicomm, brickrange; periodicity = periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )
    model = DryAtmosModel{dim}(PeriodicOrientation())
    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = EntropyConservative(),
    )

    # determine the time step
    dt = 1 / (Ne[1] * polynomialorder^2)^2
    timeend = 100dt
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

    compute_entropy(esdg, Q)
    solve!(Q, odesolver; callbacks = callbacks, timeend = timeend)

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
