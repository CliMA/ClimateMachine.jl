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
using ClimateMachine.VariableTemplates: flattenednames
import ClimateMachine.NumericalFluxes: normal_boundary_flux_second_order!
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

struct PeriodicOrientation <: Orientation end

function init_state_auxiliary!(
    ::DryAtmosModel,
    ::PeriodicOrientation,
    state_auxiliary,
    geom,
) where {dim}
    FT = eltype(state_auxiliary)
    @inbounds state_auxiliary.Φ = sin(2π * geom.coord[1])
end

Base.@kwdef struct IsothermalPeriodicExample{FT} <: AbstractDryAtmosProblem
    η::FT = 1 // 10000
end

function init_state_prognostic!(
    m::DryAtmosModel,
    problem::IsothermalPeriodicExample,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
    args...,
)
    FT = eltype(state)
    @inbounds x = localgeo.coord[1]
    state.ρ = exp(-aux.Φ)
    state.ρu = @SVector [FT(0), FT(0), FT(0)]
    p = state.ρ + problem.η * exp(-100 * (x - FT(1 / 2))^2)
    state.ρe = totalenergy(state.ρ, state.ρu, p, aux.Φ)
    nothing
end

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD
    polynomialorder = 4
    Ne = 10

    timeend = 1
    FT = Float64
    result = run(mpicomm, polynomialorder, Ne, timeend, ArrayType, FT)
end

function run(mpicomm, polynomialorder, Nelem, timeend, ArrayType, FT)

    problem = IsothermalPeriodicExample{FT}()

    dim = 2
    Ne = [Nelem, Nelem, Nelem]
    brickrange = ntuple(j -> range(FT(0); length = Ne[j] + 1, stop = 1), dim)
    periodicity = ntuple(j -> true, dim)
    warpfun =
        (x1, x2, x3) -> begin
            x1 = 2x1 - 1
            x2 = 2x2 - 1
            dim == 3 && (x3 = 2x3 - 1)
            α = (4 / π) * (1 - x1^2) * (1 - x2^2) * (1 - x3^2)
            # Rotate by α with x1 and x2
            x1, x2 = cos(α) * x1 - sin(α) * x2, sin(α) * x1 + cos(α) * x2
            # Rotate by α with x1 and x3
            if dim == 3
                x1, x3 = cos(α) * x1 - sin(α) * x3, sin(α) * x1 + cos(α) * x3
            end
            x1 = (x1 + 1) / 2
            x2 = (x2 + 1) / 2
            dim == 3 && (x3 = (x3 + 1) / 2)
            return (x1, x2, x3)
        end

    topl = BrickTopology(mpicomm, brickrange; periodicity = periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = warpfun,
    )
    model = DryAtmosModel{dim}(PeriodicOrientation(), problem)
    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = EntropyConservative(),
    )

    # determine the time step
    dt = 1 / (Ne[1] * polynomialorder^2)^2
    Q = init_ode_state(esdg, FT(0))
    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem         = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder Ne[1] dt eng0

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

    output_vtk = true
    if output_vtk
        # create vtk dir
        vtkdir =
            "psuedo1D_isothermal" *
            "_poly$(polynomialorder)_dims$(dim)_$(ArrayType)_$(FT)_nelem$(Nelem)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)

        # setup the output callback
        outputtime = 1 / 10
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(esdg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

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

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    esdg,
    Q,
    model,
    testname = "psuedo1D_isothermal",
)
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

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames)#; number_sample_points = 10)

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
