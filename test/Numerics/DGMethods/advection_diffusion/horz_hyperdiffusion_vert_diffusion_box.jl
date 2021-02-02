using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.Mesh.Grids: min_node_distance

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

include("advection_diffusion_model.jl")

struct HorzHyperDiffVertDiff{FT} <: AdvectionDiffusionProblem
    α::FT
    β::FT
end

function init_velocity_diffusion!(
    problem::HorzHyperDiffVertDiff,
    aux::Vars,
    geom::LocalGeometry,
)
    x = geom.coord
    k = SVector(0, 0, 1)
    aux.hyperdiffusion.H = problem.α * (I - k * k')
    aux.hyperdiffusion.P = I - k * k'
    aux.diffusion.D = problem.β * k * k'
end

function initial_condition!(
    problem::HorzHyperDiffVertDiff,
    state,
    aux,
    localgeo,
    t,
)
    FT = eltype(aux)
    m = SVector{3, FT}(1, 1, 1)
    @inbounds c = problem.α * (m[1]^2 + m[2]^2)^2 + problem.β * m[3]^2
    x = localgeo.coord
    state.ρ = sin(dot(m, x)) * exp(-c * t)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    exactnames = statenames .* "_exact"

    writevtk(filename, Q, dg, statenames, Qe, exactnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames..., exactnames...),
            eltype(Q),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end


function test_run(
    mpicomm,
    ArrayType,
    dim,
    topl,
    N,
    timeend,
    FT,
    vtkdir,
    outputtime,
)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    dx = min_node_distance(grid)
    α = FT(1 // 100)
    β = FT(1 // 200)
    dt = dx^4 / 100 / (α + β)
    @info "time step" dt
    dt = outputtime / ceil(Int64, outputtime / dt)

    model = AdvectionDiffusion{dim}(
        HorzHyperDiffVertDiff(α, β);
        advection = false,
        diffusion = true,
        hyperdiffusion = true,
    )
    dg = DGModel(
        model,
        grid,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                gettime(lsrk),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    callbacks = (cbinfo,)
    if ~isnothing(vtkdir)
        # create vtk dir
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "hyperdiffusion")

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(lsrk))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dg,
                Q,
                Qe,
                model,
                "hyperdiffusion",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = callbacks)

    # Print some end of the simulation information
    engf = norm(Q)
    Qe = init_ode_state(dg, FT(timeend))

    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 3 :
        1

    polynomialorder = 4
    base_num_elem = 4
    dim = 3

    expected_result = Dict()

    #numlevels = integration_testing ? 3 : 1
    numlevels = 3
    for FT in (Float64,)
        result = zeros(FT, numlevels)
        for l in 1:numlevels
            Ne = 2^(l - 1) * base_num_elem
            xrange = range(FT(0); length = Ne + 1, stop = FT(2pi))
            brickrange = ntuple(j -> xrange, dim)
            periodicity = ntuple(j -> true, dim)
            topl = StackedBrickTopology(
                mpicomm,
                brickrange;
                periodicity = periodicity,
            )
            timeend = 1
            outputtime = 1

            @info (ArrayType, FT, dim)
            vtkdir =
                output ?
                "vtk_hyperdiffusion" *
                "_poly$(polynomialorder)" *
                "_dim$(dim)_$(ArrayType)_$(FT)" *
                "_level$(l)" :
                nothing
            result[l] = test_run(
                mpicomm,
                ArrayType,
                dim,
                topl,
                polynomialorder,
                timeend,
                FT,
                vtkdir,
                outputtime,
            )
        end
        @info begin
            msg = ""
            for l in 1:(numlevels - 1)
                rate = log2(result[l]) - log2(result[l + 1])
                msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
            end
            msg
        end
    end
end

nothing
