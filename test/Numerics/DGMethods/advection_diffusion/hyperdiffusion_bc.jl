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

struct ConstantHyperDiffusion{FT} <: AdvectionDiffusionProblem
    μ::FT
    k::SVector{3, FT}
end

function init_velocity_diffusion!(
    problem::ConstantHyperDiffusion,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.hyperdiffusion.H = problem.μ * SMatrix{3, 3, FT}(I)
end

function initial_condition!(
    problem::ConstantHyperDiffusion,
    state,
    aux,
    localgeo,
    t,
)
    x, y, z = localgeo.coord
    k = problem.k
    k2 = sum(k .^ 2)
    @inbounds begin
        state.ρ =
            cos(k[1] * x) *
            cos(k[2] * y) *
            cos(k[3] * z) *
            exp(-k2^2 * problem.μ * t)
    end
end

# Boundary conditions data
inhomogeneous_data!(::Val{0}, p::ConstantHyperDiffusion, x...) =
    initial_condition!(p, x...)
function inhomogeneous_data!(
    ::Val{1},
    problem::ConstantHyperDiffusion,
    ∇state,
    aux,
    x,
    t,
)
    k = problem.k
    k2 = sum(k .^ 2)
    ∇state.ρ =
        -SVector(
            k[1] * sin(k[1] * x[1]) * cos(k[2] * x[2]) * cos(k[3] * x[3]),
            k[2] * cos(k[1] * x[1]) * sin(k[2] * x[2]) * cos(k[3] * x[3]),
            k[3] * cos(k[1] * x[1]) * cos(k[2] * x[2]) * sin(k[3] * x[3]),
        ) * exp(-k2^2 * problem.μ * t)
end
function inhomogeneous_data!(
    ::Val{2},
    problem::ConstantHyperDiffusion,
    Δstate,
    aux,
    x,
    t,
)
    k = problem.k
    k2 = sum(k .^ 2)
    Δstate.ρ =
        -k2 *
        cos(k[1] * x[1]) *
        cos(k[2] * x[2]) *
        cos(k[3] * x[3]) *
        exp(-k2^2 * problem.μ * t)
end
function inhomogeneous_data!(
    ::Val{3},
    problem::ConstantHyperDiffusion,
    ∇Δstate,
    aux,
    x,
    t,
)
    k = problem.k
    k2 = sum(k .^ 2)
    ∇Δstate.ρ =
        k2 *
        SVector(
            k[1] * sin(k[1] * x[1]) * cos(k[2] * x[2]) * cos(k[3] * x[3]),
            k[2] * cos(k[1] * x[1]) * sin(k[2] * x[2]) * cos(k[3] * x[3]),
            k[3] * cos(k[1] * x[1]) * cos(k[2] * x[2]) * sin(k[3] * x[3]),
        ) *
        exp(-k2^2 * problem.μ * t)
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

    μ = 1 // 1000
    dx = min_node_distance(grid)
    dt = dx^4 / 100 / μ
    dt = outputtime / ceil(Int64, outputtime / dt)

    bcx1 = (InhomogeneousBC{0}(), InhomogeneousBC{2}())
    bcx2 = (InhomogeneousBC{0}(), InhomogeneousBC{1}())

    bcy1 = (InhomogeneousBC{3}(), InhomogeneousBC{1}())
    bcy2 = (InhomogeneousBC{3}(), InhomogeneousBC{2}())

    bcz1 = (HomogeneousBC{3}(), HomogeneousBC{1}())
    bcz2 = (HomogeneousBC{3}(), HomogeneousBC{1}())

    k = SVector(1, 1, 0)
    bcs = (bcx1, bcx2, bcy1, bcy2, bcz1, bcz2)
    model = AdvectionDiffusion{dim}(
        ConstantHyperDiffusion{FT}(μ, k),
        bcs;
        advection = false,
        diffusion = false,
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
    dim      = %d
    dt       = %.16e
    norm(Q₀) = %.16e""" dim dt eng0

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

    polynomialorder = 4
    base_num_elem = 4

    expected_result = Dict()
    expected_result[2, 1, Float64] = 1.1666787574326038e-03
    expected_result[2, 2, Float64] = 8.5948289965604964e-05
    expected_result[2, 3, Float64] = 2.5117423516568199e-06

    expected_result[3, 1, Float64] = 3.0867418520680958e-03
    expected_result[3, 2, Float64] = 2.2739780086168765e-04
    expected_result[3, 3, Float64] = 6.6454456228180395e-06

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 3 :
        1

    for FT in (Float64,)
        result = zeros(FT, numlevels)
        for dim in (2, 3)
            for l in 1:numlevels
                Ne = 2^(l - 1) * base_num_elem
                xrange = range(FT(2); length = Ne + 1, stop = FT(9))
                brickrange = ntuple(j -> xrange, dim)
                periodicity = ntuple(j -> false, dim)
                connectivity = dim == 2 ? :face : :full
                boundary = ((1, 2), (3, 4), (5, 6))[1:dim]
                topl = StackedBrickTopology(
                    mpicomm,
                    brickrange;
                    periodicity,
                    boundary,
                    connectivity,
                )
                timeend = 1
                outputtime = timeend

                @info (ArrayType, FT)
                vtkdir =
                    output ?
                    "vtk_hyperdiffusion_bc" *
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
                @test result[l] ≈ FT(expected_result[dim, l, FT])
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
end

nothing
