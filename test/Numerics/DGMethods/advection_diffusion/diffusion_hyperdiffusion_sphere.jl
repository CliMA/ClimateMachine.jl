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

struct DiffusionSphere <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    problem::DiffusionSphere,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    IM = SMatrix{3, 3, FT}(I)
    ZM = zeros(SMatrix{3, 3, FT})
    μ = FT(1 / 10000)
    aux.diffusion.D = μ * hcat(IM, ZM)
    aux.hyperdiffusion.H = μ * hcat(ZM, IM)
end

function initial_condition!(problem::DiffusionSphere, state, aux, localgeo, t)
    coord = localgeo.coord
    x, y, z = coord
    r = norm(coord)
    θ = atan(sqrt(x^2 + y^2), z)
    φ = atan(y, x)
    @inbounds begin
        # m = 1 l = 2 spherical harmonic
        ρ₀ = cos(φ) * sin(θ) * cos(θ)
        l = 2
        c = l * (l + 1) / r^2
        μ = aux.diffusion.D[1]
        state.ρ = (ρ₀ * exp(-c * μ * t), ρ₀ * exp(-c^2 * μ * t))
    end
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

function run(mpicomm, ArrayType, topl, N, timeend, FT, vtkdir, outputtime)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = cubedshellwarp,
    )

    dx = min_node_distance(grid)
    dt = 300 * dx^4
    @info "time step" dt
    dt = outputtime / ceil(Int64, outputtime / dt)

    model = AdvectionDiffusion{3}(
        DiffusionSphere(),
        advection = false,
        diffusion = true,
        hyperdiffusion = true,
        num_equations = 2,
    )
    dg = DGModel(
        model,
        grid,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        diffusion_direction = HorizontalDirection(),
    )

    Q = init_ode_state(dg, FT(0))

    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q, dims = (1, 3))
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0[1]

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
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "diffusion_sphere")

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
                "diffusion_sphere",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = callbacks)

    # Print some end of the simulation information
    engf = norm(Q, dims = (1, 3))
    Qe = init_ode_state(dg, FT(timeend))

    engfe = norm(Qe, dims = (1, 3))
    errf = norm(Q .- Qe, dims = (1, 3))

    metrics = @. (engf, engf / eng0, engf - eng0, errf, errf / engfe)

    @info @sprintf """Finished
    Diffusion:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
    HyperDiffusion:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
      """ first.(metrics)... last.(metrics)...
    errf
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    polynomialorder = 3
    base_num_elem = 4

    expected_result = Dict()
    expected_result[Diffusion, 1] = 2.6002775334282785e-06
    expected_result[Diffusion, 2] = 4.1602462623838931e-07
    expected_result[Diffusion, 3] = 5.8997889725858304e-08

    expected_result[HyperDiffusion, 1] = 9.3032674316702424e-05
    expected_result[HyperDiffusion, 2] = 1.0149377619104328e-05
    expected_result[HyperDiffusion, 3] = 1.2985297333025857e-06

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 3 :
        1
    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            result = Dict()
            for l in 1:numlevels
                Ne = 2^(l - 1) * base_num_elem
                vert_range = grid1d(1, 2, nelem = 1)
                topl = StackedCubedSphereTopology(
                    mpicomm,
                    Ne,
                    vert_range,
                    boundary = (0, 0),
                )

                timeend = FT(1)
                outputtime = FT(2)

                @info (ArrayType, FT)
                vtkdir =
                    output ?
                    "vtk_diffusion_sphere" *
                    "_poly$(polynomialorder)" *
                    "_$(ArrayType)_$(FT)" *
                    "_level$(l)" :
                    nothing
                result[l] = run(
                    mpicomm,
                    ArrayType,
                    topl,
                    polynomialorder,
                    timeend,
                    FT,
                    vtkdir,
                    outputtime,
                )
                @test result[l][1] ≈ expected_result[Diffusion, l]
                @test result[l][2] ≈ expected_result[HyperDiffusion, l]
            end
            @info begin
                msg = ""
                for l in 1:(numlevels - 1)
                    rate = @. log2(result[l]) - log2(result[l + 1])
                    msg *= @sprintf(
                        "\n  rates for level %d Diffusion = %e",
                        l,
                        rate[1]
                    )
                    msg *= @sprintf(", HyperDiffusion = %e\n", rate[2])
                end
                msg
            end
        end
    end
end

nothing
