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

include("hyperdiffusion_model.jl")

struct SphereHyperDiffusion <: HyperDiffusionProblem end

function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.D = SMatrix{3, 3, FT}(I) / 10000
end

function initial_condition!(
    problem::SphereHyperDiffusion,
    state,
    aux,
    localgeo,
    t,
)
    coord = localgeo.coord
    x, y, z = coord
    r = norm(coord)
    θ = atan(sqrt(x^2 + y^2), z)
    φ = atan(y, x)
    @inbounds begin
        # m = 0 l = 1 spherical harmonic
        l = 1
        c = l * (l + 1) / r^2
        v = aux.D[1, 1]
        α = c^2 * v
        state.ρ = -c * sqrt(3 / π) * cos(θ) / 2 * exp(-α * t)
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
    dt = dx^2 / 20
    @info "time step" dt
    dt = outputtime / ceil(Int64, outputtime / dt)

    model = HyperDiffusion{3}(SphereHyperDiffusion())
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
    ClimateMachine.init(parse_clargs = true)
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    polynomialorder = 2
    base_num_elem = 2

    expected_result = Dict()

    numlevels = 4
    for FT in (Float64,)
        result = zeros(FT, numlevels)
        for l in 1:numlevels
            Ne = 2^(l - 1) * base_num_elem
            vert_range = grid1d(1, 2, nelem = 1)
            topl = StackedCubedSphereTopology(mpicomm, Ne, vert_range)

            timeend = 1.0
            outputtime = 100

            @info (ArrayType, FT)
            vtkdir = output ?
                "vtk_sphere_hyperdiffusion" *
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
