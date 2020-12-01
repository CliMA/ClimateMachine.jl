import Printf: @sprintf
import LinearAlgebra: dot, norm
import Dates
import MPI

import ClimateMachine
import ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxGradient
import ClimateMachine.DGMethods: DGModel, init_ode_state
import ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
import ClimateMachine.MPIStateArrays: MPIStateArray, euclidean_distance
import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
import ClimateMachine.Mesh.Topologies: StackedBrickTopology
import ClimateMachine.ODESolvers: LSRK54CarpenterKennedy, solve!, gettime
import ClimateMachine.VTK: writevtk, writepvtu


if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end
const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n, α} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{n, α},
    aux::Vars,
    geom::LocalGeometry,
) where {n, α}
    # Direction of flow is n with magnitude α
    aux.advection.u = α * n
end

function initial_condition!(
    ::Pseudo1D{n, α},
    state,
    aux,
    localgeo,
    t,
) where {n, α}
    ξn = dot(n, localgeo.coord)
    state.ρ = sin((ξn - α * t) * pi)
end
Dirichlet_data!(P::Pseudo1D, x...) = initial_condition!(P, x...)

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

    ## generate the pvtu file for these vtk files
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
    dt,
    n,
    α,
    vtkdir,
    outputtime,
)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    model = AdvectionDiffusion{dim}(Pseudo1D{n, α}(), diffusion = false)
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(Dates.now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = Dates.now()
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
        do_output(
            mpicomm,
            vtkdir,
            vtkstep,
            dg,
            Q,
            Q,
            model,
            "advection_diffusion",
        )

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
                "advection_diffusion",
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

    base_num_elem = 4

    expected_result = Dict()
    expected_result[2, 1, Float64] = 3.1975803201004632e-01
    expected_result[2, 2, Float64] = 1.8627674551822404e-01
    expected_result[2, 3, Float64] = 1.0405818305399551e-01
    expected_result[2, 4, Float64] = 5.5997036645317050e-02
    expected_result[3, 1, Float64] = 2.7529128252186852e-01
    expected_result[3, 2, Float64] = 1.6737691447662753e-01
    expected_result[3, 3, Float64] = 9.6809054521915947e-02
    expected_result[3, 4, Float64] = 5.3410417324053723e-02
    expected_result[2, 1, Float32] = 3.1975793838500977e-01
    expected_result[2, 2, Float32] = 1.8627668917179108e-01
    expected_result[2, 3, Float32] = 1.0405827313661575e-01
    expected_result[2, 4, Float32] = 5.5997163057327271e-02
    expected_result[3, 1, Float32] = 2.7529123425483704e-01
    expected_result[3, 2, Float32] = 1.6737693548202515e-01
    expected_result[3, 3, Float32] = 9.6808202564716339e-02
    expected_result[3, 4, Float32] = 5.3404398262500763e-02

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            numlevels =
                integration_testing ||
                ClimateMachine.Settings.integration_testing ?
                4 : 1
            result = zeros(FT, numlevels)
            for dim in 2:3
                polynomialorder = (ntuple(j -> 2, dim - 1)..., 0)
                n =
                    dim == 2 ? SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0) :
                    SVector{3, FT}(1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))
                α = FT(1)
                for l in 1:numlevels
                    Ne = 2^(l - 1) * base_num_elem
                    brickrange = ntuple(
                        j -> range(FT(-1); length = Ne + 1, stop = 1),
                        dim,
                    )
                    periodicity = ntuple(j -> false, dim)
                    bc = ntuple(j -> (1, 2), dim)
                    topl = StackedBrickTopology(
                        mpicomm,
                        brickrange;
                        periodicity = periodicity,
                        boundary = bc,
                    )
                    dt = (α / 4) / (Ne * max(1, maximum(polynomialorder))^2)
                    @info "time step" dt

                    timeend = FT(1 // 4)
                    outputtime = timeend

                    dt = outputtime / ceil(Int64, outputtime / dt)

                    @info (ArrayType, FT, dim)
                    vtkdir =
                        output ?
                        "vtk_advection" *
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
                        dt,
                        n,
                        α,
                        vtkdir,
                        outputtime,
                    )
                    @test result[l] ≈ FT(expected_result[dim, l, FT])
                    if l > 1
                        rate = log2(result[l - 1]) - log2(result[l])
                        @info @sprintf("rate for level %d = %e", l, rate)
                    end
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
end

nothing
