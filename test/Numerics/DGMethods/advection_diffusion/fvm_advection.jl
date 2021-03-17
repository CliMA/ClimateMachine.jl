import Printf: @sprintf
import LinearAlgebra: dot, norm
import Dates
import MPI

import ClimateMachine
import ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
import ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxGradient
import ClimateMachine.DGMethods: DGFVModel, init_ode_state
import ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
import ClimateMachine.MPIStateArrays: MPIStateArray, euclidean_distance
import ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid, EveryDirection
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
inhomogeneous_data!(::Val{0}, P::Pseudo1D, x...) = initial_condition!(P, x...)

function do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Qe, model, testname)
    ## Name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    exactnames = statenames .* "_exact"

    writevtk(filename, Q, dgfvm, statenames, Qe, exactnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## Name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## Name of each of the ranks vtk files
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
    fvmethod,
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
    bcs = (InhomogeneousBC{0}(),)
    model = AdvectionDiffusion{dim}(Pseudo1D{n, α}(), bcs, diffusion = false)
    dgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = EveryDirection(),
    )

    Q = init_ode_state(dgfvm, FT(0))

    lsrk = LSRK54CarpenterKennedy(dgfvm, Q; dt = dt, t0 = 0)

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
        # Create vtk dir
        mkpath(vtkdir)

        vtkstep = 0
        # Output initial step
        do_output(
            mpicomm,
            vtkdir,
            vtkstep,
            dgfvm,
            Q,
            Q,
            model,
            "advection_diffusion",
        )

        # Setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dgfvm, gettime(lsrk))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dgfvm,
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
    Qe = init_ode_state(dgfvm, FT(timeend))

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
    expected_result[2, 1, Float64, FVConstant()] = 1.0404261715459338e-01
    expected_result[2, 2, Float64, FVConstant()] = 5.5995868545685376e-02
    expected_result[2, 3, Float64, FVConstant()] = 2.9383695610072275e-02
    expected_result[2, 4, Float64, FVConstant()] = 1.5171779426843507e-02

    expected_result[2, 1, Float64, FVLinear()] = 8.3196657944903635e-02
    expected_result[2, 2, Float64, FVLinear()] = 3.9277132273521774e-02
    expected_result[2, 3, Float64, FVLinear()] = 1.7155773433218020e-02
    expected_result[2, 4, Float64, FVLinear()] = 7.6525022056819231e-03

    expected_result[3, 1, Float64, FVConstant()] = 9.6785620234054362e-02
    expected_result[3, 2, Float64, FVConstant()] = 5.3406412788842651e-02
    expected_result[3, 3, Float64, FVConstant()] = 2.8471535157235807e-02
    expected_result[3, 4, Float64, FVConstant()] = 1.4846239937398318e-02

    expected_result[3, 1, Float64, FVLinear()] = 8.5860120005258181e-02
    expected_result[3, 2, Float64, FVLinear()] = 4.2844889694123235e-02
    expected_result[3, 3, Float64, FVLinear()] = 1.9302295207100174e-02
    expected_result[3, 4, Float64, FVLinear()] = 8.6084633401356733e-03

    expected_result[2, 1, Float32, FVConstant()] = 1.0404255986213684e-01
    expected_result[2, 2, Float32, FVConstant()] = 5.5995877832174301e-02
    expected_result[2, 3, Float32, FVConstant()] = 2.9383875429630280e-02
    expected_result[2, 4, Float32, FVConstant()] = 1.5171864069998264e-02

    expected_result[2, 1, Float32, FVLinear()] = 8.3196602761745453e-02
    expected_result[2, 2, Float32, FVLinear()] = 3.9277125149965286e-02
    expected_result[2, 3, Float32, FVLinear()] = 1.7155680805444717e-02
    expected_result[2, 4, Float32, FVLinear()] = 7.6521718874573708e-03

    expected_result[3, 1, Float32, FVConstant()] = 9.6785508096218109e-02
    expected_result[3, 2, Float32, FVConstant()] = 5.3406376391649246e-02
    expected_result[3, 3, Float32, FVConstant()] = 2.8471505269408226e-02
    expected_result[3, 4, Float32, FVConstant()] = 1.4849635772407055e-02

    expected_result[3, 1, Float32, FVLinear()] = 8.5860058665275574e-02
    expected_result[3, 2, Float32, FVLinear()] = 4.2844854295253754e-02
    expected_result[3, 3, Float32, FVLinear()] = 1.9302234053611755e-02
    expected_result[3, 4, Float32, FVLinear()] = 8.6139924824237823e-03

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            numlevels =
                integration_testing ||
                ClimateMachine.Settings.integration_testing ?
                4 : 1
            result = zeros(FT, numlevels)
            for dim in 2:3
                N = (ntuple(j -> 4, dim - 1)..., 0)
                n =
                    dim == 2 ? SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0) :
                    SVector{3, FT}(1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))
                α = FT(1)
                connectivity = dim == 2 ? :face : :full

                for fvmethod in (FVConstant(), FVLinear())
                    @info @sprintf """Configuration
                                      FT                = %s
                                      ArrayType         = %s
                                      FV Reconstruction = %s
                                      dims              = %d
                                      """ FT ArrayType fvmethod dim
                    for l in 1:numlevels
                        Ne = 2^(l - 1) * base_num_elem
                        brickrange = (
                            ntuple(
                                j -> range(FT(-1); length = Ne + 1, stop = 1),
                                dim - 1,
                            )...,
                            range(FT(-1); length = N[1] * Ne + 1, stop = 1),
                        )
                        periodicity = ntuple(j -> false, dim)
                        bc = ntuple(j -> (1, 1), dim)
                        topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                            boundary = bc,
                            connectivity = connectivity,
                        )
                        dt = (α / 4) / (Ne * max(1, maximum(N))^2)

                        timeend = FT(1 // 4)
                        outputtime = timeend

                        dt = outputtime / ceil(Int64, outputtime / dt)

                        vtkdir =
                            output ?
                            "vtk_advection" *
                            "_poly$(N)" *
                            "_dim$(dim)_$(ArrayType)_$(FT)" *
                            "_level$(l)" :
                            nothing
                        result[l] = test_run(
                            mpicomm,
                            ArrayType,
                            fvmethod,
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
                        @test result[l] ≈
                              FT(expected_result[dim, l, FT, fvmethod])
                    end
                    @info begin
                        msg = ""
                        for l in 1:(numlevels - 1)
                            rate = log2(result[l]) - log2(result[l + 1])
                            msg *= @sprintf(
                                "\n  rate for level %d = %e\n",
                                l,
                                rate
                            )
                        end
                        msg
                    end
                end
            end
        end
    end
end

nothing
