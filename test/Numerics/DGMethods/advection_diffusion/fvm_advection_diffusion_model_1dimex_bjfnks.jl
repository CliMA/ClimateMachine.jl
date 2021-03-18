using MPI
using ClimateMachine
using Logging
using Test
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.MPIStateArrays
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu

if !@isdefined integration_testing
    if length(ARGS) > 0
        const integration_testing = parse(Bool, ARGS[1])
    else
        const integration_testing = parse(
            Bool,
            lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
        )
    end
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{n, α, β},
    aux::Vars,
    geom::LocalGeometry,
) where {n, α, β}
    # Direction of flow is n with magnitude α
    aux.advection.u = α * n

    # Diffusion of strength β in the n direction
    aux.diffusion.D = β * n * n'
end

function initial_condition!(
    ::Pseudo1D{n, α, β, μ, δ},
    state,
    aux,
    localgeo,
    t,
) where {n, α, β, μ, δ}
    ξn = dot(n, localgeo.coord)
    # ξT = SVector(localgeo.coord) - ξn * n
    state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end
inhomogeneous_data!(::Val{0}, P::Pseudo1D, x...) = initial_condition!(P, x...)
function inhomogeneous_data!(
    ::Val{1},
    ::Pseudo1D{n, α, β, μ, δ},
    ∇state,
    aux,
    x,
    t,
) where {n, α, β, μ, δ}
    ξn = dot(n, x)
    ∇state.ρ =
        -(
            2n * (ξn - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
end

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
    dim,
    topl,
    N,
    fvmethod,
    timeend,
    FT,
    dt,
    n,
    α,
    β,
    μ,
    δ,
    vtkdir,
    outputtime,
    fluxBC,
)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = (N, 0),
    )

    bcs = (
        InhomogeneousBC{0}(),
        InhomogeneousBC{1}(),
        HomogeneousBC{0}(),
        HomogeneousBC{1}(),
    )
    model = AdvectionDiffusion{dim}(
        Pseudo1D{n, α, β, μ, δ}(),
        bcs,
        flux_bc = fluxBC,
    )
    dgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction = EveryDirection(),
    )

    vdgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        state_auxiliary = dgfvm.state_auxiliary,
        direction = VerticalDirection(),
    )


    Q = init_ode_state(dgfvm, FT(0))

    linearsolver = BatchedGeneralizedMinimalResidual(
        dgfvm,
        Q;
        max_subspace_size = 5,
        atol = sqrt(eps(FT)) * 0.01,
        rtol = sqrt(eps(FT)) * 0.01,
    )

    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4)

    ode_solver = ARK2ImplicitExplicitMidpoint(
        dgfvm,
        vdgfvm,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = 1000,
        ),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
    )

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
                gettime(ode_solver),
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
            Qe = init_ode_state(dgfvm, gettime(ode_solver))
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

    numberofsteps = convert(Int64, cld(timeend, dt))
    dt = timeend / numberofsteps

    @info "time step" dt numberofsteps dt * numberofsteps timeend

    solve!(
        Q,
        ode_solver;
        numberofsteps = numberofsteps,
        callbacks = callbacks,
        adjustfinalstep = false,
    )

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

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 4

    expected_result = Dict()
    # dim, refinement level, FT, vertical scheme
    expected_result[2, 1, Float64, FVConstant()] = 1.4890228213182394e-01
    expected_result[2, 2, Float64, FVConstant()] = 1.1396608915201276e-01
    expected_result[2, 3, Float64, FVConstant()] = 7.9360293305962212e-02

    expected_result[2, 1, Float64, FVLinear()] = 1.0755278583097065e-01
    expected_result[2, 2, Float64, FVLinear()] = 4.5924219102504410e-02
    expected_result[2, 3, Float64, FVLinear()] = 1.0775256661783415e-02

    expected_result[3, 1, Float64, FVConstant()] = 2.1167998484526718e-01
    expected_result[3, 2, Float64, FVConstant()] = 1.5936623436529485e-01
    expected_result[3, 3, Float64, FVConstant()] = 1.1069147173311458e-01

    expected_result[3, 1, Float64, FVLinear()] = 1.5361757743888277e-01
    expected_result[3, 2, Float64, FVLinear()] = 6.3622331921472902e-02
    expected_result[3, 3, Float64, FVLinear()] = 1.4836612761110498e-02


    numlevels = integration_testing ? 3 : 1

    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            result = zeros(FT, numlevels)
            for dim in 2:3
                connectivity = dim == 3 ? :full : :face
                for fvmethod in (FVConstant(), FVLinear())


                    for fluxBC in (true, false)
                        d = dim == 2 ? FT[1, 10, 0] : FT[1, 1, 10]
                        n = SVector{3, FT}(d ./ norm(d))

                        α = FT(1)
                        β = FT(1 // 100)
                        μ = FT(-1 // 2)
                        δ = FT(1 // 10)

                        solvertype = "HEVI_Nolinearsolver"
                        for l in 1:numlevels
                            Ne = 2^(l - 1) * base_num_elem
                            brickrange = (
                                ntuple(
                                    j -> range(
                                        FT(-1);
                                        length = Ne + 1,
                                        stop = 1,
                                    ),
                                    dim - 1,
                                )...,
                                range(
                                    FT(-5);
                                    length = 5Ne * polynomialorder + 1,
                                    stop = 5,
                                ),
                            )

                            periodicity = ntuple(j -> false, dim)
                            topl = StackedBrickTopology(
                                mpicomm,
                                brickrange;
                                periodicity = periodicity,
                                boundary = (
                                    ntuple(j -> (1, 2), dim - 1)...,
                                    (3, 4),
                                ),
                                connectivity = connectivity,
                            )
                            dt = 2 * (α / 4) / (Ne * polynomialorder^2)

                            outputtime = 0.01
                            timeend = 0.5

                            @info (
                                ArrayType,
                                FT,
                                dim,
                                solvertype,
                                l,
                                polynomialorder,
                                fvmethod,
                                fluxBC,
                            )
                            vtkdir =
                                output ?
                                "vtk_fvm_advection" *
                                "_poly$(polynomialorder)" *
                                "_dim$(dim)_$(ArrayType)_$(FT)" *
                                "_fvmethod$(fvmethod)" *
                                "_$(solvertype)_level$(l)" :
                                nothing
                            result[l] = test_run(
                                mpicomm,
                                ArrayType,
                                dim,
                                topl,
                                polynomialorder,
                                fvmethod,
                                timeend,
                                FT,
                                dt,
                                n,
                                α,
                                β,
                                μ,
                                δ,
                                vtkdir,
                                outputtime,
                                fluxBC,
                            )
                            # Test the errors significantly larger than floating point epsilon
                            if !(dim == 2 && l == 4 && FT == Float32)
                                @test result[l] ≈
                                      FT(expected_result[dim, l, FT, fvmethod])
                            end
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
end
