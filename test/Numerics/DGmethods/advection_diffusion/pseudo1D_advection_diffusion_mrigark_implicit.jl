using MPI
using ClimateMachine
using Logging
using Test
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.LinearSolvers
using ClimateMachine.GeneralizedMinimalResidualSolver
using ClimateMachine.ColumnwiseLUSolver:
    SingleColumnLU, ManyColumnLU, banded_matrix, banded_matrix_vector_product!
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
    aux.u = α * n

    # diffusion of strength β in the n direction
    aux.D = β * n * n'
end

function initial_condition!(
    ::Pseudo1D{n, α, β, μ, δ},
    state,
    aux,
    x,
    t,
) where {n, α, β, μ, δ}
    ξn = dot(n, x)
    # ξT = SVector(x) - ξn * n
    state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end
Dirichlet_data!(P::Pseudo1D, x...) = initial_condition!(P, x...)
function Neumann_data!(
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

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
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

        writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

        @info "Done writing VTK: $pvtuprefix"
    end
end


function run(
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
    β,
    μ,
    δ,
    vtkdir,
    outputtime,
    linearsolvertype,
    fluxBC,
)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    model = AdvectionDiffusion{dim, fluxBC}(Pseudo1D{n, α, β, μ, δ}())
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    vdg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
    )

    Q = init_ode_state(dg, FT(0))

    # With diffussion the Vertical + Horizontal ≠ Full because of 2nd
    # derivative mixing. So we define a custom RHS
    dQ2 = similar(Q)
    function rhs!(dQ, Q, p, time; increment)
        dg(dQ, Q, p, time; increment = increment)
        vdg(dQ2, Q, p, time; increment = false)
        dQ .= dQ .- dQ2
    end

    fastsolver = LSRK144NiegemannDiehlBusch(rhs!, Q; dt = dt)

    # We're dominated by spatial error, so we can get away with a low order time
    # integrator
    ode_solver = MRIGARKESDIRK46aSandu(
        vdg,
        LinearBackwardEulerSolver(linearsolvertype(); isadjustable = false),
        fastsolver,
        Q;
        dt = dt,
        t0 = 0,
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
            Qe = init_ode_state(dg, gettime(ode_solver))
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

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 4

    expected_result = Dict()
    expected_result[2, 1, Float64] = 7.3188989633310442e-02
    expected_result[2, 2, Float64] = 6.8155958327716232e-03
    expected_result[2, 3, Float64] = 1.4832570828897563e-04
    expected_result[2, 4, Float64] = 3.3905353801669396e-06
    expected_result[3, 1, Float64] = 1.0501469629884301e-01
    expected_result[3, 2, Float64] = 1.0341917570778314e-02
    expected_result[3, 3, Float64] = 2.1032014288411172e-04
    expected_result[3, 4, Float64] = 4.5797013335024617e-06
    expected_result[2, 1, Float32] = 7.3186099529266357e-02
    expected_result[2, 2, Float32] = 6.8112355656921864e-03
    expected_result[2, 3, Float32] = 1.4748815738130361e-04
    # This is near roundoff so we will not check it
    # expected_result[2, 4, Float32] = 2.9435863325488754e-05
    expected_result[3, 1, Float32] = 1.0500905662775040e-01
    expected_result[3, 2, Float32] = 1.0342594236135483e-02
    expected_result[3, 3, Float32] = 4.2716524330899119e-04
    # This is near roundoff so we will not check it
    # expected_result[3, 4, Float32] = 1.9564463291317225e-03

    numlevels = integration_testing ? 4 : 1

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            result = zeros(FT, numlevels)
            for dim in 2:3
                for fluxBC in (true, false)
                    for linearsolvertype in (SingleColumnLU,)# ManyColumnLU)
                        d = dim == 2 ? FT[1, 10, 0] : FT[1, 1, 10]
                        n = SVector{3, FT}(d ./ norm(d))

                        α = FT(1)
                        β = FT(1 // 100)
                        μ = FT(-1 // 2)
                        δ = FT(1 // 10)
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
                                range(FT(-5); length = 5Ne + 1, stop = 5),
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
                            )
                            dt = 32 * (α / 4) / (Ne * polynomialorder^2)

                            outputtime = 0.01
                            timeend = 0.5

                            @info (
                                ArrayType,
                                FT,
                                dim,
                                linearsolvertype,
                                l,
                                fluxBC,
                            )
                            vtkdir = output ?
                                "vtk_advection" *
                            "_poly$(polynomialorder)" *
                            "_dim$(dim)_$(ArrayType)_$(FT)" *
                            "_$(linearsolvertype)_level$(l)" :
                                nothing
                            result[l] = run(
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
                                β,
                                μ,
                                δ,
                                vtkdir,
                                outputtime,
                                linearsolvertype,
                                fluxBC,
                            )
                            # test the errors significantly larger than floating point epsilon
                            if !(l == 4 && FT == Float32)
                                @test result[l] ≈
                                      FT(expected_result[dim, l, FT])
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

nothing
