using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
import ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Test
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{u, v, ν} <: AdvectionDiffusionProblem end

# This test has two 2D equations : the first one is an advection equation with
# the Gaussian initial condition the second one is an advection diffusion
# equation with the Gaussian initial condition
function init_velocity_diffusion!(
    ::Pseudo1D{u, v, ν},
    aux::Vars,
    geom::LocalGeometry,
) where {u, v, ν}
    # advection velocity of the flow is [u, v]
    uvec = SVector(u, v, 0)
    aux.advection.u = hcat(uvec, uvec)

    # diffusion of the flow is νI (isentropic diffusivity)
    I3 = @SMatrix [1 0 0; 0 1 0; 0 0 1]
    aux.diffusion.D = hcat(0 * I3, ν * I3)

end

function initial_condition!(
    ::Pseudo1D{u, v, ν},
    state,
    aux,
    localgeo,
    t,
) where {u, v, ν}
    FT = typeof(u)
    # the computational domain is [-1.5 1.5]×[-1.5 1.5]
    Lx, Ly = 3, 3
    x, y, _ = localgeo.coord

    μx, μz, σ = 0, 0, Lx / 10
    ρ1 = exp.(-(((x - μx) / σ)^2 + ((y - μz) / σ)^2) / 2) / (σ * sqrt(2 * pi))
    ρ2 = exp.(-(((x - μx) / σ)^2 + ((y - μz) / σ)^2) / 2) / (σ * sqrt(2 * pi))

    state.ρ = (ρ1, ρ2)
end

Dirichlet_data!(P::Pseudo1D, x...) = initial_condition!(P, x...)


function do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Qe, model, testname)
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

    writevtk(filename, Q, dgfvm, statenames, Qe, exactnames)

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
    vtkdir,
    fvmethod,
    polynomialorders,
    level,
    ArrayType,
    FT,
)

    dim = 2
    Lx, Ly = FT(3), FT(3)
    u, v = FT(1), FT(1)
    ν = FT(1 // 100)


    # Grid/topology information
    base_num_elem = 4
    Ne = 2^(level - 1) * base_num_elem

    # match number of points 
    N_dg_point, N_fvm_point = Ne + 1, Ne * polynomialorders[1] + 1


    brickrange = (
        range(-Lx / 2; length = N_dg_point, stop = Lx / 2),
        range(-Ly / 2; length = N_fvm_point, stop = Ly / 2),
    )

    periodicity = ntuple(j -> true, dim)
    bc = ntuple(j -> (1, 2), dim)

    topl = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = bc,
    )

    # one period 
    timeend = Lx / u #
    # dt ≤ CFL (Δx / Np²)/u   u = √2  CFL = 1/√2
    dt = Lx / (Ne * polynomialorders[1]^2)

    Nt = (Ne * polynomialorders[1]^2)

    outputtime = Ne * dt

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorders,
    )

    # Model being tested
    model = AdvectionDiffusion{dim}(Pseudo1D{u, v, ν}(), num_equations = 2)

    # Main DG discretization
    dgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = EveryDirection(),
    )

    # Initialize all relevant state arrays and create solvers
    Q = init_ode_state(dgfvm, FT(0))

    solver = LSRK54CarpenterKennedy(dgfvm, Q; dt = dt, t0 = 0)


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
                gettime(solver),
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
            dgfvm,
            Q,
            Q,
            model,
            "advection_diffusion_periodic",
        )

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dgfvm, gettime(solver))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dgfvm,
                Q,
                Qe,
                model,
                "advection_diffusion_periodic",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    numberofsteps = convert(Int64, cld(timeend, dt))
    dt = timeend / numberofsteps

    @info "time step" dt numberofsteps dt * numberofsteps timeend

    solve!(Q, solver; timeend = timeend, callbacks = callbacks)

    engf = norm(Q, dims = (1, 3))

    # Reference solution
    Q_ref = init_ode_state(dgfvm, FT(0))
    engfe = norm(Q_ref, dims = (1, 3))

    errf = norm(Q_ref .- Q, dims = (1, 3))

    metrics = [engf; engfe; errf]

    @info @sprintf """Finished
    Advection equation:
    norm(Q)                 = %.16e
    norm(Qe)                = %.16e
    norm(Q - Qe)            = %.16e
    Advection diffusion equation:
    norm(Q)                 = %.16e
    norm(Qe)                = %.16e
    norm(Q - Qe)            = %.16e
    """ metrics[:]...

    return errf
end

# Run this test problem
function main()

    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    # Dictionary keys: dim, level, polynomial order, FT, and direction
    expected_result = Dict()

    # Dim 2, degree 4 in the horizontal, FV order 1, refinement level, Float64, equation number
    expected_result[2, 4, FVConstant(), 1, Float64, 1] = 4.5196354911392578e-01
    expected_result[2, 4, FVConstant(), 1, Float64, 2] = 4.8622171647472179e-01
    expected_result[2, 4, FVConstant(), 2, Float64, 1] = 3.5051983693457450e-01
    expected_result[2, 4, FVConstant(), 2, Float64, 2] = 4.1168975732563706e-01
    expected_result[2, 4, FVConstant(), 3, Float64, 1] = 2.5130141068332995e-01
    expected_result[2, 4, FVConstant(), 3, Float64, 2] = 3.4635415661628755e-01
    expected_result[2, 4, FVConstant(), 4, Float64, 1] = 1.6320856055402777e-01
    expected_result[2, 4, FVConstant(), 4, Float64, 2] = 2.9647989774687805e-01
    expected_result[2, 4, FVConstant(), 5, Float64, 1] = 9.6641259241910610e-02
    expected_result[2, 4, FVConstant(), 5, Float64, 2] = 2.6372336617960646e-01

    expected_result[2, 4, FVConstant(), 1, Float32, 1] = 4.5196333527565002e-01
    expected_result[2, 4, FVConstant(), 1, Float32, 2] = 4.8622152209281921e-01
    expected_result[2, 4, FVConstant(), 2, Float32, 1] = 3.5051953792572021e-01
    expected_result[2, 4, FVConstant(), 2, Float32, 2] = 4.1168937087059021e-01
    expected_result[2, 4, FVConstant(), 3, Float32, 1] = 2.5130018591880798e-01
    expected_result[2, 4, FVConstant(), 3, Float32, 2] = 3.4635293483734131e-01
    expected_result[2, 4, FVConstant(), 4, Float32, 1] = 1.6320574283599854e-01
    expected_result[2, 4, FVConstant(), 4, Float32, 2] = 2.9647585749626160e-01
    expected_result[2, 4, FVConstant(), 5, Float32, 1] = 9.6632070839405060e-02
    expected_result[2, 4, FVConstant(), 5, Float32, 2] = 2.6370745897293091e-01


    # Dim 2, degree 4 in the horizontal, FV order 1, refinement level, Float64, equation number

    expected_result[2, 4, FVLinear(), 1, Float64, 1] = 2.2783152269907422e-01
    expected_result[2, 4, FVLinear(), 1, Float64, 2] = 3.1823753821941969e-01
    expected_result[2, 4, FVLinear(), 2, Float64, 1] = 9.2628269590376469e-02
    expected_result[2, 4, FVLinear(), 2, Float64, 2] = 2.4517823755458742e-01
    expected_result[2, 4, FVLinear(), 3, Float64, 1] = 3.1401247771425542e-02
    expected_result[2, 4, FVLinear(), 3, Float64, 2] = 2.2608977452273774e-01
    expected_result[2, 4, FVLinear(), 4, Float64, 1] = 9.8710350648653390e-03
    expected_result[2, 4, FVLinear(), 4, Float64, 2] = 2.2377315397364847e-01
    expected_result[2, 4, FVLinear(), 5, Float64, 1] = 2.9619222883137744e-03
    expected_result[2, 4, FVLinear(), 5, Float64, 2] = 2.2359698753564772e-01


    expected_result[2, 4, FVLinear(), 1, Float32, 1] = 2.2783152269907422e-01
    expected_result[2, 4, FVLinear(), 1, Float32, 2] = 3.1823753821941969e-01
    expected_result[2, 4, FVLinear(), 2, Float32, 1] = 9.2628269590376469e-02
    expected_result[2, 4, FVLinear(), 2, Float32, 2] = 2.4517823755458742e-01
    expected_result[2, 4, FVLinear(), 3, Float32, 1] = 3.1401247771425542e-02
    expected_result[2, 4, FVLinear(), 3, Float32, 2] = 2.2608977452273774e-01
    expected_result[2, 4, FVLinear(), 4, Float32, 1] = 9.8710350648653390e-03
    expected_result[2, 4, FVLinear(), 4, Float32, 2] = 2.2377315397364847e-01
    expected_result[2, 4, FVLinear(), 5, Float32, 1] = 2.9614968225359917e-03
    expected_result[2, 4, FVLinear(), 5, Float32, 2] = 2.2358775138854980e-01

    polynomialorders = (4, 0)
    numlevels = integration_testing ? 5 : 1

    # Dictionary keys: dim, level, polynomial order, FT, and direction
    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            result = Dict()
            for fvmethod in (FVConstant(), FVLinear())
                @info @sprintf """Test parameters:
                ArrayType                   = %s
                FloatType                   = %s
                FV Reconstruction           = %s
                Dimension                   = %s
                Horizontal polynomial order = %s
                Vertical polynomial order   = %s
                """ ArrayType FT fvmethod 2 polynomialorders[1] polynomialorders[end]
                for level in 1:numlevels
                    result[level] = test_run(
                        mpicomm,
                        output ?
                        "vtk_advection_diffusion_2d" *
                        "_poly$(polynomialorders)" *
                        "_$(ArrayType)_$(FT)" *
                        "_level$(level)" :
                        nothing,
                        fvmethod,
                        polynomialorders,
                        level,
                        ArrayType,
                        FT,
                    )


                    @test result[level][1] ≈ FT(expected_result[
                        2,
                        polynomialorders[1],
                        fvmethod,
                        level,
                        FT,
                        1,
                    ])
                    @test result[level][2] ≈ FT(expected_result[
                        2,
                        polynomialorders[1],
                        fvmethod,
                        level,
                        FT,
                        2,
                    ])
                end

                @info begin
                    msg = "advection equation"
                    for l in 1:(numlevels - 1)
                        rate = log2(result[l][1]) - log2(result[l + 1][1])
                        msg *= @sprintf("\n  rate for level %d = %e", l, rate)
                    end
                    msg
                end
                # Not an analytic solution, convergence doesn't make sense
                # @info begin
                #     msg = "advection-diffusion equation"
                #     for l in 1:(numlevels - 1)
                #         rate = log2(result[l][2]) - log2(result[l + 1][2])
                #         msg *= @sprintf("\n  rate for level %d = %e", l, rate)
                #     end
                #     msg
                # end
            end
        end
    end
end

main()
