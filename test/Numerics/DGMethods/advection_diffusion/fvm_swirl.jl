# This tutorial uses the TMAR Filter from [Light2016](@cite)
#
# to reproduce the tutorial in section 4b.  It is a shear swirling
# flow deformation of a transported quantity from LeVeque 1996.  The exact
# solution at the final time is the same as the initial condition.

using MPI
using Test
using ClimateMachine
ClimateMachine.init()
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu

using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)))

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end
const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

Base.@kwdef struct SwirlingFlow{FT} <: AdvectionDiffusionProblem
    period::FT = 5
end

init_velocity_diffusion!(::SwirlingFlow, aux::Vars, geom::LocalGeometry) =
    nothing

function initial_condition!(::SwirlingFlow, state, aux, localgeo, t)
    FT = eltype(state)
    x, y, _ = aux.coord
    x0, y0 = FT(1 // 3), FT(1 // 3)
    τ = 6hypot(x - x0, y - y0)
    state.ρ = exp(-τ^2)
end

has_variable_coefficients(::SwirlingFlow) = true
function update_velocity_diffusion!(
    problem::SwirlingFlow,
    ::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    t::Real,
)
    x, y, _ = aux.coord
    sx, cx = sinpi(x), cospi(x)
    sy, cy = sinpi(y), cospi(y)
    ct = cospi(t / problem.period)

    u = 2 * sx^2 * sy * cy * ct
    v = -2 * sy^2 * sx * cx * ct
    aux.advection.u = SVector(u, v, 0)
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname;
    number_sample_points = 0,
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

    writevtk(
        filename,
        Q,
        dg,
        statenames;
        number_sample_points = number_sample_points,
    )

    ## generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames...,), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

function test_run(
    mpicomm,
    ArrayType,
    fvmethod,
    topl,
    problem,
    dt,
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

    model = AdvectionDiffusion{2}(problem, diffusion = false)

    dg = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    ## We integrate so that the final solution is equal to the initial solution
    Qe = copy(Q)
    odesolver = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

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
                gettime(odesolver),
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
        if MPI.Comm_rank(mpicomm) == 0
            mkpath(vtkdir)
        end
        MPI.Barrier(mpicomm)

        vtkstep = 0
        # output initial step
        do_output(
            mpicomm,
            vtkdir,
            vtkstep,
            dg,
            Q,
            model,
            "swirling_flow";
            number_sample_points = N[1] + 1,
        )

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dg,
                Q,
                model,
                "swirling_flow";
                number_sample_points = N[1] + 1,
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; timeend = timeend, callbacks = callbacks)

    error = euclidean_distance(Q, Qe)

    # Print some end of the simulation information
    eng0 = norm(Qe)
    engf = norm(Q)

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
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    expected_result = Dict()
    expected_result[1, FVConstant()] = 1.2437314516997458e-01
    expected_result[2, FVConstant()] = 9.8829388561567838e-02
    expected_result[3, FVConstant()] = 7.2096312912198937e-02
    expected_result[4, FVConstant()] = 4.7720226730379636e-02
    expected_result[1, FVLinear()] = 4.3807967239640234e-02
    expected_result[2, FVLinear()] = 1.4913083518239653e-02
    expected_result[3, FVLinear()] = 4.3666055848495802e-03
    expected_result[4, FVLinear()] = 1.2749240022458719e-03

    @testset "$(@__FILE__)" begin
        numlevels =
            integration_testing || ClimateMachine.Settings.integration_testing ?
            4 : 1
        FT = Float64
        for fvmethod in (FVConstant(), FVLinear())
            result = zeros(FT, numlevels)
            for l in 1:numlevels
                Ne = 20 * 2^(l - 1)
                polynomialorder = (4, 0)

                problem = SwirlingFlow()

                brickrange = (
                    range(FT(0); length = Ne + 1, stop = 1),
                    range(
                        FT(0);
                        length = polynomialorder[1] * Ne + 1,
                        stop = 1,
                    ),
                )

                topology = StackedBrickTopology(
                    mpicomm,
                    brickrange,
                    boundary = ((3, 3), (3, 3)),
                )

                maxvelocity = 2
                elementsize = 1 / Ne
                dx = elementsize / polynomialorder[1]^2
                CFL = 1
                dt = CFL * dx / maxvelocity

                vtkdir = abspath(joinpath(
                    ClimateMachine.Settings.output_dir,
                    "fvm_swirl_lvl_$l",
                ))

                timeend = problem.period

                outputtime = timeend / 10
                dt = outputtime / ceil(Int64, outputtime / dt)

                @info @sprintf """Starting
                FT               = %s
                ArrayType        = %s
                FV Reconstuction = %s
                dim              = %d
                Ne               = %d
                polynomial order = %d
                final time       = %.16e
                time step        = %.16e
                """ FT ArrayType fvmethod 2 Ne polynomialorder[1] timeend dt

                result[l] = test_run(
                    mpicomm,
                    ArrayType,
                    fvmethod,
                    topology,
                    problem,
                    dt,
                    polynomialorder,
                    timeend,
                    FT,
                    output ? vtkdir : nothing,
                    outputtime,
                )
                @test result[l] ≈ FT(expected_result[l, fvmethod])
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
