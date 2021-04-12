using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.SystemSolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
using CLIMAParameters.Planet: kappa_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

include("isentropicvortex_setup.jl")

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output_vtk = false

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numlevels = integration_testing ? 4 : 1

    expected_error = Dict()

    # Just to make it shorter and aligning
    Roe = RoeNumericalFlux

    # Float64, Dim 2, degree 4 in the horizontal, FV order 1, refinement level
    expected_error[Float64, FVConstant(), Roe, 1] = 3.5317756615538940e+01
    expected_error[Float64, FVConstant(), Roe, 2] = 2.5104217086071472e+01
    expected_error[Float64, FVConstant(), Roe, 3] = 1.6169569358521223e+01
    expected_error[Float64, FVConstant(), Roe, 4] = 9.4731749125284708e+00

    expected_error[Float64, FVLinear(), Roe, 1] = 2.6132907774912638e+01
    expected_error[Float64, FVLinear(), Roe, 2] = 8.8198392537283006e+00
    expected_error[Float64, FVLinear(), Roe, 3] = 2.4517109427575416e+00
    expected_error[Float64, FVLinear(), Roe, 4] = 7.4154384427579900e-01


    dims = 2
    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            for NumericalFlux in (Roe,)
                setup = IsentropicVortexSetup{FT}()
                for fvmethod in (FVConstant(), FVLinear())
                    @info @sprintf """Configuration
                                      FT                = %s
                                      ArrayType         = %s
                                      FV Reconstruction = %s
                                      NumericalFlux     = %s
                                      dims              = %d
                                      """ FT ArrayType fvmethod NumericalFlux dims
                    errors = Vector{FT}(undef, numlevels)

                    for level in 1:numlevels

                        # Match element numbers
                        numelems = (
                            2^(level - 1) * 5,
                            2^(level - 1) * 5 * polynomialorder,
                        )

                        errors[level] = test_run(
                            mpicomm,
                            ArrayType,
                            fvmethod,
                            polynomialorder,
                            numelems,
                            NumericalFlux,
                            setup,
                            FT,
                            dims,
                            level,
                        )

                        @test errors[level] ≈
                              expected_error[FT, fvmethod, NumericalFlux, level]
                    end
                    @info begin
                        msg = ""
                        for l in 1:(numlevels - 1)
                            rate = log2(errors[l]) - log2(errors[l + 1])
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

function test_run(
    mpicomm,
    ArrayType,
    fvmethod,
    polynomialorder,
    numelems,
    NumericalFlux,
    setup,
    FT,
    dims,
    level,
)
    brickrange = ntuple(dims) do dim
        range(
            -setup.domain_halflength;
            length = numelems[dim] + 1,
            stop = setup.domain_halflength,
        )
    end
    connectivity = dims == 3 ? :full : :face

    topology = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = ntuple(_ -> true, dims),
        connectivity = connectivity,
    )

    grid = SpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = (polynomialorder, 0),
    )

    problem =
        AtmosProblem(boundaryconditions = (), init_state_prognostic = setup)

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        problem = problem,
        orientation = NoOrientation(),
        source = (),
    )

    dgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        NumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(2 * setup.domain_halflength / 10 / setup.translation_speed)

    # Determine the time step
    elementsize = minimum(step.(brickrange))
    dt = elementsize / soundspeed_air(param_set, setup.T∞) / polynomialorder^2
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dgfvm, FT(0))
    lsrk = LSRK54CarpenterKennedy(dgfvm, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    dims == 2 && (numelems = (numelems..., 0))
    @info @sprintf """Starting refinement level %d
                      numelems  = (%d, %d, %d)
                      dt        = %.16e
                      norm(Q₀)  = %.16e
                      """ level numelems... dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(lsrk) runtime energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # Create vtk dir
        vtkdir =
            "vtk_isentropicvortex" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)"
        mkpath(vtkdir)

        vtkstep = 0
        # Output initial step
        do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Q, model)

        # Setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dgfvm, gettime(lsrk), setup)
            do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = callbacks)

    # Final statistics
    Qe = init_ode_state(dgfvm, timeend, setup)
    engf = norm(Q)
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished refinement level %d
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ level engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dgfvm,
    Q,
    Qe,
    model,
    testname = "isentropicvortex",
)
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
        ## name of the pvtu file
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

main()
