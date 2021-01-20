using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
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

include("bickley_jet_setup.jl")

const output_vtk = true

function main()
    ClimateMachine.init(parse_clargs = true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numlevels = integration_testing ? 4 : 1

    expected_error = Dict()

    # just to make it shorter and aligning
    Rusanov = RusanovNumericalFlux()
    Central = CentralNumericalFluxFirstOrder()
    Roe = RoeNumericalFlux()
    HLLC = HLLCNumericalFlux()
    RoeMoist = RoeNumericalFluxMoist()
    RoeMoistLM = RoeNumericalFluxMoist(; LM = true)
    RoeMoistHH = RoeNumericalFluxMoist(; HH = true)
    RoeMoistLV = RoeNumericalFluxMoist(; LV = true)
    RoeMoistLVPP = RoeNumericalFluxMoist(; LVPP = true)

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32), dims in (2,)
            for NumericalFlux in (
                Rusanov,
                Central,
                Roe,
                HLLC,
                RoeMoist,
                RoeMoistLM,
                RoeMoistHH,
                RoeMoistLV,
                RoeMoistLVPP,
            )
                @info @sprintf """Configuration
                                  ArrayType     = %s
                                  FT        = %s
                                  NumericalFlux = %s
                                  dims          = %d
                                  """ ArrayType "$FT" "$NumericalFlux" dims

                setup = BickleyJetSetup{FT}()
                errors = Vector{FT}(undef, numlevels)

                for level in 1:numlevels
                    numelems =
                        ntuple(dim -> dim == 3 ? 1 : 2^(level - 1) * 5, dims)
                    errors[level] = test_run(
                        mpicomm,
                        ArrayType,
                        polynomialorder,
                        numelems,
                        NumericalFlux,
                        setup,
                        FT,
                        dims,
                        level,
                    )
                end
            end
        end
    end
end

function test_run(
    mpicomm,
    ArrayType,
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

    topology = BrickTopology(
        mpicomm,
        brickrange;
        periodicity = ntuple(_ -> true, dims),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem =
        AtmosProblem(boundaryconditions = (), init_state_prognostic = setup)
    if NumericalFlux isa RoeNumericalFluxMoist
        moisture = EquilMoist{FT}()
    else
        moisture = DryModel()
    end
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        orientation = NoOrientation(),
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = moisture,
        source = (),
    )

    dg = DGModel(
        model,
        grid,
        NumericalFlux,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(2 * setup.domain_halflength / 10 / setup.translation_speed)

    # determine the time step
    elementsize = minimum(step.(brickrange))
    dt =
        elementsize / soundspeed_air(model.param_set, setup.T∞) /
        polynomialorder^2
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

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
        # create vtk dir
        vtkdir =
            "vtk_bickleyjet" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(lsrk), setup)
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, timeend, setup)
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
    dg,
    Q,
    Qe,
    model,
    testname = "bickleyjet",
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

main()
