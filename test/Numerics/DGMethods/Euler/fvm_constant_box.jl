using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.TemperatureProfiles
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates

const output_vtk = true
function main()
    ClimateMachine.init(parse_clargs = true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = (4, 0)
    FT = Float64
    dims = 2
    NumericalFlux = RoeNumericalFlux
    @info @sprintf """Configuration
                      ArrayType     = %s
                      FT        = %s
                      NumericalFlux = %s
                      dims          = %d
                      """ ArrayType "$FT" "$NumericalFlux" dims

    numelem_horz = 2
    numelem_vert = 32
    test_run(
        mpicomm,
        ArrayType,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        NumericalFlux,
        FT,
        dims,
    )
end

function test_run(
    mpicomm,
    ArrayType,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    NumericalFlux,
    FT,
    dims,
)
    domain_width = 20e3
    domain_height = 30e3
    horz_range =
        range(FT(0), length = numelem_horz + 1, stop = FT(domain_width))
    vert_range = range(0, length = numelem_vert + 1, stop = domain_height)
    brickrange = (horz_range, vert_range)

    # periodicity = (true, false)
    periodicity = (true, true)
    topology =
        StackedBrickTopology(mpicomm, brickrange; periodicity = periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem = AtmosProblem(
        init_state_prognostic = initialcondition!,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        orientation = NoOrientation(),
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (),
    )



    dg = DGFVModel(
        model,
        grid,
        FVLinear(),
        NumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(86400 * 10)

    # determine the time step
    cfl = 0.8
    dz = step(vert_range)
    dx = step(horz_range)
    dt = min(cfl/polynomialorder[1]^2 * dx / 330, cfl * dz / 330)
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      numelem_horz  = %d
                      numelem_vert  = %d
                      dt            = %.16e
                      norm(Q₀)      = %.16e
                      """ numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(10, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @views begin
                ρu = extrema(Array(Q.data[:, 2, :]))
                ρv = extrema(Array(Q.data[:, 3, :]))
                ρw = extrema(Array(Q.data[:, 4, :]))
            end
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              ρu = %.16e, %.16e
                              ρv = %.16e, %.16e 
                              ρw = %.16e, %.16e 
                              norm(Q) = %.16e
                              """ gettime(lsrk) runtime ρu... ρv... ρw... energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # Create vtk dir
        vtkdir = "vtk_fvm_constant_box"
        mkpath(vtkdir)

        vtkstep = 0
        # Output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # Setup the output callback
        outputtime = timeend/10
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(lsrk))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, timeend)
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

function initialcondition!(problem, bl, state, aux, coords, t, args...)
    state.ρ = 1
    state.ρu = SVector(0, 0, 0)
    state.ρe = 10000
end


function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "constant",
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

    writevtk(filename, Q, dg, statenames, Qe, exactnames)

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
