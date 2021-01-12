using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.FVReconstructions: FVLinear
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
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using Test, MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates

output_vtk = true
# 3D box configuration
# 2 × 1 × numelem_vert elements
# the horizontal polynomial order is N
function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    FT = Float64
    NumericalFlux = RoeNumericalFlux
    @info @sprintf """Configuration
                      ArrayType     = %s
                      FT            = %s
                      NumericalFlux = %s
                      """ ArrayType FT NumericalFlux

    numelem_horz = 2
    N = 4
    @testset for discretization_type in (:DG, :DGFV)
        for direction in (EveryDirection(), VerticalDirection())
            polynomialorder = discretization_type == :DG ? (N, 4) : (N, 0)
            numelem_vert = discretization_type == :DG ? 8 : 32

            err = test_run(
                mpicomm,
                ArrayType,
                polynomialorder,
                numelem_horz,
                numelem_vert,
                NumericalFlux,
                FT,
                direction,
                discretization_type,
            )
            @test err < FT(6e-12)
        end
    end
end

function test_run(
    mpicomm,
    ArrayType,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    NumericalFlux,
    FT,
    direction,
    discretization_type,
)
    domain_height = 30e3
    domain_width = 20e3

    horz_x_range =
        range(FT(0), length = numelem_horz + 1, stop = FT(domain_width))
    # one single element
    horz_y_range = range(FT(0), length = 2, stop = FT(domain_width))
    vert_range = range(0, length = numelem_vert + 1, stop = domain_height)

    brickrange = (horz_x_range, horz_y_range, vert_range)
    periodicity = (true, true, false)
    topology =
        StackedBrickTopology(mpicomm, brickrange; periodicity = periodicity)
    meshwarp = (x...) -> identity(x)


    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = meshwarp,
    )

    problem = AtmosProblem(init_state_prognostic = initialcondition!)

    T_virt_surf = FT(290)
    T_min_ref = FT(220)
    temp_profile = DecayingTemperatureProfile{FT}(
        param_set,
        T_virt_surf,
        T_min_ref,
        FT(8e3),
    )
    ref_state = HydrostaticState(
        temp_profile;
        subtract_off = (discretization_type == :DG),
    )

    configtype = AtmosLESConfigType
    orientation = FlatOrientation()
    source = (Gravity(),)


    model = AtmosModel{FT}(
        configtype,
        param_set;
        problem = problem,
        orientation = orientation,
        ref_state = ref_state,
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = source,
    )

    dg =
        discretization_type == :DG ?
        DGModel(
            model,
            grid,
            NumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction,
        ) :
        DGFVModel(
            model,
            grid,
            HBFVReconstruction(model, FVLinear()),
            NumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction,
        )

    nday = 10
    timeend = FT(nday * 86400)

    # determine the time step
    cfl = 0.5
    dz = step(vert_range)
    dt =
        cfl * dz / max(polynomialorder[end], 1)^2 /
        soundspeed_air(param_set, T_virt_surf)
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      discretization_type   = %s
                      direction = %s
                      numelem_horz  = %d
                      numelem_vert  = %d
                      dt            = %.16e
                      norm(Q₀)      = %.16e
                      """ discretization_type direction numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
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
        vtkdir =
            "vtk_hydrostaticbalance" *
            "$(discretization_type)_$(direction)_horzpoly$(polynomialorder[1])"
        mkpath(vtkdir)

        vtkstep = 0
        # Output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # Setup the output callback
        total_output = 10
        cbvtk = EveryXSimulationSteps(floor(timeend / dt / total_output)) do
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
    errr = errf / engfe
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf / eng0 engf - eng0 errf errr
    errr
end

function initialcondition!(problem, bl, state, aux, coords, t, args...)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector(0, 0, 0)
    state.ρe = aux.ref_state.ρe
end


function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "hydrostatic_balance",
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
