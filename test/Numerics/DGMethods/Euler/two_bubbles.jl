using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
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
using ClimateMachine.VTK: writevtk, writepvtu

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius, R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using Test, MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates

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

    Ndof = 400
    N = 6
    Ne = round(Int, Ndof / (N+1))

    test_run(
        mpicomm,
        ArrayType,
        N,
        Ne,
        Ne,
        NumericalFlux,
        FT,
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
)
    domain_height = 1000
    domain_width = 1000
    horz_range =
        range(FT(0), length = numelem_horz + 1, stop = FT(domain_width))
    vert_range = range(0, length = numelem_vert + 1, stop = domain_height)
    brickrange = (horz_range, vert_range)
    periodicity = (false, false)
    topology =
        StackedBrickTopology(mpicomm, brickrange; periodicity = periodicity,
                             boundary = ((1,1), (1, 1)))

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem = AtmosProblem(init_state_prognostic = initialcondition!)

    T_surface = FT(303.15)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile; subtract_off = false)

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

    dg = DGModel(
        model,
        grid,
        NumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(10 * 60)

    # determine the time step
    cfl = 1.5
    dx = min_node_distance(grid)
    dt = cfl * dx / 330
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))
    odesolver = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      numelem_horz  = %d
                      numelem_vert  = %d
                      dt            = %.16e
                      norm(Q₀)      = %.16e
                      """ numelem_horz numelem_vert dt eng0

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
                              """ gettime(odesolver) runtime ρu... ρv... ρw... energy
        end
    end
    callbacks = (cbinfo,)

    output_vtk = true
    outputtime = 6
    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_two_bubbles" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; numberofsteps = nsteps, callbacks = callbacks)

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

function initialcondition!(problem, bl, state, aux, localgeo, t, args...)
    (x, z, _) = localgeo.coord
    FT = eltype(state)
    param_set = bl.param_set

    R_gas::FT = R_d(param_set)
    c_p::FT = cp_d(param_set)
    c_v::FT = cv_d(param_set)
    p0::FT = MSLP(param_set)
    _grav::FT = grav(param_set)
    
    ## Reference temperature
    θ_ref::FT = 303.15
    Δθ::FT = 0

    # First bubble
    A₁ = FT(0.5)
    a₁ = 150
    s₁ = 50
    x₁ = 500
    z₁ = 300

    r₁ = sqrt((x - x₁) ^ 2 + (z - z₁) ^ 2)
    if r₁ <= a₁
      Δθ += A₁
    else
      Δθ += A₁ * exp(-(r₁ - a₁) ^ 2 / s₁ ^ 2)
    end
    
    # Second bubble
    A₂ = FT(-0.15)
    a₂ = 0
    s₂ = 50
    x₂ = 560
    z₂ = 640
    
    r₂ = sqrt((x - x₂) ^ 2 + (z - z₂) ^ 2)
    if r₂ <= a₂
      Δθ += A₂
    else
      Δθ += A₂ * exp(-(r₂ - a₂) ^ 2 / s₂ ^ 2)
    end

    ## Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      # density
    T = θ * π_exner
    e_int = internal_energy(param_set, T)
    ts = PhaseDry(param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    ## State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = aux.orientation.Φ                                       # potential energy
    ρe = ρ * (e_kin + e_pot + e_int)

    ## Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.energy.ρe = ρe
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "twobubbles",
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
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
