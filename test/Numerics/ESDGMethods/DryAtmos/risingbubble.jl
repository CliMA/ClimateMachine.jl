using MPI
using ClimateMachine
using Logging
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid, min_node_distance
using ClimateMachine.Thermodynamics
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.VariableTemplates: flattenednames
import ClimateMachine.ODESolvers: LSRK144NiegemannDiehlBusch, solve!, gettime
using StaticArrays: @SVector
using LazyArrays

using ClimateMachine.TemperatureProfiles: DryAdiabaticProfile

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("DryAtmos.jl")
include("diagnostics.jl")

struct RisingBubble <: AbstractDryAtmosProblem end

function init_state_prognostic!(bl::DryAtmosModel, 
                                ::RisingBubble,
                                state, aux, localgeo, t)
    (x, y, z) = localgeo.coord
    ## Problem float-type
    FT = eltype(state)

    ## Unpack constant parameters
    R_gas::FT = R_d(param_set)
    c_p::FT = cp_d(param_set)
    c_v::FT = cv_d(param_set)
    p0::FT = MSLP(param_set)
    _grav::FT = grav(param_set)
    γ::FT = c_p / c_v

    ## Define bubble center and background potential temperature
    xc::FT = 5000
    yc::FT = 1000
    zc::FT = 2000
    r = sqrt((x - xc)^2 + (z - zc)^2)
    rc::FT = 2000
    θamplitude::FT = 2

    ## Reference temperature
    θ_ref::FT = 300

    ## Add the thermal perturbation:
    Δθ::FT = 0
    if r <= rc
        Δθ = θamplitude * (1.0 - r / rc)
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
    e_pot = aux.Φ                                       # potential energy
    # ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy
    _cv_d = cv_d(param_set)
    e_int = _cv_d * T
    ρe = ρ * (e_kin + e_pot + e_int)

    ## Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe
    p = pressure(ρ, ρu, ρe, e_pot)
end

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD
    polynomialorder = 4
    Ne = (10, 1, 10)
    FT = Float64
    xmax = FT(10000)
    ymax = FT(500)
    zmax = FT(10000)

    timeend = 5000
    FT = Float64
    result = run(
        mpicomm,
        polynomialorder,
        Ne,
        xmax,
        ymax,
        zmax,
        timeend,
        ArrayType,
        FT,
    )
end

function run(
    mpicomm,
    polynomialorder,
    Ne,
    xmax,
    ymax,
    zmax,
    timeend,
    ArrayType,
    FT,
)

    dim = 3
    brickrange = (
        range(FT(0), stop = xmax, length = Ne[1] + 1),
        range(FT(0), stop = ymax, length = Ne[2] + 1),
        range(FT(0), stop = zmax, length = Ne[3] + 1),
    )
    boundary = ((0, 0), (0, 0), (1, 2))
    periodicity = (true, true, false)
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )
    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = DryReferenceState(T_profile)

    problem = RisingBubble()
    model = DryAtmosModel{dim}(FlatOrientation(),
                               problem;
                               ref_state=ref_state)

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        #surface_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = MatrixFlux(),
    )

    # determine the time step
    dx = min_node_distance(grid)
    cfl = FT(1.7)
    dt = cfl * dx / 330

    Q = init_ode_state(esdg, FT(0))

    η = similar(Q, vars = @vars(η::FT), nstate=1)

    ∫η0 = entropy_integral(esdg, model, η, Q)

    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem         = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      ∫η              = %.16e
                      """ "$ArrayType" "$FT" polynomialorder Ne[1] dt eng0 ∫η0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXSimulationSteps(100) do (s = false)
        if s
            starttime[] = now()
        else
            ∫η = entropy_integral(esdg, model, η, Q)
            dη = (∫η - ∫η0) / ∫η0
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime          = %.16e
                              runtime          = %s
                              norm(Q)          = %.16e
                              ∫η               = %.16e
                              (∫η - ∫η0) / ∫η0 = %.16e 
                              """ gettime(odesolver) runtime energy ∫η dη
        end
    end
    callbacks = (cbinfo,)

    output_vtk = true
    if output_vtk
        # create vtk dir
        Nelem = Ne[1]
        vtkdir =
            "test_RTB" *
            "_poly$(polynomialorder)_dims$(dim)_$(ArrayType)_$(FT)_nelem$(Nelem)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)

        # setup the output callback
        outputtime = 50
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; callbacks = callbacks, timeend = timeend)

    # final statistics
    engf = norm(Q)
    ∫ηf = entropy_integral(esdg, model, η, Q)
    dηf = (∫ηf - ∫η0) / ∫η0
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    ∫η                      = %.16e
    (∫η - ∫η0) / ∫η0        = %.16e 
    """ engf engf / eng0 engf - eng0 ∫ηf dηf
    engf
end

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, testname = "RTB")
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

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames)#; number_sample_points = 10)

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
