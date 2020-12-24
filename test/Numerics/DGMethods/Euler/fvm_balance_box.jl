using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
import ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
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

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates

function main()
    ClimateMachine.init(parse_clargs = true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = (4, 0)
    FT = Float64
    dims = 3
    NumericalFlux = RoeNumericalFlux
    @info @sprintf """Configuration
                      ArrayType     = %s
                      FT        = %s
                      NumericalFlux = %s
                      dims          = %d
                      """ ArrayType "$FT" "$NumericalFlux" dims

    numelem_horz = 10
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
    domain_height = 10e3
    horz_range =
        range(FT(0), length = numelem_horz + 1, stop = FT(domain_width))
    vert_range = range(0, length = numelem_vert + 1, stop = domain_height)
    brickrange = (horz_range, horz_range, vert_range)

    periodicity = (true, true, false)
    topology =
        StackedBrickTopology(mpicomm, brickrange; periodicity = periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem = AtmosProblem(init_state_prognostic = initialcondition!)

    temp_profile = IsothermalProfile(param_set, FT(300))
    ref_state = HydrostaticState(temp_profile; subtract_off = false)
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        orientation = FlatOrientation(),
        ref_state = ref_state,
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (Gravity(),),
    )

    dg = DGFVModel(
        model,
        grid,
        HBFVReconstruction(model, FVLinear()),
        NumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(10000)

    # determine the time step
    cfl = 0.2
    dx = step(vert_range)
    dt = cfl * dx / 330
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
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector(0, 0, 0)
    state.ρe = aux.ref_state.ρe
end

main()
