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

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using Test, MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = (4, 0)
    FT = Float64
    NumericalFlux = RoeNumericalFlux
    @info @sprintf """Configuration
                      ArrayType     = %s
                      FT            = %s
                      NumericalFlux = %s
                      """ ArrayType FT NumericalFlux

    numelem_horz = 10
    numelem_vert = 32

    @testset for domain_type in (:box, :sphere)
        err = test_run(
            mpicomm,
            ArrayType,
            polynomialorder,
            numelem_horz,
            numelem_vert,
            NumericalFlux,
            FT,
            domain_type,
        )
        @test err < FT(6e-12)
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
    domain_type,
)
    domain_height = 10e3
    if domain_type === :box
        domain_width = 20e3
        horz_range =
            range(FT(0), length = numelem_horz + 1, stop = FT(domain_width))
        vert_range = range(0, length = numelem_vert + 1, stop = domain_height)
        brickrange = (horz_range, horz_range, vert_range)
        periodicity = (true, true, false)
        topology =
            StackedBrickTopology(mpicomm, brickrange; periodicity = periodicity)
        meshwarp = (x...) -> identity(x)
    elseif domain_type === :sphere
        _planet_radius::FT = planet_radius(param_set)
        vert_range = grid1d(
            _planet_radius,
            FT(_planet_radius + domain_height),
            nelem = numelem_vert,
        )
        topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)
        meshwarp = equiangular_cubed_sphere_warp
    end

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = meshwarp,
    )

    problem = AtmosProblem(init_state_prognostic = initialcondition!)

    T0 = FT(300)
    temp_profile = IsothermalProfile(param_set, T0)
    ref_state = HydrostaticState(temp_profile; subtract_off = false)

    if domain_type === :box
        configtype = AtmosLESConfigType
        source = (Gravity(),)
    elseif domain_type === :sphere
        configtype = AtmosGCMConfigType
        source = (Gravity(), Coriolis())
    end

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
    )
    model =
        AtmosModel{FT}(configtype, physics; problem = problem, source = source)

    dg = DGFVModel(
        model,
        grid,
        HBFVReconstruction(model, FVLinear()),
        NumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(100)

    # determine the time step
    cfl = 0.2
    dz = step(vert_range)
    dt = cfl * dz / soundspeed_air(param_set, T0)
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
                      domain_type   = %s
                      numelem_horz  = %d
                      numelem_vert  = %d
                      dt            = %.16e
                      norm(Q₀)      = %.16e
                      """ domain_type numelem_horz numelem_vert dt eng0

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
    state.energy.ρe = aux.ref_state.ρe
end

main()
