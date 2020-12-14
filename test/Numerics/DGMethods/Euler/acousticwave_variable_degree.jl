using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid, VerticalDirection
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: DGModel, init_ode_state, remainder_DGModel
using ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics:
    air_density, soundspeed_air, internal_energy, PhaseDry_pT, PhasePartition
using ClimateMachine.TemperatureProfiles: IsothermalProfile
using ClimateMachine.Atmos:
    AtmosModel,
    DryModel,
    NoPrecipitation,
    NoRadiation,
    NTracers,
    vars_state,
    Gravity,
    HydrostaticState,
    AtmosAcousticGravityLinearModel
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Orientations:
    SphericalOrientation, gravitational_potential, altitude, latitude, longitude
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const ntracers = 1

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    # 5th order in the horizontal, cubic in the vertical
    polynomialorder = (5, 3)
    numelem_horz = 10
    numelem_vert = 6

    timeend = 60 * 60
    # timeend = 33 * 60 * 60 # Full simulation
    outputtime = 60 * 60

    expected_result = Dict()
    expected_result[Float32] = 9.5075065f13
    expected_result[Float64] = 9.507349773781483e13

    for FT in (Float64, Float32)
        for split_explicit_implicit in (false, true)
            result = test_run(
                mpicomm,
                polynomialorder,
                numelem_horz,
                numelem_vert,
                timeend,
                outputtime,
                ArrayType,
                FT,
                split_explicit_implicit,
            )
            @test result ≈ expected_result[FT]
        end
    end
end

function test_run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
    split_explicit_implicit,
)

    setup = AcousticWaveSetup{FT}()

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + setup.domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = cubedshellwarp,
    )

    T_profile = IsothermalProfile(param_set, setup.T_ref)
    δ_χ = @SVector [FT(ii) for ii in 1:ntracers]

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = setup,
        orientation = SphericalOrientation(),
        ref_state = HydrostaticState(T_profile),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (Gravity(),),
        tracers = NTracers{length(δ_χ), FT}(δ_χ),
    )

    linearmodel = AtmosAcousticGravityLinearModel(model)

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )

    # determine the time step
    element_size = (setup.domain_height / numelem_vert)
    acoustic_speed = soundspeed_air(model.param_set, FT(setup.T_ref))
    dt_factor = 445
    Nmax = maximum(polynomialorder)
    dt = dt_factor * element_size / acoustic_speed / Nmax^2
    # Adjust the time step so we exactly hit 1 hour for VTK output
    dt = 60 * 60 / ceil(60 * 60 / dt)
    nsteps = ceil(Int, timeend / dt)

    Q = init_ode_state(dg, FT(0))

    linearsolver = ManyColumnLU()

    if split_explicit_implicit
        rem_dg = remainder_DGModel(
            dg,
            (lineardg,);
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (lineardg.numerical_flux_first_order,),
            ),
        )
    end
    odesolver = ARK2GiraldoKellyConstantinescu(
        split_explicit_implicit ? rem_dg : dg,
        lineardg,
        LinearBackwardEulerSolver(
            linearsolver;
            isadjustable = true,
            preconditioner_update_freq = -1,
        ),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = split_explicit_implicit,
    )
    @test getsteps(odesolver) == 0

    eng0 = norm(Q)
    @info @sprintf """Starting
    ArrayType       = %s
    FT              = %s
    poly order horz = %d
    poly order vert = %d
    numelem_horz    = %d
    numelem_vert    = %d
    dt              = %.16e
    norm(Q₀)        = %.16e
    """ "$ArrayType" "$FT" polynomialorder... numelem_horz numelem_vert dt eng0

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
                              """ gettime(odesolver) runtime energy
        end
    end

    # Look ma, no filters!
    callbacks = (cbinfo,)

    solve!(
        Q,
        odesolver;
        numberofsteps = nsteps,
        adjustfinalstep = false,
        callbacks = callbacks,
    )

    @test getsteps(odesolver) == nsteps

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    return engf
end

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(problem, bl, state, aux, localgeo, t)
    # callable to set initial conditions
    FT = eltype(state)

    λ = longitude(bl, aux)
    φ = latitude(bl, aux)
    z = altitude(bl, aux)

    β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(setup.nv * FT(π) * z / setup.domain_height)
    Δp = setup.γ * f * g
    p = aux.ref_state.p + Δp

    ts = PhaseDry_pT(bl.param_set, p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)

    state.tracers.ρχ = @SVector [FT(ii) for ii in 1:ntracers]
    nothing
end

main()
