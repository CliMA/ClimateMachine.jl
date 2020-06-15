using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Checkpoint
using ClimateMachine.ConfigTypes
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates
using ClimateMachine.Grids
using ClimateMachine.ODESolvers

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(bl, state, aux, coords, t)
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

    ts = PhaseDry_given_pT(bl.param_set, p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)
    return nothing
end

function main()
    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution
    nelem_horz = 4
    nelem_vert = 6
    resolution = (nelem_horz, nelem_vert)

    t0 = FT(0)
    timeend = FT(1800)
    # Timestep size (s)
    dt = FT(600)

    setup = AcousticWaveSetup{FT}()
    T_profile = IsothermalProfile(param_set, setup.T_ref)
    orientation = SphericalOrientation()
    ref_state = HydrostaticState(T_profile)
    turbulence = ConstantViscosityWithDivergence(FT(0))
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        orientation = orientation,
        ref_state = ref_state,
        turbulence = turbulence,
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = setup,
    )

    ode_solver = ClimateMachine.MultirateSolverType(
        fast_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 180,
    )
    driver_config = ClimateMachine.AtmosGCMConfiguration(
        "GCM Driver test",
        N,
        resolution,
        setup.domain_height,
        param_set,
        setup;
        solver_type = ode_solver,
        model = model,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

    cb_test = 0
    result = ClimateMachine.invoke!(
        solver_config;
        user_info_callback = (init) -> cb_test += 1,
    )
    @test cb_test > 0
end

main()
