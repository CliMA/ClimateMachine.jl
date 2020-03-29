using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.PlanetParameters: grav
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: grav
using CLIMA.VariableTemplates
using CLIMA.Grids
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks: EveryXSimulationSteps
using CLIMA.Mesh.Filters
using CLIMA.Parameters
const clima_dir = dirname(pathof(CLIMA))
include(joinpath(clima_dir, "..", "Parameters", "Parameters.jl"))
param_set = ParameterSet()

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

    λ = longitude(bl.orientation, aux)
    φ = latitude(bl.orientation, aux)
    z = altitude(bl.orientation, aux)

    β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(setup.nv * FT(π) * z / setup.domain_height)
    Δp = setup.γ * f * g
    p = aux.ref_state.p + Δp

    ts = PhaseDry_given_pT(p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)
    return nothing
end

function main()
    CLIMA.init()

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
    orientation = SphericalOrientation()
    ref_state = HydrostaticState(IsothermalProfile(setup.T_ref), FT(0))
    turbulence = ConstantViscosityWithDivergence(FT(0))
    model = AtmosModel{FT}(
        AtmosGCMConfigType;
        orientation = orientation,
        ref_state = ref_state,
        turbulence = turbulence,
        moisture = DryModel(),
        source = Gravity(),
        init_state = setup,
        param_set = param_set,
    )

    ode_solver = CLIMA.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 180,
    )
    driver_config = CLIMA.AtmosGCMConfiguration(
        "GCM Driver test",
        N,
        resolution,
        setup.domain_height,
        setup;
        solver_type = ode_solver,
        model = model,
    )
    solver_config =
        CLIMA.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt)

    # Set up the filter callback
    filterorder = 18
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            1:size(solver_config.Q, 2),
            solver_config.dg.grid,
            filter,
            VerticalDirection(),
        )
        return nothing
    end

    cb_test = 0
    result = CLIMA.invoke!(
        solver_config;
        user_callbacks = (cbfilter,),
        user_info_callback = (init) -> cb_test += 1,
        check_euclidean_distance = true,
    )
    @test cb_test > 0
end

main()
