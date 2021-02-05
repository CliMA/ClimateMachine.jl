# # [Acoustic Wave Configuration](@id Acoustic-Wave-Configuration)

#
# In this example, we demonstrate the usage of the `ClimateMachine`
# [AtmosModel](@ref AtmosModel-docs) machinery to solve the fluid
# dynamics of an acoustic wave.

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.Checkpoint
using ClimateMachine.ConfigTypes
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.Grids
using ClimateMachine.ODESolvers

using CLIMAParameters

using StaticArrays

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(problem, bl, state, aux, localgeo, t)
    ## callable to set initial conditions
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
    state.energy.ρe = state.ρ * (e_int + e_pot)
    return nothing
end

function run_acousticwave(ode_solver, CFL, CFL_direction, timeend)
    FT = Float64

    ## DG polynomial orders
    N = (4, 4)

    ## Domain resolution
    nelem_horz = 6
    nelem_vert = 4
    resolution = (nelem_horz, nelem_vert)

    t0 = FT(0)

    setup = AcousticWaveSetup{FT}()
    T_profile = IsothermalProfile(param_set, setup.T_ref)
    orientation = SphericalOrientation()
    ref_state = HydrostaticState(T_profile)
    turbulence = ConstantDynamicViscosity(FT(0))
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = setup,
        orientation = orientation,
        ref_state = ref_state,
        turbulence = turbulence,
        moisture = DryModel(),
        source = (Gravity(),),
    )

    driver_config = ClimateMachine.AtmosGCMConfiguration(
        "GCM Driver: Acoustic wave test",
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
        Courant_number = CFL,
        init_on_cpu = true,
        ode_solver_type = ode_solver,
        CFL_direction = CFL_direction,
    )

    ClimateMachine.invoke!(solver_config)
end
