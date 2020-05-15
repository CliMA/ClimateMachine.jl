using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MoistThermodynamics
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
import ClimateMachine.Atmos:
    MoistureModel,
    temperature,
    pressure,
    soundspeed,
    total_specific_enthalpy,
    thermo_state

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using LinearAlgebra
using MPI
using StaticArrays
using Test

include("mms_solution_generated.jl")

"""
    MMSDryModel

Assumes the moisture components is in the dry limit.
"""
struct MMSDryModel <: MoistureModel end

function total_specific_enthalpy(
    bl::AtmosModel,
    moist::MMSDryModel,
    state::Vars,
    aux::Vars,
)
    zero(eltype(state))
end
function thermo_state(
    bl::AtmosModel,
    moist::MMSDryModel,
    state::Vars,
    aux::Vars,
)
    PS = typeof(bl.param_set)
    return PhaseDry{eltype(state), PS}(
        bl.param_set,
        internal_energy(bl, state, aux),
        state.ρ,
    )
end
function pressure(bl::AtmosModel, moist::MMSDryModel, state::Vars, aux::Vars)
    T = eltype(state)
    γ = T(7) / T(5)
    ρinv = 1 / state.ρ
    return (γ - 1) * (state.ρe - ρinv / 2 * sum(abs2, state.ρu))
end
function soundspeed(bl::AtmosModel, moist::MMSDryModel, state::Vars, aux::Vars)
    T = eltype(state)
    γ = T(7) / T(5)
    ρinv = 1 / state.ρ
    p = pressure(bl, bl.moisture, state, aux)
    sqrt(ρinv * γ * p)
end

function mms3_init_state!(bl, state::Vars, aux::Vars, (x1, x2, x3), t)
    state.ρ = ρ_g(t, x1, x2, x3, Val(3))
    state.ρu = SVector(
        U_g(t, x1, x2, x3, Val(3)),
        V_g(t, x1, x2, x3, Val(3)),
        W_g(t, x1, x2, x3, Val(3)),
    )
    state.ρe = E_g(t, x1, x2, x3, Val(3))
end

function mms3_source!(
    bl,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)
    x1, x2, x3 = aux.coord
    source.ρ = Sρ_g(t, x1, x2, x3, Val(3))
    source.ρu = SVector(
        SU_g(t, x1, x2, x3, Val(3)),
        SV_g(t, x1, x2, x3, Val(3)),
        SW_g(t, x1, x2, x3, Val(3)),
    )
    source.ρe = SE_g(t, x1, x2, x3, Val(3))
end

function main()
    FT = Float64

    # DG polynomial order
    N = 4

    t0 = FT(0)
    timeend = FT(1)

    ode_dt = 0.00125
    nsteps = ceil(Int64, timeend / ode_dt)
    ode_dt = timeend / nsteps

    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )

    expected_result = FT(3.403104838700577e-02)

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = NoOrientation(),
        ref_state = NoReferenceState(),
        turbulence = ConstantViscosityWithDivergence(FT(μ_exact)),
        moisture = MMSDryModel(),
        source = mms3_source!,
        boundarycondition = InitStateBC(),
        init_state_conservative = mms3_init_state!,
    )

    brickrange = (
        range(FT(0); length = 5, stop = 1),
        range(FT(0); length = 5, stop = 1),
        range(FT(0); length = 5, stop = 1),
    )
    topl = BrickTopology(
        MPI.COMM_WORLD,
        brickrange,
        periodicity = (false, false, false),
    )
    warpfun =
        (x1, x2, x3) -> begin
            (
                x1 + (x1 - 1 / 2) * cos(2 * π * x2 * x3) / 4,
                x2 + exp(sin(2π * (x1 * x2 + x3))) / 20,
                x3 + x1 / 4 + x2^2 / 2 + sin(x1 * x2 * x3),
            )
        end
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ClimateMachine.array_type(),
        polynomialorder = N,
        meshwarp = warpfun,
    )
    driver_config = ClimateMachine.DriverConfiguration(
        AtmosLESConfigType(),
        "MMS3",
        N,
        FT,
        ClimateMachine.array_type(),
        ode_solver,
        model,
        MPI.COMM_WORLD,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        ClimateMachine.AtmosLESSpecificInfo(),
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_solver_type = ode_solver,
        ode_dt = ode_dt,
    )
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    ClimateMachine.invoke!(solver_config)

    Qe = init_ode_state(dg, timeend)
    result = euclidean_distance(solver_config.Q, Qe)
    @test result ≈ expected_result
end

main()
