using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.Orientations: NoOrientation
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates

import ClimateMachine.Thermodynamics: total_specific_enthalpy
import ClimateMachine.BalanceLaws: source

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
# Assume zero reference temperature
CLIMAParameters.Planet.T_0(::EarthParameterSet) = 0

using LinearAlgebra
using MPI
using StaticArrays
using Test
using UnPack

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "test",
    "Numerics",
    "DGMethods",
    "compressible_Navier_Stokes",
    "mms_solution_generated.jl",
))

total_specific_enthalpy(ts::PhaseDry{FT}, e_tot::FT) where {FT <: Real} =
    zero(FT)

function mms3_init_state!(problem, bl, state::Vars, aux::Vars, localgeo, t)
    (x1, x2, x3) = localgeo.coord
    state.ρ = ρ_g(t, x1, x2, x3, Val(3))
    state.ρu = SVector(
        U_g(t, x1, x2, x3, Val(3)),
        V_g(t, x1, x2, x3, Val(3)),
        W_g(t, x1, x2, x3, Val(3)),
    )
    state.energy.ρe = E_g(t, x1, x2, x3, Val(3))
end

struct MMSSource{PV <: Union{Mass, Momentum, Energy}, N} <:
       TendencyDef{Source, PV} end

MMSSource(N::Int) =
    (MMSSource{Mass, N}(), MMSSource{Momentum, N}(), MMSSource{Energy, N}())


function source(s::MMSSource{Mass, N}, m, args) where {N}
    @unpack aux, t = args
    x1, x2, x3 = aux.coord
    return Sρ_g(t, x1, x2, x3, Val(N))
end
function source(s::MMSSource{Momentum, N}, m, args) where {N}
    @unpack aux, t = args
    x1, x2, x3 = aux.coord
    return SVector(
        SU_g(t, x1, x2, x3, Val(N)),
        SV_g(t, x1, x2, x3, Val(N)),
        SW_g(t, x1, x2, x3, Val(N)),
    )
end
function source(s::MMSSource{Energy, N}, m, args) where {N}
    @unpack aux, t = args
    x1, x2, x3 = aux.coord
    return SE_g(t, x1, x2, x3, Val(N))
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

    problem = AtmosProblem(
        boundaryconditions = (InitStateBC(),),
        init_state_prognostic = mms3_init_state!,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        orientation = NoOrientation(),
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(μ_exact), WithDivergence()),
        moisture = DryModel(),
        source = (MMSSource(3)...,),
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
        (N, N),
        FT,
        ClimateMachine.array_type(),
        ode_solver,
        param_set,
        model,
        MPI.COMM_WORLD,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        nothing,
        nothing, # filter
        ClimateMachine.AtmosLESSpecificInfo(),
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_solver_type = ode_solver,
        ode_dt = ode_dt,
    )
    Q₀ = solver_config.Q

    # turn on checkpointing
    ClimateMachine.Settings.checkpoint = "300steps"
    ClimateMachine.Settings.checkpoint_keep_one = false

    # run the simulation
    ClimateMachine.invoke!(solver_config)

    # turn off checkpointing and set up a restart
    ClimateMachine.Settings.checkpoint = "never"
    ClimateMachine.Settings.restart_from_num = 2

    # the solver configuration is where the restart is set up
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_solver_type = ode_solver,
        ode_dt = ode_dt,
    )

    # run the restarted simulation
    ClimateMachine.invoke!(solver_config)

    # test correctness
    dg = DGModel(driver_config)
    Qe = init_ode_state(dg, timeend)
    result = euclidean_distance(Q₀, Qe)
    @test result ≈ expected_result
end

main()
