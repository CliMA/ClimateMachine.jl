using ClimateMachine

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.NumericalFluxes
using ClimateMachine.VTK

using StaticArrays
using Test
using Printf
using MPI
using ArgParse

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ------------------------ Description ------------------------- #
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries - `All Walls` : Impenetrable(FreeSlip())
#                               Laterally periodic
# 3) Domain - 20000m[horizontal] x 10000m[vertical] (2-dimensional)
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 2:1
# 7) Overrides defaults for
#               `init_on_cpu`
#               `solver_type`
#               `sources`
#               `C_smag`
# 8) Default settings can be found in `src/Driver/Configurations.jl`
# ------------------------ Description ------------------------- #
function init_risingbubble!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    γ::FT = c_p / c_v
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)

    xc::FT = 10000
    zc::FT = 2000
    r = sqrt((x - xc)^2 + (z - zc)^2)
    rc::FT = 2000
    θ_ref::FT = 300
    Δθ::FT = 0

    if r <= rc
        Δθ = FT(2) * cospi(0.5 * r / rc)^2
    end

    # Perturbed state:
    θ = θ_ref + Δθ # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density
    q_tot = FT(0)
    ts = PhaseEquil_ρθq(bl.param_set, ρ, θ, q_tot)
    q_pt = PhasePartition(ts)

    ρu = SVector(FT(0), FT(0), FT(0))

    # State (prognostic) variable assignment
    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_pt.tot
end

function config_risingbubble(FT, N, resolution, xmax, ymax, zmax, fast_method)

    # Choose fast solver
    if fast_method == "LowStorageRungeKutta2N"
        ode_solver = ClimateMachine.MISSolverType(
            splitting_type = ClimateMachine.SlowFastSplitting(),
            fast_model = AtmosAcousticGravityLinearModel,
            mis_method = MIS2,
            fast_method = LSRK54CarpenterKennedy,
            nsubsteps = (50,),
        )
    elseif fast_method == "StrongStabilityPreservingRungeKutta"
        ode_solver = ClimateMachine.MISSolverType(
            splitting_type = ClimateMachine.SlowFastSplitting(),
            fast_model = AtmosAcousticGravityLinearModel,
            mis_method = MIS2,
            fast_method = SSPRK33ShuOsher,
            nsubsteps = (12,),
        )
    elseif fast_method == "MultirateInfinitesimalStep"
        ode_solver = ClimateMachine.MISSolverType(
            splitting_type = ClimateMachine.HEVISplitting(),
            fast_model = AtmosAcousticGravityLinearModel,
            mis_method = MIS2,
            fast_method = (dg, Q, nsubsteps) -> MultirateInfinitesimalStep(
                MISKWRK43,
                dg,
                (dgi, Qi) -> LSRK54CarpenterKennedy(dgi, Qi),
                Q,
                nsubsteps = nsubsteps,
            ),
            nsubsteps = (12, 2),
        )
    elseif fast_method == "MultirateRungeKutta"
        ode_solver = ClimateMachine.MISSolverType(
            splitting_type = ClimateMachine.HEVISplitting(),
            fast_model = AtmosAcousticGravityLinearModel,
            mis_method = MIS2,
            fast_method = (dg, Q, nsubsteps) -> MultirateRungeKutta(
                LSRK144NiegemannDiehlBusch,
                dg,
                Q,
                steps = nsubsteps,
            ),
            nsubsteps = (12, 4),
        )
    elseif fast_method == "AdditiveRungeKutta"
        ode_solver = ClimateMachine.MISSolverType(
            splitting_type = ClimateMachine.HEVISplitting(),
            fast_model = AtmosAcousticGravityLinearModel,
            mis_method = MISRK3,
            fast_method = (dg, Q, dt, nsubsteps) -> AdditiveRungeKutta(
                ARK548L2SA2KennedyCarpenter,
                dg,
                LinearBackwardEulerSolver(ManyColumnLU(), isadjustable = true),
                Q,
                dt = dt,
                nsubsteps = nsubsteps,
            ),
            nsubsteps = (12,),
        )
    else
        error("Invalid --fast_method=$fast_method")
    end
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    # Set up the model
    C_smag = FT(0.23)
    ref_state =
        HydrostaticState(DryAdiabaticProfile{FT}(param_set, FT(300), FT(0)))
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag),
        source = (Gravity(),),
        ref_state = ref_state,
        init_state_prognostic = init_risingbubble!,
    )

    # Problem configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "DryRisingBubbleMIS",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_risingbubble!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()

    rbb_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(rbb_args, "RisingBubbleBryan")
    @add_arg_table! rbb_args begin
        "--fast_method"
        help = "Choice of fast solver for the MIS method"
        metavar = "<name>"
        arg_type = String
        default = "AdditiveRungeKutta"
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = rbb_args)
    fast_method = cl_args["fast_method"]

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δx = FT(250)
    Δy = FT(250)
    Δz = FT(250)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = FT(20000)
    ymax = FT(1000)
    zmax = FT(10000)
    # Simulation time
    t0 = FT(0)
    timeend = FT(1000)

    # Time-step size (s)
    Δt = FT(0.4)

    driver_config =
        config_risingbubble(FT, N, resolution, xmax, ymax, zmax, fast_method)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = Δt,
    )
    dgn_config = config_diagnostics(driver_config)

    # Invoke solver (calls solve! function for time-integrator)
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
