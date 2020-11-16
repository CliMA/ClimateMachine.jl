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
using CLIMAParameters.Planet
using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end
struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
  liq::L
  ice::I
  rain::R
  snow::S
end
struct EarthParameterSet{M} <: AbstractEarthParameterSet
  microphys::M
end
const microphys = MicropysicsParameterSet(
  LiquidParameterSet(),
  IceParameterSet(),
  RainParameterSet(),
  SnowParameterSet(),
)
const param_set = EarthParameterSet(microphys)

using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet:
    R_d, cp_d, cv_d, R_v, cp_v, cv_v, cp_l, MSLP, grav, LH_v0, T_freeze, T_0, e_int_v0

include("moist_bubble_profile.jl")
const profile = MoistBubbleProfile(param_set, Float64);

# ------------------------ Description ------------------------- #
# 1) Moist Rising Bubble (circular potential temperature perturbation)
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
# 9) Citation
#    @article{
#       author = {Bryan, George H. and Fritsch, J. Michael},
#       title = "{A Benchmark Simulation for Moist Nonhydrostatic Numerical Models}",
#       journal = {Monthly Weather Review},
#       volume = {130},
#       number = {12},
#       pages = {2917-2928},
#       year = {2002},
#       month = {12},
#       issn = {0027-0644},
#       doi = {10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2},
#       url = {https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2},
#       eprint = {https://journals.ametsoc.org/mwr/article-pdf/130/12/2917/4198716/1520-0493(2002)130\_2917\_absfmn\_2\_0\_co\_2.pdf},
#    }
# ------------------------ Description ------------------------- #

function init_moistbubble!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    FT = eltype(state)
    _R_d::FT = R_d(bl.param_set)
    _R_v::FT = R_v(bl.param_set)
    _cp_d::FT = cp_d(bl.param_set)
    _cv_d::FT = cv_d(bl.param_set)
    _cp_v::FT = cp_v(bl.param_set)
    _cv_v::FT = cv_v(bl.param_set)
    _cp_l::FT = cp_l(bl.param_set)
    _T_0::FT = T_0(bl.param_set)
    _e_int_v0::FT = e_int_v0(bl.param_set)
    κ::FT = _R_d / _cp_d
    γ::FT = _cp_d / _cv_d
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    L00::FT = LH_v0(param_set) + (_cp_l - _cp_v) * T_freeze(param_set)
    _LH_v0::FT = LH_v0(param_set)

    xc::FT = 10000
    zc::FT = 2000
    rc::FT = 2000
    Δθ::FT = 2

    iz = 1000
    for i = 2:size(profile.z, 1)
        if z <= profile.z[i]
            iz = i - 1
            break
        end
    end
    z_l = profile.z[iz]
    ρ_l = profile.Val[iz, 1]
    ρ_θ_l = profile.Val[iz, 2]
    ρ_qv_l = profile.Val[iz, 3]
    ρ_qc_l = profile.Val[iz, 4]
    z_r = profile.z[iz+1]
    ρ_r = profile.Val[iz+1, 1]
    ρ_θ_r = profile.Val[iz+1, 2]
    ρ_qv_r = profile.Val[iz+1, 3]
    ρ_qc_r = profile.Val[iz+1, 4]

    ρ = (ρ_r * (z - z_l) + ρ_l * (z_r - z)) / profile.Δz
    ρ_θ = ρ * (ρ_θ_r / ρ_r * (z - z_l) + ρ_θ_l / ρ_l * (z_r - z)) / profile.Δz
    ρ_qv = ρ * (ρ_qv_r / ρ_r * (z - z_l) + ρ_qv_l / ρ_l * (z_r - z)) / profile.Δz
    ρ_qc = ρ * (ρ_qc_r / ρ_r * (z - z_l) + ρ_qc_l / ρ_l * (z_r - z)) / profile.Δz

    r = sqrt((x - xc)^2 + (z - zc)^2)
    ρ_d = ρ - ρ_qv - ρ_qc
    κ_M = (_R_d * ρ_d + _R_v * ρ_qv) / (_cp_d * ρ_d + _cp_v * ρ_qv + _cp_l * ρ_qc)
    p_loc = p0 * (_R_d * ρ_θ / p0)^(1 / (1 - κ_M))
    T_loc = p_loc / (_R_d * ρ_d + _R_v * ρ_qv)
    ρ_e = (_cv_d * ρ_d + _cv_v * ρ_qv + _cp_l * ρ_qc) * (T_loc - _T_0) + _e_int_v0 * ρ_qv

    if r < rc && Δθ > 0
        θ_dens = ρ_θ / ρ * (p_loc / p0)^(κ_M - κ)
        θ_dens_new = θ_dens * (1 + Δθ * cospi(0.5 * r / rc)^2 / 300)
        rt = (ρ_qv + ρ_qc) / ρ_d
        rv = ρ_qv / ρ_d
        θ_loc = θ_dens_new * (1 + rt) / (1 + (_R_v / _R_d) * rv)
        if rt > 0
            while true
                T_loc = θ_loc * (p_loc / p0)^κ
                T_C = T_loc - 273.15
                # SaturVapor
                pvs = saturation_vapor_pressure(param_set, T_loc, _LH_v0, _cp_v - _cp_l)
                ρ_d_new = (p_loc - pvs) / (_R_d * T_loc)
                rvs = pvs / (_R_v * ρ_d_new * T_loc)
                θ_new = θ_dens_new * (1 + rt) / (1 + (_R_v / _R_d) * rvs)
                if abs(θ_new - θ_loc) <= θ_loc * 1.0e-12
                    break
                else
                    θ_loc = θ_new
                end
            end
        else
            rvs = 0
            T_loc = θ_loc * (p_loc / p0)^κ
            ρ_d_new = p_loc / (_R_d * T_loc)
            θ_new = θ_dens_new * (1 + rt) / (1 + (_R_v / _R_d) * rvs)
        end
        ρ_qv = rvs * ρ_d_new
        ρ_qc = (rt - rvs) * ρ_d_new
        ρ = ρ_d_new * (1 + rt)
        ρ_d = ρ - ρ_qv - ρ_qc
        κ_M = (_R_d * ρ_d + _R_v * ρ_qv) / (_cp_d * ρ_d + _cp_v * ρ_qv + _cp_l * ρ_qc)
        ρ_θ = ρ * θ_dens_new * (p_loc / p0)^(κ - κ_M)
        ρ_e = (_cv_d * ρ_d + _cv_v * ρ_qv + _cp_l * ρ_qc) * (T_loc - _T_0) + _e_int_v0 * ρ_qv

    end

    u = SVector(FT(0), FT(0), FT(0))
    ρu = ρ .* u
    ρ_e =
        ρ_e +
        1 / 2 * ρ * (u[1]^2 + u[3]^2) +
        ρ * gravitational_potential(bl.orientation, aux)

    #State (prognostic) variable assignment
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρ_e
    state.moisture.ρq_tot = ρ_qv + ρ_qc
    if bl.moisture isa NonEquilMoist
        state.moisture.ρq_liq = ρ_qc
        state.moisture.ρq_ice = FT(0)
    end
end

function config_moistbubble(FT, N, resolution, xmax, ymax, zmax, fast_method)

    # Choose fast solver
    println("fast_method  ",fast_method)
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

    # Set up the model
    C_smag = FT(0.23)
    C_smag = FT(0.0)
    ref_state = HydrostaticState(DryAdiabaticProfile{FT}(param_set, FT(300), FT(0)))
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag),
#       source = (Gravity(),),
#       moisture = EquilMoist{FT}(; maxiter = 10, tolerance = FT(0.1)),
        source = (Gravity(),CreateClouds(),),
        moisture = NonEquilMoist(),
        ref_state = ref_state,
        init_state_prognostic = init_moistbubble!,
    )
    println("model.ref_state ",model.ref_state)

    # Problem configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "MoistBubbleMIS",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_moistbubble!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp =
        setup_atmos_default_diagnostics(AtmosLESConfigType(), interval, driver_config.name)
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
    Δx = FT(125)
    Δy = FT(125)
    Δz = FT(125)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = FT(20000)
    ymax = FT(500)
    zmax = FT(10000)
    # Simulation time
    t0 = FT(0)
    timeend = FT(1000)

    # Time-step size (s)
    Δt = FT(1.0)

    fast_method = "StrongStabilityPreservingRungeKutta"
    driver_config = config_moistbubble(FT, N, resolution, xmax, ymax, zmax, fast_method)
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
