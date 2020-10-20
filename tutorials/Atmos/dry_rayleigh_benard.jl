# # Dry Rayleigh Benard

# ## Problem description
#
# 1) Dry Rayleigh Benard Convection (re-entrant channel configuration)
# 2) Boundaries - `Sides` : Periodic (Default `bctuple` used to identify bot,top walls)
#                 `Top`   : Prescribed temperature, no-slip
#                 `Bottom`: Prescribed temperature, no-slip
# 3) Domain - 250m[horizontal] x 250m[horizontal] x 500m[vertical]
# 4) Timeend - 100s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 6) Random values in initial condition (Requires `init_on_cpu=true` argument)
# 7) Overrides defaults for
#               `C_smag`
#               `Courant_number`
#               `init_on_cpu`
#               `ref_state`
#               `solver_type`
#               `bc`
#               `sources`
# 8) Default settings can be found in src/Driver/Configurations.jl

# ## Loading code
using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using Printf

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Thermodynamics:
    PhaseEquil_pTq, internal_energy
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# Convenience struct for sharing data between kernels
struct DryRayleighBenardConvectionDataConfig{FT}
    xmin::FT
    ymin::FT
    zmin::FT
    xmax::FT
    ymax::FT
    zmax::FT
    T_bot::FT
    T_lapse::FT
    T_top::FT
end

# Define initial condition kernel
function init_problem!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    dc = bl.data_config
    FT = eltype(state)

    _R_d::FT = R_d(bl.param_set)
    _cp_d::FT = cp_d(bl.param_set)
    _grav::FT = grav(bl.param_set)
    _cv_d::FT = cv_d(bl.param_set)
    _MSLP::FT = MSLP(bl.param_set)

    γ::FT = _cp_d / _cv_d
    δT =
        sinpi(6 * z / (dc.zmax - dc.zmin)) *
        cospi(6 * z / (dc.zmax - dc.zmin)) + rand()
    δw =
        sinpi(6 * z / (dc.zmax - dc.zmin)) *
        cospi(6 * z / (dc.zmax - dc.zmin)) + rand()
    ΔT = _grav / _cv_d * z + δT
    T = dc.T_bot - ΔT
    P = _MSLP * (T / dc.T_bot)^(_grav / _R_d / dc.T_lapse)
    ρ = P / (_R_d * T)

    q_tot = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ts = PhaseEquil_pTq(bl.param_set, P, T, q_tot)

    ρu, ρv, ρw = FT(0), FT(0), ρ * δw

    e_int = internal_energy(ts)
    e_kin = FT(1 / 2) * δw^2

    ρe_tot = ρ * (e_int + e_pot + e_kin)
    state.ρ = ρ
    state.ρu = SVector(ρu, ρv, ρw)
    state.ρe = ρe_tot
    state.moisture.ρq_tot = FT(0)
    ρχ = zero(FT)
    if z <= 100
        ρχ += FT(0.1) * (cospi(z / 2 / 100))^2
    end
    state.tracers.ρχ = SVector{1, FT}(ρχ)
end

# Define problem configuration kernel
function config_problem(FT, N, resolution, xmax, ymax, zmax)

    ## Boundary conditions
    T_bot = FT(299)

    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    T_lapse = FT(_grav / _cp_d)
    T_top = T_bot - T_lapse * zmax

    ntracers = 1
    δ_χ = SVector{ntracers, FT}(1)

    ## Turbulence
    C_smag = FT(0.23)
    data_config = DryRayleighBenardConvectionDataConfig{FT}(
        0,
        0,
        0,
        xmax,
        ymax,
        zmax,
        T_bot,
        T_lapse,
        FT(T_bot - T_lapse * zmax),
    )

    ## Set up the problem
    problem = AtmosProblem(
        boundarycondition = (
            AtmosBC(
                momentum = Impenetrable(NoSlip()),
                energy = PrescribedTemperature((state, aux, t) -> T_bot),
            ),
            AtmosBC(
                momentum = Impenetrable(NoSlip()),
                energy = PrescribedTemperature((state, aux, t) -> T_top),
            ),
        ),
        init_state_prognostic = init_problem!,
    )

    ## Set up the model
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        turbulence = Vreman(C_smag),
        source = (Gravity(),),
        tracers = NTracers{ntracers, FT}(δ_χ),
        data_config = data_config,
    )

    ## Set up the time-integrator, using a multirate infinitesimal step
    ## method. The option `splitting_type = ClimateMachine.SlowFastSplitting()`
    ## separates fast-slow modes by splitting away the acoustic waves and
    ## treating them via a sub-stepped explicit method.
    ode_solver = ClimateMachine.MISSolverType(;
        splitting_type = ClimateMachine.SlowFastSplitting(),
        mis_method = MIS2,
        fast_method = LSRK144NiegemannDiehlBusch,
        nsubsteps = 10,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "DryRayleighBenardConvection",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_problem!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

# Define diagnostics configuration kernel
function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

# Define main entry point kernel
function main()
    FT = Float64
    ## DG polynomial order
    N = 4
    ## Domain resolution and size
    Δh = FT(10)
    ## Time integrator setup
    t0 = FT(0)
    ## Courant number
    CFLmax = FT(20)
    timeend = FT(1000)
    xmax, ymax, zmax = FT(250), FT(250), FT(500)

    @testset "DryRayleighBenardTest" begin
        for Δh in Δh
            Δv = Δh
            resolution = (Δh, Δh, Δv)
            driver_config = config_problem(FT, N, resolution, xmax, ymax, zmax)
            solver_config = ClimateMachine.SolverConfiguration(
                t0,
                timeend,
                driver_config,
                init_on_cpu = true,
                Courant_number = CFLmax,
            )
            dgn_config = config_diagnostics(driver_config)
            ## User defined callbacks (TMAR positivity preserving filter)
            cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
                Filters.apply!(
                    solver_config.Q,
                    ("moisture.ρq_tot",),
                    solver_config.dg.grid,
                    TMARFilter(),
                )
                nothing
            end
            result = ClimateMachine.invoke!(
                solver_config;
                diagnostics_config = dgn_config,
                user_callbacks = (cbtmarfilter,),
                check_euclidean_distance = true,
            )
            ## result == engf/eng0
            @test isapprox(result, FT(1); atol = 1.5e-2)
        end
    end
end

# Run
main()
