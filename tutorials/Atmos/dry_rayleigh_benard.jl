using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using Printf

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Diagnostics
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics: TemperatureSHumEquil, internal_energy
using CLIMA.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ------------------- Description ---------------------------------------- #
# 1) Dry Rayleigh Benard Convection (re-entrant channel configuration)
# 2) Boundaries - `Sides` : Periodic (Default `bctuple` used to identify bot,top walls)
#                 `Top`   : Prescribed temperature, no-slip
#                 `Bottom`: Prescribed temperature, no-slip
# 3) Domain - 250m[horizontal] x 250m[horizontal] x 500m[vertical]
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 6) Random seed in initial condition (Requires `init_on_cpu=true` argument)
# 7) Overrides defaults for
#               `C_smag`
#               `Courant_number`
#               `init_on_cpu`
#               `ref_state`
#               `solver_type`
#               `bc`
#               `sources`
# 8) Default settings can be found in src/Driver/Configurations.jl

const randomseed = MersenneTwister(1)

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

function init_problem!(bl, state, aux, (x, y, z), t)
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
        cospi(6 * z / (dc.zmax - dc.zmin)) + rand(randomseed)
    δw =
        sinpi(6 * z / (dc.zmax - dc.zmin)) *
        cospi(6 * z / (dc.zmax - dc.zmin)) + rand(randomseed)
    ΔT = _grav / _cv_d * z + δT
    T = dc.T_bot - ΔT
    P = _MSLP * (T / dc.T_bot)^(_grav / _R_d / dc.T_lapse)
    ρ = P / (_R_d * T)

    q_tot = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ts = TemperatureSHumEquil(bl.param_set, T, P, q_tot)

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

function config_problem(FT, N, resolution, xmax, ymax, zmax)

    # Boundary conditions
    T_bot = FT(299)

    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    T_lapse = FT(_grav / _cp_d)
    T_top = T_bot - T_lapse * zmax

    ntracers = 1
    δ_χ = SVector{ntracers, FT}(1)

    # Turbulence
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

    # Set up the model
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = Vreman(C_smag),
        source = (Gravity(),),
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
        tracers = NTracers{ntracers, FT}(δ_χ),
        init_state = init_problem!,
        data_config = data_config,
    )
    ode_solver =
        CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)
    config = CLIMA.AtmosLESConfiguration(
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

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.DiagnosticsConfiguration([dgngrp])
end

function main()
    CLIMA.init()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(10)
    # Time integrator setup
    t0 = FT(0)
    CFLmax = FT(0.90)
    timeend = FT(1000)
    xmax, ymax, zmax = FT(250), FT(250), FT(500)

    @testset "DryRayleighBenardTest" begin
        for Δh in Δh
            Δv = Δh
            resolution = (Δh, Δh, Δv)
            driver_config = config_problem(FT, N, resolution, xmax, ymax, zmax)
            solver_config = CLIMA.SolverConfiguration(
                t0,
                timeend,
                driver_config,
                init_on_cpu = true,
                Courant_number = CFLmax,
            )
            dgn_config = config_diagnostics(driver_config)
            # User defined callbacks (TMAR positivity preserving filter)
            cbtmarfilter =
                GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
                    Filters.apply!(
                        solver_config.Q,
                        6,
                        solver_config.dg.grid,
                        TMARFilter(),
                    )
                    nothing
                end
            result = CLIMA.invoke!(
                solver_config;
                diagnostics_config = dgn_config,
                user_callbacks = (cbtmarfilter,),
                check_euclidean_distance = true,
            )
            # result == engf/eng0
            @test isapprox(result, FT(1); atol = 1.5e-2)
        end
    end
end

main()
