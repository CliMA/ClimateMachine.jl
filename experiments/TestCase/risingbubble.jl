#!/usr/bin/env julia --project
using ArgParse

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using Thermodynamics.TemperatureProfiles
using Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using StaticArrays
using Test
using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

function init_risingbubble!(problem, bl, state, aux, localgeo, t)
    ## Problem float-type
    FT = eltype(state)

    (x, y, z) = localgeo.coord
    param_set = parameter_set(bl)

    ## Unpack constant parameters
    R_gas::FT = R_d(param_set)
    c_p::FT = cp_d(param_set)
    c_v::FT = cv_d(param_set)
    p0::FT = MSLP(param_set)
    _grav::FT = grav(param_set)
    γ::FT = c_p / c_v

    ## Define bubble center and background potential temperature
    xc::FT = 5000
    yc::FT = 1000
    zc::FT = 2000
    r = sqrt((x - xc)^2 + (z - zc)^2)
    rc::FT = 2000
    θamplitude::FT = 2
    q_tot_amplitude::FT = 1e-3

    ## TODO: clean this up, or add convenience function:
    ## This is configured in the reference hydrostatic state
    ref_state = reference_state(bl)
    θ_ref::FT = ref_state.virtual_temperature_profile.T_surface

    ## Add the thermal perturbation:
    Δθ::FT = 0
    Δq_tot::FT = 0
    if r <= rc
        Δθ = θamplitude * (1.0 - r / rc)
        Δq_tot = q_tot_amplitude * (1.0 - r / rc)
    end

    ## Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      # potential temperature
    q_pt = PhasePartition(Δq_tot)
    R_m = gas_constant_air(param_set, q_pt)
    _cp_m = cp_m(param_set, q_pt)
    _cv_m = cv_m(param_set, q_pt)
    π_exner = FT(1) - _grav / (_cp_m * θ) * z           # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(_cv_m / R_m)      # density
    T = θ * π_exner

    if moisture_model(bl) isa EquilMoist
        e_int = internal_energy(param_set, T, q_pt)
        ts = PhaseEquil_ρeq(param_set, ρ, e_int, Δq_tot)
    else
        e_int = internal_energy(param_set, T)
        ts = PhaseDry(param_set, e_int, ρ)
    end
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    ## State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl, aux)            # potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy
    ρq_tot = ρ * Δq_tot                                 # total water specific humidity

    ## Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.energy.ρe = ρe_tot
    if !(moisture_model(bl) isa DryModel)
        state.moisture.ρq_tot = ρq_tot
    end
end

function config_risingbubble(FT, N, resolution, xmax, ymax, zmax, with_moisture)

    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    _C_smag = FT(C_smag(param_set))

    if with_moisture
        moisture = EquilMoist()
    else
        moisture = DryModel()
    end
    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = SmagorinskyLilly(_C_smag),
        moisture = moisture,
        tracers = NoTracers(),
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        init_state_prognostic = init_risingbubble!,
        source = (Gravity(),),
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "RisingBubble",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_risingbubble!,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    FT = Float64
    interval = "50ssecs"
    boundaries = [
        FT(0.0) FT(0.0) FT(0.0)
        FT(10000) FT(500) FT(10000)
    ]
    resolution = (FT(100), FT(100), FT(100))
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    state_dgngrp = setup_dump_state_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    aux_dgngrp = setup_dump_aux_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        dgngrp,
        state_dgngrp,
        aux_dgngrp,
    ])
end

function main()
    # add a command line argument to specify whether to use a moist setup
    rb_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(rb_args, "RisingBubble")
    @add_arg_table! rb_args begin
        "--with-moisture"
        help = "use a moist setup"
        action = :store_const
        constant = true
        default = false
    end
    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = rb_args)
    with_moisture = cl_args["with_moisture"]

    FT = Float64
    N = 4
    Δh = FT(125)
    Δv = FT(125)
    resolution = (Δh, Δh, Δv)
    xmax = FT(10000)
    ymax = FT(500)
    zmax = FT(10000)
    t0 = FT(0)
    timeend = FT(1000)

    CFL = FT(1.7)

    driver_config =
        config_risingbubble(FT, N, resolution, xmax, ymax, zmax, with_moisture)

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_solver_type = ode_solver_type,
        init_on_cpu = true,
        Courant_number = CFL,
    )
    dgn_config = config_diagnostics(driver_config)

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
