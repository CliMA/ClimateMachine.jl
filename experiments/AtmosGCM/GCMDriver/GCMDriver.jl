#!/usr/bin/env julia --project

# This file is the entry point for a generalized GCM driver that is used
# to run a variety of GCM experiments.
# Currently available default experiments are:
#   1. `baroclinicwave_problem`: initial value problem (no sources to
#      maintain equilibration)
#   2. `heldsuarez_problem`: runs to equilibration
#
# Experiment name is required to be specified in the command line:
#   e.g.: julia --project experiments/AtmosGCM/GCMDriver/jl --experiment=heldsuarez_problem
#
# The initial / boundary conditions of each experiment (defaults of which are
# defined in the experiment`_problem.jl` files) can be mixed and matched, as long
# as they are specified in:
#   - gcm_bcs.jl (use `--surface-flux`)
#   - gcm_perturbations.jl (use `--init-perturbation`)
#   - gcm_base_states.jl (use `--init-base-state`)
#   - gcm_moisture_profiles.jl (use `--init-moisture-profile`)
# Default sources cannot be currently overriden from command line and are defined in `gcm_sources.jl`

using ArgParse
using LinearAlgebra
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Atmos: recover_thermo_state
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.SystemSolvers
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet
using CLIMAParameters.Atmos.SubgridScale

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# Menu for initial conditions, boundary conditions and sources
include("gcm_bcs.jl")                # Boundary conditions
include("gcm_perturbations.jl")      # Initial perturbation
include("gcm_base_states.jl")        # Initial base state
include("gcm_moisture_profiles.jl")  # Initial moisture profile
include("gcm_sources.jl")            # GCM-specific sources and parametrizations

# Set default initial conditions, boundary conditions and sources for each experiment type
include("baroclinicwave_problem.jl") # initial value problem (no sources to maintain equilibration)
include("heldsuarez_problem.jl")     # runs to equilibration

# Initial conditions (common to all GCM experiments)
function init_gcm_experiment!(problem, bl, state, aux, coords, t)
    FT = eltype(state)

    # General parameters
    M_v::FT = molmass_ratio(bl.param_set) - 1 # constant for virtual temperature conversion - FIX: this assumes no initial liq/ice

    # Select initial perturbation
    u′, v′, w′, rand_pert =
        init_perturbation(problem.perturbation, bl, state, aux, coords, t)

    # Select initial base state
    T_v, p, u_ref, v_ref, w_ref =
        init_base_state(problem.base_state, bl, state, aux, coords, t)

    # Select initial moisture profile
    q_tot = init_moisture_profile(
        problem.moisture_profile,
        bl,
        state,
        aux,
        coords,
        t,
        p,
    )

    # Calculate initial total winds
    u_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
    u_cart = sphr_to_cart_vec(bl.orientation, u_sphere, aux)

    # Calculate initial temperature and density
    phase_partition = PhasePartition(q_tot)
    T::FT = T_v / (1 + M_v * q_tot) # this needs to be adapted for ice and liq

    ρ::FT = air_density(bl.param_set, T, p, phase_partition)

    ## potential & kinetic energy
    e_pot::FT = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * u_cart' * u_cart
    e_tot::FT = total_energy(bl.param_set, e_kin, e_pot, T, phase_partition)

    ## Assign state variables
    state.ρ = ρ
    state.ρu = ρ * u_cart
    state.ρe = ρ * e_tot * rand_pert

    if bl.moisture isa EquilMoist
        state.moisture.ρq_tot = ρ * q_tot
    end

    return nothing
end

lowercasearg(arg::Nothing) = nothing
lowercasearg(arg) = lowercase(arg)

# Helper for parsing `--experiment` command line argument to initialize defaults for the specified problem
function parse_experiment_arg(arg)
    if arg == "baroclinic_wave"
        problem_type = BaroclinicWaveProblem
    elseif arg == "heldsuarez"
        problem_type = HeldSuarezProblem
    else
        error("unknown experiment: " * arg)
    end

    return problem_type
end

# General GCM configuration setup
function config_gcm_experiment(
    ::Type{FT},
    poly_order,
    resolution,
    experiment_arg,
    perturbation_arg,
    base_state_arg,
    moisture_profile_arg,
    surface_flux_arg,
) where {FT}
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Distance between surface and top of atmosphere (m)
    domain_height::FT = 30e3

    # make the orientation explicit (for surface fluxes)
    orientation = SphericalOrientation()

    # Determine the problem type
    problem_type = parse_experiment_arg(experiment_arg)

    # Set up problem components
    perturbation = parse_perturbation_arg(perturbation_arg)
    base_state = parse_base_state_arg(base_state_arg)
    moisture_profile = parse_moisture_profile_arg(moisture_profile_arg)

    # Choose the moisture model
    if moisture_profile isa NoMoistureProfile
        hyperdiffusion = DryBiharmonic(FT(8 * 3600))
        moisture = DryModel()
    else
        hyperdiffusion = EquilMoistBiharmonic(FT(8 * 3600))
        moisture = EquilMoist{FT}()
    end

    # Set up the boundary conditions
    boundarycondition = parse_surface_flux_arg(
        surface_flux_arg,
        FT,
        param_set,
        orientation,
        moisture,
    )

    # Set up the problem
    problem = problem_type(
        boundarycondition = boundarycondition,
        perturbation = perturbation,
        base_state = base_state,
        moisture_profile = moisture_profile,
    )

    # Create the Atmosphere model
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        problem = problem,
        orientation = orientation,
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        hyperdiffusion = hyperdiffusion,
        moisture = moisture,
        source = setup_source(problem),
    )

    # Create the GCM driver configuration
    config = ClimateMachine.AtmosGCMConfiguration(
        problem_name(problem),
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_gcm_experiment!;
        model = model,
    )

    return config
end

# Diagnostics configuration setup
function config_diagnostics(::Type{FT}, driver_config) where {FT}
    interval = "40000steps"

    _planet_radius = planet_radius(param_set)::FT

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(1), FT(1), FT(1000)) # in (deg, deg, m)
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    dgngrp = setup_atmos_default_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

# Entry point
function main()
    # Helper for command-line arguments to specify the experiment and
    # the override experiment defaults
    # various options
    # TODO: some of this must move to a future namelist functionality
    exp_args = ArgParseSettings(autofix_names = true) # hyphens replaced with underscores in cl_args
    add_arg_group!(exp_args, "GCMDriver")
    @add_arg_table! exp_args begin
        "--experiment"
        help = """
            baroclinic_wave defaults:
                init-perturbation: deterministic,
                init-base-state: bc_wave,
                init-moisture-profile: moist_low_tropics;
            heldsuarez defaults:
                init-perturbation: deterministic,
                init-base-state: heldsuarez,
                init-moisture-profile: moist_low_tropics
            """
        metavar = "baroclinic_wave|heldsuarez"
        arg_type = String
        required = true
        "--init-perturbation"
        help = "select the perturbation to use"
        metavar = "deterministic|random|zero"
        arg_type = String
        "--init-base-state"
        help = "select the initial state to use"
        metavar = "bc_wave|heldsuarez|zero"
        arg_type = String
        "--init-moisture-profile"
        help = "specify the moisture profile"
        metavar = "moist_low_tropics|zero|dry"
        arg_type = String
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "default|bulk"
        arg_type = String
        default = "default"
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = exp_args)
    experiment_arg = lowercasearg(cl_args["experiment"])
    perturbation_arg = lowercasearg(cl_args["init_perturbation"]) # use "_" not "-" in cl_args
    base_state_arg = lowercasearg(cl_args["init_base_state"])
    moisture_profile_arg = lowercasearg(cl_args["init_moisture_profile"])
    surface_flux_arg = lowercasearg(cl_args["surface_flux"])

    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 3                           # discontinuous Galerkin polynomial order
    n_horz = 12                              # horizontal element number
    n_vert = 6                               # vertical element number
    n_days::FT = 0.1
    timestart::FT = 0                        # start time (s)
    timeend::FT = n_days * day(param_set)    # end time (s)

    # Set up driver configuration
    driver_config = config_gcm_experiment(
        FT,
        poly_order,
        (n_horz, n_vert),
        experiment_arg,
        perturbation_arg,
        base_state_arg,
        moisture_profile_arg,
        surface_flux_arg,
    )

    # Choose time stepper
    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = true,
        discrete_splitting = false,
    )

    # The target acoustic CFL number; time step is computed such that the
    # horizontal acoustic Courant number is CFL.
    CFL = FT(0.1)

    # Set up the solver
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = HorizontalDirection(),
        diffdir = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up filters as user-defined callbacks
    filterorder = 20
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter, cbfilter),
        check_euclidean_distance = true,
    )

    return result, solver_config
end

result, solver_config = main()
