# # Dry atmosphere GCM with Held-Suarez forcing
#
# The Held-Suarez setup (Held and Suarez, 1994) is a textbook example for a
# simplified atmospheric global circulation model configuration which has been
# used as a benchmark experiment for development of the dynamical cores (i.e.,
# GCMs without continents, moisture or parametrization schemes of the physics)
# for atmospheric models.  It is forced by a thermal relaxation to a reference
# state and damped by linear (Rayleigh) friction. This example demonstrates how
#
#   * to set up a ClimateMachine-Atmos GCM configuration;
#   * to select and save GCM diagnostics output.
#
# To begin, we load ClimateMachine and a few miscellaneous useful Julia packages.
using Distributions
using Random
using StaticArrays
using UnPack

# ClimateMachine specific modules needed to make this example work (e.g., we will need
# spectral filters, etc.).
using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.TemperatureProfiles
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws

import ClimateMachine.BalanceLaws: source
import ClimateMachine.Atmos: filter_source, atmos_source!

# [ClimateMachine parameters](https://github.com/CliMA/CLIMAParameters.jl) are
# needed to have access to Earth's physical parameters.
using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, cp_d, cv_d, planet_radius

# We need to load the physical parameters for Earth to have an Earth-like simulation :).
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

"""
    HeldSuarezForcingTutorial{PV <: Union{Momentum,Energy}} <: TendencyDef{Source, PV}

Defines a forcing that parametrises radiative and frictional effects using
Newtonian relaxation and Rayleigh friction, following Held and Suarez (1994)
"""
struct HeldSuarezForcingTutorial{PV <: Union{Momentum, Energy}} <:
       TendencyDef{Source, PV} end

HeldSuarezForcingTutorial() =
    (HeldSuarezForcingTutorial{Momentum}(), HeldSuarezForcingTutorial{Energy}())

filter_source(pv::PV, m, s::HeldSuarezForcingTutorial{PV}) where {PV} = s
atmos_source!(::HeldSuarezForcingTutorial, args...) = nothing

function held_suarez_forcing_coefficients(bl, args)
    @unpack state, aux = args
    @unpack ts = args.precomputed
    FT = eltype(state)

    ## Parameters
    T_ref = FT(255)

    _R_d = FT(R_d(bl.param_set))
    _day = FT(day(bl.param_set))
    _grav = FT(grav(bl.param_set))
    _cp_d = FT(cp_d(bl.param_set))
    _p0 = FT(MSLP(bl.param_set))

    ## Held-Suarez parameters
    k_a = FT(1 / (40 * _day))
    k_f = FT(1 / _day)
    k_s = FT(1 / (4 * _day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)

    ## Held-Suarez forcing
    φ = latitude(bl, aux)
    p = air_pressure(ts)

    ##TODO: replace _p0 with dynamic surface pressure in Δσ calculations to account
    ##for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor
    return (k_v = k_v, k_T = k_T, T_equil = T_equil)
end

function source(s::HeldSuarezForcingTutorial{Energy}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    nt = held_suarez_forcing_coefficients(m, args)
    _cv_d = FT(cv_d(m.param_set))
    @unpack k_T, T_equil = nt
    T = air_temperature(ts)
    return -k_T * state.ρ * _cv_d * (T - T_equil)
end

function source(s::HeldSuarezForcingTutorial{Momentum}, m, args)
    @unpack state, aux = args
    nt = held_suarez_forcing_coefficients(m, args)
    return -nt.k_v * projection_tangential(m, aux, state.ρu)
end


# ## Set initial condition
# When using ClimateMachine, we need to define a function that sets the initial
# state of our model run. In our case, we use the reference state of the
# simulation (defined below) and add a little bit of noise. Note that the
# initial states includes a zero initial velocity field.
function init_heldsuarez!(problem, balance_law, state, aux, localgeo, time)
    FT = eltype(state)

    ## Set initial state to reference state with random perturbation
    rnd = FT(1 + rand(Uniform(-1e-3, 1e-3)))
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = rnd * aux.ref_state.ρe
end;


# ## Initialize ClimateMachine
# Before we do anything further, we need to initialize ClimateMachine. Among
# other things, this will initialize the MPI us.
ClimateMachine.init();


# ## Setting the floating-type precision
# ClimateMachine allows us to run a model with different floating-type
# precisions, with lower precision we get our results faster, and with higher
# precision, we may get more accurate results, depending on the questions we
# are after.
const FT = Float32;


# ## Setup model configuration
# Now that we have defined our forcing and initialization functions, and have
# initialized ClimateMachine, we can set up the model.
#
# ## Set up a reference state
# We start by setting up a reference state. This is simply a vector field that
# we subtract from the solutions to the governing equations to both improve
# numerical stability of the implicit time stepper and enable faster model
# spin-up. The reference state assumes hydrostatic balance and ideal gas law,
# with a pressure $p_r(z)$ and density $\rho_r(z)$ that only depend on altitude
# $z$ and are in hydrostatic balance with each other.
#
# In this example, the reference temperature field smoothly transitions from a
# linearly decaying profile near the surface to a constant temperature profile
# at the top of the domain.
temp_profile_ref = DecayingTemperatureProfile{FT}(param_set)
ref_state = HydrostaticState(temp_profile_ref);

# ## Set up a Rayleigh sponge layer
# To avoid wave reflection at the top of the domain, the model applies a sponge
# layer that linearly damps the momentum equations.
domain_height = FT(30e3)               # height of the computational domain (m)
z_sponge = FT(12e3)                    # height at which sponge begins (m)
α_relax = FT(1 / 60 / 15)              # sponge relaxation rate (1/s)
exponent = FT(2)                       # sponge exponent for squared-sinusoid profile
u_relax = SVector(FT(0), FT(0), FT(0)) # relaxation velocity (m/s)
sponge = RayleighSponge(FT, domain_height, z_sponge, α_relax, u_relax, exponent);

# ## Set up turbulence models
# In order to produce a stable simulation, we need to dissipate energy and
# enstrophy at the smallest scales of the developed flow field. To achieve this
# we set up diffusive forcing functions.
c_smag = FT(0.21);   # Smagorinsky constant
τ_hyper = FT(4 * 3600); # hyperdiffusion time scale
turbulence_model = SmagorinskyLilly(c_smag);
hyperdiffusion_model = DryBiharmonic(FT(4 * 3600));


# ## Instantiate the model
# The Held Suarez setup was designed to produce an equilibrated state that is
# comparable to the zonal mean of the Earth’s atmosphere.
model = AtmosModel{FT}(
    AtmosGCMConfigType,
    param_set;
    init_state_prognostic = init_heldsuarez!,
    ref_state = ref_state,
    turbulence = turbulence_model,
    hyperdiffusion = hyperdiffusion_model,
    moisture = DryModel(),
    source = (Gravity(), Coriolis(), HeldSuarezForcingTutorial()..., sponge),
);

# This concludes the setup of the physical model!

# ## Set up the driver
# We just need to set up a few parameters that define the resolution of the
# discontinuous Galerkin method and for how long we want to run our model
# setup.
poly_order = 5;                        ## discontinuous Galerkin polynomial order
n_horz = 2;                            ## horizontal element number
n_vert = 2;                            ## vertical element number
resolution = (n_horz, n_vert)
n_days = 0.1;                          ## experiment day number
timestart = FT(0);                     ## start time (s)
timeend = FT(n_days * day(param_set)); ## end time (s);


# The next lines set up the spatial grid.
driver_config = ClimateMachine.AtmosGCMConfiguration(
    "HeldSuarez",
    poly_order,
    resolution,
    domain_height,
    param_set,
    init_heldsuarez!;
    model = model,
);

# The next lines set up the time stepper. Since the resolution
# in the vertical is much finer than in the horizontal,
# the 'stiff' parts of the PDE will be in the vertical.
# Setting `splitting_type = HEVISplitting()` will treat
# vertical acoustic waves implicitly, while all other dynamics
# are treated explicitly.
ode_solver_type = ClimateMachine.IMEXSolverType(
    splitting_type = HEVISplitting(),
    implicit_model = AtmosAcousticGravityLinearModel,
    implicit_solver = ManyColumnLU,
    solver_method = ARK2GiraldoKellyConstantinescu,
);

solver_config = ClimateMachine.SolverConfiguration(
    timestart,
    timeend,
    driver_config,
    Courant_number = FT(0.1),
    ode_solver_type = ode_solver_type,
    init_on_cpu = true,
    CFL_direction = HorizontalDirection(),
    diffdir = HorizontalDirection(),
);

# ## Set up spectral exponential filter
# After every completed time step we apply a spectral filter to remove
# remaining small-scale noise introduced by the numerical procedures. This
# assures that our run remains stable.
filterorder = 10;
filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder);
cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
    Filters.apply!(
        solver_config.Q,
        AtmosFilterPerturbations(model),
        solver_config.dg.grid,
        filter,
        state_auxiliary = solver_config.dg.state_auxiliary,
    )
    nothing
end;

# ## Setup diagnostic output
#
# Choose frequency and resolution of output, and a diagnostics group (dgngrp)
# which defines output variables. This needs to be defined in
# [`Diagnostics`](@ref ClimateMachine.Diagnostics).
interval = "1000steps";
_planet_radius = FT(planet_radius(param_set));
info = driver_config.config_info;
boundaries = [
    FT(-90.0) FT(-180.0) _planet_radius
    FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
];
resolution = (FT(10), FT(10), FT(1000)); # in (deg, deg, m)
interpol = ClimateMachine.InterpolationConfiguration(
    driver_config,
    boundaries,
    resolution,
);

dgngrps = [
    setup_dump_state_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    ),
    setup_dump_aux_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    ),
];
dgn_config = ClimateMachine.DiagnosticsConfiguration(dgngrps);

# ## Run the model
# Finally, we can run the model using the physical setup and solvers from
# above. We use the spectral filter in our callbacks after every time step, and
# collect the diagnostics output.
result = ClimateMachine.invoke!(
    solver_config;
    diagnostics_config = dgn_config,
    user_callbacks = (cbfilter,),
    check_euclidean_distance = true,
);


# ## References
#
# - [Held1994](@cite)
