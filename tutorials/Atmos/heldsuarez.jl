# # Dry atmosphere GCM with Held-Suarez forcing
#
# The Held-Suarez setup (Held and Suarez, 1994) is a textbook example for a simplified atmospheric global circulation model configuration
# which has been used as a benchmark experiment for development of the dynamical cores (i.e., GCMs without continents,
# moisture or parametrization schemes of the physics) for atmospheric models.
# It is forced by a thermal relaxation to a reference state and damped by linear (Rayleigh) friction. This example demonstrates how
#
#   * to set up a ClimateMachine-Atmos GCM configuration;
#   * to select and save GCM diagnostics output.
#
# To begin, we load ClimateMachine and a few miscellaneous useful Julia packages.
using Distributions
using Random
using StaticArrays

# ClimateMachine specific modules needed to make this example work (e.g., we will need
# spectral filters, etc.).
using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.MoistThermodynamics
using ClimateMachine.VariableTemplates

# [ClimateMachine parameters](https://github.com/CliMA/CLIMAParameters.jl) are needed to have access to Earth's physical parameters
#
using CLIMAParameters
using CLIMAParameters.Planet: R_d, day, grav, cp_d, cv_d, planet_radius

# We need to load the physical parameters for Earth to have an Earth-like simulation :).
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
nothing # hide

# Construct the Held-Suarez forcing function
# We can view this as part the right-hand-side of our governing equations. It
# forces the total energy field in a way that the resulting steady-state velocity
# and temperature fields of the simluation resemble those of an idealized dry
# planet.
function held_suarez_forcing!(
    balance_law,
    source,
    state,
    diffusive,
    aux,
    time::Real,
    direction,
)
    FT = eltype(state)

    ## Parameters
    T_ref::FT = 255 # reference temperature for Held-Suarez forcing (K)

    ## Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe

    coord = aux.coord
    e_int = internal_energy(
        balance_law.moisture,
        balance_law.orientation,
        state,
        aux,
    )
    T = air_temperature(balance_law.param_set, e_int)
    _R_d = FT(R_d(balance_law.param_set))
    _day = FT(day(balance_law.param_set))
    _grav = FT(grav(balance_law.param_set))
    _cp_d = FT(cp_d(balance_law.param_set))
    _cv_d = FT(cv_d(balance_law.param_set))

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
    φ = latitude(balance_law.orientation, aux)
    p = air_pressure(balance_law.param_set, T, ρ)

    ##TODO: replace _p0 with dynamic surfce pressure in Δσ calculations to account
    #for topography, but leave unchanged for calculations of σ involved in T_equil
    _p0 = 1.01325e5
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor

    ## Apply Held-Suarez forcing
    source.ρu -= k_v * projection_tangential(balance_law, aux, ρu)
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end
nothing # hide

# ## Set initial condition
# When using ClimateMachine, we need to define a function that sets the intial state of our
# model run. In our case, we use the reference state of the simulation (defined
# below) and add a little bit of noise. Note that the initial states includes a
# zero initial velocity field. 
function init_heldsuarez!(balance_law, state, aux, coordinates, time)
    FT = eltype(state)

    ## Set initial state to reference state with random perturbation
    rnd = FT(1.0 + rand(Uniform(-1e-3, 1e-3)))
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = rnd * aux.ref_state.ρe

    nothing
end
nothing # hide

# ## Initialize ClimateMachine
# Before we do anything further, we need to initialize ClimateMachine. Among other things,
# this will initialize the MPI for us.
ClimateMachine.init()
nothing # hide

# ## Setting the floating-type precision
# ClimateMachine allows us to run a model with different floating-type precisions, with
# lower precision we get our results faster, and with higher precision, we may get
# more accurate results, depending on the questions we are after.
FT = Float32
nothing # hide

# ## Setup model configuration
# Now that we have definied our forcing and initialization functions, and have
# initialized ClimateMachine, we can set up the model. 
# 
# ## Set up a reference state
# We start by setting up a
# reference state. This is simply a vector field that we subtract from the
# solutions to the governing equations to both improve numerical stability of the
# implicit time stepper and enable faster model spin-up. The reference state
# assumes hydrostatic balance and ideal gas law, with a pressure $p_r(z)$ and 
# density $\rho_r(z)$ that only depend on altitude $z$ and are in hydrostatic balance
# with each other. 
# In this example, the reference temperature field smoothly transitions from a
# linearly decaying profile near the surface to a constant temperature profile at the top
# of the domain.
T_surface = FT(290) ## surface temperature (K)
ΔT = FT(60)  ## temperature drop between surface and top of atmosphere (K)
H_t = FT(8e3) ## height scale over which temperature drops (m)
temp_profile_ref = DecayingTemperatureProfile(T_surface, ΔT, H_t)
ref_state = HydrostaticState(temp_profile_ref, FT(0))
nothing # hide

# ## Set up a Rayleigh sponge layer
# To avoid wave reflection at the top of the domain, the model
# applies a sponge layer that linearly damps the momentum equations.
domain_height = FT(30e3)               ## height of the computational domain (m)
z_sponge = FT(12e3)                    ## height at which sponge begins (m)
α_relax = FT(1 / 60 / 15)              ## sponge relaxation rate (1/s)
exponent = FT(2)                       ## sponge exponent for squared-sinusoid profile
u_relax = SVector(FT(0), FT(0), FT(0)) ## relaxation velocity (m/s)
sponge = RayleighSponge(domain_height, z_sponge, α_relax, u_relax, exponent)
nothing # hide

# ## Set up turbulence models
# In order to produce a stable simulation, we need to dissipate energy and
# enstrophy at the smallest scales of the developed flow field. To achieve this we
# set up diffusive forcing functions.
c_smag = FT(0.21)   ## Smagorinsky constant
τ_hyper = FT(4 * 3600) ## hyperdiffusion time scale
turbulence_model = SmagorinskyLilly(c_smag)
hyperdiffusion_model = StandardHyperDiffusion(FT(4 * 3600))
nothing # hide

# ## Instantiate the model
# The Held Suarez setup was designed to produce an equilibrated state
# that is comparable to the zonal mean of the Earth’s atmosphere. 
model = AtmosModel{FT}(
    AtmosGCMConfigType,
    param_set;
    ref_state = ref_state,
    turbulence = turbulence_model,
    hyperdiffusion = hyperdiffusion_model,
    moisture = DryModel(),
    source = (Gravity(), Coriolis(), held_suarez_forcing!, sponge),
    init_state = init_heldsuarez!,
)
nothing # hide
# This concludes the setup of the physical model!

# ## Set up the driver 
# We just need to set up a few parameters that define the resolution of the
# discontinuous Galerkin method and how long we want to run our model setup for.
poly_order = 5                        ## discontinuous Galerkin polynomial order
n_horz = 2                            ## horizontal element number
n_vert = 2                            ## vertical element number
resolution = (n_horz, n_vert)
n_days = 1                            ## experiment day number
timestart = FT(0)                     ## start time (s)
timeend = FT(n_days * day(param_set)) ## end time (s)
nothing # hide

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

# The next lines set up the time stepper.
solver_config = ClimateMachine.SolverConfiguration(
    timestart,
    timeend,
    driver_config,
    Courant_number = FT(0.2),
    init_on_cpu = true,
    CFL_direction = HorizontalDirection(),
    diffdir = HorizontalDirection(),
);

# ## Set up spectral exponential filter
# After every completed time step we apply a spectral filter to remove remaining
# small-scale noise introduced by the numerical procedures. This assures that our
# run remains stable.
filterorder = 10
filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
    Filters.apply!(
        solver_config.Q,
        1:size(solver_config.Q, 2),
        solver_config.dg.grid,
        filter,
    )
    nothing
end

# ## Setup diagnostic output
#
# Choose frequency and resolution of output, and a diagnostics group (dgngrp)
# which defines output variables. This needs to be defined
# in [diagnostics](https://CliMA.github.io/ClimateMachine.jl/latest/generated/Diagnostics).
interval = "1000steps"
_planet_radius = FT(planet_radius(param_set))
info = driver_config.config_info
boundaries = [
    FT(-90.0) FT(-180.0) _planet_radius
    FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
]
resolution = (FT(10), FT(10), FT(1000)) # in (deg, deg, m)
interpol = ClimateMachine.InterpolationConfiguration(
    driver_config,
    boundaries,
    resolution,
)

dgn_config = setup_dump_state_and_aux_diagnostics(
    interval,
    driver_config.name,
    interpol = interpol,
    project = true,
)
nothing # hide

# ## Run the model
# Finally, we can run the model using the physical setup and solvers from above. We
# use the spectral filter in our callbacks after every time step, and collect
# the diagnostics output.
result = ClimateMachine.invoke!(
    solver_config;
    diagnostics_config = dgn_config,
    user_callbacks = (cbfilter,),
    check_euclidean_distance = true,
)
nothing # hide

# ## References
#
# - Held, I.M. and M.J. Suarez, 1994: A Proposal for the Intercomparison
# the Dynamical Cores of Atmospheric General Circulation Models. Bull. #
# Amer. Meteor. Soc., 75, 1825–1830, https://doi.org/10.1175/1520-0477(1994)075<1825:APFTIO>2.0.CO;2
