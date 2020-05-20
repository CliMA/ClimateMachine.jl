# # [Rising Thermal Bubble](@id EX-RTB-docs)
#
# In this example, we demonstrate the usage of the `ClimateMachine`
# [AtmosModel](@ref AtmosModel-docs) machinery to solve the fluid
# dynamics of a thermal perturbation in a neutrally stratified background state
# defined by its uniform potential temperature. We solve a flow in a
# [`FlatOrientation`](@ref LESConfig) (Box) configuration - this is
# representative of a large-eddy simulation. Several versions of the problem
# setup may be found in literature, but the general idea is to examine the
# vertical ascent of a thermal _bubble_ (we can interpret these as simple
# representation of convective updrafts).
#
# ## Description of experiment
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries
#    - `Impenetrable(FreeSlip())` - no momentum flux, no mass flux through
#      walls.
#    - `Impermeable()` - non-porous walls, i.e. no diffusive fluxes through
#       walls.
#    - Laterally periodic
# 3) Domain - 2500m (horizontal) x 2500m (horizontal) x 2500m (vertical)
# 4) Resolution - 50m effective resolution
# 5) Total simulation time - 1000s
# 6) Mesh Aspect Ratio (Effective resolution) 1:1
# 7) Overrides defaults for
#    - CPU Initialisation
#    - Time integrator
#    - Sources
#    - Smagorinsky Coefficient
# 8) Default settings can be found in `src/Driver/<files>.jl`

#md # !!! note
#md #     This experiment setup assumes that you have installed the
#md #     `ClimateMachine` according to the instructions on the landing page.
#md #     We assume the users' familiarity with the conservative form of the
#md #     equations of motion for a compressible fluid (see the
#md #     [`AtmosModel`](@ref AtmosModel-docs) page).
#md #
#md #     The following topics are covered in this example
#md #     - Package requirements
#md #     - Defining a `model` subtype for the set of conservation equations
#md #     - Defining the initial conditions
#md #     - Applying boundary conditions
#md #     - Applying source terms
#md #     - Choosing a turbulence model
#md #     - Adding tracers to the model
#md #     - Choosing a time-integrator
#md #     - Choosing diagnostics (output) configurations
#md #
#md #     The following topics are not covered in this example
#md #     - Defining new boundary conditions
#md #     - Defining new turbulence models
#md #     - Building new time-integrators
#md #     - Adding diagnostic variables (beyond a standard pre-defined list of
#md #       variables)
#
# ## Boilerplate (Using Modules)
#
# #### [Skip Section](@ref init)
#
# Before setting up our experiment, we recognize that we need to import some
# pre-defined functions from other packages. Julia allows us to use existing
# modules (variable workspaces), or write our own to do so.  Complete
# documentation for the Julia module system can be found
# [here](https://docs.julialang.org/en/v1/manual/modules/#).

# We need to use the `ClimateMachine` module! This imports all functions
# specific to atmospheric and ocean flow modelling.  While we do not cover the
# ins-and-outs of the contents of each of these we provide brief descriptions
# of the utility of each of the loaded packages.
using ClimateMachine
ClimateMachine.init()

using ClimateMachine.Atmos
# - Required so that we inherit the appropriate model types for the large-eddy
#   simulation (LES) and global-circulation-model (GCM) configurations.
using ClimateMachine.ConfigTypes
# - Required so that we may define diagnostics configurations, e.g. choice of
#   file-writer, choice of output variable sets, output-frequency and directory,
using ClimateMachine.Diagnostics
# - Required so that we may define (or utilise existing functions) functions
#   that are `called-back` or executed at frequencies of either timesteps,
#   simulation-time, or wall-clock time.
using ClimateMachine.GenericCallbacks
# - Required so we load the appropriate functions for the time-integration
#   component. Contains ODESolver methods.
using ClimateMachine.ODESolvers
# - Required for utility of spatial filtering functions (e.g. positivity
#   preservation)
using ClimateMachine.Mesh.Filters
# - Required so functions for computation of moist thermodynamic quantities is
#   enabled.
using ClimateMachine.MoistThermodynamics
# - Required so we may access our variable arrays by a sensible naming
#   convention rather than by numerical array indices.
using ClimateMachine.VariableTemplates
# - Required so we may access planet parameters
#   ([CLIMAParameters](https://CliMA.github.io/CLIMAParameters.jl/latest/)
#   specific to this problem include the gas constant, specific heats,
#   mean-sea-level pressure, gravity and the Smagorinsky coefficient)

# In ClimateMachine we use `StaticArrays` for our variable arrays.
using StaticArrays
# We also use the `Test` package to help with unit tests and continuous
# integration systems to design sensible tests for our experiment to ensure new
# / modified blocks of code don't damage the fidelity of the physics. The test
# defined within this experiment is not a unit test for a specific
# subcomponent, but ensures time-integration of the defined problem conditions
# within a reasonable tolerance. Immediately useful macros and functions from
# this include `@test` and `@testset` which will allow us to define the testing
# parameter sets.
using Test

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ## [Initial Conditions](@id init)
# This example of a rising thermal bubble can be classified as an initial value
# problem. We must (at the very least) assign values for the initial variables
# in a sensible manner. This example demonstrates the use of functions defined
# in the [`MoistThermodynamics`](@ref MoistThermodynamics-docs) package to
# generate the appropriate initial state for our problem.

#md # !!! note
#md #     The following variables are assigned in the initial condition
#md #     - `state.ρ` = Scalar quantity for initial density profile
#md #     - `state.ρu`= 3-component vector for initial momentum profile
#md #     - `state.ρe`= Scalar quantity for initial total-energy profile
#md #       humidity
#md #     - `state.tracers.ρχ` = Vector of four tracers (here, for demonstration
#md #       only; we can interpret these as dye injections for visualisation
#md #       purposes)
function init_risingbubble!(bl, state, aux, (x, y, z), t)
    # Problem float-type
    FT = eltype(state)

    # Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    # Define bubble center and background potential temperature
    xc::FT = 1250
    yc::FT = 1250
    zc::FT = 1000
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    rc::FT = 500
    # TODO: clean this up, or add convenience function:
    # This is configured in the reference hydrostatic state
    θ_ref::FT = bl.ref_state.virtual_temperature_profile.T_surface
    Δθ::FT = 0

    # Compute temperature difference over bubble region
    if r <= rc
        Δθ = FT(5) * cospi(r / rc / 2)
    end

    # Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      # density
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    #State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)# potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    ρχ = FT(0)                                          # tracer

    # We inject tracers at the initial condition at some specified z coordinates
    if 500 < z <= 550
        ρχ += FT(0.05)
    end

    # We want 4 tracers
    ntracers = 4

    # Define 4 tracers, (arbitrary scaling for this demo problem)
    ρχ = SVector{ntracers, FT}(ρχ, ρχ / 2, ρχ / 3, ρχ / 4)

    # Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.tracers.ρχ = ρχ
end

# ## [Model Configuration](@id config-helper)
# We define a configuration function to assist in prescribing the physical
# model. The purpose of this is to populate the
# [`ClimateMachine.AtmosLESConfiguration`](@ref LESConfig) with arguments
# appropriate to the problem being considered.
function config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

    # Choose an Explicit Multi-rate Solver from the existing [ODESolvers](@ref
    # ODESolvers-docs) options Apply the outer constructor to define the
    # `ode_solver`. Here `AtmosAcousticGravityLinearModel` splits the
    # acoustic-gravity wave components from the advection-diffusion dynamics.
    # The 1D-IMEX method is less appropriate for the problem given the current
    # mesh aspect ratio (1:1)
    ode_solver = ClimateMachine.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )

    # Since we want four tracers, we specify this and include the appropriate
    # diffusivity scaling coefficients (normally these would be physically
    # informed but for this demonstration we use integers corresponding to the
    # tracer index identifier)
    ntracers = 4
    δ_χ = SVector{ntracers, FT}(1, 2, 3, 4)

    # The model coefficient for the turbulence closure is defined via the
    # [CLIMAParameters
    # package](https://CliMA.github.io/CLIMAParameters.jl/latest/) A reference
    # state for the linearisation step is also defined.
    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    # The fun part! Here we assemble the `AtmosModel`.
    #md # !!! note
    #md #     Docs on model subcomponent options can be found here:
    #md #     - [`param_set`](https://CliMA.github.io/CLIMAParameters.jl/latest/)
    #md #     - [`turbulence`](@ref Turbulence-Closures-docs)
    #md #     - [`hyperdiffusion`](@ref Hyperdiffusion-docs)
    #md #     - [`source`](@ref atmos-sources)
    #md #     - [`tracers`](@ref Tracers-docs)
    #md #     - [`init_state`](@ref init)

    _C_smag = FT(C_smag(param_set))
    model = AtmosModel{FT}(
        AtmosLESConfigType,                           # Flow in a box, requires the AtmosLESConfigType
        param_set;                                    # Parameter set corresponding to earth parameters
        turbulence = SmagorinskyLilly(_C_smag),       # Turbulence closure model
        moisture = DryModel(),                        # Exclude moisture variables
        hyperdiffusion = StandardHyperDiffusion(60),  # Hyperdiffusion (4th order) model
        source = (Gravity(),),                        # Gravity is the only source term here
        tracers = NTracers{ntracers, FT}(δ_χ),        # Tracer model with diffusivity coefficients
        ref_state = ref_state,                        # Reference state
        init_state_conservative = init_risingbubble!, # Apply the initial condition
    )

    # Finally, we pass a `Problem Name` string, the mesh information, and the
    # model type to  the [`AtmosLESConfiguration`] object.
    config = ClimateMachine.AtmosLESConfiguration(
        "DryRisingBubble",       # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        xmax,                    # Domain maximum size [m]
        ymax,                    # Domain maximum size [m]
        zmax,                    # Domain maximum size [m]
        param_set,               # Parameter set.
        init_risingbubble!,      # Function specifying initial condition
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
    )
    return config
end

#md # !!! note
#md #     `Keywords` are used to specify some arguments (see appropriate source
#md #     files).

# ## [Diagnostics](@id config_diagnostics)
# Here we define the diagnostic configuration specific to this problem.
function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    # These are essentially arguments passed to the
    # [`config_risingbubble`](@ref config-helper) function.  For type
    # consistency we explicitly define the problem floating-precision.
    FT = Float64
    # We need to specify the polynomial order for the DG discretization,
    # effective resolution, simulation end-time, the domain bounds, and the
    # courant-number for the time-integrator. Note how the time-integration
    # components `solver_config` are distinct from the spatial / model
    # components in `driver_config`. `init_on_cpu` is a helper keyword argument
    # that forces problem initialisation on CPU (thereby allowing the use of
    # random seeds, spline interpolants and other special functions at the
    # initialisation step.)
    N = 4
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    xmax = FT(2500)
    ymax = FT(2500)
    zmax = FT(2500)
    t0 = FT(0)
    timeend = FT(1000)
    CFL = FT(20)

    # Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )
    dgn_config = config_diagnostics(driver_config)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    # Invoke solver (calls `solve!` function for time-integrator), pass the driver, solver and diagnostic config
    # information.
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

    # Check that the solution norm is reasonable.
    @test isapprox(result, FT(1); atol = 1.5e-3)
end

# The experiment definition is now complete. Time to run it.

# ## Running the Experiment
# `julia --project /experiments/AtmosLES/risingbubble.jl` will run the
# experiment from the main ClimateMachine.jl directory, with diagnostics output
# at the intervals specified in [`config_diagnostics`](@ref
# config_diagnostics).  You can also prescribe command line arguments (docs
# pending, `Driver.jl`) for simulation update and output specifications.  For
# rapid turnaround, we recommend that you run this experiment on a GPU.

# ## [Output Visualisation](@id output-viz)
#md # See the `ClimateMachine` [command line arguments](@ref ClimateMachine-args)
#md # for generating output.
#md #
#md # Given VTK output,
#md # - [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/)
#md # - [Paraview](https://wci.llnl.gov/simulation/computer-codes/visit/)
#md # are two commonly used programs for `.vtu` files.
#md #
#md # For NetCDF or JLD2 diagnostics you may use Julia's
#md # [`NCDatasets`](https://github.com/Alexander-Barth/NCDatasets.jl) and
#md # [`JLD2`](https://github.com/JuliaIO/JLD2.jl) packages with a suitable
#md # plotting program.

#

main()
