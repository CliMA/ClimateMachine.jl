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
#    Top and Bottom boundaries:
#    - `Impenetrable(FreeSlip())` - Top and bottom: no momentum flux, no mass flux through
#      walls.
#    - `Impermeable()` - non-porous walls, i.e. no diffusive fluxes through
#       walls.
#    Lateral boundaries
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

## !!! note
##     This experiment setup assumes that you have installed the
##     `ClimateMachine` according to the instructions on the landing page.
##     We assume the users' familiarity with the conservative form of the
##     equations of motion for a compressible fluid (see the
##     [`AtmosModel`](@ref AtmosModel-docs) page).
##
##     The following topics are covered in this example
##     - Package requirements
##     - Defining a `model` subtype for the set of conservation equations
##     - Defining the initial conditions
##     - Applying source terms
##     - Choosing a turbulence model
##     - Adding tracers to the model
##     - Choosing a time-integrator
##     - Choosing diagnostics (output) configurations
##
##     The following topics are not covered in this example
##     - Defining new boundary conditions
##     - Defining new turbulence models
##     - Building new time-integrators
##     - Adding diagnostic variables (beyond a standard pre-defined list of
##       variables)
#
# ## Code loading
#
# Before setting up our experiment, we recognize that we need to import some
# pre-defined functions from other packages. Julia allows us to use existing
# modules (variable workspaces), or write our own to do so.  Complete
# documentation for the Julia module system can be found
# [here](https://docs.julialang.org/en/v1/manual/modules/#).

# We need to use the `ClimateMachine` module! This imports all functions
# specific to atmospheric and ocean flow modeling.

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates

# In ClimateMachine we use `StaticArrays` for our variable arrays.
# We also use the `Test` package to help with unit tests and continuous
# integration systems to design sensible tests for our experiment to ensure new
# / modified blocks of code don't damage the fidelity of the physics. The test
# defined within this experiment is not a unit test for a specific
# subcomponent, but ensures time-integration of the defined problem conditions
# within a reasonable tolerance. Immediately useful macros and functions from
# this include `@test` and `@testset` which will allow us to define the testing
# parameter sets.
using StaticArrays
using Test
using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

# ## [Initial Conditions](@id init)
# This example of a rising thermal bubble can be classified as an initial value
# problem. We must (at the very least) assign values for the initial variables
# in a sensible manner. This example demonstrates the use of functions defined
# in the [`Thermodynamics`](@ref Thermodynamics-docs) package to
# generate the appropriate initial state for our problem.

## !!! note
##     The following variables are assigned in the initial condition
##     - `state.ρ` = Scalar quantity for initial density profile
##     - `state.ρu`= 3-component vector for initial momentum profile
##     - `state.ρe`= Scalar quantity for initial total-energy profile
##       humidity
##     - `state.tracers.ρχ` = Vector of four tracers (here, for demonstration
##       only; we can interpret these as dye injections for visualization
##       purposes)
function init_risingbubble!(bl, state, aux, (x, y, z), t)
    ## Problem float-type
    FT = eltype(state)

    ## Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    ## Define bubble center and background potential temperature
    xc::FT = 5000
    yc::FT = 1000
    zc::FT = 2000
    r = sqrt((x - xc)^2 + (z - zc)^2)
    rc::FT = 2000
    θamplitude::FT = 2

    ## TODO: clean this up, or add convenience function:
    ## This is configured in the reference hydrostatic state
    θ_ref::FT = bl.ref_state.virtual_temperature_profile.T_surface

    ## Add the thermal perturbation:
    Δθ::FT = 0
    if r <= rc
        Δθ = θamplitude * (1.0 - r / rc)
    end

    ## Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      # density
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    ## State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)# potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    ρχ = FT(0)                                          # tracer

    ## We inject tracers at the initial condition at some specified z coordinates
    if 500 < z <= 550
        ρχ += FT(0.05)
    end

    ## We want 4 tracers
    ntracers = 4

    ## Define 4 tracers, (arbitrary scaling for this demo problem)
    ρχ = SVector{ntracers, FT}(ρχ, ρχ / 2, ρχ / 3, ρχ / 4)

    ## Assign State Variables
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

    ## Choose an Explicit Single-rate Solver from the existing [ODESolvers](@ref
    ## ODESolvers-docs) options. Apply the outer constructor to define the
    ## `ode_solver`.
    ## The 1D-IMEX method is less appropriate for the problem given the current
    ## mesh aspect ratio (1:1).
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )
    ## If the user prefers a multi-rate explicit time integrator,
    ## the ode_solver above can be replaced with
    ##
    ## `ode_solver = ClimateMachine.MultirateSolverType(
    ##    fast_model = AtmosAcousticGravityLinearModel,
    ##    slow_method = LSRK144NiegemannDiehlBusch,
    ##    fast_method = LSRK144NiegemannDiehlBusch,
    ##    timestep_ratio = 10,
    ## )`
    ## See [ODESolvers](@ref ODESolvers-docs) for all of the available solvers.


    ## Since we want four tracers, we specify this and include the appropriate
    ## diffusivity scaling coefficients (normally these would be physically
    ## informed but for this demonstration we use integers corresponding to the
    ## tracer index identifier)
    ntracers = 4
    δ_χ = SVector{ntracers, FT}(1, 2, 3, 4)
    ## To assemble `AtmosModel` with no tracers, set `tracers = NoTracers()`.

    ## The model coefficient for the turbulence closure is defined via the
    ## [CLIMAParameters
    ## package](https://CliMA.github.io/CLIMAParameters.jl/latest/) A reference
    ## state for the linearisation step is also defined.
    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    ## The fun part! Here we assemble the `AtmosModel`.
    ## !!! note
    ##     Docs on model subcomponent options can be found here:
    ##     - [`param_set`](https://CliMA.github.io/CLIMAParameters.jl/latest/)
    ##     - [`turbulence`](@ref Turbulence-Closures-docs)
    ##     - [`hyperdiffusion`](@ref Hyperdiffusion-docs)
    ##     - [`source`](@ref atmos-sources)
    ##     - [`tracers`](@ref Tracers-docs)
    ##     - [`init_state`](@ref init)

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

    ## Finally, we pass a `Problem Name` string, the mesh information, and the
    ## model type to  the [`AtmosLESConfiguration`] object.
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

## !!! note
##     `Keywords` are used to specify some arguments (see appropriate source
##     files).

# ## [Diagnostics](@id config_diagnostics)
# Here we define the diagnostic configuration specific to this problem.
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
    ## These are essentially arguments passed to the
    ## [`config_risingbubble`](@ref config-helper) function.  For type
    ## consistency we explicitly define the problem floating-precision.
    FT = Float64
    ## We need to specify the polynomial order for the DG discretization,
    ## effective resolution, simulation end-time, the domain bounds, and the
    ## courant-number for the time-integrator. Note how the time-integration
    ## components `solver_config` are distinct from the spatial / model
    ## components in `driver_config`. `init_on_cpu` is a helper keyword argument
    ## that forces problem initialization on CPU (thereby allowing the use of
    ## random seeds, spline interpolants and other special functions at the
    ## initialization step.)
    N = 4
    Δh = FT(125)
    Δv = FT(125)
    resolution = (Δh, Δh, Δv)
    xmax = FT(10000)
    ymax = FT(500)
    zmax = FT(10000)
    t0 = FT(0)
    timeend = FT(1000)

    ## Use up to 20 if ode_solver is the multi-rate LRRK144.
    ## CFL = FT(15)

    ## Use up to 1.7 if ode_solver is the single rate LSRK144.
    CFL = FT(1.7)

    ## Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )
    dgn_config = config_diagnostics(driver_config)

    ## Invoke solver (calls `solve!` function for time-integrator), pass the driver,
    ## solver and diagnostic config information.
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (),
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
## See the `ClimateMachine` [command line arguments](@ref ClimateMachine-args)
## for generating output.
##
## Given VTK output,
## - [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/)
## - [Paraview](https://wci.llnl.gov/simulation/computer-codes/visit/)
## are two commonly used programs for `.vtu` files.
##
## For NetCDF or JLD2 diagnostics you may use any of the following tools:
## Julia's
## [`NCDatasets`](https://github.com/Alexander-Barth/NCDatasets.jl) and
## [`JLD2`](https://github.com/JuliaIO/JLD2.jl) packages with a suitable
##
## or the known and quick NCDF visualization tool:
## [`ncview`](https://meteora.ucsd.edu/~pierce/ncview_home_page.html)
## plotting program.

#

main()
