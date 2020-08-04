# # [Density Current](@id EX-DC-docs)
#
# In this example, we demonstrate the usage of the `ClimateMachine`
#  to solve the density current test by Straka 1993.
# We solve a flow in a box configuration, which is
# representative of a large-eddy simulation. Several versions of the problem
# setup may be found in literature, but the general idea is to examine the
# vertical ascent of a thermal _bubble_ (we can interpret these as simple
# representation of convective updrafts).
#
# ## Description of experiment
# The setup described below is such that the simulation reaches completion
# (timeend = 900 s) in approximately 4 minutes of wall-clock time on 1 GPU
#
# 1) Dry Density Current (circular potential temperature perturbation)
# 2) Boundaries
#    - `Impenetrable(FreeSlip())` - no momentum flux, no mass flux through
#      walls.
#    - `Impermeable()` - non-porous walls, i.e. no diffusive fluxes through
#       walls.
# 3) Domain - 25600m (horizontal) x 10000m (horizontal) x 6400m (vertical)
# 4) Resolution - 100m effective resolution
# 5) Total simulation time - 900s
# 6) Mesh Aspect Ratio (Effective resolution) 1:1
# 7) Overrides defaults for
#    - CPU Initialisation
#    - Time integrator
#    - Sources
#    - Smagorinsky Coefficient _C_smag
# 8) Default settings can be found in `src/Driver/<files>.jl`

#md # !!! note
#md #     This experiment setup assumes that you have installed the
#md #     `ClimateMachine` according to the instructions on the landing page.
#md #     We assume the users' familiarity with the conservative form of the
#md #     equations of motion for a compressible fluid
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
#md #
#md #     The following topics are not covered in this example
#md #     - Defining new boundary conditions
#md #     - Defining new turbulence models
#md #     - Building new time-integrators
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
# specific to atmospheric and ocean flow modeling.  While we do not cover the
# ins-and-outs of the contents of each of these we provide brief descriptions
# of the utility of each of the loaded packages.
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

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
# - Required so functions for computation of temperature profiles.
using ClimateMachine.TemperatureProfiles
# - Required so functions for computation of moist thermodynamic quantities and turbulence closures 
# are available.
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
# - Required so we may access our variable arrays by a sensible naming
#   convention rather than by numerical array indices.
using ClimateMachine.VariableTemplates
# - Required so we may access planet parameters
#   ([CLIMAParameters](https://github.com/CliMA/CLIMAParameters.jl)
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
#md # !!! note
#md #     The following variables are assigned in the initial condition
#md #     - `state.ρ` = Scalar quantity for initial density profile
#md #     - `state.ρu`= 3-component vector for initial momentum profile
#md #     - `state.ρe`= Scalar quantity for initial total-energy profile
#md #       humidity
#md #     - `state.tracers.ρχ` = Vector of four tracers (here, for demonstration
#md #       only; we can interpret these as dye injections for visualisation
#md #       purposes)
function init_densitycurrent!(bl, state, aux, (x, y, z), t)
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
    xc::FT = 0
    yc::FT = 0
    zc::FT = 3000
    rx::FT = 4000
    rz::FT = 2000
    r = sqrt(((x - xc)^2) / rx^2 + ((z - zc)^2) / rz^2)

    ## TODO: clean this up, or add convenience function:
    ## This is configured in the reference hydrostatic state
    θ_ref::FT = bl.ref_state.virtual_temperature_profile.T_surface
    Δθ::FT = 0
    θamplitude::FT = -15.0

    ## Compute temperature difference over bubble region
    if r <= 1
        Δθ = 0.5 * θamplitude * (1 + cospi(r))
    end

    ## Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      ## potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             ## exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      ## density
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   ## momentum
    ## State (prognostic) variable assignment
    e_kin = FT(0)                                       ## kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)## potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         ## total energy

    ## Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
end

# ## [Model Configuration](@id config-helper)
# We define a configuration function to assist in prescribing the physical
# model.
function config_densitycurrent(FT, N, resolution, xmax, ymax, zmax)

    ## Choose an Explicit Single-rate Solver LSRK144 from the existing [ODESolvers](@ref
    ## ODESolvers-docs) options Apply the outer constructor to define the
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    ## The model coefficient for the turbulence closure is defined via the
    ## [CLIMAParameters
    ## package](https://CliMA.github.io/CLIMAParameters.jl/dev/) A reference
    ## state for the linearisation step is also defined.
    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    ## The fun part! Here we assemble the `AtmosModel`.
    ##md # !!! note
    ##md #     Docs on model subcomponent options can be found here:
    ##md #     - [`param_set`](https://CliMA.github.io/CLIMAParameters.jl/dev/)
    ##md #     - [`turbulence`](@ref Turbulence-Closures-docs)
    ##md #     - [`source`](@ref atmos-sources)
    ##md #     - [`init_state`](@ref init)

    _C_smag = FT(0.21)
    model = AtmosModel{FT}(
        AtmosLESConfigType,                             # Flow in a box, requires the AtmosLESConfigType
        param_set;                                      # Parameter set corresponding to earth parameters
        turbulence = Vreman(_C_smag),                   # Turbulence closure model
        moisture = DryModel(),                          # Exclude moisture variables
        source = (Gravity(),),                          # Gravity is the only source term here
        tracers = NoTracers(),                          # Tracer model with diffusivity coefficients
        ref_state = ref_state,                          # Reference state
        init_state_prognostic = init_densitycurrent!, # Apply the initial condition
    )

    ## Finally, we pass a `Problem Name` string, the mesh information, and the
    ## model type to  the [`AtmosLESConfiguration`](@ref ClimateMachine.AtmosLESConfiguration) object.
    config = ClimateMachine.AtmosLESConfiguration(
        "DryDensitycurrent",      # Problem title [String]
        N,                        # Polynomial order [Int]
        resolution,               # (Δx, Δy, Δz) effective resolution [m]
        xmax,                     # Domain maximum size [m]
        ymax,                     # Domain maximum size [m]
        zmax,                     # Domain maximum size [m]
        param_set,                # Parameter set.
        init_densitycurrent!,     # Function specifying initial condition
        solver_type = ode_solver, # Time-integrator type
        model = model,            # Model type
        periodicity = (false, false, false),
        boundary = ((1, 1), (1, 1), (1, 1)),   # Set all boundaries to solid walls
    )
    return config
end

#md # !!! note
#md #     `Keywords` are used to specify some arguments (see appropriate source
#md #     files).

function main()
    ## These are essentially arguments passed to the
    ## [`config_densitycurrent`](@ref config-helper) function.  For type
    ## consistency we explicitly define the problem floating-precision.
    FT = Float64
    ## We need to specify the polynomial order for the DG discretization,
    ## effective resolution, simulation end-time, the domain bounds, and the
    ## courant-number for the time-integrator. Note how the time-integration
    ## components `solver_config` are distinct from the spatial / model
    ## components in `driver_config`. `init_on_cpu` is a helper keyword argument
    ## that forces problem initialisation on CPU (thereby allowing the use of
    ## random seeds, spline interpolants and other special functions at the
    ## initialisation step.)
    N = 4
    Δx = FT(100)
    Δy = FT(250)
    Δv = FT(100)
    resolution = (Δx, Δy, Δv)
    xmax = FT(25600)
    ymax = FT(1000)
    zmax = FT(6400)
    t0 = FT(0)
    timeend = FT(100)
    CFL = FT(1.5)

    ## Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_densitycurrent(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )

    ## Invoke solver (calls `solve!` function for time-integrator), pass the driver, solver and diagnostic config
    ## information.
    result =
        ClimateMachine.invoke!(solver_config; check_euclidean_distance = true)

    ## Check that the solution norm is reasonable.
    @test isapprox(result, FT(1); atol = 1.5e-2)
end

# The experiment definition is now complete. Time to run it.
# `julia --project=$CLIMA_HOME tutorials/Atmos/densitycurrent.jl --vtk 1smins`
# to run with VTK output enabled at intervals of 1 simulation minute.
#
# ## References
#
# [1] J. Straka, R. Wilhelmson, L. Wicker, J. Anderson, K. Droegemeier,
# Numerical solution of a nonlinear density current: a benchmark solution and comparisons
# Int. J. Numer. Methods Fluids 17 (1993) 1–22,  https://doi.org/10.1002/fld.1650170103
#
# [2] R. Carpenter, K. Droegemeier, P. Woodward, C. Hane,
# Application of the piecewise parabolic method (PPM) to meteorological modeling,
# Mon. Weather Rev. 118 (1990) 586–612, https://doi.org/10.1175/1520-0493(1990)118<0586:AOTPPM>2.0.CO;2

main()
