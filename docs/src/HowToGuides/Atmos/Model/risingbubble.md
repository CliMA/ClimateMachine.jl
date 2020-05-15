```@meta
EditURL = "<unknown>/experiments/AtmosLES/risingbubble.jl"
```

# [Example - Rising Thermal Bubble](@id EX-RTB-docs)

In this example, we demonstrate the usage of the ClimateMachine [AtmosModel](@ref AtmosModel-docs)
machinery to solve for a the fluid dynamics of a thermal perturbation
in a neutrally stratified background state defined by its uniform
potential temperature. We solve a flow in a [`FlatOrientation`](@ref LESConfig) (Box) configuration - this is representative of a large-eddy simulation. Several versions of the problem setup
may be found in literature, but the general idea is to examine the vertical ascent of a
thermal `bubble` (we can interpret these as simple representation of convective updrafts).
The example is essentially a line-by-line walk-through.

## Description of experiment
1) Dry Rising Bubble (circular potential temperature perturbation)
2) Boundaries
   - `Impenetrable(FreeSlip())` - no momentum flux, no mass flux through walls.
   - `Impermeable()` - non-porous walls, i.e. no diffusive fluxes through walls.
   - Laterally periodic
3) Domain - 2500m (horizontal) x 2500m (horizontal) x 2500m (vertical)
4) Resolution - 50m effective resolution
5) Total simulation time - 1000s
6) Mesh Aspect Ratio (Effective resolution) 1:1
7) Overrides defaults for
   - CPU Initialisation
   - Time integrator
   - Sources
   - Smagorinsky Coefficient
8) Default settings can be found in `src/Driver/<files>.jl`

!!! note
    This experiment setup assumes that you have installed ClimateMachine according to
    the instructions on the landing page. We suggest that you attempt to run the
    `runtests.jl` file in the main `ClimateMachine` directory to ensure that packages and
    `git` histories are consistently installed / managed. We assume the users' familiarity
    with the conservative form of the equations of motion for a compressible fluid (see the [`AtmosModel`](@ref AtmosModel-docs) page).

    The following topics are covered in this example
    - Package requirements
    - Defining a `model` subtype for the set of conservation equations
    - Defining the initial conditions
    - Applying boundary conditions
    - Applying source terms
    - Choosing a turbulence model
    - Adding tracers to the model
    - Choosing a time-integrator
    - Choosing diagnostics (output) configurations

    The following topics are not covered in this example
    - Defining new boundary conditions
    - Defining new turbulence models
    - Building new time-integrators
    - Adding diagnostic variables (beyond a standard pre-defined list of variables)

## Boilerplate (Using Modules)

#### [Skip Section](@ref init)

Before setting up our experiment, we recognize that we need to import
some pre-defined functions from other packages. Julia allows us to use existing modules (variable workspaces), or write our own to do so.
Complete documentation for the Julia module system can be found [here](https://docs.julialang.org/en/v1/manual/modules/#).

In ClimateMachine we use `StaticArrays` for our variable arrays.

```julia
using StaticArrays
```

We also use the `Test` package to help with unit tests and continuous integration systems to design sensible tests
for our experiment to ensure new / modified blocks of code don't damage the fidelity of the physics. The test defined within this experiment is not a unit test for a specific subcomponent, but ensures time-integration of the defined problem conditions within a reasonable tolerance. Immediately useful macros and functions from this include `@test` and `@testset` which will allow us to define the testing parameter sets.

```julia
using Test
```

We then need to use the ClimateMachine module! This imports all functions specific to atmospheric and ocean flow modelling.
While we do not cover the ins-and-outs of the contents of each of these we provide brief descriptions of the utility of each of the loaded packages.

```julia
using ClimateMachine
using ClimateMachine.Atmos
```

- Required so that we inherit the appropriate model types for the large-eddy simulation (LES) and global-circulation-model (GCM) configurations.

```julia
using ClimateMachine.ConfigTypes
```

- Required so that we may define diagnostics configurations, e.g. choice of file-writer, choice of output variable sets, output-frequency and directory,

```julia
using ClimateMachine.Diagnostics
```

- Required so that we may define (or utilise existing functions) functions that are `called-back` or executed at frequencies of either timesteps, simulation-time, or wall-clock time.

```julia
using ClimateMachine.GenericCallbacks
```

- Required so we load the appropriate functions for the time-integration component. Contains ODESolver methods.

```julia
using ClimateMachine.ODESolvers
```

- Required for utility of spatial filtering functions (e.g. positivity preservation)

```julia
using ClimateMachine.Mesh.Filters
```

- Required so functions for computation of moist thermodynamic quantities is enabled.

```julia
using ClimateMachine.MoistThermodynamics
```

- Required so we may access our variable arrays by a sensible naming convention rather than by numerical array indices.

```julia
using ClimateMachine.VariableTemplates
```

- Required so we may access planet parameters ([CLIMAParameters](https://github.com/CliMA/CLIMAParameters.jl) specific to this problem include the gas constant, specific heats, mean-sea-level pressure, gravity and the Smagorinsky coefficient)

```julia
using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
```

## [Initial Conditions](@id init)
This example of a rising thermal bubble can be classified as an initial value problem. We must (at the very least) assign values for the initial variables in a sensible manner. This example demonstrates the use of functions defined in the [`MoistThermodynamics`](@ref MoistThermodynamics-docs) package to generate the appropriate initial state for our problem.

!!! note
    The following variables are assigned in the initial condition
    - `state.ρ` = Scalar quantity for initial density profile
    - `state.ρu`= 3-component vector for initial momentum profile
    - `state.ρe`= Scalar quantity for initial total-energy profile
    - `state.moisture.ρq_tot` = Scalar quantity for the total specific humidity
    - `state.tracers.ρχ` = Vector of four tracers (here, for demonstration only; we can interpret these as dye injections for visualisation purposes)

```julia
"""
    init_risingbubble!(bl, state, aux, (x,y,z), t)

Arguments

- bl = `BalanceLaw` (for this example, this is the `AtmosModel <: BalanceLaw`)
- state = Array of prognostic (or state) variables. These are conserved quantities. Indices accessed using the `VariableTemplates` system. (elements are floating-point values)
- aux = Array of auxiliary (or helper) variables. These may include auxiliary thermodynamic variables / diagnostics / variables for debug assistance (and other variables for which we don't need time-integration) (elements are floating-point values)
- (x,y,z) = `Tuple` specifying the `local Cartesian coordinates`. (elements are floating point values)
- t = Simulation time in seconds, a `Real` value.

Returns
- Updated values for `state.<variable_name>`
"""
function init_risingbubble!(bl, state, aux, (x, y, z), t)

    FT = eltype(state) # Problem float-type

    ## Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    ## Define bubble center and background potential temperature

    xc::FT = 1250
    yc::FT = 1250
    zc::FT = 1000
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    rc::FT = 500
    θ_ref::FT = 300
    Δθ::FT = 0

    ## Compute temperature difference over bubble region

    if r <= rc
        Δθ = FT(5) * cospi(r / rc / 2)
    end

    ## Compute perturbed thermodynamic state:

    θ = θ_ref + Δθ                                      # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      # density
    q_tot = FT(0)                                       # total specific humidity
    ts = LiquidIcePotTempSHumEquil(bl.param_set, θ, ρ, q_tot) # thermodynamic state
    q_pt = PhasePartition(ts)                           # phase partition
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    #State (prognostic) variable assignment
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
    state.moisture.ρq_tot = ρ * q_pt.tot
    state.tracers.ρχ = ρχ
end
```

## [Model Configuration](@id config-helper)
We define a configuration function to assist in prescribing the physical model. The purpose of this is to populate the [`ClimateMachine.AtmosLESConfiguration`](@ref LESConfig) with arguments appropriate to the problem being considered.

```julia
"""
    config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

Arguments
- FT = Floating-point type. Currently `Float64` or `Float32`
- N  = DG Polynomial order
- resolution = 3-component Tuple (Δx, Δy, Δz) with effective resolutions for Cartesian directions
- xmax, ymax, zmax = Domain maximum extents. Assumes (0,0,0) to be the domain minimum extents unless otherwise specified.

Returns
- `config` = Object using the constructor for the `AtmosLESConfiguration`
"""
function config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

    ## Choose an Explicit Multi-rate Solver from the existing [ODESolvers](@ref ODESolvers-docs) options
    ## Apply the outer constructor to define the `ode_solver`. Here
    ## `AtmosAcousticGravityLinearModel` splits the acoustic-gravity wave components
    ## from the advection-diffusion dynamics. The 1D-IMEX method is less appropriate for the problem given the current mesh aspect ratio (1:1)

    ode_solver = ClimateMachine.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )

    ## Since we want four tracers, we specify this and include
    ## the appropriate diffusivity scaling coefficients (normally these
    ## would be physically informed but for this demonstration we use
    ## integers corresponding to the tracer index identifier)

    ntracers = 4
    δ_χ = SVector{ntracers, FT}(1, 2, 3, 4)

    ## The model coefficient for the turbulence closure is defined via the [CLIMAParameters
    ## package](https://github.com/CliMA/CLIMAParameters.jl)
    ## A reference state for the linearisation step is also defined.

    ref_state =
        HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))

    ## The fun part! Here we assemble the `AtmosModel`.

    #md # !!! note
    #md #     Docs on model subcomponent options can be found here:
    #md #     - [`param_set`](https://github.com/CliMA/CLIMAParameters.jl)
    #md #     - [`turbulence`](@ref Turbulence-Closures-docs)
    #md #     - [`hyperdiffusion`](@ref Hyperdiffusion-docs)
    #md #     - [`source`](@ref atmos-sources)
    #md #     - [`tracers`](@ref Tracers-docs)
    #md #     - [`init_state`](@ref init)

    model = AtmosModel{FT}(
        AtmosLESConfigType,                          # Flow in a box, requires the AtmosLESConfigType
        param_set;                                   # Parameter set corresponding to earth parameters
        turbulence = SmagorinskyLilly(C_smag),       # Turbulence closure model
        hyperdiffusion = StandardHyperDiffusion(60), # Hyperdiffusion (4th order) model
        source = (Gravity(),),                       # Gravity is the only source term here
        tracers = NTracers{ntracers, FT}(δ_χ),       # Tracer model with diffusivity coefficients
        ref_state = ref_state,                       # Reference state
        init_state = init_risingbubble!,             # Apply the initial condition
    )

    ## Finally,  we pass a `Problem Name` string, the mesh information, and the model type to  the [`AtmosLESConfiguration`] object.

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
```

!!! note
    `Keywords` are used to specify some arguments (see appropriate source files).

## [Diagnostics](@id config_diagnostics)
Here we define the diagnostic configuration specific to this problem.

```julia
function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    ClimateMachine.init()

    ## These are essentially arguments passed to the [`config_risingbubble`](@ref config-helper) function.
    ## For type consistency we explicitly define the problem floating-precision.

    FT = Float64

    ## We need to specify the polynomial order for the DG discretization, effective resolution, simulation end-time,
    ## the domain bounds, and the courant-number for the time-integrator. Note how the time-integration components
    ## `solver_config` are distinct from the spatial / model components in `driver_config`. `init_on_cpu` is a helper
    ## keyword argument that forces problem initialisation on CPU (thereby allowing the use of random seeds, spline interpolants and other special functions at the initialisation step.)

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

    ## User defined filter (TMAR positivity preserving filter)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    ## Invoke solver (calls `solve!` function for time-integrator), pass the driver, solver and diagnostic config
    ## information.

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

    ## Check that the solution norm is reasonable.

    @test isapprox(result, FT(1); atol = 1.5e-3)
end
```

The experiment definition is now complete. Time to run it.

## Running the Experiment
`julia --project /experiments/AtmosLES/risingbubble.jl` will run the experiment from the main ClimateMachine directory,
with diagnostics output at the intervals specified in [`config_diagnostics`](@ref config_diagnostics). You
can also prescribe command line arguments (docs pending, `Driver.jl`) for simulation update and output specifications.
For rapid turnaround, we recommend that you run this experiment on a GPU.

## [Output Visualisation](@id output-viz)
See command line output arguments listed [here](https://github.com/CliMA/ClimateMachine.jl/wiki/ClimateMachine-command-line-arguments).
For VTK output,
- [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/)
- [Paraview](https://wci.llnl.gov/simulation/computer-codes/visit/)
are two commonly used programs for `.vtu` files.
For NetCDF or JLD2 diagnostics you may use Julia's [`NCDatasets`](https://github.com/Alexander-Barth/NCDatasets.jl) and [`JLD2`](https://github.com/JuliaIO/JLD2.jl) packages with
a suitable plotting program.

```julia
main();
```
