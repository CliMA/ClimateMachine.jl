using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using CLIMA
using CLIMA.Atmos
# - Required so that we inherit the appropriate model types for the large-eddy simulation (LES) and global-circulation-model (GCM) configurations.
using CLIMA.ConfigTypes
# - Required so that we may define diagnostics configurations, e.g. choice of file-writer, choice of output variable sets, output-frequency and directory,
using CLIMA.Diagnostics
# - Required so that we may define (or utilise existing functions) functions that are `called-back` or executed at frequencies of either timesteps, simulation-time, or wall-clock time.
using CLIMA.GenericCallbacks
# - Required so we load the appropriate functions for the time-integration component. Contains ODESolver methods.
using CLIMA.ODESolvers
# - Required for utility of spatial filtering functions (e.g. positivity preservation)
using CLIMA.Mesh.Filters
# - Required so functions for computation of moist thermodynamic quantities is enabled.
using CLIMA.MoistThermodynamics
# - Required so we may access our variable arrays by a sensible naming convention rather than by numerical array indices.
using CLIMA.VariableTemplates
# - Required so we may access planet parameters ([CLIMAParameters](https://climate-machine.github.io/CLIMAParameters.jl/latest/) specific to this problem include the gas constant, specific heats, mean-sea-level pressure, gravity and the Smagorinsky coefficient)
using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
function init_greenvortex!(bl, state, aux, (x, y, z), t)
    # Problem float-type
    FT = eltype(state)

    # Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v



    # Compute perturbed thermodynamic state:
    ρ = FT(1.178)      # density
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    #State (prognostic) variable assignment
    e_pot = FT(0)# potential energy
    Pinf = 101325
    Uzero = FT(100)
    p = Pinf + (ρ * Uzero / 16) * (2 + cos(z)) * (cos(x) + cos(y)) 
    u = Uzero * sin(x) * cos(y) * cos(z)
    v = - Uzero * cos(x) * sin(y) * cos(z)
    e_kin = 0.5 * (u^2+v^2)
    T = p / (ρ * R_gas)
    e_int = internal_energy(bl.param_set, T, PhasePartition(FT(0)))
    ρe_tot = ρ * (e_kin + e_pot + e_int)
    # Assign State Variables
    state.ρ = ρ
    state.ρu = SVector(FT(ρ * u),FT(ρ * v),FT(0))
    state.ρe = ρe_tot
end

# ## [Model Configuration](@id config-helper)
# We define a configuration function to assist in prescribing the physical model. The purpose of this is to populate the [`CLIMA.AtmosLESConfiguration`](@ref LESConfig) with arguments appropriate to the problem being considered.
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
function config_greenvortex(FT, N, resolution, xmax, ymax, zmax,xmin,ymin,zmin)

    ode_solver = CLIMA.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    # Since we want four tracers, we specify this and include
    # the appropriate diffusivity scaling coefficients (normally these
    # would be physically informed but for this demonstration we use
    # integers corresponding to the tracer index identifier)

    # The model coefficient for the turbulence closure is defined via the [CLIMAParameters
    # package](https://climate-machine.github.io/CLIMAParameters.jl/latest/)
    # A reference state for the linearisation step is also defined.

    # The fun part! Here we assemble the `AtmosModel`.
    #md # !!! note
    #md #     Docs on model subcomponent options can be found here:
    #md #     - [`param_set`](https://climate-machine.github.io/CLIMAParameters.jl/latest/)
    #md #     - [`turbulence`](@ref Turbulence-Closures-docs)
    #md #     - [`hyperdiffusion`](@ref Hyperdiffusion-docs)
    #md #     - [`source`](@ref atmos-sources)
    #md #     - [`tracers`](@ref Tracers-docs)
    #md #     - [`init_state`](@ref init)

    _C_smag = FT(C_smag(param_set))
    model = AtmosModel{FT}(
        AtmosLESConfigType,                          # Flow in a box, requires the AtmosLESConfigType
        param_set;                                   # Parameter set corresponding to earth parameters
        turbulence = Vreman(_C_smag),       # Turbulence closure model
	    moisture = DryModel(),
        source = (),                       
        init_state_conservative = init_greenvortex!,             # Apply the initial condition
    )

    # Finally,  we pass a `Problem Name` string, the mesh information, and the model type to  the [`AtmosLESConfiguration`] object.
    config = CLIMA.AtmosLESConfiguration(
        "GreenVortex",       # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        xmax,                    # Domain maximum size [m]
        ymax,                    # Domain maximum size [m]
        zmax,                    # Domain maximum size [m]
        param_set,               # Parameter set.
        init_greenvortex!,      # Function specifying initial condition
        boundary = ((0,0),(0,0),(0,0)),
	    periodicity = (true,true,true),
	    xmin = xmin,
	    ymin = ymin,
	    zmin = zmin,
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
    )
    return config
end

#md # !!! note
#md #     `Keywords` are used to specify some arguments (see appropriate source files).

# ## [Diagnostics](@id config_diagnostics)
# Here we define the diagnostic configuration specific to this problem.
function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.DiagnosticsConfiguration([dgngrp])
end

function main()
    CLIMA.init()

    # These are essentially arguments passed to the [`config_risingbubble`](@ref config-helper) function.
    # For type consistency we explicitly define the problem floating-precision.
    FT = Float64
    # We need to specify the polynomial order for the DG discretization, effective resolution, simulation end-time,
    # the domain bounds, and the courant-number for the time-integrator. Note how the time-integration components
    # `solver_config` are distinct from the spatial / model components in `driver_config`. `init_on_cpu` is a helper
    # keyword argument that forces problem initialisation on CPU (thereby allowing the use of random seeds, spline interpolants and other special functions at the initialisation step.)
    N = 4
    Ncellsx = 64
    Ncellsy = 64
    Ncellsz = 64
    Δx = FT(2*pi/Ncellsx) #FT(0.09817477042)
    Δy = Δx
    Δz = Δx
    resolution = (Δx, Δy, Δz)
    xmax = FT(pi)
    ymax = FT(pi)
    zmax = FT(pi)
    xmin = FT(-pi)
    ymin = FT(-pi)
    zmin = FT(-pi)
    t0 = FT(0)
    timeend = FT(100)
    CFL = FT(0.8)

    # Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_greenvortex(FT, N, resolution, xmax, ymax, zmax,xmin,ymin,zmin)
    solver_config = CLIMA.SolverConfiguration(
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
    result = CLIMA.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        #user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

end

# The experiment definition is now complete. Time to run it.

# ## Running the Experiment
# `julia --project /experiments/AtmosLES/taylor-green.jl` will run the experiment from the main CLIMA directory,
# with diagnostics output at the intervals specified in [`config_diagnostics`](@ref config_diagnostics). You
# can also prescribe command line arguments (docs pending, `Driver.jl`) for simulation update and output specifications.
# For rapid turnaround, we recommend that you run this experiment on a GPU.

# ## [Output Visualisation](@id output-viz)
#md # See command line output arguments listed [here](https://github.com/climate-machine/CLIMA/wiki/CLIMA-command-line-arguments).
#md # For VTK output,
#md # - [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/)
#md # - [Paraview](https://wci.llnl.gov/simulation/computer-codes/visit/)
#md # are two commonly used programs for `.vtu` files.
#md # For NetCDF or JLD2 diagnostics you may use Julia's [`NCDatasets`](https://github.com/Alexander-Barth/NCDatasets.jl) and [`JLD2`](https://github.com/JuliaIO/JLD2.jl) packages with
#md # a suitable plotting program.
#
#

main()
