# # Geostrophic adjustment in the hydrostatic Boussinesq equations
#
# This example simulates a one-dimensional geostrophic adjustement problem
# using the `ClimateMachine.Ocean` subcomponent to solve the hydrostatic
# Boussinesq equations.
#
# First we `ClimateMachine.init()`.

using ClimateMachine

ClimateMachine.init()

# # Domain setup
#
# We formulate our problem in a Cartesian domain 100 km in ``x, y`` and 400 m
# deep, and discretized on a grid 100 fourth-order elements in ``x``, and 1
# fourth-order element in ``y, z``,

## Domain
Lx = 1e6   # m
Ly = 1e6   # m
Lz = 400.0 # m

## Numerical parameters
Np = 4           # Polynomial order
Ne = (100, 1, 1) # Number of elements in (x, y, z)
nothing # hide

# # Physical parameters
#
# We use a Coriolis parameter appropriate for mid-latitudes,

f = 1e-4 # s⁻¹, Coriolis parameter

# and Earth's gravitational acceleration,

using CLIMAParameters: AbstractEarthParameterSet, Planet

gravitational_acceleration = Planet.grav

struct EarthParameters <: AbstractEarthParameterSet end

g = gravitational_acceleration(EarthParameters()) # m s⁻²

# # An unbalanced initial state
#
# We use a Gaussian, partially-balanced initial condition with parameters

U = 0.1            # geostrophic velocity (m s⁻¹)
L = Lx / 40        # Gaussian width (m)
a = f * U * L / g  # amplitude of the geostrophic surface displacement (m)
x₀ = Lx / 4        # Gaussian origin (m, recall that x ∈ [0, Lx])

# and functional form

Ψ(x, L) = exp(-x^2 / 2L^2) # a Gaussian

## Geostrophic ``y``-velocity
vᵍ(x, y, z) = - U * (x - x₀) / L * Ψ(x - x₀, L)

## Geostrophic surface displacement
ηᵍ(x, y, z) = a * Ψ(x - x₀, L)

# We double the initial surface displacement so that the surface is half-balanced,
# half unbalanced,

ηⁱ(x, y, z) = 2 * ηᵍ(x, y, z)

# In summary,

using ClimateMachine.Ocean.OceanProblems: InitialConditions

initial_conditions = InitialConditions(v=vᵍ, η=ηⁱ)

@info """ The parameters for the Geostrophic adjustment problem are...

    Coriolis parameter:                            $f s⁻¹
    Gravitational acceleration:                    $g m s⁻²
    Geostrophic velocity:                          $U m s⁻¹
    Width of the initial geostrophic perturbation: $L m
    Amplitude of the initial surface perturbation: $a m
    Rossby number (U / f L):                       $(U / (f * L))

"""

# # Driver configuration
#
# Next, we configure the `HydrostaticBoussinesqModel` and build the `DriverConfiguration`,

using ClimateMachine.Ocean.HydrostaticBoussinesq: HydrostaticBoussinesqModel

using ClimateMachine.Ocean.OceanProblems: InitialValueProblem

problem = InitialValueProblem(dimensions = (Lx, Ly, Lz), initial_conditions = initial_conditions)

equations = HydrostaticBoussinesqModel{Float64}(
    EarthParameters(),
    problem,           # The InitialValueProblem
    κʰ = 1e-4,         # Horizontal diffusivity (m² s⁻²)
    fₒ = f             # Coriolis parameter
)

driver_configuration = ClimateMachine.OceanBoxGCMConfiguration(
    "geostrophic_adjustment",             # The name of the experiment
    Np,                                   # The polynomial order
    Ne,                                   # The number of elements
    EarthParameters(),                    # The CLIMAParameters.AbstractParameterSet to use
    equations;                            # The equations to solve, represented by a `BalanceLaw`
    periodicity = (false, false, false),  # Topology of the domain
    boundary = ((0, 0), (0, 0), (1, 2))   # (?)
)  
nothing # hide

# # Filters
#
# The `HydrostaticBoussinesqModel` currently assumes that both `vert_filter`
# and `exp_filter` are supplied to `modeldata` in `SolverConfiguration`.
# We define these required objects,

using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids: polynomialorder

grid = driver_configuration.grid

filters = (
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1),
    exp_filter = ExponentialFilter(grid, 1, 8)
)

# # Configuring the solver
#
# We configure our solver with the `LSRK144NiegemannDiehlBusch` time-stepping method.

using ClimateMachine.ODESolvers

minute = 60.0
hours = 60minute

solver_configuration = ClimateMachine.SolverConfiguration(
    0.0,                    # start time (s)
    4hours,                 # stop time (s)
    driver_configuration,
    init_on_cpu = true,
    ode_dt = 1minute,       # time step size (s)
    ode_solver_type = ClimateMachine.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch),
    modeldata = filters
)

# and use a simple callback to log the progress of our simulation,

using ClimateMachine.GenericCallbacks: EveryXSimulationSteps

print_every = 10 # iterations
solver = solver_configuration.solver

tiny_progress_printer = EveryXSimulationSteps(print_every) do
    @info "Steps: $(solver.steps), time: $(solver.t), time step: $(solver.dt)"
end

# # Animating the solution
#
# To animate the `ClimateMachine.Ocean` solution, we'll create a callback
# that draws a plot and stores it in an array. When the simulation is finished,
# we'll string together the plotted frames into an animation.

using Plots

using ClimateMachine.Ocean.CartesianDomains: CartesianDomain, CartesianField, join

## CartesianDomain and CartesianField objects to help with plotting
domain = CartesianDomain(solver_configuration.dg.grid, Ne)

u = CartesianField(solver_configuration, 1, domain)
v = CartesianField(solver_configuration, 2, domain)
η = CartesianField(solver_configuration, 3, domain)

## Container to hold the plotted frames
movie_plots = []

plot_every = 10 # iterations

plot_maker = EveryXSimulationSteps(plot_every) do
    joined_u = join(u.elements)
    joined_v = join(v.elements)
    joined_η = join(η.elements)
    
    umax = 0.5 * max(maximum(abs, u), maximum(abs, v))
    ulim = (-umax, umax)
    
    u_plot = plot(joined_u.x, [joined_u.data[:, 1, 1] joined_v.data[:, 1, 1]],
                  xlim=domain.x, ylim=(-U, U), label=["u" "v"],
                  xlabel="x (m)", ylabel="u and v (m s⁻¹)")

    η_plot = plot(joined_η.x, joined_η.data[:, 1, 1], xlim=domain.x, ylim=(0, a),
                  label=nothing, xlabel="x (m)", ylabel="η (m)")

    push!(movie_plots, (u=u_plot, η=η_plot))
    
    return nothing
end

# # Running the simulation and animating the results
#
# Finally, we run the simulation,

result = ClimateMachine.invoke!(solver_configuration;
                                user_callbacks = [tiny_progress_printer, plot_maker])

# and animate the results,

animation = @animate for p in movie_plots
    frame = plot(p.u, p.η, layout=(2, 1), size=(800, 600), title=["u and v" "η"])
end

gif(animation, "geostrophic_adjustment.gif", fps = 8) # hide
