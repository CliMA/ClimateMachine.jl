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
# deep, and discretized on a grid with 100 fourth-order elements in ``x``, and 1
# fourth-order element in ``y, z``,

using ClimateMachine.Ocean.Domains

domain = RectangularDomain(
    Ne = (25, 1, 1),
    Np = 4,
    x = (0, 1e6),
    y = (0, 1e6),
    z = (-400, 0),
    periodicity = (false, true, false),
)

# # Physical parameters
#
# We use a Coriolis parameter appropriate for mid-latitudes,

f = 1e-4 # s⁻¹, Coriolis parameter
nothing # hide

# and Earth's gravitational acceleration,

using CLIMAParameters: AbstractEarthParameterSet, Planet
struct EarthParameters <: AbstractEarthParameterSet end

g = Planet.grav(EarthParameters()) # m s⁻²

# # An unbalanced initial state
#
# We use a Gaussian, partially-balanced initial condition with parameters

U = 0.1              # geostrophic velocity (m s⁻¹)
L = domain.L.x / 40  # Gaussian width (m)
a = f * U * L / g    # amplitude of the geostrophic surface displacement (m)
x₀ = domain.L.x / 4  # Gaussian origin (m, recall that x ∈ [0, Lx])

# and functional form

Gaussian(x, L) = exp(-x^2 / (2 * L^2))

## Geostrophic ``y``-velocity: f V = g ∂_x η
vᵍ(x, y, z) = -U * (x - x₀) / L * Gaussian(x - x₀, L)

## Geostrophic surface displacement
ηᵍ(x, y, z) = a * Gaussian(x - x₀, L)

# We double the initial surface displacement so that the surface is half-balanced,
# half unbalanced,

ηⁱ(x, y, z) = 2 * ηᵍ(x, y, z)

# In summary,

using ClimateMachine.Ocean.OceanProblems: InitialConditions

initial_conditions = InitialConditions(v = vᵍ, η = ηⁱ)

@info """ Parameters for the Geostrophic adjustment problem are...

    Coriolis parameter:                            $f s⁻¹
    Gravitational acceleration:                    $g m s⁻²
    Geostrophic velocity:                          $U m s⁻¹
    Width of the initial geostrophic perturbation: $L m
    Amplitude of the initial surface perturbation: $a m
    Rossby number (U / f L):                       $(U / (f * L))

"""

# # Boundary conditions, Driver configuration, and Solver configuration
#
# Next, we configure the `HydrostaticBoussinesqModel` and build the `DriverConfiguration`.
# We configure our model in a domain which is bounded in the ``x`` direction.
# Both the boundary conditions in ``x`` and in ``z`` require boundary conditions,
# which we define:

using ClimateMachine.Ocean:
    Impenetrable, Penetrable, FreeSlip, Insulating, OceanBC

solid_surface_boundary_conditions = OceanBC(
    Impenetrable(FreeSlip()), # Velocity boundary conditions
    Insulating(),             # Temperature boundary conditions
)

free_surface_boundary_conditions = OceanBC(
    Penetrable(FreeSlip()),   # Velocity boundary conditions
    Insulating(),             # Temperature boundary conditions
)

boundary_conditions =
    (solid_surface_boundary_conditions, free_surface_boundary_conditions)

# We refer to these boundary conditions by their indices in the `boundary_tags` tuple
# when specifying the boundary conditions for the `state`; in other words, "1" corresponds to
# `solid_surface_boundary_conditions`, while `2` corresponds to `free_surface_boundary_conditions`,

boundary_tags = (
    (1, 1), # (west, east) boundary conditions
    (0, 0), # (south, north) boundary conditions
    (1, 2), # (bottom, top) boundary conditions
)

# We're now ready to build the model.

using ClimateMachine.Ocean

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = 2.0,
    initial_conditions = initial_conditions,
    parameters = EarthParameters(),
    turbulence_closure = (νʰ = 0, κʰ = 0, νᶻ = 0, κᶻ = 0),
    coriolis = (f₀ = f, β = 0),
    boundary_tags = boundary_tags,
    boundary_conditions = boundary_conditions,
);
nothing

# !!! info "Horizontallly-periodic boundary conditions"
#     To set horizontally-periodic boundary conditions with
#     `(solid_surface_boundary_conditions, free_surface_boundary_conditions)`
#     in the vertical direction use `periodicity = (true, true, false)` in
#     the `domain` constructor and `boundary_tags = ((0, 0), (0, 0), (1, 2))`
#     in the constructor for `HydrostaticBoussinesqSuperModel`.

# # Animating the solution
#
# To animate the `ClimateMachine.Ocean` solution, we'll create a callback
# that draws a plot and stores it in an array. When the simulation is finished,
# we'll string together the plotted frames into an animation.

using Printf
using Plots
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using ClimateMachine.Ocean.Fields: assemble
using ClimateMachine.Ocean: current_step, current_time

u, v, η, θ = model.fields

## Container to hold the plotted frames
movie_plots = []

plot_every = 200 # iterations

plot_maker = EveryXSimulationSteps(plot_every) do

    @info "Steps: $(current_step(model)), time: $(current_time(model))"

    assembled_u = assemble(u.elements)
    assembled_v = assemble(v.elements)
    assembled_η = assemble(η.elements)

    umax = 0.5 * max(maximum(abs, u), maximum(abs, v))
    ulim = (-umax, umax)

    u_plot = plot(
        assembled_u.x[:, 1, 1],
        [assembled_u.data[:, 1, 1] assembled_v.data[:, 1, 1]],
        xlim = domain.x,
        ylim = (-0.7U, 0.7U),
        label = ["u" "v"],
        linewidth = 2,
        xlabel = "x (m)",
        ylabel = "Velocities (m s⁻¹)",
    )

    η_plot = plot(
        assembled_η.x[:, 1, 1],
        assembled_η.data[:, 1, 1],
        xlim = domain.x,
        ylim = (-0.01a, 1.2a),
        linewidth = 2,
        label = nothing,
        xlabel = "x (m)",
        ylabel = "η (m)",
    )

    push!(movie_plots, (u = u_plot, η = η_plot, time = current_time(model)))

    return nothing
end

# # Running the simulation and animating the results
#
# Finally, we run the simulation,

hours = 3600.0

model.solver_configuration.timeend = 2hours

result = ClimateMachine.invoke!(
    model.solver_configuration;
    user_callbacks = [plot_maker],
)

# and animate the results,

animation = @animate for p in movie_plots
    title = @sprintf("Geostrophic adjustment at t = %.2f hours", p.time / hours)
    frame = plot(
        p.u,
        p.η,
        layout = (2, 1),
        size = (800, 600),
        title = [title ""],
    )
end

gif(animation, "geostrophic_adjustment.mp4", fps = 5) # hide
