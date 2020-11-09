# # Shear instability of a free-surface flow
#
# This script simulates the instability of a sheared, free-surface
# flow using `ClimateMachine.Ocean.HydrostaticBoussinesqSuperModel`.

using Printf
using Plots
using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean
using ClimateMachine.Ocean.CartesianDomains

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.Ocean: steps, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet

# We begin by specifying the domain and mesh,

domain = CartesianDomain(
    elements = (24, 24, 1),
    polynomialorder = 4,
    x = (-3π, 3π),
    y = (-3π, 3π),
    z = (0, 1),
    periodicity = (true, false, false),
    boundary = ((0, 0), (1, 1), (1, 2)),
)

# Note that the default solid-wall boundary conditions are free-slip and
# insulating on tracers. Next, we specify model parameters and the sheared
# initial conditions

struct NonDimensionalParameters <: AbstractEarthParameterSet end
Planet.grav(::NonDimensionalParameters) = 1

initial_conditions = InitialConditions(
    u = (x, y, z) -> tanh(y) + 0.1 * cos(x / 3) + 0.01 * randn(),
    v = (x, y, z) -> 0.1 * sin(y / 3),
    θ = (x, y, z) -> x
)
                                       
model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = 0.1,
    initial_conditions = initial_conditions,
    parameters = NonDimensionalParameters(),
    turbulence_closure = (νʰ = 1e-2, νᶻ = 1e-2, κʰ = 1e-2, κᶻ = 1e-2),
    rusanov_wave_speeds = (cʰ = 0.1, cᶻ = 1),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

u, v, η, θ = model.fields
fetched_states = []

data_fetcher = EveryXSimulationTime(1) do
    @info "Step: $(steps(model)), t: $(current_time(model)), max|u|: $(maximum(abs, u))"

    push!(fetched_states, (u = assemble(u), θ = assemble(θ), time = current_time(model)))
end

# and then run the simulation.
        
model.solver_configuration.timeend = 100.0

result = ClimateMachine.invoke!(model.solver_configuration; user_callbacks = [data_fetcher])

# Finally, we make an animation of the evolving shear instability.

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states))..."

    kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1)

    x, y, = state.u.x, state.u.y

    u_plot = contourf(x, y, state.u.data[:, :, 1]'; color=:balance, kwargs...)
    θ_plot = contourf(x, y, state.θ.data[:, :, 1]'; color=:thermal, kwargs...)

    u_title = @sprintf("u at t = %.2f", state.time)
    θ_title = @sprintf("θ at t = %.2f", state.time)

    plot(u_plot, θ_plot, layout = (1, 2), title = [u_title θ_title], size = (1200, 500))
end

gif(animation, "shear_instability.gif", fps = 8)
