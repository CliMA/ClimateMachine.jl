# # Shear instability of a free-surface flow
#
# This script simulates the instability of a sheared, free-surface
# flow using `ClimateMachine.Ocean.HydrostaticBoussinesqSuperModel`.

using Printf
using Plots
using Revise
using ClimateMachine

ClimateMachine.init()

ClimateMachine.Settings.array_type = Array

using ClimateMachine.Ocean
using ClimateMachine.Ocean.Domains
using ClimateMachine.Ocean.Fields

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.Ocean: current_step, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet

# We begin by specifying the domain and mesh,

domain = RectangularDomain(
    Ne = (32, 32, 1),
    Np = 4,
    x = (-4π, 4π),
    y = (-4π, 4π),
    z = (0, 1),
    periodicity = (true, true, false),
)

# Note that the default solid-wall boundary conditions are free-slip and
# insulating on tracers. Next, we specify model parameters and the sheared
# initial conditions

struct NonDimensionalParameters <: AbstractEarthParameterSet end
Planet.grav(::NonDimensionalParameters) = 1

initial_conditions = InitialConditions(
    u = (x, y, z) -> sech(y) * (1 + 0.1 * cos(x))
    v = (x, y, z) -> 0.1 * sech(y) * sin(y),
    θ = (x, y, z) -> x,
)

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = 0.05,
    initial_conditions = initial_conditions,
    parameters = NonDimensionalParameters(),
    turbulence_closure = (νʰ = 1e-3, κʰ = 1e-3,
                          νᶻ = 1e-1, κᶻ = 1e-1),
    rusanov_wave_speeds = (cʰ = 1, cᶻ = 1),
    boundary_tags = ((0, 0), (0, 0), (1, 2)),
    boundary_conditions = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    ),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

u, v, η, θ = model.fields
fetched_states = []

start_time = time_ns()

data_fetcher = EveryXSimulationTime(1) do

    # Print a helpful message
    step = @sprintf("Step: %d", current_step(model))
    time = @sprintf("time: %.2f min", current_time(model) / 60)
    max_u = @sprintf("max|u|: %.6f", maximum(abs, u))

    elapsed = (time_ns() - start_time) * 1e-9
    wall_time = @sprintf("elapsed wall time: %.2f min", elapsed / 60)  

    @info "$step, $time, $max_u, $wall_time"

    # Fetch some data
    push!(
        fetched_states,
        (u = assemble(u), θ = assemble(θ), time = current_time(model)),
    )
end

# and then run the simulation.

model.solver_configuration.timeend = 100.0

result = ClimateMachine.invoke!(
    model.solver_configuration;
    user_callbacks = [data_fetcher],
)

# Finally, we make an animation of the evolving shear instability.

animation = @animate for (i, state) in enumerate(fetched_states)

    local u
    local θ

    @info "Plotting frame $i of $(length(fetched_states))..."

    kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1)

    x, y = state.u.x[:, 1, 1], state.u.y[1, :, 1]

    u = state.u.data[:, :, 1]
    θ = state.θ.data[:, :, 1]

    ulim = 1
    θlim = 8

    ulevels = range(-ulim, ulim, length=31)
    θlevels = range(-θlim, θlim, length=31)

    u_plot = contourf(x, y, clamp.(u, -ulim, ulim)'; levels = ulevels, color = :balance, kwargs...)
    θ_plot = contourf(x, y, clamp.(θ, -θlim, θlim)'; levels = θlevels, color = :thermal, kwargs...)

    u_title = @sprintf("u at t = %.2f", state.time)
    θ_title = @sprintf("θ at t = %.2f", state.time)

    plot(u_plot, θ_plot, title = [u_title θ_title], size = (1200, 500))
end

gif(animation, "shear_instability.gif", fps = 8)
