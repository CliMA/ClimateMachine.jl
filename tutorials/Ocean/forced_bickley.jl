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
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using ClimateMachine.Ocean: current_step, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet

# We begin by specifying the domain and mesh,

domain = RectangularDomain(
    Ne = (64, 64, 1),
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
Planet.grav(::NonDimensionalParameters) = 0.1

ϵ = 0.01
ψ(y) = tanh(y)
f = 0.0
g = Planet.grav(NonDimensionalParameters())

initial_conditions = InitialConditions(
    u = (x, y, z) -> sech(y)^2 * (1 + ϵ * randn()),
    v = (x, y, z) -> ϵ * sech(y)^2 * randn(),
    θ = (x, y, z) -> sin(y / 4),
    η = (x, y, z) -> - f / g * ψ(y),
)

Δx = domain.L.x / (domain.Ne.x * domain.Np)
c = sqrt(g * domain.L.z)
@show dt = Δx / c / 10

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = dt,
    initial_conditions = initial_conditions,
    parameters = NonDimensionalParameters(),
    turbulence_closure = (νʰ = 1e-4, κʰ = 1e-4,
                          νᶻ = 1e-4, κᶻ = 1e-4),
    rusanov_wave_speeds = (cʰ = sqrt(g * domain.L.z), cᶻ = 1e-2),
    coriolis = (f₀ = f, β = 0),
    buoyancy = (αᵀ = 0,),
    boundary_tags = ((0, 0), (0, 0), (1, 2)),
    boundary_conditions = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    ),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

realdata = convert(Array, model.state.realdata)
u = SpectralElementField(domain, model.grid, view(realdata, :, 1, :))

volume = assemble(u)
x = volume.x[:, 1, 1]
y = volume.y[1, :, 1]

u, v, η, θ = model.fields
fetched_states = []

start_time = time_ns()

data_fetcher = EveryXSimulationSteps(10) do

    realdata = convert(Array, model.state.realdata)
    u = SpectralElementField(domain, model.grid, view(realdata, :, 1, :))
    v = SpectralElementField(domain, model.grid, view(realdata, :, 2, :))
    η = SpectralElementField(domain, model.grid, view(realdata, :, 3, :))
    θ = SpectralElementField(domain, model.grid, view(realdata, :, 4, :))

    # Print a helpful message
    step = @sprintf("Step: %d", current_step(model))
    time = @sprintf("time: %.2f", current_time(model))
    max_u = @sprintf("max|u|: %.6f", maximum(abs, u))

    elapsed = (time_ns() - start_time) * 1e-9
    wall_time = @sprintf("elapsed wall time: %.2f min", elapsed / 60)  

    isnan(maximum(abs, u)) && error("Your simulation NaN'd out.") 

    @info "$step, $time, $max_u, $wall_time"


    # Fetch some data
    push!(
        fetched_states,
        (u = assemble(u),
         η = assemble(η),
         θ = assemble(θ),
         time = current_time(model)),
    )
end

# and then run the simulation.

model.solver_configuration.timeend = 1000 * dt

try
    result = ClimateMachine.invoke!(
        model.solver_configuration;
        user_callbacks = [data_fetcher],
    )
catch err
    @warn "Simulation ended prematurely due to $(sprint(showerror, err))"
end

# Finally, we make an animation of the evolving shear instability.

animation = @animate for (i, state) in enumerate(fetched_states)

    local u
    local η
    local θ

    @info "Plotting frame $i of $(length(fetched_states))..."

    kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1)

    u = state.u.data[:, :, 1]
    η = state.η.data[:, :, 1]
    θ = state.θ.data[:, :, 1]

    ulim = 1
    ηlim = 1 / g
    θlim = 1

    ulevels = range(-ulim, ulim, length=31)
    ηlevels = range(-ηlim, ηlim, length=31)
    θlevels = range(-θlim, θlim, length=31)

    u_plot = contourf(x, y, clamp.(u, -ulim, ulim)'; levels = ulevels, color = :balance, kwargs...)
    η_plot = contourf(x, y, clamp.(η, -ηlim, ηlim)'; levels = ηlevels, color = :balance, kwargs...)
    θ_plot = contourf(x, y, clamp.(θ, -θlim, θlim)'; levels = θlevels, color = :thermal, kwargs...)

    u_title = @sprintf("u at t = %.2f", state.time)
    θ_title = @sprintf("θ at t = %.2f", state.time)

    plot(u_plot, θ_plot, title = [u_title θ_title], size = (1200, 500))
end

gif(animation, "forced_bickley.gif", fps = 8)
