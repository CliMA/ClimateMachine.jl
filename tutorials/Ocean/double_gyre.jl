# # Shear instability of a free-surface flow
#
# This script simulates the instability of a sheared, free-surface
# flow using `ClimateMachine.Ocean.HydrostaticBoussinesqSuperModel`.

using Printf
using Plots
using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean
using ClimateMachine.Ocean.Domains
using ClimateMachine.Ocean.Fields

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.Ocean: steps, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet
using StaticArrays

# We begin by specifying the domain and mesh,

τ  = 1e-4
Lx = 4e6
Ly = 6e6
H = -3000

domain = RectangularDomain(
    elements = (10, 15, 2),
    polynomialorder = 4,
    x = (0, Lx),
    y = (0, Ly),
    z = (H, 0),
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

# Note that the default solid-wall boundary conditions are free-slip and
# insulating on tracers. Next, we specify model parameters and the sheared
# initial conditions

struct EarthParameters <: AbstractEarthParameterSet end
Planet.grav(::EarthParameters) = 0.1

linear_gradient(x, y, z) = 20 * (1 - z / H)
constant_temperature(x, y, z) = 20

initial_conditions = InitialConditions(θ=linear_gradient)

wind_stress(y) = @SVector [τ * cos(2π * y / Ly), 0]

top_boundary_condition = OceanBC(Penetrable(KinematicStress(wind_stress)), Insulating())

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = 30, # seconds
    initial_conditions = initial_conditions,
    parameters = EarthParameters(),
    turbulence_closure = (νʰ = 5e3, νᶻ = 5e-3, κʰ = 1e3, κᶻ = 1e-4),
    rusanov_wave_speeds = (cʰ = 1, cᶻ = 1.e-3),
    buoyancy = (αᵀ = 0.,),
    coriolis = (f₀=1.e-4, β=1.e-11),
    boundary_conditions = (
      OceanBC(Impenetrable(NoSlip()), Insulating()),
      OceanBC(Impenetrable(FreeSlip()), Insulating()),
      top_boundary_condition,
    ),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

u, v, η, θ = model.fields
fetched_states = []

data_fetcher = EveryXSimulationTime(1) do
    @info "Step: $(steps(model)), t: $(current_time(model)), max|u|: $(maximum(abs, u))"

    push!(
        fetched_states,
        (u = assemble(u), θ = assemble(θ), time = current_time(model)),
    )
end

# and then run the simulation.

model.solver_configuration.timeend = 100 * model.solver_configuration.dt

result = ClimateMachine.invoke!(
    model.solver_configuration;
    user_callbacks = [data_fetcher],
)

# Finally, we make an animation of the evolving shear instability.

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states))..."

    kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1)

    x, y, = state.u.x, state.u.y

    u_plot = contourf(x, y, state.u.data[:, :, 1]'; color = :balance, kwargs...)
    θ_plot = contourf(x, y, state.θ.data[:, :, 1]'; color = :thermal, kwargs...)

    u_title = @sprintf("u at t = %.2f", state.time)
    θ_title = @sprintf("θ at t = %.2f", state.time)

    plot(u_plot, θ_plot, title = [u_title θ_title], size = (1200, 500))
end

gif(animation, "shear_instability.gif", fps = 8)
