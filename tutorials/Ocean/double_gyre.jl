# # Shear instability of a free-surface flow
#
# This script simulates the spin-up of a double-gyre
# using `ClimateMachine.Ocean.HydrostaticBoussinesqSuperModel`.

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
H = 3000

domain = RectangularDomain(
    elements = (24, 24, 2),
    polynomialorder = 4,
    x = (0, Lx),
    y = (0, Ly),
    z = (-H, 0),
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

# Specify initial condition

struct EarthParameters <: AbstractEarthParameterSet end
Planet.grav(::EarthParameters) = 0.1

linear_gradient(x, y, z) = 20 * (1 + z / H)
constant_temperature(x, y, z) = 20
initial_conditions = InitialConditions(θ=linear_gradient)

# Specify wind stress at the top boundary

wind_stress(y) = @SVector [- τ * cos(2π * y / Ly), 0]
top_boundary_condition = OceanBC(Penetrable(KinematicStress(wind_stress)), Insulating())

hour = 3600.0

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = hour / 2, # seconds
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

data_fetcher = EveryXSimulationTime(24hour) do
    @info "Step: $(steps(model)), hours: $(current_time(model) / hour), max|u|: $(maximum(abs, u))"

    push!(
        fetched_states,
        (u = assemble(u),
         θ = assemble(θ),
         η = assemble(η),
         time = current_time(model)),
    )
end

# and then run the simulation.

day = 24hour

model.solver_configuration.timeend = 300day

result = ClimateMachine.invoke!(
    model.solver_configuration;
    user_callbacks = [data_fetcher],
)

# Finally, we make an animation of the double gyre spinning up.

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states))..."

    kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1)

    x, y, = state.u.x, state.u.y

    u_plot = contourf(x, y, state.u.data[:, :, 1]'; color = :balance, kwargs...)
    η_plot = contourf(x, y, state.η.data[:, :, 1]'; color = :balance, kwargs...)
    θ_plot = contourf(x, y, state.θ.data[:, :, 1]'; color = :thermal, kwargs...)

    u_title = @sprintf("u (m s⁻¹) at t = %.2f days", state.time / day)
    η_title = @sprintf("η (m) at t = %.2f days", state.time / day)
    θ_title = @sprintf("θ (ᵒC) at t = %.2f days", state.time / day)

    plot(u_plot, η_plot, θ_plot, title = [u_title η_title θ_title], size = (1200, 500), layout=(1, 3))
end

gif(animation, "double_gyre.gif", fps = 8)
