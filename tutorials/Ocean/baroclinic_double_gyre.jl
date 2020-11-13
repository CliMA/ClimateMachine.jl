# # Shear instability of a free-surface flow
#
# This script simulates the spin-up of a double-gyre
# using `ClimateMachine.Ocean.HydrostaticBoussinesqSuperModel`.

using Printf
using Plots
using StaticArrays
using ClimateMachine

ClimateMachine.init()

using ClimateMachine.Ocean
using ClimateMachine.Ocean.Domains
using ClimateMachine.Ocean.Fields

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.Ocean: steps, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet

# We begin by specifying a few parameters:

wind_stress_magnitude = 2e-4 # m² s⁻²
Lx = 4e6 # m
Ly = 6e6 # m
H = 3000 # m
N² = 1e-5 # s⁻²
hour = 3600.0 # comes in handy

# Next we specify the domain

domain = RectangularDomain(
    elements = (16, 24, 3),
    polynomialorder = 4,
    x = (0, Lx),
    y = (0, Ly),
    z = (-H, 0),
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

# # Boundary conditions
#
# The ``x``-component of the surface stress varies in ``y``:

wind_stress(y) = @SVector [- wind_stress_magnitude * cos(2π * y / Ly), 0]

top_bc = OceanBC(Penetrable(KinematicStress(wind_stress)), Insulating())

# while we use no-slip at side walls and free-slip at the bottom,

side_wall_bc = OceanBC(Impenetrable(NoSlip()), Insulating())
bottom_bc = OceanBC(Impenetrable(FreeSlip()), Insulating())
      
boundary_conditions = (side_wall_bc, bottom_bc, top_bc)

# The indices of `boundary_conditions` correspond to the integers
# specified above in the kwarg `boundary` passed to `RectangularDomain`.
# We use temperature as a buoyancy variable, and initialize with linear stratification,

struct EarthParameters <: AbstractEarthParameterSet end
Planet.grav(::EarthParameters) = 0.1

linear_gradient(x, y, z) = N² * z
initial_conditions = InitialConditions(θ=linear_gradient)

# We're now ready to build the model. We use large viscosity and diffusion
# coefficients, and Coriolis parameters appropriate for mid-latitudes on Earth.

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = hour,
    initial_conditions = initial_conditions,
    parameters = EarthParameters(),
    turbulence_closure = (νʰ = 5e3, νᶻ = 5e-3, κʰ = 1e3, κᶻ = 1e-4),
    rusanov_wave_speeds = (cʰ = 1, cᶻ = 1e-3),
    buoyancy = (αᵀ = 1 / Planet.grav(EarthParameters()),),
    coriolis = (f₀ = 1e-4, β = 1e-11),
    boundary_conditions = boundary_conditions,
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

model.solver_configuration.timeend = 365day

result = ClimateMachine.invoke!(
    model.solver_configuration;
    user_callbacks = [data_fetcher],
)

# Finally, we make an animation of the double gyre spinning up.

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states))..."

    j = round(Int, domain.Ne.y/2)

    xy_kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1)
    xz_kwargs = (xlim = domain.x, ylim = domain.z, linewidth = 0, aspectratio = 1000)

    x, y, z = state.u.x, state.u.y, state.u.z

    u_xy_plot = contourf(x, y, state.u.data[:, :, 1]'; color = :balance, xy_kwargs...)
    η_xy_plot = contourf(x, y, state.η.data[:, :, 1]'; color = :balance, xy_kwargs...)
    θ_xy_plot = contourf(x, y, state.θ.data[:, :, 1]'; color = :thermal, xy_kwargs...)

    u_xz_plot = contourf(x, z, state.u.data[:, j, :]'; color = :balance, xz_kwargs...)
    η_xz_plot = contourf(x, z, state.η.data[:, j, :]'; color = :balance, xz_kwargs...)
    θ_xz_plot = contourf(x, z, state.θ.data[:, j, :]'; color = :thermal, xz_kwargs...)

    u_title = @sprintf("u (m s⁻¹) at t = %.2f days", state.time / day)
    η_title = @sprintf("η (m) at t = %.2f days", state.time / day)
    θ_title = @sprintf("θ (ᵒC) at t = %.2f days", state.time / day)

    plot(
        u_xy_plot,
        η_xy_plot,
        θ_xy_plot,
        u_xz_plot,
        η_xz_plot,
        θ_xz_plot,
        title = [u_title η_title θ_title],
        size = (1200, 500),
        layout = (2, 3),
    )
end

gif(animation, "baroclinic_double_gyre.gif", fps = 8)
