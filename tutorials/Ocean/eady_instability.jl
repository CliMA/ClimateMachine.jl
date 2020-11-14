# # Shear instability of a free-surface flow
#
# This script simulates the instability of a sheared, free-surface
# flow using `ClimateMachine.Ocean.HydrostaticBoussinesqSuperModel`.

using Printf

ENV["GKSwstype"] = "nul"
using Plots
using JLD2
using Plots
using Revise
using ClimateMachine

ClimateMachine.init()

ClimateMachine.Settings.array_type = Array

using ClimateMachine.Ocean
using ClimateMachine.Ocean.Domains
using ClimateMachine.Ocean.Fields

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.Ocean: steps, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet

# # Parameters

Nh = 32
Nz = 4
L = 1e6 # Domain width (m)
H = 1e3 # Domain height (m)
f = 1e-4 # Coriolis parameter (s⁻¹)
α = 10f # Geostrophic shear (s⁻¹)
N² = 1e-5 # Initial buoyancy gradient (s⁻²)
νh = κh = 1e3 # Horizontal viscosity and diffusivity (m² s⁻¹)
νz = κh = 1e-2 # Vertical viscosity and diffusivity (m² s⁻¹)

hour = 3600.0
day = 24hour
year = 365day
stop_time = 30day # Simulation stop time

struct EarthParameters <: AbstractEarthParameterSet end
Planet.grav(::EarthParameters) = 0.1

# # The domain

domain = RectangularDomain(
    elements = (Nh, Nh, Nz),
    polynomialorder = 4,
    x = (0, L),
    y = (0, L),
    z = (-H, 0),
    periodicity = (true, false, false),
    boundary = ((0, 0), (1, 1), (1, 2)),
)

# # Boundary conditions

no_slip = OceanBC(Impenetrable(NoSlip()), Insulating())
free_surface = OceanBC(Penetrable(FreeSlip()), Insulating())

# # Initial conditions
#
# We idealize θ as buoyancy. We initialize the flow with a
# vertically-sheared, horizontally-sinusoidal jet in geostrophic
# balance. We add a small amount of noise to the buoyancy field.

Ξ(z) = randn() * z / H * (z / H + 1)

λ = L / π # Sinusoidal jet width (m)

# Ψ(y, z) = - α * λ * cos(y / λ) * (z + H)
U(x, y, z) = + α * sin(y / λ) * (z + H)
Θ(x, y, z) = - α * f * λ * cos(y / λ) + N² * z + α * f * L * 1e-3 * Ξ(z)

initial_conditions = InitialConditions(u = U, θ = Θ)

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = hour / 4,
    initial_conditions = initial_conditions,
    parameters = EarthParameters(),
    buoyancy = (αᵀ = 1 / Planet.grav(EarthParameters()),),
    coriolis = (f₀ = f, β = 0),
    turbulence_closure = (νʰ = νh, κʰ = κh,
                          νᶻ = νz, κᶻ = νz),
    rusanov_wave_speeds = (cʰ = 1.0, cᶻ = 1e-3),
    boundary_conditions = (no_slip, free_surface),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

u, v, η, θ = model.fields
fetched_states = []

volume = assemble(u)
x = volume.x[:, 1, 1]
y = volume.y[1, :, 1]

@assert issorted(x)
@assert issorted(y)

start_time = time_ns()

#data_fetcher = EveryXSimulationTime(day) do
data_fetcher = EveryXSimulationSteps(1) do
    umax = maximum(abs, u)
    elapsed = (time_ns() - start_time) * 1e-9

    step = @sprintf("Step: %d", steps(model))
    sim_time = @sprintf("time: %.2f days", current_time(model) / day)
    wall_time = @sprintf("wall time: %.2f min", elapsed / 60)

    @info "$step, $sim_time, $wall_time"

    isnan(umax) && error("NaN'd out.")

    push!(
        fetched_states,
        (u = assemble(data.(u.elements)), θ = assemble(data.(θ.elements)), time = current_time(model)),
    )
end

# and then run the simulation.

#model.solver_configuration.timeend = stop_time
model.solver_configuration.timeend = model.solver_configuration.dt * 10

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

    u = state.u.data[:, :, end]
    θ = state.θ.data[:, :, end]

    @show ulim = maximum(abs, u) + 1e-9
    @show θlim = maximum(abs, θ) + 1e-9

    ulevels = range(-ulim, ulim, length=31)
    θlevels = range(-θlim, θlim, length=31)

    u_plot = contourf(
        x,
        y,
        clamp.(u, -ulim, ulim)';
        levels = ulevels,
        color = :balance,
        clim = (-ulim, ulim),
        kwargs...
    )

    θ_plot = contourf(
        x,
        y,
        clamp.(θ, -θlim, θlim)';
        levels = θlevels,
        color = :thermal,
        clim = (-θlim, θlim),
        kwargs...
    )

    u_title = @sprintf("u at t = %d days", state.time / day)
    θ_title = @sprintf("θ at t = %d days", state.time / day)

    plot(u_plot, θ_plot, title = [u_title θ_title], size = (1200, 500))
end

gif(animation, "eady_instability.gif", fps = 8)
