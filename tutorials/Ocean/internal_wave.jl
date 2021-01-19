# # Mode-1 internal wave reflection
#
# This example simulates the propagation of a mode-1 internal wave
# using the `ClimateMachine.Ocean` subcomponent to solve the hydrostatic
# Boussinesq equations.
#
# First we `ClimateMachine.init()`.

using ClimateMachine

ClimateMachine.init()

# # Domain setup
#
# We formulate a non-dimension problem in a Cartesian domain with oceanic anisotropy,

using ClimateMachine.Ocean.Domains

domain = RectangularDomain(
    Ne = (32, 1, 4),
    Np = 4,
    x = (-128, 128),
    y = (-128, 128),
    z = (-1, 0),
    periodicity = (false, false, false),
)

# # Parameters
#
# We choose parameters appropriate for a hydrostatic internal wave,

# Non-dimensional internal wave parameters
f = 1  # Coriolis
N = 10 # Buoyancy frequency

# Note that the validity of the hydrostatic approximation requires
# small aspect ratio motions with ``k / m \\ll 1``.
# The hydrostatic dispersion relation for inertia-gravity waves then implies that

λ = 8      # horizontal wave-length
k = 2π / λ # horizontal wavenumber
m = π

ω² = f^2 + N^2 * k^2 / m^2 # and

ω = √(ω²)

# # Internal wave initial condition
# 
# We impose modest gravitational acceleration to render time-stepping feasible,

using CLIMAParameters: AbstractEarthParameterSet, Planet
struct NonDimensionalParameters <: AbstractEarthParameterSet end

Planet.grav(::NonDimensionalParameters) = 256.0

# we'd like to use `θ` as a buoyancy variable, which requires
# setting the thermal expansion coefficient ``αᵀ`` to

g = Planet.grav(NonDimensionalParameters())

αᵀ = 1 / g

# We then use the "polarization relations" for vertically-standing, horizontally-
# propagating hydrostatic internal waves to initialze two wave packets.
# The hydrostatic polarization relations require
#
# ```math
# \begin{gather}
# (∂_t^2 + f^2) u = - ∂_x ∂_t p
# ∂_t v = - f u
# b = ∂_z p
# \end{gather}
# ```
#
# Thus given ``p = \cos (k x - ω t) \cos (m z)``, we find

δ = domain.L.x / 15
a(x) = 1e-6 * exp(-x^2 / 2 * δ^2)

ũ(x, z, t) = +a(x) * ω * sin(k * x - ω * t) * cos(m * z)
ṽ(x, z, t) = -a(x) * f * cos(k * x - ω * t) * cos(m * z)
θ̃(x, z, t) = -a(x) * m / k * (ω^2 - f^2) * sin(k * x - ω * t) * sin(m * z)

uᵢ(x, y, z) = ũ(x, z, 0)
vᵢ(x, y, z) = ṽ(x, z, 0)
θᵢ(x, y, z) = θ̃(x, z, 0) + N^2 * z

using ClimateMachine.Ocean.OceanProblems: InitialConditions

initial_conditions = InitialConditions(u = uᵢ, v = vᵢ, θ = θᵢ)

# # Model configuration
# 
# We choose a time-step that resolves the gravity wave phase speed,

time_step = 0.005 # close to Δx / c = 0.5 * 1/16, where Δx is nominal resolution

# and build a model with a smidgeon of viscosity and diffusion,

using ClimateMachine.Ocean.SuperModels: SplitExplicitOceanSuperModel

model = SplitExplicitOceanSuperModel(
    domain = domain,
    parameters = NonDimensionalParameters(),
    barotropic_time_step = time_step,
    baroclinic_time_step = time_step / 24,
    initial_conditions = initial_conditions,
    turbulence_closure = (νʰ = 1e-6, νᶻ = 1e-6, κʰ = 1e-6, κᶻ = 1e-6, κᶜ = 1.0),
    coriolis = (f₀ = f, β = 0),
    buoyancy = (αᵀ = αᵀ,),
    boundary_tags = ((1, 1), (1, 1), (1, 2)),
)

# # Fetching data for an animation
#
# To animate the `ClimateMachine.Ocean` solution, we assemble and
# cache the horizontal velocity ``u`` at periodic intervals:

using ClimateMachine.Ocean: current_time
using ClimateMachine.Ocean.Fields: assemble
using ClimateMachine.GenericCallbacks: EveryXSimulationTime

fetched_states = []
fetch_every = 0.2 * 2π / ω # time

data_fetcher = EveryXSimulationTime(fetch_every) do
    push!(
        fetched_states,
        (
            u = assemble(model.fields.u.elements),
            θ = assemble(model.fields.θ.elements),
            η = assemble(model.fields.η.elements),
            time = current_time(model),
        ),
    )
    return nothing
end

# We also build a callback to log the progress of our simulation,

using Printf
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using ClimateMachine.Ocean: current_time, current_step, Δt

print_every = 100 # iterations
wall_clock = [time_ns()]

tiny_progress_printer = EveryXSimulationSteps(print_every) do

    @info(@sprintf(
        "Steps: %d, time: %.2f, Δt: %.2f, max(|u|): %.2e, elapsed time: %.2f secs",
        current_step(model),
        current_time(model),
        Δt(model),
        maximum(abs, model.fields.u),
        1e-9 * (time_ns() - wall_clock[1])
    ))

    wall_clock[1] = time_ns()
end

# # Running the simulation and animating the results
#
# We're ready to launch.

## model.solver.dt = 0.05 # make this work

@info """ Simulating a hydrostatic Gaussian wave packet with parameters

    f (Coriolis parameter):       $f
    N (buoyancy frequency):       $N
    Internal wave frequency:      $(abs(ω))
    Surface wave frequency:       $(k * sqrt(g * domain.L.z))
    Surface wave group velocity:  $(sqrt(g * domain.L.z))
    Internal wave group velocity: $(N^2 * k / (ω * m))
    Domain width:                 $(domain.L.x)  
    Domain height:                $(domain.L.z)  

"""

using ClimateMachine.ODESolvers: solve!

solve!(
    (slow = model.baroclinic.state, fast = model.barotropic.state),
    model.solver,
    timeend = 6 * 2π / ω,
    callbacks = (tiny_progress_printer, data_fetcher),
)

# # Animating the result
#
# We first analye the results to generate plotting limits and contour levels

ηmax = maximum([maximum(abs, state.η.data) for state in fetched_states])
umax = maximum([maximum(abs, state.u.data) for state in fetched_states])

ηlim = (-ηmax, ηmax)
ulim = (-umax, umax)
ulevels = range(ulim[1], ulim[2], length = 31)

# and then animate both fields in a loop,

using Plots

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states))..."

    η_plot = plot(
        state.u.x[:, 1, 1],
        state.η.data[:, 1, 1],
        ylim = ηlim,
        label = nothing,
        title = @sprintf("η at t = %.2f", state.time),
    )

    u_plot = contourf(
        state.u.x[:, 1, 1],
        state.u.z[1, 1, :],
        clamp.(state.u.data[:, 1, :], ulim[1], ulim[2])';
        aspectratio = 64,
        linewidth = 0,
        xlim = domain.x,
        ylim = domain.z,
        xlabel = "x",
        ylabel = "z",
        color = :balance,
        colorbar = false,
        clim = ulim,
        levels = ulevels,
        title = @sprintf("u at t = %.2f", state.time),
    )

    plot(
        η_plot,
        u_plot,
        layout = Plots.grid(2, 1, heights = (0.3, 0.7)),
        link = :x,
        size = (1200, 600),
    )
end

gif(animation, "internal_wave.gif", fps = 8) # hide
