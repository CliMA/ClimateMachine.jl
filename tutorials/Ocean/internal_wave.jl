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
# We formulate a non-dimension problem in a Cartesian domain with oceanic anisotropy,

using ClimateMachine.Ocean.CartesianDomains: CartesianDomain

domain = CartesianDomain(
    elements = (32, 1, 32),
    polynomialorder = 4,
    x = (-64π, 64π),
    y = (-64π, 64π),
    z = (-π, 0)
)

# # Parameters
#
# We choose parameters appropriate for a hydrostatic internal wave,

# Non-dimensional internal wave parameters
f = 1   # Coriolis
N = 10  # Buoyancy frequency
m = 16  # vertical wavenumber
k = 1   # horizontal wavenumber

# Note that the validity of the hydrostatic approximation requires
# small aspect ratio motions with ``k / m \\ll 1``.
# The hydrostatic dispersion relation for inertia-gravity waves then implies that

ω² = f^2 + N^2 * k^2 / m^2 # and 
ω = sqrt(ω²)

# # Internal wave initial condition
# 
# We impose modest gravitational acceleration to render time-stepping feasible,

using CLIMAParameters: AbstractEarthParameterSet, Planet
const gravitational_acceleration = Planet.grav
struct NonDimensionalParameters <: AbstractEarthParameterSet end

gravitational_acceleration(::NonDimensionalParameters) = 16.0

# and set the thermal expansion coefficient to inverse the gravitational
# accleration so that our temperature variable ``θ`` corresponds to buoyancy,

αᵀ = 1 / gravitational_acceleration(NonDimensionalParameters())

# We then use the "polarization relations" for hydrostatic internal waves to
# create initial conditions for a Gaussian wave-packet 
# centered at ``(x, z) = (0, -π/2)``,

## A Gaussian envelope
A = 1e-9
δ = 16π / k

a(x, z) = A * exp( -( x^2 + z^2 ) / 2δ^2 )

## "Polarization relations" (these are non-hydrostatic; adapt for hydrostatic)
uᵢ(x, y, z) = a(x, z + π/2) * k * ω   / (ω^2 - f^2) * cos(k*x + m*z)
vᵢ(x, y, z) = a(x, z + π/2) * k * f   / (ω^2 - f^2) * sin(k*x + m*z)
wᵢ(x, y, z) = a(x, z + π/2) * m * ω   / (f^2 - N^2) * cos(k*x + m*z)
θᵢ(x, y, z) = a(x, z + π/2) * m * N^2 / (f^2 - N^2) * sin(k*x + m*z)

# The total temperature profile includes a linear stratification,

Tᵢ(x, y, z) = N^2 * z + θᵢ(x, y, z)

using ClimateMachine.Ocean.OceanProblems: InitialConditions

initial_conditions = InitialConditions(u=uᵢ, v=vᵢ, θ=Tᵢ)

# # Model configuration
#
# We're ready to build the model:

using ClimateMachine.Ocean: HydrostaticBoussinesqSuperModel
using ClimateMachine.Ocean.HydrostaticBoussinesq: NonLinearAdvectionTerm

model = HydrostaticBoussinesqSuperModel(
    domain = domain,
    initial_conditions = initial_conditions,
    advection = (momentum=nothing, tracers=NonLinearAdvectionTerm()),
    turbulence = (νʰ=1e-4, νᶻ=1e-4, κʰ=1e-4, κᶻ=1e-4),
    coriolis = (f₀=f, β=0),
    buoyancy = (αᵀ=αᵀ,),
    parameters = NonDimensionalParameters(),
)

# # Fetching the data to make a movie
#
# To animate the `ClimateMachine.Ocean` solution, we assemble and
# cache the horizontal velocity ``u`` at periodic intervals:

using ClimateMachine.Ocean.CartesianDomains: glue
using ClimateMachine.GenericCallbacks: EveryXSimulationTime

fetched_states = []
fetch_every = 1 # unit of simulation time

data_fetcher = EveryXSimulationTime(fetch_every) do
    push!(fetched_states, (u=glue(model.fields.u.elements), time=time(model)))
    return nothing
end

# We also build a callback to log the progress of our simulation,

using Printf
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using ClimateMachine.Ocean: steps, time, Δt

print_every = 10 # iterations
wall_clock = [time_ns()]

tiny_progress_printer = EveryXSimulationSteps(print_every) do

    @info(
        @sprintf("Steps: %d, time: %.2f, Δt: %.2f, max(|u|): %.4f, elapsed time: %.2f secs",
            steps(model),
            time(model),
            Δt(model),
            maximum(abs, model.fields.u),
            1e-9 * (time_ns() - wall_clock[1])
        )
    )

    wall_clock[1] = time_ns()
end

# # Running the simulation and animating the results
#
# Finally, we run the simulation,

model.solver.timeend = 20.0
model.solver.dt = 0.02 # resolves the gravity wave speed c = 4

result = ClimateMachine.invoke!(
    model.solver;
    user_callbacks = [tiny_progress_printer, data_fetcher])

# # Animating the result
#
# Time see what happened inside `ClimateMachine.invoke!`...

using Plots

animation = @animate for (i, state) in enumerate(fetched_states)

    @info "Plotting frame $i of $(length(fetched_states))..."

    umax = maximum(abs, state.u)
    ulim = (-umax, umax)
    ulevels = range(ulim[1], ulim[2], length=31)

    u_plot = contourf(
        state.u.x,
        state.u.y,
        clamp.(state.u.data[:, :, 1]', ulim[1], ulim[2]);
        aspectratio = 64,
        linewidth = 0,
        xlim = domain.x,
        ylim = domain.y,
        xlabel = "x",
        ylabel = "y",
        color = :balance,
        clim = ulim,
        levels = ulevels,
        @sprintf("u at t = %.2f", state.time)
    )
end

gif(animation, "internal_waves.gif", fps = 8) # hide
