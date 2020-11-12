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
# We formulate a non-dimension problem in a rectangular domain. We normalize
# length scales by the initial width of the radiating eddy and choose
# the domain dimensions

Lx = Ly = 16
Lz = 1
nothing # hide

# We discretize the domain with 32² fourth-order elements in ``x, y``,
# and a single fourth-order element in ``z``,

Np = 4           # Polynomial order
Ne = (16, 16, 1) # Number of elements in (x, y, z)
nothing # hide

# # Non-dimensional parameters
#
# There are three non-dimensional parameters in this problem: the initial
# eddy amplitude, ``ϵ``, the initial planetary vorticity gradient, ``β``,
# and gravitational acceleration, ``g``. For the eddy amplitude we choose

ϵ = 0.01
nothing # hide

# while for the planetary vorticity gradient we choose

β = 0.2 # Planetary vorticity gradient ∂_y f
nothing # hide

# The gravitational acceleration ``g`` determines the surface gravity wave
# phase speed, ``c = \sqrt{g L_z}``. Because the eddy scale is 1,
# the characteristic Rossby wave speed associated with the eddy
# is just ``cᵝ = β``. We thus set

using CLIMAParameters: AbstractEarthParameterSet, Planet

const gravitational_acceleration = Planet.grav

struct NonDimensionalParameters <: AbstractEarthParameterSet end

gravitational_acceleration(::NonDimensionalParameters) = 16.0

# and so that the gravity wave speed is slightly faster than the Rossby
# wave speed, rendering our simulations feasible.

# # A balanced initial state
#
# Our initial condition consists of a Gaussian surface perturbation ``η``,
# with the corresponding geostrophically-balanced velocity field
# ``u = - g ∂_y η``, ``v = + g ∂_x η``,

using ClimateMachine.Ocean.OceanProblems: InitialConditions

x₀ = Lx / 2 # Gaussian origin (m, recall that x ∈ [0, Lx])
y₀ = Ly / 2 # Gaussian origin (m, recall that y ∈ [0, Ly])

Ψ(x, y) = exp(-(x^2 + y^2)) # a Gaussian

## Geostrophic surface displacement
ηᵍ(x, y, z) = ϵ * Ψ(x - x₀, y - y₀)

## Geostrophic x- and y-velocity:
g = gravitational_acceleration(NonDimensionalParameters())

uᵍ(x, y, z) = +ϵ * g * (y - y₀) * Ψ(x - x₀, y - y₀) # - g ∂_y η
vᵍ(x, y, z) = -ϵ * g * (x - x₀) * Ψ(x - x₀, y - y₀) # + g ∂_x η

initial_conditions = InitialConditions(u = uᵍ, v = vᵍ, η = ηᵍ)

@info """ The parameters for the radiating eddy problem...

    Planetary vorticity gradient: $β
    Eddy amplitude:               $ϵ
    Gravitational acceleration:   $(gravitational_acceleration(NonDimensionalParameters()))
    Surface gravity wave speed:   $(sqrt(Lz * gravitational_acceleration(NonDimensionalParameters())))

"""

# # Driver configuration
#
# Next, we configure the `HydrostaticBoussinesqModel` and build the `DriverConfiguration`.
# We use small horizontal viscosities and diffusivities to stabilize the model.

using ClimateMachine.Ocean.HydrostaticBoussinesq: HydrostaticBoussinesqModel

using ClimateMachine.Ocean.OceanProblems: InitialValueProblem

problem = InitialValueProblem{Float64}(
    dimensions = (Lx, Ly, Lz),
    initial_conditions = initial_conditions,
)

equations = HydrostaticBoussinesqModel{Float64}(
    NonDimensionalParameters(),
    problem,
    νʰ = 1e-2, # Horizontal viscosity (m² s⁻¹) 
    κʰ = 1e-2, # Horizontal diffusivity (m² s⁻¹) 
    fₒ = 1,    # Coriolis parameter (s⁻¹)
    β = β,      # Coriolis parameter gradient (m⁻¹ s⁻¹)
)

driver_configuration = ClimateMachine.OceanBoxGCMConfiguration(
    "radiating eddy tutorial",            # The name of the experiment
    Np,                                   # The polynomial order
    Ne,                                   # The number of elements
    NonDimensionalParameters(),
    equations;                            # The equations to solve, represented by a `BalanceLaw`
    periodicity = (false, false, false),  # Topology of the domain
    boundary = ((1, 1), (1, 1), (1, 2)),   # (?)
)
nothing # hide

# # Filters
#
# The `HydrostaticBoussinesqModel` currently assumes that both `vert_filter`
# and `exp_filter` are supplied to `modeldata` in `SolverConfiguration`.
# We define these required objects,

using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids: polynomialorder

grid = driver_configuration.grid

filters = (
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1),
    exp_filter = ExponentialFilter(grid, 1, 8),
)

# # Configuring the solver
#
# We configure our solver with the `LSRK144NiegemannDiehlBusch` time-stepping method.

using ClimateMachine.ODESolvers

start_time = 0.0
stop_time = 20.0
time_step = 0.02 # resolves the gravity wave speed c = 4

solver_configuration = ClimateMachine.SolverConfiguration(
    start_time,
    stop_time,
    driver_configuration,
    init_on_cpu = true,
    ode_dt = time_step,
    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    ),
    modeldata = filters,
)

# # Fetching the data to make a movie
#
# To animate the `ClimateMachine.Ocean` solution, we create
# `SpectralElementField`s to `view` the data corresponding to prognostic fields
# in `solver_configuration.solver.Q.realdata`:

using ClimateMachine.Ocean.Domains: RectangularDomain
using ClimateMachine.Ocean.Fields: SpectralElementField

## RectangularDomain and SpectralElementField objects to help with plotting
domain = RectangularDomain(solver_configuration.dg.grid, Ne)

u = SpectralElementField(domain, solver_configuration.Q, 1)
v = SpectralElementField(domain, solver_configuration.Q, 2)
η = SpectralElementField(domain, solver_configuration.Q, 3)

# and then and builda callback that copies and assembles the data 
# for each state variable, and stores it in `fetched_states`.

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.Ocean.Fields: assemble

fetched_states = []
fetch_every = 1 # unit of simulation time

data_fetcher = EveryXSimulationTime(fetch_every) do
    assembled_u = assemble(u.elements)
    assembled_v = assemble(v.elements)
    assembled_η = assemble(η.elements)

    push!(
        fetched_states,
        (u = assembled_u, v = assembled_v, η = assembled_η, time = solver.t),
    )

    return nothing
end

# We also build a simple callback to log the progress of our simulation,

using Printf
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps

print_every = 10 # iterations
solver = solver_configuration.solver
wall_clock = [time_ns()]

tiny_progress_printer = EveryXSimulationSteps(print_every) do
    @info @sprintf(
        "Steps: %d, time: %.2f, Δt: %.2f, max(|η|): %.4f, elapsed time: %.2f secs",
        solver.steps,
        solver.t,
        solver.dt,
        maximum(abs, η),
        1e-9 * (time_ns() - wall_clock[1])
    )

    wall_clock[1] = time_ns()
end

# # Running the simulation and animating the results
#
# Finally, we run the simulation,

result = ClimateMachine.invoke!(
    solver_configuration;
    user_callbacks = [tiny_progress_printer, data_fetcher],
)

# # Animating the result
#
# Time see what happened inside `ClimateMachine.invoke!`...

using Plots

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states))..."

    umax = maximum(abs, state.u)
    vmax = maximum(abs, state.v)
    Umax = max(umax, vmax)

    ηmax = maximum(abs, state.η)

    ulim = (-ϵ * g / 10, ϵ * g / 10)
    ηlim = (-ηmax, ηmax)

    ulevels = range(ulim[1], ulim[2], length = 31)
    ηlevels = range(ηlim[1], ηlim[2], length = 31)

    kwargs = (
        aspectratio = 1,
        linewidth = 0,
        xlim = domain.x,
        ylim = domain.y,
        xlabel = "x (m)",
        ylabel = "y (m)",
    )

    u_plot = contourf(
        state.u.x,
        state.u.y,
        clamp.(state.u.data[:, :, 1]', ulim[1], ulim[2]);
        color = :balance,
        clim = ulim,
        levels = ulevels,
        kwargs...,
    )

    v_plot = contourf(
        state.v.x,
        state.v.y,
        clamp.(state.v.data[:, :, 1]', ulim[1], ulim[2]);
        color = :balance,
        clim = ulim,
        levels = ulevels,
        kwargs...,
    )

    η_plot = contourf(
        state.η.x,
        state.η.y,
        clamp.(state.η.data[:, :, 1]', ηlim[1], ηlim[2]);
        color = :balance,
        clim = ηlim,
        levels = ηlevels,
        kwargs...,
    )

    u_title = @sprintf("u at t = %.2f", state.time)
    v_title = @sprintf("v at t = %.2f", state.time)
    η_title = @sprintf("η at t = %.2f", state.time)

    plot(
        u_plot,
        v_plot,
        η_plot,
        layout = (1, 3),
        size = (1600, 400),
        title = [u_title v_title η_title],
    )
end

gif(animation, "radiating_eddy.gif", fps = 8) # hide
