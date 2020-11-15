# # Eady instability
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
using CUDA

ClimateMachine.init()

using ClimateMachine.Ocean
using ClimateMachine.Ocean.Domains
using ClimateMachine.Ocean.Fields

using ClimateMachine.GenericCallbacks: EveryXSimulationTime
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using ClimateMachine.Ocean: steps, Δt, current_time
using CLIMAParameters: AbstractEarthParameterSet, Planet

# # The Eady instability problem
#
# Our objective is to model the instability
# of a sinusoidal geostrophic jet with the streamfunction
#
# ```math
# ψ(y, z) = α λ cos(y / λ) (z + H)
# ```
#
# where ``α`` is geostrophic shear, ``λ`` is the width
# of the sinusoid, and ``H`` is depth. We set ``λ = L / π``,
# where ``L`` is the width of the domain, which fits a
# half wavelength of the sinusoid.
#
# The jet is vertically sheared with the total velocity
# field ``u = - ∂_y ψ``, such that
#
# ```math
# u = α sin(y / λ) (z + H) \, .
# ```
#
# The total buoyancy field is ``b = f ∂_z ψ + N² z``, where
# ``f ∂_z ψ`` is the geostrophic component of buoyancy and
# ``N² z`` refelcts a stable stratification with buoyancy
# frequency ``N``. We thus have
#
# ```math
# b = α λ cos(y / λ) + N² z
# ```
# 
# The baroclinic component of ``u`` is the "thermal wind"
# associated with ``b``. The barotropic component of ``u``,
# ``⟨u⟩ = \frac{1}{H} ∫ u \rm{d} z``, however, is balanced by
# a surface displacement via
# 
# ```math
# f ⟨u⟩ = - g H ∂_y η \, ,
# ```
#
# where ``g`` is gravitational acceleration. Using
# ``⟨u⟩ = α sin(y / λ) / 2``, we obtain
#
# ```math
# η = α f λ / (2 g H) cos(y / λ) 
# ```
#
# # Parameters
#
# We now choose parameters for the Eady instability problem.

Nh = 64
Nz = 8
L = 1e6 # Domain width (m)
H = 1e3 # Domain height (m)
f = 1e-4 # Coriolis parameter (s⁻¹)
α = 10f # Geostrophic shear (s⁻¹)
N² = 1e-5 # Initial buoyancy gradient (s⁻²)
νh = κh = 1e3 # Horizontal viscosity and diffusivity (m² s⁻¹)
νz = κh = 1e-2 # Vertical viscosity and diffusivity (m² s⁻¹)

minutes = 60.0
hours = 60minutes
days = 24hours
years = 365days

stop_time = 30days # Simulation stop time

struct EarthParameters <: AbstractEarthParameterSet end
Planet.grav(::EarthParameters) = 0.1
g = Planet.grav(EarthParameters())

# # The domain

domain = RectangularDomain(
    elements = (Nh, Nh, Nz),
    polynomialorder = 4,
    x = (0, L),
    y = (0, L),
    z = (-H, 0),
    periodicity = (true, false, false),
    array_type = CuArray,
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

uᵢ(x, y, z) = + α * sin(y / λ) * (z + H)
θᵢ(x, y, z) = + α * f * λ * cos(y / λ) + N² * z + α * f * L * 1e-3 * Ξ(z)
ηᵢ(x, y, z) = α * f * λ / (2 * g * H) * cos(y / λ)

initial_conditions = InitialConditions(u = uᵢ, θ = θᵢ, η = ηᵢ)

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = 2minutes,
    initial_conditions = initial_conditions,
    parameters = EarthParameters(),
    buoyancy = (αᵀ = 1 / g,),
    coriolis = (f₀ = f, β = 0),
    turbulence_closure = (νʰ = νh, κʰ = κh,
                          νᶻ = νz, κᶻ = νz),
    rusanov_wave_speeds = (cʰ = 1.0, cᶻ = 1e-3),
    boundary_conditions = (no_slip, free_surface),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

fetched_states = []

realdata = Array(model.state.realdata)
u = SpectralElementField(domain, view(realdata, :, 1, :))

volume = assemble(u)
x = volume.x[:, 1, 1]
y = volume.y[1, :, 1]

@assert issorted(x)
@assert issorted(y)

start_time = time_ns()

data_fetcher = EveryXSimulationTime(days / 10) do

    realdata = Array(model.state.realdata)
    u = SpectralElementField(domain, view(realdata, :, 1, :))
    v = SpectralElementField(domain, view(realdata, :, 2, :))
    η = SpectralElementField(domain, view(realdata, :, 3, :))
    θ = SpectralElementField(domain, view(realdata, :, 4, :))

    umax = maximum(abs, u)
    elapsed = (time_ns() - start_time) * 1e-9

    step = @sprintf("Step: %d", steps(model))
    sim_time = @sprintf("time: %.2f days", current_time(model) / days)
    wall_time = @sprintf("wall time: %.2f min", elapsed / 60)

    @info "$step, $sim_time, $wall_time"

    isnan(umax) && error("NaN'd out.")

    u_assembly = assemble(data.(u.elements))
    θ_assembly = assemble(data.(θ.elements))

    push!(
        fetched_states,
        (u = u_assembly, θ = θ_assembly, time = current_time(model)),
    )
end

# and then run the simulation.

model.solver_configuration.timeend = stop_time

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

    u = state.u[:, :, end]
    θ = state.θ[:, :, end]

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
        color = :balance,
        clim = (-θlim, θlim),
        kwargs...
    )

    u_title = @sprintf("u at t = %d days", state.time / days)
    θ_title = @sprintf("θ at t = %d days", state.time / days)

    plot(u_plot, θ_plot, title = [u_title θ_title], size = (1200, 500))
end

gif(animation, "eady_instability.gif", fps = 8)
