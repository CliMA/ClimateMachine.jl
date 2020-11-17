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
using ClimateMachine.Ocean: current_step, Δt, current_time
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
# ``N² z`` reflects a stable stratification with buoyancy
# frequency ``N``. We thus have
#
# ```math
# b = α λ cos(y / λ) + N² z
# ```
# 
# The baroclinic component of ``u`` is the "thermal wind"
# associated with ``b``. The barotropic component of ``u``,
# ``⟨u⟩ = \frac{1}{H} ∫ u \rm{d} z``, however, is 
# geostrophically-balanced with the sea surface displacement
# via the north-south (``y``) momentum balance,
# 
# ```math
# f ⟨u⟩ = - g ∂_y η \, ,
# ```
#
# where ``g`` is gravitational acceleration. Using
# ``⟨u⟩ = α H sin(y / λ) / 2``, we obtain
#
# ```math
# η = α f λ H / (2 g) cos(y / λ) \, .
# ```
#
# We add a periodic barotropic perturbation to
# this unstable state to simulate the growth of 
# instability. The periodic perturbation has the 
# streamfunction
#
# ```math
# ψ̃(x, y) = \cos(k x) \cos(ℓ y) \exp((y - y_0)^2 / 2L̃^2) \, ,
# ```
#
# where ``k = 12π / L``, ``ℓ = 6π / L``, ``y_0 = L/2``, and
# ``L̃ = 5 \times 10^4`` m. The velocity field and height
# perturbation associated with this streamfunction is
#
# ```math
# \begin{gather}
# ũ = - ∂_y ψ̃ = (ℓ \tan(ℓ y) + (y - y₀) / L̃^2 ) ψ̃
# ṽ = + ∂_x ψ̃ = k \tan(k x) ψ̃
# η = + f ψ̃  / g
# \end{gather}
#```
#
# We set the amplitude of the perturbation to 1% of the 
# basic state.
#
# # Parameters
#
# We now choose parameters for the Eady instability problem.

Nh = 128
Nz = 16
Np = 2

L = 1e6 # Domain width (m)
H = 1e3 # Domain height (m)
f = 1e-4 # Coriolis parameter (s⁻¹)
α = 2f # Geostrophic shear (s⁻¹)
N² = 1e-5 # Initial buoyancy gradient (s⁻²)
νh = κh = 1e3 # Horizontal viscosity and diffusivity (m² s⁻¹)
νz = κh = 1e-2 # Vertical viscosity and diffusivity (m² s⁻¹)

minutes = 60.0
hours = 60minutes
days = 24hours
years = 365days

stop_time = 60days # Simulation stop time

struct EarthParameters <: AbstractEarthParameterSet end
Planet.grav(::EarthParameters) = 0.1
g = Planet.grav(EarthParameters())

# # The domain

domain = RectangularDomain(
    Ne = (Int(Nh / Np), Int(Nh / Np), Int(Nz / Np)),
    Np = Np,
    x = (0, L),
    y = (0, L),
    z = (-H, 0),
    periodicity = (true, false, false),
)

# # Boundary conditions

no_slip = OceanBC(Impenetrable(NoSlip()), Insulating())
free_surface = OceanBC(Penetrable(FreeSlip()), Insulating())

# # Initial conditions
#
# We idealize θ as buoyancy. We initialize the flow with a
# vertically-sheared, horizontally-sinusoidal jet in geostrophic
# balance. We add a small amount of noise to the buoyancy field.

## Geostrophic jet:
λ = L / π # Sinusoidal jet width (m)

U(y, z) = + α * sin(y / λ) * (z + H)
B(y, z) = + α * f * λ * cos(y / λ) + N² * z
h(y, z) = α * f * λ * H / 2g * cos(y / λ)

# Geostrophic mode-3 perturbation:
width = 5e4
kx = 12π / L
ky = 6π / L
y₀ = L / 2

ψ̃(x, y) = cos(kx * x) * cos(ky * y) * exp(-(y - y₀)^2 / 2width^2)

# ũ = - ∂y ψ̃
ũ(x, y) = (ky * tan(ky * y) + (y - y₀)^2 / width^2) * ψ̃(x, y)

# ṽ = ∂x ψ̃
ṽ(x, y) = - kx * tan(kx * x) * ψ̃(x, y)

a = 0.01
uᵢ(x, y, z) = U(y, z) + a * ũ(x, y)
vᵢ(x, y, z) = a * ṽ(x, y)
ηᵢ(x, y, z) = h(y, z) + a * f / g * ψ̃(x, y)
θᵢ(x, y, z) = B(y, z)

initial_conditions = InitialConditions(u = uᵢ, v = vᵢ, θ = θᵢ, η = ηᵢ)

minutes = 60.0

model = Ocean.HydrostaticBoussinesqSuperModel(
    domain = domain,
    time_step = 5minutes,
    initial_conditions = initial_conditions,
    parameters = EarthParameters(),
    # array_type = CuArray,
    buoyancy = (αᵀ = 1 / g,),
    coriolis = (f₀ = f, β = 0),
    turbulence_closure = (νʰ = νh, κʰ = κh,
                          νᶻ = νz, κᶻ = νz),
    rusanov_wave_speeds = (cʰ = sqrt(g*H), cᶻ = 1e-2),
    boundary_tags = ((0, 0), (1, 1), (1, 2)),
    boundary_conditions = (no_slip, free_surface),
)

# We prepare a callback that periodically fetches the horizontal velocity and
# tracer concentration for later animation,

fetched_states = []

realdata = Array(model.state.realdata)
u = SpectralElementField(domain, model.grid, view(realdata, :, 1, :))

volume = assemble(u)
x = volume.x[:, 1, 1]
y = volume.y[1, :, 1]

start_time = time_ns()

data_fetcher = EveryXSimulationTime(days) do

    realdata = Array(model.state.realdata)
    u = SpectralElementField(domain, model.grid, view(realdata, :, 1, :))
    v = SpectralElementField(domain, model.grid, view(realdata, :, 2, :))
    η = SpectralElementField(domain, model.grid, view(realdata, :, 3, :))
    θ = SpectralElementField(domain, model.grid, view(realdata, :, 4, :))

    umax = maximum(abs, u)
    elapsed = (time_ns() - start_time) * 1e-9

    step = @sprintf("Step: %d", current_step(model))
    sim_time = @sprintf("time: %.2f days", current_time(model) / days)
    wall_time = @sprintf("wall time: %.2f min", elapsed / 60)

    @info "$step, $sim_time, $wall_time"

    isnan(umax) && error("NaN'd out.")

    u_assembly = assemble(data.(u.elements))
    v_assembly = assemble(data.(v.elements))
    θ_assembly = assemble(data.(θ.elements))

    push!(
        fetched_states,
        (u = u_assembly,
         v = v_assembly,
         θ = θ_assembly,
         time = current_time(model)),
    )
end

# and then run the simulation.

model.solver_configuration.timeend = stop_time

try
    result = ClimateMachine.invoke!(
        model.solver_configuration;
        user_callbacks = [data_fetcher],
    )
catch err
    @warn "Simulation stopped early because " print(showerror, err)
end

# Finally, we make an animation of the evolving shear instability.

animation = @animate for (i, state) in enumerate(fetched_states)

    local u
    local v
    local θ

    @info "Plotting frame $i of $(length(fetched_states))..."

    kwargs = (xlim = domain.x, ylim = domain.y, linewidth = 0, aspectratio = 1,
              xlabel = "x (m)", ylabel = "y (m)")

    u = state.u[:, :, end]
    v = state.v[:, :, end]
    θ = state.θ[:, :, end]

    ulim = 1.2 * α * H
    θmin = minimum(θ)
    θmax = maximum(θ)

    ulevels = range(-ulim, ulim, length=31)
    θlevels = range(θmin, θmax, length=31)

    u_plot = heatmap(
        x,
        y,
        clamp.(u, -ulim, ulim)';
        #levels = ulevels,
        color = :balance,
        clim = (-ulim, ulim),
        kwargs...
    )

    v_plot = contourf(
        x,
        y,
        clamp.(v, -ulim, ulim)';
        #levels = ulevels,
        color = :balance,
        clim = (-ulim, ulim),
        kwargs...
    )

    θ_plot = heatmap(
        x,
        y,
        clamp.(θ, θmin, θmax)';
        #levels = θlevels,
        color = :thermal,
        clim = (θmin, θmax),
        kwargs...
    )

    u_title = @sprintf("surface u (m s⁻¹) at t = %.2f days", state.time / days)
    v_title = @sprintf("surface v (m s⁻¹) at t = %.2f days", state.time / days)
    θ_title = @sprintf("surface b (m s⁻²) at t = %.2f days", state.time / days)

    plot(u_plot, v_plot, θ_plot, title = [u_title v_title θ_title],
         layout=(1, 3), size = (1500, 400))
end

name = @sprintf("eady_dg_Np%d_Nh%d_Nz%d_αf%.2e_ν%.2e.gif", Np, Nh, Nz, α/f, νh)

gif(animation, name, fps = 8)
