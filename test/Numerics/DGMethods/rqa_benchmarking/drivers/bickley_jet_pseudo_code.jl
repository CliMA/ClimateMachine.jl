@boilerplate

########
# Set up parameters
########
parameters = (
    g  = 0.0,
    ρₒ = 1.0,  # reference density
    cₛ = 1e-2, # sound speed
    ℓᵐ = 10,   # jet thickness, (larger is thinner)
    ℓ  = 20,   # perturbation thickness, (larger is thinner)
    m  = 2,    # number of sign changes on equator for perturbation
    ϕᵖ = π/2 * 0.05, # of centerdness of perturbation
    ϵ  = 0.3,  # perturbation amplitude
    vˢ = 5e-4, # velocity scale
    α  = 2e-4,
    Ω  = 1e-3,
)

########
# Set up domain
########
domain = SphericalShell(radius = 1.0, height = 0.2)

########
# Set up model physics
########
physics = Physics(
    orientation = SphericalOrientation(),
    advection   = NonlinearAdvection{(:ρ, :ρu, :ρθ)}(),
    coriolis    = DeepShellCoriolis(),
    eos         = BarotropicFluid{(:ρ, :ρu)}(),
    parameters  = parameters,
)

########
# Set up inital condition
########
uᵐ(p, λ, ϕ, r) =  p.ℓᵐ * sech(p.ℓᵐ * ϕ)^2 
vᵐ(p, λ, ϕ, r) =  0.0
hᵐ(p, λ, ϕ, r) =  0.0

u1(p, λ, ϕ, r) =  p.ℓ * 2 * (ϕ - p.ϕᵖ)* exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(ϕ) * cos(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
u2(p, λ, ϕ, r) =  exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * sin(ϕ) * cos(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
u3(p, λ, ϕ, r) =  2*exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(ϕ) * sin(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
uᵖ(p, λ, ϕ, r) =  u1(p, λ, ϕ, r) + u2(p, λ, ϕ, r) + u3(p, λ, ϕ, r)
vᵖ(p, λ, ϕ, r) =  p.m * exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(2 * (ϕ - p.ϕᵖ)) * cos(p.m * λ)
hᵖ(p, λ, ϕ, r) =  0.0

ρ₀(p, λ, ϕ, r)    = p.ρₒ 
ρuʳᵃᵈ(p, λ, ϕ, r) = 0
ρuˡᵃᵗ(p, λ, ϕ, r) = p.vˢ * ρ₀(p, λ, ϕ, r) * (p.ϵ * vᵖ(p, λ, ϕ, r))
ρuˡᵒⁿ(p, λ, ϕ, r) = p.vˢ * ρ₀(p, λ, ϕ, r) * (uᵐ(p, λ, ϕ, r) + p.ϵ * uᵖ(p, λ, ϕ, r))
ρθ₀(p, λ, ϕ, r) = ρ₀(p, λ, ϕ, r) * tanh(p.ℓᵐ * ϕ)

########
# Set up boundary conditions
########
bcs = (
    bottom = (ρu = FreeSlip(), ρθ = NoFlux(), ρ = NoFlux()),
    top =    (ρu = FreeSlip(), ρθ = NoFlux(), ρ = NoFlux()),
)

########
# Set up model
########
model = ModelSetup(
    physics = physics,
    boundary_conditions = bcs,
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀),
)

########
# Set up time steppers
########
Δt          = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
start_time  = 0
end_time    = 8*1600.0
callbacks   = (
    Info(), 
    StateCheck(10), 
    VTKState(iteration = 160, filepath = "./out/"),
)

########
# Set up simulation
########
numerics_backend = (vertical = FiniteVolume(), horizontal = ContinuousGalerkin())
# backend = DiscontinuousGalerkin()
# backend = (vertical = DiscontinuousGalerkin(), horizontal = DiscontinuousGalerkin())
#=
aliases:
DG() = DiscontinuousGalerkin()
CG() = ContinuousGalerkin()
FV() = FiniteVolume()
FD() = FiniteDifferences()
WENO() = WeightedEssentiallyNonOscillatory()
OldBackend() = (vertical = DiscontinuousGalerkin(), horizontal = DiscontinuousGalerkin())
NewBackend() = (vertical = WENO(), horizontal = CG())
OceananigansBackend() = (vertical = FiniteVolume(), horizontal = FiniteVolume())
=#
simulation = Simulation(
    model;
    numerics = numerics_backend,
    grid = grid,
    timestepper = (method = SSPRK22Heuns, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
)

########
# Run the simulation
########
initialize!(simulation)
evolve!(simulation)