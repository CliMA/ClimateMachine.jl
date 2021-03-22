using ClimateMachine, MPI
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods

using ClimateMachine.ODESolvers

using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, Omega, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

ClimateMachine.init()
FT = Float64

# Shared functions
include("domains.jl")
include("interface.jl")
include("abstractions.jl")
include("callbacks.jl")

# Main balance law and its components
include("test_model.jl") # umbrella model: TestEquations

# Problems and relevant parameters to be included in the balance law
diffusion_params = (; D = 1, H = 1, l = 7, m = 4,) 

hyperdiffusion = HyperDiffusionCubedSphereProblem{FT}(diffusion_params...,)
diffusion = DiffusionCubedSphereProblem{FT}(diffusion_params...,)
advection = AdvectionCubedSphereProblem()

# Domain
Ω = AtmosGCMDomain(radius = FT(planet_radius(param_set)), height = FT(30e3))

# Grid
nelem = (;horizontal = 8, vertical = 3)
polynomialorder = (;horizontal = 3, vertical = 3)
grid = DiscontinuousSpectralElementGrid(Ω, nelem, polynomialorder)
dx = min_node_distance(grid, HorizontalDirection())

# Numerics-specific options
numerics = (; flux = CentralNumericalFluxFirstOrder() ) # add  , overintegration = 1

# Timestepping
u_rad = FT(2π / (nelem.horizontal) / (polynomialorder.horizontal+1) / 4 ) 

Δt_ = min( Δt(hyperdiffusion, dx) , Δt(diffusion, dx), Δt(advection, dx, u = u_rad * planet_radius(param_set) ) )
timestepper = TimeStepper(method = LSRK54CarpenterKennedy, timestep = Δt_ )
start_time, end_time = ( 0  , 2π / u_rad )

# Initial conditions
#   Earth Spherical Representation
#       longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
#       latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
#       radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of earth; hᵐⁱⁿ, hᵐᵃˣ ≥ 0
ρ₀(p, λ, ϕ, r) = calc_Ylm(ϕ, λ, diffusion_params.l, diffusion_params.m)

uʳᵃᵈ(p, λ, ϕ, r) = 0
uˡᵃᵗ(p, λ, ϕ, r) = 0
uˡᵒⁿ(p, λ, ϕ, r) = u_rad * r * cos(ϕ)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))

u⃗₀ᶜᵃʳᵗ(p, x...) = (   uʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                    + uˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                    + uˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 

aux = (u=u⃗₀ᶜᵃʳᵗ, uˡᵒⁿ=uˡᵒⁿ,)
init_state = (ρ=ρ₀ᶜᵃʳᵗ,)
initial_conditions = (state=init_state, aux=aux)
initial_problem = InitialValueProblem(diffusion_params, initial_conditions)

# Boundary conditions
ρu_bcs = ()
boundary_problem = BoundaryProblem()

# Callbacks (TODO)
callbacks = (
    #JLD2State((
    #    iteration = 1,
    #    filepath ="output/tmp.jld2",
    #    overwrite = true
    #    )...,),
    VTKOutput((
         iteration = string(100Δt_)*"ssecs" ,
         overdir ="output",
         overwrite = true,
         number_sample_points = 0
         )...,),
    )

# Specify RHS terms and any useful parameters
balance_law = TestEquations{FT}(
    Ω;
    advection = advection, #advection, 
    turbulence = nothing, # diffusion
    hyperdiffusion = nothing, #hyperdiffusion
    coriolis = nothing, # cori
    params = nothing,
    param_set = param_set,
    initial_value_problem = initial_problem,
    boundary_problem = boundary_problem,
)

# Collect all spatial info of the model
model = SpatialModel(
    balance_law = balance_law,
    numerics = numerics,
    grid = grid,
)

# Initialize simulation with time info
simulation = Simulation(
    model = model,
    timestepper = timestepper,
    callbacks = callbacks,
    simulation_time = (start_time, end_time),
    name = "HyperdiffusionUnitTest"
)

cbvector = create_callbacks(simulation, simulation.odesolver)

@time solve!(
            simulation.state,
            simulation.odesolver;
            timeend = end_time,
            callbacks = cbvector,
        )

errors = TestEquationsSimErrors(simulation)
# # err of Q
# Qend = simulation.state.ρ
# Qana = simulation.dgmodel.state_auxiliary.ρ_analytical

# errQ = norm(Qend.-Qana)/norm(Qend)

# # err of RHS
# rhsDG = similar(Qinit)
# simulation.dgmodel(rhsDG, Qinit, nothing, 0)
# DD = simulation.dgmodel.state_auxiliary.D 
# HH = simulation.dgmodel.state_auxiliary.H 
# cD = simulation.dgmodel.state_auxiliary.cD 
# cH = simulation.dgmodel.state_auxiliary.cH
# rhsAna = .- (DD.*cD .+ HH.*cH) .* Qinit.ρ
# errRHS = norm(rhsDG .- rhsAna)/norm(rhsDG)

#  TODO
# Next PR
# - put all spatial model & simulation info in one struct
# - add over-integration callback
# - Viz
# - add Maciek's h-d boundary condtions
# - cleanup
