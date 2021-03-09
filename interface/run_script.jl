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
# include("interface/domains.jl")
# include("interface/interface.jl")
# include("interface/abstractions.jl")
# include("interface/callbacks.jl")
include("domains.jl")
include("interface.jl")
include("abstractions.jl")
include("callbacks.jl")

# Main balance law and its components
# include("interface/test_model.jl") # umbrella model: TestEquations
include("test_model.jl") # umbrella model: TestEquations

# BL and problem for this
hyperdiffusion = HyperDiffusionCubedSphereProblem{FT}((;
                            τ = 8*60*60,
                            l = 7,
                            m = 4,
                            )...,
                        )
                     
# Domain
Ω = AtmosDomain(radius = planet_radius(param_set), height = 30e3)

# Grid
nelem = (;horizontal = 8, vertical = 4)
polynomialorder = (;horizontal = 5, vertical = 5)
grid = DiscontinuousSpectralElementGrid(Ω, nelem, polynomialorder)
dx = min_node_distance(grid, HorizontalDirection())

# Numerics-specific options
numerics = (; flux = CentralNumericalFluxFirstOrder() ) # add  , overintegration = 1

# Timestepping
Δt_ = Δt(hyperdiffusion, dx)
timestepper = TimeStepper(method = LSRK54CarpenterKennedy, timestep = Δt_ )
start_time, end_time = ( 0  , 2Δt_ )

# Callbacks (TODO)
callbacks = (
            #JLD2State((
            #    interation = 1,
            #    filepath ="output/tmp.jld2", 
            #    overwrite = true
            #    )...,),
            VTKOutput((
                interation = string(Δt_)*"ssecs" ,
                overdir ="output", 
                overwrite = true,
                number_sample_points = 0
                )...,),    
            )

# Specify RHS terms and any useful parameters
balance_law = TestEquations{FT}(
        Ω;
        advection = nothing, # adv 
        turbulence = nothing, # turb
        hyperdiffusion = hyperdiffusion, # hyper
        coriolis = nothing, # cori
        params = nothing,
        param_set = param_set,
    )

# Collect all spatial info of the model 
model = SpatialModel( 
    balance_law = balance_law,
    numerics = numerics,
    grid = grid,
    boundary_conditions = nothing,
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

# Run the model
@time solve!( 
    simulation.state, 
    simulation.odesolver; 
    timeend = end_time,
    callbacks = cbvector,
)
