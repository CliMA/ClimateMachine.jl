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

function main(::Type{FT}) where {FT}

    # BL and problem for this
    hyperdiffusion = HyperDiffusionCubedSphereProblem{FT}((;
                                τ = 8*60*60,
                                l = 7,
                                m = 4,
                                )...,
                            )

    diffusion = DiffusionCubedSphereProblem{FT}((;
                                τ = 8*60*60,
                                l = 7,
                                m = 4,
                                )...,
                            )

    advection = AdvectionCubedSphereProblem()
    
    # Domain
    Ω = AtmosDomain(radius = FT(planet_radius(param_set)), height = FT(30e3))

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
            advection = advection, # adv
            turbulence = diffusion, # turb
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
    return simulation, end_time, cbvector
end


function run(simulation, end_time, cbvector)
    # Run the model
    solve!(
        simulation.state,
        simulation.odesolver;
        timeend = end_time,
        callbacks = cbvector,
    )
end

simulation, end_time, cbvector = main(Float64);
println("Initialized. Running...")
@time run(simulation, end_time, cbvector)
