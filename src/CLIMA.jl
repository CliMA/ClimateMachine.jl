module CLIMA

include("Utilities/ParametersType/ParametersType.jl")
include("Utilities/PlanetParameters/PlanetParameters.jl")
include("Utilities/RootSolvers/RootSolvers.jl")
include("Utilities/MoistThermodynamics/MoistThermodynamics.jl")
include("Atmos/Parameterizations/SurfaceFluxes/SurfaceFluxes.jl")
include("Atmos/Parameterizations/TurbulenceConvection/TurbulenceConvection.jl")
include("Mesh/Topologies.jl")
include("Mesh/Grids.jl")
include("Arrays/MPIStateArrays.jl")
include("DGmethods/SpaceMethods.jl")
include("DGmethods/DGBalanceLawDiscretizations.jl")
include("ODESolvers/ODESolvers.jl")
include("ODESolvers/LowStorageRungeKuttaMethod.jl")
include("ODESolvers/StrongStabilityPreservingRungeKuttaMethod.jl")
include("ODESolvers/AdditiveRungeKuttaMethod.jl")
include("ODESolvers/GenericCallbacks.jl")
include("InputOutput/Vtk/Vtk.jl")
include("misc.jl")

include("Mesh/Topography.jl")

end
