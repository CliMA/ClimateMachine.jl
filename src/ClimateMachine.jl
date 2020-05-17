module ClimateMachine

using Pkg.TOML

const CLIMATEMACHINE_VERSION =
    VersionNumber(TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))["version"])

include(joinpath("Utilities", "TicToc", "TicToc.jl"))
include(joinpath("Utilities", "ArtifactWrappers", "ArtifactWrappers.jl"))
include(joinpath("InputOutput", "Writers", "Writers.jl"))
include(joinpath("Common", "ConfigTypes", "ConfigTypes.jl"))
include(joinpath("Utilities", "VariableTemplates", "VariableTemplates.jl"))
include(joinpath("Common", "MoistThermodynamics", "MoistThermodynamics.jl"))
include(joinpath(
    "Atmos",
    "Parameterizations",
    "CloudPhysics",
    "Microphysics.jl",
))
include(joinpath("Common", "SurfaceFluxes", "SurfaceFluxes.jl"))
include(joinpath("Arrays", "MPIStateArrays.jl"))
include(joinpath("Numerics", "Mesh", "Mesh.jl"))
include(joinpath("Numerics", "DGmethods", "Courant.jl"))
include(joinpath("Numerics", "DGmethods", "DGmethods.jl"))
include(joinpath("Utilities", "SingleStackUtils", "SingleStackUtils.jl"))
include(joinpath("Ocean", "ShallowWater", "ShallowWaterModel.jl"))
include(joinpath(
    "Ocean",
    "HydrostaticBoussinesq",
    "HydrostaticBoussinesqModel.jl",
))
include(joinpath("Numerics", "LinearSolvers", "LinearSolvers.jl"))
include(joinpath(
    "Numerics",
    "LinearSolvers",
    "GeneralizedConjugateResidualSolver.jl",
))
include(joinpath(
    "Numerics",
    "LinearSolvers",
    "GeneralizedMinimalResidualSolver.jl",
))
include(joinpath("Numerics", "LinearSolvers", "ColumnwiseLUSolver.jl"))
include(joinpath("Numerics", "LinearSolvers", "ConjugateGradientSolver.jl"))
include(joinpath(
    "Numerics",
    "LinearSolvers",
    "BatchedGeneralizedMinimalResidualSolver.jl",
))
include(joinpath("Numerics", "ODESolvers", "ODESolvers.jl"))
include(joinpath("Numerics", "ODESolvers", "GenericCallbacks.jl"))
include(joinpath("Atmos", "Model", "AtmosModel.jl"))
include(joinpath("InputOutput", "VTK", "VTK.jl"))
include(joinpath("Diagnostics", "Diagnostics.jl"))
include(joinpath("Utilities", "Checkpoint", "Checkpoint.jl"))
include(joinpath("Utilities", "Callbacks", "Callbacks.jl"))
include(joinpath("Driver", "Driver.jl"))

end
