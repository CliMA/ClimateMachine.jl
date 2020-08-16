module Ocean

include("HydrostaticBoussinesq/HydrostaticBoussinesqModel.jl")
include("ShallowWater/ShallowWaterModel.jl")
include("SplitExplicit/SplitExplicitModel.jl")
include("SplitExplicit01/SplitExplicitModel.jl")
# include("OceanProblems/SimpleBoxProblem.jl")

end
