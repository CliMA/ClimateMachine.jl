using Test
using StaticArrays
using ClimateMachine.VariableTemplates

@testset "VariableTemplates" begin
    include("test_base_functionality.jl")
    include("varsindex.jl")
    include("test_complex_models.jl")
end
