using Test
using StaticArrays
using ClimateMachine.VariableTemplates

@testset "VariableTemplates - GPU" begin
    include("test_complex_models_gpu.jl")
end
