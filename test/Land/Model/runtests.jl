using Test, Pkg
@testset "Land" begin
    include("test_water_parameterizations.jl")
    include("prescribed_twice.jl")
end
