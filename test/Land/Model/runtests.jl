using Test, Pkg
@testset "Land" begin
    include("test_water_parameterizations.jl")
    #    include("haverkamp_test.jl")
    #    include("test_bc.jl")
    include("prescribed_twice.jl")
end
