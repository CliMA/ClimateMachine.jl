using Test


@testset "PySDMCall" begin
    include(joinpath("test1.jl"))
    include(joinpath("test2.jl"))
    include(joinpath("test3.jl"))
end