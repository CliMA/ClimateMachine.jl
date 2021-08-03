using Test


@testset "PySDMCall tests" begin
    include(joinpath("PySDMCallback_invocation.jl"))
    include(joinpath("PyCall_invocation.jl"))
    include(joinpath("PySDM_presence.jl"))
end