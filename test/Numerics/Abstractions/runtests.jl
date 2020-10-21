using Test
@testset "CLIMA abstraction tests" begin
    include("impero_algebra_test.jl")
    include("impero_curl_test.jl")
    include("impero_grid_test.jl")
end
