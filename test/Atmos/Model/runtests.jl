using Test

@testset "Atmos Model" begin
    include("ref_state.jl")
    include("./driver_spacing_test.jl")
end
