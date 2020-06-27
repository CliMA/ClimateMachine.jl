using Test
using ClimateMachine.BalanceLaws:
    number_state_conservative, number_state_auxiliary, number_state_entropy
using Random
Random.seed!(8)

include("DryAtmos.jl")

let
    model = DryAtmosModel()
    num_state = number_state_conservative(model)
    num_aux = number_state_auxiliary(model)
    num_entropy = number_state_entropy(model)

    @testset "state to entropy variable transforms" begin
        for FT in (BigFloat, Float32, Float64)
            state_in =
                [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
            aux_in = rand(FT, num_aux)
            state_out = similar(state_in)
            aux_out = similar(aux_in)
            entropy = similar(state_in, num_entropy)

            state_to_entropy_variables!(model, entropy, state_in, aux_in)
            entropy_variables_to_state!(model, state_out, aux_out, entropy)

            @test all(state_in .≈ state_out)
            @test all(aux_in .≈ aux_out)
        end
    end
end
