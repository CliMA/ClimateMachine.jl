using Test
using ClimateMachine.BalanceLaws:
    number_state_conservative, number_state_auxiliary, number_state_entropy
using ClimateMachine.DGMethods.NumericalFluxes:
    numerical_volume_flux_first_order!
using StaticArrays: MArray
using Random
using DoubleFloats

Random.seed!(7)

include("DryAtmos.jl")

let
    model = DryAtmosModel()
    num_state = number_state_conservative(model)
    num_aux = number_state_auxiliary(model)
    num_entropy = number_state_entropy(model)

    @testset "state to entropy variable transforms" begin
        for FT in (Float32, Float64, Double64, BigFloat)
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

    @testset "test numerical flux for Tadmor shuffle" begin
        # Vars doesn't work with BigFloat so we will use Double64
        for FT in (Float32, Float64, Double64)
            # Create some random states
            state_1 =
                [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
            aux_1 = 0 * rand(FT, num_aux)

            state_2 =
                [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
            aux_2 = 0 * rand(FT, num_aux)

            # Get the entropy variables for the two states
            entropy_1 = similar(state_1, num_entropy)
            state_to_entropy_variables!(model, entropy_1, state_1, aux_1)

            entropy_2 = similar(state_1, num_entropy)
            state_to_entropy_variables!(model, entropy_2, state_2, aux_2)

            # Get the values of Ψ_j = β^T f_j - ζ_j = ρu_j where β is the
            # entropy variables, f_j is the conservative flux, and ζ_j is the
            # entropy flux. For conservation laws this is the entropy potential.
            Ψ_1 = Vars{vars_state_conservative(model, FT)}(state_1).ρu
            Ψ_2 = Vars{vars_state_conservative(model, FT)}(state_2).ρu

            # Evaluate the flux with both orders of the two states
            H_12 = fill!(MArray{Tuple{3, num_state}, FT}(undef), -zero(FT))
            numerical_volume_flux_first_order!(
                EntropyConservative(),
                model,
                H_12,
                state_1,
                aux_1,
                state_2,
                aux_2,
            )

            H_21 = fill!(MArray{Tuple{3, num_state}, FT}(undef), -zero(FT))
            numerical_volume_flux_first_order!(
                EntropyConservative(),
                model,
                H_21,
                state_2,
                aux_2,
                state_1,
                aux_1,
            )

            # Check that we satisfy the Tadmor shuffle
            @test all(
                H_12 * entropy_1[1:num_state] - H_21 * entropy_2[1:num_state] .≈
                Ψ_1 - Ψ_2,
            )
        end
    end
end
