using Random
Random.seed!(8)
using StaticArrays: MArray

include("DryAtmos.jl")

let
    model = DryAtmosphereModel()
    FT = Float64
    num_state = number_state_conservative(model, FT)
    num_aux = number_state_auxiliary(model, FT)

    state_1 = [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
    state_2 = [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]

    aux_1 = rand(FT, num_aux)
    aux_2 = rand(FT, num_aux)

    H = MArray{Tuple{3, num_state}, FT}(undef)

    numerical_volume_fluctuation!(model, H, state_1, aux_1, state_2, aux_2)
end
