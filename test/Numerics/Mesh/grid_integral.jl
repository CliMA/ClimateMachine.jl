using Test
using ClimateMachine

let
    for N in 1:10
        for FloatType in (Float64, Float32)
            (ξ, ω) = ClimateMachine.Mesh.Elements.lglpoints(FloatType, N)
            I∫ =
                ClimateMachine.Mesh.Grids.indefinite_integral_interpolation_matrix(
                    ξ,
                    ω,
                )
            for n in 1:N
                if N == 1
                    @test sum(abs.(I∫ * ξ)) < 10 * eps(FloatType)
                else
                    @test I∫ * ξ .^ n ≈
                          (ξ .^ (n + 1) .- (-1) .^ (n + 1)) / (n + 1)
                end
            end
        end
    end
end
