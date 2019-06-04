using Test
using CLIMA
using Canary

let
  for N = 1:10
    for FloatType in (Float64, Float32)
      (ξ, ω) = Canary.lglpoints(FloatType, N)
      I∫ = CLIMA.Grids.indefinite_integral_interpolation_matrix(ξ, ω)
      for n = 1:N
        if N == 1
          @test sum(abs.(I∫ * ξ)) < 10*eps(FloatType)
        else
          @test I∫ * ξ.^n ≈ (ξ.^(n+1) .- (-1).^(n+1)) / (n+1)
        end
      end
    end
  end
end

