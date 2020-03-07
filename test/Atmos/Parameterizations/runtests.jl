using Test, Pkg

@testset "Parameterizations" begin
    if isempty(ARGS) || "all" in ARGS
        all_tests = true
    else
        all_tests = false
    end
    for submodule in [
                      "Microphysics",
                      "SurfaceFluxes",
                      "TurbulenceConvection",
                     ]

      if all_tests || "$submodule" in ARGS || "Parameterizations" in ARGS || "Atmos" in ARGS
        include_test(submodule)
      end
    end

end
