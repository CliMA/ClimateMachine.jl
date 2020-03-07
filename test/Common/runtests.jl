using Test, Pkg

@testset "Common" begin
    if isempty(ARGS) || "all" in ARGS
        all_tests = true
    else
        all_tests = false
    end
    for submodule in [
                      "MoistThermodynamics",
                      "PlanetParameters",
                     ]

      if all_tests || "$submodule" in ARGS || "Common" in ARGS
        include_test(submodule)
      end
    end

end
