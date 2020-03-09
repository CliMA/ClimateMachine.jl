using Test, Pkg

@testset "Common" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in [
                      "MoistThermodynamics",
                      "PlanetParameters",
                     ]

      if all_tests || "$submodule" in ARGS || "Common" in ARGS
        include_test(submodule)
      end
    end

end
