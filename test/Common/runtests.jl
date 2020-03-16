using Test, Pkg

@testset "Common" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in [
                      "MoistThermodynamics",
                      "PlanetParameters",
                     ]

      if all_tests || "$submodule" in ARGS || "Common" in ARGS
          println("Starting tests for $submodule")
          t = @elapsed include(joinpath(submodule,"runtests.jl"))
          println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
      end
    end

end
