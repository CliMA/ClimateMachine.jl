using Test, Pkg

@testset "Parameterizations" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in [
                      "Microphysics",
                      "SurfaceFluxes",
                      "TurbulenceConvection",
                     ]

      if all_tests || "$submodule" in ARGS || "Parameterizations" in ARGS || "Atmos" in ARGS
        println("Starting tests for $submodule")
        t = @elapsed include(joinpath(submodule,"runtests.jl"))
        println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
      end
    end

end
