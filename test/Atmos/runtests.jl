using Test, Pkg

@testset "Atmos" begin
    if isempty(ARGS) || "all" in ARGS
        all_tests = true
    else
        all_tests = false
    end
    for submodule in [
                      "Parameterizations",
                     ]

      if all_tests || "$submodule" in ARGS || "Atmos" in ARGS
        include_test(submodule)
      end
    end

end
