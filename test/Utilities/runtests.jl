using Test, Pkg

@testset "Utilities" begin
    if isempty(ARGS) || "all" in ARGS
        all_tests = true
    else
        all_tests = false
    end
    for submodule in [
                      "TicToc",
                      "VariableTemplates",
                      "ParametersType",
                      "RootSolvers",
                     ]

      if all_tests || "$submodule" in ARGS || "Utilities" in ARGS
        include_test(submodule)
      end
    end

end
