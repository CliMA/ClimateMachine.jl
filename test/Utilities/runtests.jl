using Test, Pkg

@testset "Utilities" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
<<<<<<< HEAD
    for submodule in ["TicToc", "VariableTemplates", "ParametersType"]
        if all_tests || "$submodule" in ARGS || "Utilities" in ARGS
=======
    for submodule in ["TicToc", "VariableTemplates", "SingleStackUtils"]
        if all_tests ||
           "$submodule" in ARGS ||
           "Utilities/$submodule" in ARGS ||
           "Utilities" in ARGS
>>>>>>> fc35827415a62e04fb36c4abf171fca2504c6c90
            include_test(submodule)
        end
    end

end
