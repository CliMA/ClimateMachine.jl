using Test, Pkg

@testset "InputOutput" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["Writers"]
        if all_tests || "$submodule" in ARGS || "Utilities" in ARGS
            include_test(submodule)
        end
    end

end
