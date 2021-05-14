using Test, Pkg

@testset "InputOutput" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["VTK", "Writers"]
        if all_tests ||
           "$submodule" in ARGS ||
           "InputOutput/$submodule" in ARGS ||
           "InputOutput" in ARGS
            include_test(submodule)
        end
    end

end
