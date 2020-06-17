using Test, Pkg

@testset "Land" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["Model"]
        if all_tests ||
           "$submodule" in ARGS ||
           "Land/$submodule" in ARGS ||
           "Land" in ARGS
            include_test(submodule)
        end
    end
end
