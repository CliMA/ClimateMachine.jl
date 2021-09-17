using Test, Pkg

@testset "Common" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["CartesianDomains", "CartesianFields"]
        if all_tests ||
           "$submodule" in ARGS ||
           "Common/$submodule" in ARGS ||
           "Common" in ARGS
            include_test(submodule)
        end
    end

end
