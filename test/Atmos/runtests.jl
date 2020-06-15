using Test, Pkg

@testset "Atmos" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["Parameterizations", "TemperatureProfiles"]
        if all_tests ||
           "$submodule" in ARGS ||
           "Atmos/$submodule" in ARGS ||
           "Atmos" in ARGS
            include_test(submodule)
        end
    end
end
