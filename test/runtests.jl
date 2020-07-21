using Test, Pkg

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
ENV["JULIA_LOG_LEVEL"] = "WARN"

function include_test(_module)
    println("Starting tests for $_module")
    t = @elapsed include(joinpath(_module, "runtests.jl"))
    println("Completed tests for $_module, $(round(Int, t)) seconds elapsed")
    return nothing
end


@testset "ClimateMachine" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false

    function has_submodule(sm)
        any(ARGS) do a
            a == sm && return true
            first(split(a, '/')) == sm && return true
            return false
        end
    end

    for submodule in [
        "InputOutput",
        "Utilities",
        "Common",
        "Arrays",
        "Atmos",
        "Land",
        "Numerics",
        "Diagnostics",
        "Ocean",
        "Driver",
    ]
        if all_tests || has_submodule(submodule) || "ClimateMachine" in ARGS
            include_test(submodule)
        end
    end
end
