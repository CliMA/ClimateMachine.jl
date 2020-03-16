using Test

for submodule in [
                  "FiniteDifferenceGrids",
                  "DomainDecomp",
                  "StateVecs",
                  "TDMA",
                  "PDEs",
                  # "BOMEX",
                  ]

    println("Starting tests for $submodule")
    t = @elapsed include("$(submodule).jl")
    println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")

end
