using Test

for submodule in [
                  "FiniteDifferenceGrids",
                  "DomainDecomp",
                  "StateVecs",
                  "TDMA",
                  "PDEs",
                  # "BOMEX",
                  ]

  println("Testing $submodule")
  include(joinpath(submodule*".jl"))
end
