using Test

for submodule in [
                  "FiniteDifferenceGrids",
                  "DomainDecomp",
                  "StateVecs",
                  "StateVecFuncs",
                  "TDMA",
                  "PDEs",
                  # "BOMEX",
                  ]

  println("Testing $submodule")
  include(joinpath(submodule*".jl"))
end
