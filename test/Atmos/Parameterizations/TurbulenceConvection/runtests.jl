using Test

for submodule in [
                  "FiniteDifferenceGrids",
                  "DomainDecomp",
                  "StateVecs",
                  "TDMA",
                  "PDEs",
                  # "BOMEX",
                  ]

  include_test(submodule)

end
