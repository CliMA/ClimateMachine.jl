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

cmd = `julia --project $(joinpath(dirname(Base.active_project()), "test", "Atmos", "Parameterizations", "TurbulenceConvection", "BOMEX.jl"))`
run(cmd)
