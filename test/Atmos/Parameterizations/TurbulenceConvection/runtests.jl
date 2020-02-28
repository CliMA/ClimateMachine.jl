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

@testset "Separate session BOMEX" begin
  f = joinpath(dirname(Base.active_project()), "test", "Atmos", "Parameterizations", "TurbulenceConvection", "BOMEX.jl")
  cmd = `$(Base.julia_cmd()) --project $f`
  run(cmd)
end
