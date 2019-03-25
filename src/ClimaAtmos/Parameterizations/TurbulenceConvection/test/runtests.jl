using Test

for submodule in ["Grids",
                  "GridOperators",
                  "StateVecs",
                  ]

  println("Testing $submodule")
  include(joinpath("../src",submodule,"runtests.jl"))
end
