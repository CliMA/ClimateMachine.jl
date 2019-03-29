using Test

for submodule in ["Grids",
                  "StateVecs",
                  ]

  println("Testing $submodule")
  include(joinpath("../test",submodule*".jl"))
end
