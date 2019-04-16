using Test

for submodule in ["Grids",
                  "StateVecs",
                  "StateVecFuncs",
                  ]

  println("Testing $submodule")
  include(joinpath("../test",submodule*".jl"))
end
