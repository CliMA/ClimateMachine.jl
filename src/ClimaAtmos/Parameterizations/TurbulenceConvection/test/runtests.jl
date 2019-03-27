using Test

for submodule in ["Grids",
                  "StateVecs",
                  ]

  println("Testing $submodule")
  include(joinpath("../src",submodule,"runtests.jl"))
end
