using Test

for submodule in ["Grids",
                  "GridOperators",
                  ]

  println("Testing $submodule")
  include(joinpath("../src",submodule,"runtests.jl"))
end
