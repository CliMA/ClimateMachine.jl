using Test

for submodule in ["Grids",
                  ]

  println("Testing $submodule")
  include(joinpath("../src",submodule,"runtests.jl"))
end
