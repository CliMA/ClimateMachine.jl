using Test, Pkg

for submodule in ["ParametersType",
                  "PlanetParameters",
                  "Utilities/RootSolvers",
                  "Utilities/MoistThermodynamics",
                  "ClimaAtmos/Parameterizations/SurfaceFluxes",
                  "ClimaAtmos/Dycore"]

  println("Testing $submodule")
  include(joinpath("../src",submodule,"test/runtests.jl"))
end
