using Test, Pkg

for submodule in ["Utilities/ParametersType",
                  "Utilities/PlanetParameters",
                  "Utilities/RootSolvers",
                  "Utilities/MoistThermodynamics",
                  "Atmos/Parameterizations/SurfaceFluxes",
                  "Atmos/Parameterizations/TurbulenceConvection",
                  "Mesh",
                  "DGmethods",
                  ]

  println("Testing $submodule")
  include(joinpath("../src",submodule,"test/runtests.jl"))
end
