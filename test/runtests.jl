using Test, Pkg

test_set = try
  parse(Int, ENV["TEST_SET"])
catch
  0
end

if test_set == 0 || test_set == 1
  for submodule in ["Utilities/ParametersType",
                    "Utilities/PlanetParameters",
                    "Utilities/RootSolvers",
                    "Utilities/MoistThermodynamics",
                    "Atmos/Parameterizations/SurfaceFluxes",
                    "Atmos/Parameterizations/TurbulenceConvection",
                    "Mesh",
                   ]

    println("Testing $submodule")
    include(joinpath("../src",submodule,"test/runtests.jl"))
  end
end

if test_set == 0 || test_set == 2
  for submodule in ["DGmethods",
                   ]

    println("Testing $submodule")
    include(joinpath("../src",submodule,"test/runtests.jl"))
  end
end
