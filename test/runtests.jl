using Test, Pkg

ENV["JULIA_LOG_LEVEL"] = "WARN"
test_set = parse(Int, get(ENV, "JULIA_CLIMA_TEST_SET", "0"))

if test_set == 0 || test_set == 1
  for submodule in ["Utilities/ParametersType",
                    "Utilities/PlanetParameters",
                    "Utilities/RootSolvers",
                    "Utilities/MoistThermodynamics",
                    "Atmos/Parameterizations/SurfaceFluxes",
                    "Atmos/Parameterizations/TurbulenceConvection",
                    "Mesh"
                   ]

    println("Testing $submodule")
    include(joinpath("../src",submodule,"test/runtests.jl"))
  end
end

if test_set == 0 || test_set == 2
  for submodule in ["DGmethods",
                    "Atmos/Dycore"
                   ]

    println("Testing $submodule")
    include(joinpath("../src",submodule,"test/runtests.jl"))
  end
end
