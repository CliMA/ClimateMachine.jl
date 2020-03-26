using Test, Pkg, CuArrays

ENV["JULIA_LOG_LEVEL"] = "WARN"

@test CuArrays.functional()

for submodule in [#"Utilities/ParametersType",
    #"Common/PlanetParameters",
    #"Common/MoistThermodynamics",
    #"Atmos/Parameterizations/SurfaceFluxes",
    #"Atmos/Parameterizations/TurbulenceConvection",
    #"Mesh",
    #"DGmethods",
    "ODESolvers",
    "Arrays",
]
    println("Starting tests for $submodule")
    t = @elapsed include(joinpath(submodule, "runtests.jl"))
    println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end
