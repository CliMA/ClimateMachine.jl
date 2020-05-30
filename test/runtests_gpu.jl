using Test, Pkg, CuArrays

ENV["JULIA_LOG_LEVEL"] = "WARN"

@test CuArrays.functional()

for submodule in [
    #"Common/MoistThermodynamics",
    #"Common/SurfaceFluxes",
    "Arrays",
    #"Numerics/Mesh",
    #"Numerics/DGMethods",
    "Numerics/ODESolvers",
]
    println("Starting tests for $submodule")
    t = @elapsed include(joinpath(submodule, "runtests.jl"))
    println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end
