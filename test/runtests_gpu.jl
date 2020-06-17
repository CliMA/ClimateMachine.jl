using Test, Pkg, CUDA

ENV["JULIA_LOG_LEVEL"] = "WARN"

@test CUDA.functional()

for submodule in [
    #"Common/Thermodynamics",
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
