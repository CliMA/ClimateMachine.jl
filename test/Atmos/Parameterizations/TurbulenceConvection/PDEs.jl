using Pkg, Test

using CLIMA.TurbulenceConvection.FiniteDifferenceGrids
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.TriDiagSolvers
using CLIMA.TurbulenceConvection.haspkg

if haspkg.plots()
  using Plots
end

output_root = joinpath("output", "tests", "PDEs")

include("HeatEquation.jl")
include("AdvectionEquation.jl")
