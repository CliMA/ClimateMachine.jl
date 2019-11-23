using Pkg, Test

using CLIMA.TurbulenceConvection.FiniteDifferenceGrids
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.TriDiagSolvers

@static if haskey(Pkg.installed(), "Plots")
  using Plots
  export_plots = true
else
  export_plots = false
end

output_root = joinpath("output", "tests", "PDEs")

include("HeatEquation.jl")
include("AdvectionEquation.jl")
