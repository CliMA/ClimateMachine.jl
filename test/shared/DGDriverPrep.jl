using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.StrongStabilityPreservingRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver
using CLIMA.SpaceMethods
using CLIMA.Vtk
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using Random
using Test
using DelimitedFiles
using Dierckx

@static if haspkg("CUDAnative")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  @assert VERSION >= v"1.2-pre.25"
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end


function print_norm!(starttime, TimeIntegrator, Q_norm)

  @info @sprintf("""Update
                 simtime = %.16e
                 runtime = %s
                 Δmass   = %.16e""",
                 ODESolvers.gettime(TimeIntegrator),
                 Dates.format(convert(Dates.DateTime,
                                      Dates.now()-starttime[]),
                              Dates.dateformat"HH:MM:SS"),
                 Q_norm)
end
