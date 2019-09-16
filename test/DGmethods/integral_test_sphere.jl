using MPI
using StaticArrays
using CLIMA
using CLIMA.VariableTemplates
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.MPIStateArrays
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using Printf
using LinearAlgebra
using Logging
using GPUifyLoops

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, vars_integrals, integrate_aux!,
                        flux_nondiffusive!, flux_diffusive!, source!, wavespeed,
                        update_aux!, indefinite_stack_integral!,
                        reverse_indefinite_stack_integral!,  boundary_state!,
                        gradvariables!, init_aux!, init_state!, init_ode_param,
                        init_ode_state, LocalGeometry


struct IntegralTestSphereModel{T} <: BalanceLaw
  Rinner::T
  Router::T
end

function update_aux!(dg::DGModel, m::IntegralTestSphereModel, Q::MPIStateArray,
                     auxstate::MPIStateArray, t::Real)
  indefinite_stack_integral!(dg, m, Q, auxstate, t)
  reverse_indefinite_stack_integral!(dg, m, Q, auxstate, t)
end

vars_integrals(::IntegralTestSphereModel, T) = @vars(v::T)
vars_aux(m::IntegralTestSphereModel,T) = @vars(int::vars_integrals(m,T), rev_int::vars_integrals(m,T), r::T, θ::T, ϕ::T)

vars_state(::IntegralTestSphereModel, T) = @vars()
vars_diffusive(::IntegralTestSphereModel, T) = @vars()

flux_nondiffusive!(::IntegralTestSphereModel, _...) = nothing
flux_diffusive!(::IntegralTestSphereModel, _...) = nothing
source!(::IntegralTestSphereModel, _...) = nothing
boundary_state!(_, ::IntegralTestSphereModel, _...) = nothing
init_state!(::IntegralTestSphereModel, _...) = nothing
wavespeed(::IntegralTestSphereModel,_...) = 1

function init_aux!(m::IntegralTestSphereModel, aux::Vars, g::LocalGeometry)

  x,y,z = g.coord
  aux.r = hypot(x, y, z)
  aux.θ = atan(y , x)
  aux.ϕ = asin(z / aux.r)

  # Exact integral
  a = 1 + sin(aux.θ)^2 + sin(aux.ϕ)^2
  aux.int.v = exp(-a * aux.r^2) - exp(-a * m.Rinner^2)
  aux.rev_int.v = exp(-a * m.Router^2) - exp(-a * aux.r^2)
end

@inline function integrate_aux!(m::IntegralTestSphereModel, integrand::Vars, state::Vars, aux::Vars)
  a = 1 + sin(aux.θ)^2 + sin(aux.ϕ)^2
  integrand.v = -2aux.r * a * exp(-a * aux.r^2)
end



if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

using Test
function run(mpicomm, topl, ArrayType, N, DFloat, Rinner, Router)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp,
                                         )
  dg = DGModel(IntegralTestSphereModel(Rinner, Router),
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)
  Q = init_ode_state(dg, param, DFloat(0))
  dQdt = similar(Q)

  exact_aux = copy(param.aux)

  dg(dQdt, Q, param, 0.0)
  
  euclidean_distance(exact_aux, param.aux)
end

let
  MPI.Initialized() || MPI.Init()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  polynomialorder = 4

  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1//2
  Router = 1

  polynomialorder = 4

  expected_result = [6.228615762850257e-7
                     9.671308320438864e-9
                     1.5102832678375277e-10
                     2.359860999112363e-12]

  lvls = integration_testing ? length(expected_result) : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      err = zeros(DFloat, lvls)
      for l = 1:lvls
        @info (ArrayType, DFloat, "sphere", l)
        Nhorz = 2^(l-1) * base_Nhorz
        Nvert = 2^(l-1) * base_Nvert
        Rrange = range(DFloat(Rinner); length=Nvert+1, stop=Router)
        topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
        err[l] = run(mpicomm, topl, ArrayType, polynomialorder, DFloat,
                     DFloat(Rinner), DFloat(Router))
        @test expected_result[l] ≈ err[l]
      end
      if integration_testing
        @info begin
          msg = ""
          for l = 1:lvls-1
            rate = log2(err[l]) - log2(err[l+1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
      end
    end
  end
end

nothing

