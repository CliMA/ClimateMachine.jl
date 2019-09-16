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
                        init_aux!, init_state!, init_ode_param, init_ode_state,
                        LocalGeometry


struct IntegralTestModel{dim} <: BalanceLaw
end

vars_integrals(::IntegralTestModel, T) = @vars(a::T,b::T)
vars_aux(m::IntegralTestModel,T) = @vars(int::vars_integrals(m,T),
                                         rev_int::vars_integrals(m,T),
                                         coord::SVector{3,T}, a::T, b::T)

vars_state(::IntegralTestModel, T) = @vars()
vars_diffusive(::IntegralTestModel, T) = @vars()

flux_nondiffusive!(::IntegralTestModel, _...) = nothing
flux_diffusive!(::IntegralTestModel, _...) = nothing
source!(::IntegralTestModel, _...) = nothing
boundary_state!(_, ::IntegralTestModel, _...) = nothing
init_state!(::IntegralTestModel, _...) = nothing
wavespeed(::IntegralTestModel,_...) = 1

function init_aux!(::IntegralTestModel{dim}, aux::Vars,
                   g::LocalGeometry) where {dim}
  x,y,z = aux.coord = g.coord
  if dim == 2
    aux.a = x*y + z*y
    aux.b = 2*x*y + sin(x)*y^2/2 - (z-1)^2*y^3/3
  else
    aux.a = x*z + z^2/2
    aux.b = 2*x*z + sin(x)*y*z - (1+(z-1)^3)*y^2/3
  end
end

function update_aux!(dg::DGModel, m::IntegralTestModel, Q::MPIStateArray,
                     auxstate::MPIStateArray, t::Real)
  indefinite_stack_integral!(dg, m, Q, auxstate, t)
  reverse_indefinite_stack_integral!(dg, m, Q, auxstate, t)
end

@inline function integrate_aux!(m::IntegralTestModel, integrand::Vars,
                                state::Vars, aux::Vars)
  x,y,z = aux.coord
  integrand.a = x + z
  integrand.b = 2*x + sin(x)*y - (z-1)^2*y^2
end



using Test
function run(mpicomm, dim, ArrayType, Ne, N, DFloat)

  brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=3), dim)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  dg = DGModel(IntegralTestModel{dim}(),
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)
  Q = init_ode_state(dg, param, DFloat(0))
  dQdt = similar(Q)

  dg(dQdt, Q, param, 0.0)

  # Wrapping in Array ensure both GPU and CPU code use same approx
  @test Array(param.aux.Q[:, 1, :]) ≈ Array(param.aux.Q[:, 8, :])
  @test Array(param.aux.Q[:, 2, :]) ≈ Array(param.aux.Q[:, 9, :])
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

  numelem = (5, 5, 5)
  lvls = 1

  polynomialorder = 4

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      for dim = 2:3
        err = zeros(DFloat, lvls)
        for l = 1:lvls
          @info (ArrayType, DFloat, dim)
          run(mpicomm, dim, ArrayType, ntuple(j->2^(l-1) * numelem[j], dim),
              polynomialorder, DFloat)
        end
      end
    end
  end
end

nothing
