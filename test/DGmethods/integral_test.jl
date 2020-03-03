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

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux_nondiffusive!, flux_diffusive!,
                        source!, wavespeed, LocalGeometry, boundary_state!,
                        init_aux!, init_state!, init_ode_state, update_aux!,
                        vars_integrals, vars_reverse_integrals,
                        indefinite_stack_integral!,
                        reverse_indefinite_stack_integral!,
                        integral_load_aux!, integral_set_aux!,
                        reverse_integral_load_aux!,
                        reverse_integral_set_aux!


struct IntegralTestModel{dim} <: BalanceLaw
end

vars_reverse_integrals(::IntegralTestModel, T) = @vars(a::T,b::T)
vars_integrals(::IntegralTestModel, T) = @vars(a::T,b::T)
vars_aux(m::IntegralTestModel,T) = @vars(int::vars_integrals(m,T),
                                         rev_int::vars_reverse_integrals(m,T),
                                         coord::SVector{3,T}, a::T, b::T,
                                         rev_a::T, rev_b::T)

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
    y_top = 3
    a_top = x*y_top + z*y_top
    b_top = 2*x*y_top + sin(x)*y_top^2/2 - (z-1)^2*y_top^3/3
    aux.rev_a = a_top - aux.a
    aux.rev_b = b_top - aux.b
  else
    aux.a = x*z + z^2/2
    aux.b = 2*x*z + sin(x)*y*z - (1+(z-1)^3)*y^2/3
    zz_top = 3
    a_top = x*zz_top + zz_top^2/2
    b_top = 2*x*zz_top + sin(x)*y*zz_top - (1+(zz_top-1)^3)*y^2/3
    aux.rev_a = a_top - aux.a
    aux.rev_b = b_top - aux.b
  end
end

function update_aux!(dg::DGModel, m::IntegralTestModel, Q::MPIStateArray, t::Real)
  indefinite_stack_integral!(dg, m, Q, dg.auxstate, t)
  reverse_indefinite_stack_integral!(dg, m, Q, dg.auxstate, t)

  return true
end

@inline function integral_load_aux!(m::IntegralTestModel, integrand::Vars,
                                state::Vars, aux::Vars)
  x,y,z = aux.coord
  integrand.a = x + z
  integrand.b = 2*x + sin(x)*y - (z-1)^2*y^2
end

@inline function integral_set_aux!(m::IntegralTestModel, aux::Vars,
                                    integral::Vars)
  aux.int.a = integral.a
  aux.int.b = integral.b
end

@inline function reverse_integral_load_aux!(m::IntegralTestModel, integral::Vars,
                                            state::Vars, aux::Vars)
  integral.a = aux.int.a
  integral.b = aux.int.b
end

@inline function reverse_integral_set_aux!(m::IntegralTestModel, aux::Vars,
                                           integral::Vars)
  aux.rev_int.a = integral.a
  aux.rev_int.b = integral.b
end

using Test
function run(mpicomm, dim, Ne, N, FT, ArrayType)

  brickrange = ntuple(j->range(FT(0); length=Ne[j]+1, stop=3), dim)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  dg = DGModel(IntegralTestModel{dim}(),
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))
  dQdt = similar(Q)

  dg(dQdt, Q, nothing, 0.0)

  # Wrapping in Array ensure both GPU and CPU code use same approx
  @test Array(dg.auxstate.data[:, 1, :]) ≈ Array(dg.auxstate.data[:, 8, :])
  @test Array(dg.auxstate.data[:, 2, :]) ≈ Array(dg.auxstate.data[:, 9, :])
  @test Array(dg.auxstate.data[:, 3, :]) ≈ Array(dg.auxstate.data[:, 10, :])
  @test Array(dg.auxstate.data[:, 4, :]) ≈ Array(dg.auxstate.data[:, 11, :])
end

let
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  numelem = (5, 5, 5)
  lvls = 1

  polynomialorder = 4

    for FT in (Float64,) #Float32)
      for dim = 2:3
        err = zeros(FT, lvls)
        for l = 1:lvls
          @info (ArrayType, FT, dim)
          run(mpicomm, dim, ntuple(j->2^(l-1) * numelem[j], dim),
              polynomialorder, FT, ArrayType)
        end
      end
    end
end

nothing
