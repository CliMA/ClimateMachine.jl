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
                        vars_diffusive, vars_integrals, integral_load_aux!,
                        flux_nondiffusive!, flux_diffusive!, source!, wavespeed,
                        update_aux!, indefinite_stack_integral!,
                        reverse_indefinite_stack_integral!,  boundary_state!,
                        gradvariables!, init_aux!, init_state!,
                        init_ode_state, LocalGeometry,
                        integral_set_aux!,
                        vars_reverse_integrals,
                        reverse_integral_load_aux!,
                        reverse_integral_set_aux!


struct IntegralTestSphereModel{T} <: BalanceLaw
  Rinner::T
  Router::T
end

function update_aux!(dg::DGModel, m::IntegralTestSphereModel, Q::MPIStateArray, t::Real)
  indefinite_stack_integral!(dg, m, Q, dg.auxstate, t)
  reverse_indefinite_stack_integral!(dg, m, Q, dg.auxstate, t)

  return true
end

vars_integrals(::IntegralTestSphereModel, T) = @vars(v::T)
vars_reverse_integrals(::IntegralTestSphereModel, T) = @vars(v::T)
vars_aux(m::IntegralTestSphereModel,T) = @vars(int::vars_integrals(m,T), rev_int::vars_integrals(m,T), r::T, a::T)

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
  θ = atan(y, x)
  ϕ = asin(z / aux.r)
  # Exact integral
  aux.a = 1 + cos(ϕ)^2 * sin(θ)^2 + sin(ϕ)^2
  aux.int.v = exp(-aux.a * aux.r^2) - exp(-aux.a * m.Rinner^2)
  aux.rev_int.v = exp(-aux.a * m.Router^2) - exp(-aux.a * aux.r^2)
end

@inline function integral_load_aux!(m::IntegralTestSphereModel, integrand::Vars, state::Vars, aux::Vars)
  integrand.v = -2aux.r * aux.a * exp(-aux.a * aux.r^2)
end

@inline function integral_set_aux!(m::IntegralTestSphereModel, aux::Vars,
                                    integral::Vars)
  aux.int.v = integral.v
end

@inline function reverse_integral_load_aux!(m::IntegralTestSphereModel,
                                            integral::Vars,
                                            state::Vars,
                                            aux::Vars)
  integral.v = aux.int.v
end

@inline function reverse_integral_set_aux!(m::IntegralTestSphereModel,
                                           aux::Vars,
                                           integral::Vars)
  aux.rev_int.v = integral.v
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

using Test
function run(mpicomm, topl, ArrayType, N, FT, Rinner, Router)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp,
                                         )
  dg = DGModel(IntegralTestSphereModel(Rinner, Router),
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))
  dQdt = similar(Q)

  exact_aux = copy(dg.auxstate)
  dg(dQdt, Q, nothing, 0.0)
  euclidean_distance(exact_aux, dg.auxstate)
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

  polynomialorder = 4

  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1//2
  Router = 1

  polynomialorder = 4

  expected_result = [4.662884229467401e-7,
                     7.218989778540723e-9,
                     1.1258613174916711e-10,
                     1.7587739986848968e-12]
  lvls = integration_testing ? length(expected_result) : 1

  for FT in (Float64,) # Float32)
    err = zeros(FT, lvls)
    for l = 1:lvls
      @info (ArrayType, FT, "sphere", l)
      Nhorz = 2^(l-1) * base_Nhorz
      Nvert = 2^(l-1) * base_Nvert
      Rrange = grid1d(FT(Rinner), FT(Router); nelem=Nvert)
      topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
      err[l] = run(mpicomm, topl, ArrayType, polynomialorder, FT,
                    FT(Rinner), FT(Router))
      @test expected_result[l] ≈ err[l] rtol=1e-3 atol=eps(FT)
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

nothing
