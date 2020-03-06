using MPI
using Test
using StaticArrays
using Logging, Printf

using CLIMA
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.VariableTemplates
using CLIMA.DGmethods
import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux_nondiffusive!, flux_diffusive!,
                        source!, boundary_state!,
                        numerical_boundary_flux_diffusive!, gradvariables!,
                        diffusive!, init_aux!, init_state!,
                        LocalGeometry

import CLIMA.DGmethods.NumericalFluxes: NumericalFluxDiffusive,
                                        numerical_flux_diffusive!

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

struct PoissonModel{dim} <: BalanceLaw end

vars_aux(::PoissonModel,T) = @vars(rhs_ϕ::T)
vars_state(::PoissonModel, T) = @vars(ϕ::T)
vars_gradient(::PoissonModel, T) = @vars(ϕ::T)
vars_diffusive(::PoissonModel, T) = @vars(∇ϕ::SVector{3, T})

boundary_state!(nf, bl::PoissonModel, _...) = nothing

function flux_nondiffusive!(::PoissonModel, flux::Grad, state::Vars,
                            auxstate::Vars, t::Real)
  nothing
end

function flux_diffusive!(::PoissonModel, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, auxstate::Vars, t::Real)
  flux.ϕ = diffusive.∇ϕ
end

struct PenaltyNumFluxDiffusive <: NumericalFluxDiffusive end

# There is no boundary since we are periodic
numerical_boundary_flux_diffusive!(nf::PenaltyNumFluxDiffusive, _...) = nothing

function numerical_flux_diffusive!(::PenaltyNumFluxDiffusive,
    bl::PoissonModel, fluxᵀn::Vars{S}, n::SVector,
    state⁻::Vars{S}, diff⁻::Vars{D}, hyperdiff⁻::Vars{HD}, aux⁻::Vars{A},
    state⁺::Vars{S}, diff⁺::Vars{D}, hyperdiff⁺::Vars{HD}, aux⁺::Vars{A}, t) where {S,HD,D,A}

  numerical_flux_diffusive!(CentralNumericalFluxDiffusive(),
    bl, fluxᵀn, n, state⁻, diff⁻, hyperdiff⁻, aux⁻, state⁺, diff⁺, hyperdiff⁺, aux⁺, t)

  Fᵀn = parent(fluxᵀn)
  FT = eltype(Fᵀn)
  tau = FT(1)
  Fᵀn .-= tau*(parent(state⁻) - parent(state⁺))
end

function gradvariables!(::PoissonModel, transformstate::Vars, state::Vars, auxstate::Vars, t::Real)
  transformstate.ϕ = state.ϕ
end

function diffusive!(::PoissonModel, diffusive::Vars,
                    ∇transform::Grad, state::Vars, auxstate::Vars, t::Real)
  diffusive.∇ϕ = ∇transform.ϕ
end

source!(::PoissonModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing

# note, that the code assumes solutions with zero mean
sol1d(x) = sin(2pi * x) ^ 4 - 3 / 8
dxx_sol1d(x) = -16pi ^ 2 * sin(2pi * x) ^ 2 * (sin(2pi * x) ^ 2 - 3cos(2pi * x) ^ 2)

function init_aux!(::PoissonModel{dim}, aux::Vars, g::LocalGeometry) where dim
  aux.rhs_ϕ = 0
  @inbounds for d = 1:dim
    x1 = g.coord[d]
    x2 = g.coord[1 + mod(d, dim)]
    x3 = g.coord[1 + mod(d + 1, dim)]
    x23 = SVector(x2, x3)
    aux.rhs_ϕ -= dxx_sol1d(x1) * prod(sol1d, view(x23, 1:dim-1))
  end
end

function init_state!(::PoissonModel{dim}, state::Vars, aux::Vars, coords, t) where dim
  state.ϕ = prod(sol1d, view(coords, 1:dim))
end

function run(mpicomm, ArrayType, FT, dim, polynomialorder, brickrange, periodicity, linmethod)

  topology = BrickTopology(mpicomm, brickrange, periodicity=periodicity)
  grid = DiscontinuousSpectralElementGrid(topology,
                                          polynomialorder = polynomialorder,
                                          DeviceArray = ArrayType,
                                          FloatType = FT)

  dg = DGModel(PoissonModel{dim}(),
               grid,
               CentralNumericalFluxNonDiffusive(),
               PenaltyNumFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))
  Qrhs = dg.auxstate
  Qexact = init_ode_state(dg, FT(0))

  linearoperator!(y, x) = dg(y, x, nothing, 0; increment = false)

  linearsolver = linmethod(Q)

  iters = linearsolve!(linearoperator!, linearsolver, Q, Qrhs)

  error = euclidean_distance(Q, Qexact)

  @info @sprintf """Finished
  error = %.16e
  iters = %d
  """ error iters
  error, iters
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
  base_num_elem = 4
  tol = 1e-9

  linmethods = (b -> GeneralizedConjugateResidual(3, b, rtol=tol),
                b -> GeneralizedMinimalResidual(b, M=7, rtol=tol)
               )

  expected_result = Array{Float64}(undef, 2, 2, 3) # method, dim-1, lvl

  # GeneralizedConjugateResidual
  expected_result[1, 1, 1] = 5.0540243616448058e-02
  expected_result[1, 1, 2] = 1.4802275366044011e-03
  expected_result[1, 1, 3] = 3.3852821775121401e-05
  expected_result[1, 2, 1] = 1.4957957657736219e-02
  expected_result[1, 2, 2] = 4.7282369781541172e-04
  expected_result[1, 2, 3] = 1.4697449643351771e-05

  # GeneralizedMinimalResidual
  expected_result[2, 1, 1] = 5.0540243587512981e-02
  expected_result[2, 1, 2] = 1.4802275409186211e-03
  expected_result[2, 1, 3] = 3.3852820667079927e-05
  expected_result[2, 2, 1] = 1.4957957659220951e-02
  expected_result[2, 2, 2] = 4.7282369895963614e-04
  expected_result[2, 2, 3] = 1.4697449516628483e-05

  lvls = integration_testing ? size(expected_result)[end] : 1

  for (m, linmethod) in enumerate(linmethods), FT in (Float64,)
    result = Array{Tuple{FT, Int}}(undef, lvls)
    for dim = 2:3

      for l = 1:lvls
        Ne = ntuple(d -> 2 ^ (l - 1) * base_num_elem, dim)
        brickrange = ntuple(d -> range(FT(0), length = Ne[d], stop = 1), dim)
        periodicity = ntuple(d -> true, dim)

        @info (ArrayType, FT, m, dim)
        result[l] = run(mpicomm, ArrayType, FT, dim,
                        polynomialorder, brickrange, periodicity, linmethod)

        @test isapprox(result[l][1], FT(expected_result[m, dim-1, l]), rtol = sqrt(tol))
      end

      if integration_testing
        @info begin
          msg = ""
          for l = 1:lvls-1
            rate = log2(result[l][1]) - log2(result[l + 1][1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
      end
    end
  end
end

nothing
