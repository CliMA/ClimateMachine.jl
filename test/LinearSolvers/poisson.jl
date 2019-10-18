using Test, MPI

using LinearAlgebra
using StaticArrays
using Logging, Printf

using CLIMA
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver
using CLIMA.GeneralizedMinimalResidualSolver

using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.SpaceMethods

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

function physical_flux!(F, state, visc_state, _...)
  @inbounds F[:, 1] = visc_state[:]
end

@inline function numerical_flux!(fs, nM, stateM, viscM, auxM, stateP, viscP, auxP, t)
  FT = eltype(stateM)
  tau = FT(1.0)

  @inbounds fs[1] = ( nM[1] * (viscM[1] + viscP[1])
                    + nM[2] * (viscM[2] + viscP[2])
                    + nM[3] * (viscM[3] + viscP[3]) ) / 2 
 

  @inbounds fs[1] -= tau * (stateM[1] - stateP[1])
end

@inline gradient_transform!(G, Q, _...) = (G .= Q)

@inline viscous_transform!(V, gradG, _...) = (V[:] .= gradG[:, 1])

@inline function viscous_penalty!(VF, nM, GM, QM, aM, GP, QP, aP, t)
  @inbounds begin
    n_dq = similar(VF, Size(3, 1))
    for i = 1:3
      n_dq[i, 1] = nM[i] * (GP[1] - GM[1]) / 2
    end
    viscous_transform!(VF, n_dq)
  end
end

@inline function initialcondition!(Q, xs...)
  @inbounds Q[1] = 0
end

# note, that the code assumes solutions with zero mean
sol1d(x) = sin(2pi * x) ^ 4 - 3 / 8
dxx_sol1d(x) = -16pi ^ 2 * sin(2pi * x) ^ 2 * (sin(2pi * x) ^ 2 - 3cos(2pi * x) ^ 2)

@inline function rhs!(dim, Q, xs...)
  @inbounds Q[1] = -sum(dxx_sol1d(xs[d]) *
                        prod(sol1d(xs[mod(d + dd - 1, dim) + 1]) for dd = 1:dim-1) for d = 1:dim)
end

@inline function exactsolution!(dim, Q, xs...)
  @inbounds Q[1] = prod(sol1d.(xs[1:dim]))
end

function run(mpicomm, ArrayType, FT, dim, polynomialorder, brickrange, periodicity, linmethod)

  topology = BrickTopology(mpicomm, brickrange, periodicity=periodicity)
  grid = DiscontinuousSpectralElementGrid(topology,
                                          polynomialorder = polynomialorder,
                                          FloatType = FT,
                                          DeviceArray = ArrayType)

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 1,
                           flux! = physical_flux!,
                           numerical_flux! = numerical_flux!,
                           number_gradient_states = 1,
                           number_viscous_states = 3,
                           gradient_transform! = gradient_transform!,
                           viscous_transform! = viscous_transform!,
                           viscous_penalty! = viscous_penalty!)


  Q = MPIStateArray(spacedisc, initialcondition!)
  Qrhs = MPIStateArray(spacedisc, (args...) -> rhs!(dim, args...))
  Qexact = MPIStateArray(spacedisc, (args...) -> exactsolution!(dim, args...))

  linearoperator!(y, x) = SpaceMethods.odefun!(spacedisc, y, x, nothing, 0;
                                               increment = false)

  linearsolver = linmethod(Q)

  iters = linearsolve!(linearoperator!, Q, Qrhs, linearsolver)

  error = euclidean_distance(Q, Qexact)

  @info @sprintf """Finished
  error = %.16e
  iters = %d
  """ error iters
  error, iters
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
  base_num_elem = 4
  tol = 1e-9

  linmethods = (b -> GeneralizedConjugateResidual(3, b, tol),
                b -> GeneralizedMinimalResidual(7, b, tol)
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

  for ArrayType in ArrayTypes, (m, linmethod) in enumerate(linmethods), FT in (Float64,)
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
