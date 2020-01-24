using CLIMA
using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Grids: VerticalDirection
using CLIMA.VTK: writemesh
using Logging
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using IterativeSolvers
using CLIMA.DGmethods: DGModel, Vars, vars_state, num_state, init_ode_state
using CLIMA.LinearSolvers
using CLIMA.ColumnwiseGMRESSolver
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxDiffusive,
                                       CentralGradPenalty
using CLIMA.MPIStateArrays
import CLIMA.MPIStateArrays: MPIStateArray
using CLIMA.MPIStateArrays: MPIStateArray, euclidean_distance

using Test

const ArrayType = CLIMA.array_type()

include("../DGmethods/advection_diffusion/advection_diffusion_model.jl")

# Mock constructor of MPIArrays for testing on single node for right-hand-sides
function MPIStateArrays.MPIStateArray(b::AbstractVector{FT}) where {FT<:AbstractFloat}
  data = reshape(b, length(b), 1, 1)
  realdata = view(data, ntuple(i -> Colon(), ndims(data))...)

  reqs = Vector{MPI.Request}(undef, 0)
  buffer = Matrix{FT}(undef, 0, 0)
  nabtorank = Vector{Int64}(undef, 0)
  nabrtovmap = Vector{UnitRange{Int64}}(undef, 0)
  weights = similar(data)

  MPIStateArray{FT, typeof(data), typeof(nabrtovmap), typeof(realdata), Array{FT,2}}(
    MPI.COMM_WORLD, data, realdata, 1:1, 1:0, nabrtovmap, nabrtovmap, reqs,
    reqs, buffer, buffer, nabtorank, nabrtovmap, nabrtovmap, buffer, buffer,
    weights, 888
  )
end

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(::Pseudo1D{n, α, β}, aux::Vars,
                                  geom::LocalGeometry) where {n, α, β}
  # Direction of flow is n with magnitude α
  aux.u = α * n

  # diffusion of strength β in the n direction
  aux.D = β * n * n'
end

function initial_condition!(::Pseudo1D{n, α, β, μ, δ}, state, aux, x,
                            t) where {n, α, β, μ, δ}
  ξn = dot(n, x)
  # ξT = SVector(x) - ξn * n
  state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end

let
  # boiler plate MPI stuff
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  Random.seed!(777 + MPI.Comm_rank(mpicomm))

  # Mesh generation parameters
  N = 4
  Nq = N+1
  Neh = 10
  Nev = 4

  @testset "$(@__FILE__) DGModel matrix" begin
    for FT in (Float64, Float32)
      for dim = (2, 3)
        # for single_column in (false, true)

        # Setup the topology
        if dim == 2
          brickrange = (range(FT(0); length=Neh+1, stop=1),
                        range(FT(1); length=Nev+1, stop=2))
        elseif dim == 3
          brickrange = (range(FT(0); length=Neh+1, stop=1),
                        range(FT(0); length=Neh+1, stop=1),
          range(FT(1); length=Nev+1, stop=2))
        end
        topl = StackedBrickTopology(mpicomm, brickrange)

        # Warp mesh
        function warpfun(ξ1, ξ2, ξ3)
          # single column currently requires no geometry warping

          # Even if the warping is in only the horizontal, the way we
          # compute metrics causes problems for the single column approach
          # (possibly need to not use curl-invariant computation)
          ξ1 = ξ1 + sin(2π * ξ1 * ξ2) / 10
          ξ2 = ξ2 + sin(2π * ξ1) / 5
          if dim == 3
            ξ3 = ξ3 + sin(8π * ξ1 * ξ2) / 10
          end
          (ξ1, ξ2, ξ3)
        end

        # create the actual grid
        grid = DiscontinuousSpectralElementGrid(topl,
                                                FloatType = FT,
                                                DeviceArray = ArrayType,
                                                polynomialorder = N,
                                                meshwarp = warpfun)
        d = dim == 2 ? FT[1, 10, 0] : FT[1, 1, 10]
        n = SVector{3, FT}(d ./ norm(d))

        α = FT(1)
        β = FT(1 // 100)
        μ = FT(-1 // 2)
        δ = FT(1 // 10)
        model = AdvectionDiffusion{dim}(Pseudo1D{n, α, β, μ, δ}())

        # the nonlinear model is needed so we can grab the auxstate below
        dg = DGModel(model,
                      grid,
                      Rusanov(),
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty())

        vdg = DGModel(model,
                      grid,
                      Rusanov(),
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty();
                      direction=VerticalDirection(),
                      auxstate=dg.auxstate)

        # α = FT(1 // 10)
        # function op!(LQ, Q)
        #   vdg(LQ, Q, nothing, 0; increment=false)
        #   @. LQ = Q + α * LQ
        # end

        s = 1000
        local S
        while true
          println("Generating the matrix")
          S = sprandn(FT, s, s, 0.5)
          @show cond(Array(S))
          det(S) == 0 || break
        end

        function op!(LQ, Q)
          v = reshape(Q, length(Q))
          reshape(LQ, length(LQ)) .= S * v
        end

        b = randn(s) / s
        x = zeros(s)
        Qrhs = MPIStateArray(b)
        Q1 = MPIStateArray(zeros(s) / s)
        solver = GeneralizedMinimalResidual(s, Q1, 1e-8)

        println("Starting linear solve")

        _, history = gmres!(x,S,b; maxiter=1000*s, restart=s, log=true, tol=1e-8)
        @show history
        linearsolve!(op!, solver, Q1, Qrhs)
        Q2 = S \ b
        @test all(isapprox.(Array(Q1.realdata), Q2, atol=100*eps(FT)))

        # Q1 = init_ode_state(dg, FT(0))
        # Q2 = init_ode_state(dg, FT(0))
        # dQ1 = MPIStateArray(vdg)
        #
        # solver = StackGMRES{10, Neh}(Q1)
        # solver = GeneralizedMinimalResidual(50, Q1, 1e-8)
        # N = length(Q1.data)
        # reshape(Q1.data, N) .= randn(FT, N) / N
        # Q2.data .= Q1.data
        #
        # # Test that linearsolve! inverts op!
        # op!(dQ1, Q1)
        # Q1.data .= dQ1.data
        # op!(dQ1, Q1)
        #
        # @show norm(dQ1.data)
        # fill!(Q1, FT(0))
        # linearsolve!(op!, solver, Q1, dQ1)
        # @test all(isapprox.(Array(Q1.data), Array(Q2.data), atol=100*eps(FT)))

        # Test for linearity
        # First test A(τv+ωu) - ωA(u) ≈ τA(v)
        # τ, ω = randn(FT, 2)
        # dQ2 = MPIStateArray(vdg)
        # op!(dQ2, Q1)
        # rhs = τ * dQ2
        #
        # dQ3 = MPIStateArray(vdg)
        # arg = init_ode_state(dg, FT(0))
        # @. arg.data = τ * Q1.data + ω * Q2.data
        # op!(dQ3, arg)
        # dQ4 = MPIStateArray(vdg)
        # op!(dQ4, Q2)
        # @. dQ4.data *= ω
        # @. dQ3.data -= dQ4
        # @test all(isapprox.(Array(dQ3.data), Array(rhs.data), rtol=100*eps(FT)))
        #
        # # Now test that v ≈ A\(lhs) * inv(τ)
        # Q3 = init_ode_state(dg, FT(0))
        # linearsolve!(op!, solver, Q3, dQ3)
        # @. Q3.data /= τ
        # @test all(isapprox.(Array(Q3.data), Array(Q1.data), rtol=100*eps(FT)))

        # let Aw = A(v - u)
        # Test v - u ≈ w
        # reshape(Q2.data, N) .= randn(FT, N) / N
        # dQ2 = MPIStateArray(vdg)
        # Q3 = Q1 - Q2
        # op!(dQ2, Q3)
        # Q4 = MPIStateArray(vdg)
        # linearsolve!(op!, solver, Q4, dQ2)
        # @show norm(Array(Q4.data) - Array(Q3.data), Inf)
        # @test all(isapprox.(Array(Q4.data), Array(Q3.data), rtol=1000*eps(FT)))
      end
    end
  end
end

nothing
