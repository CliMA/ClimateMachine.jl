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


        for α in FT[5e-3, 1e-3, 1e-4, 1e-5]
          function op!(LQ, Q)
            vdg(LQ, Q, nothing, 0; increment=false)
            @. LQ = Q + α * LQ
          end

          # Just using init_ode_state to get vectors of the right size
          Q1 = init_ode_state(dg, FT(0))
          Q2 = init_ode_state(dg, FT(0))
          Qrhs = MPIStateArray(vdg)

          # Generate the solver objects
          solver_general = GeneralizedMinimalResidual(Q1; rtol=√eps(FT),
                                                      atol=100eps(FT), M=15)
          solver_stack = StackGMRES(Q1; M=15, nhorzelem=Neh,
                                    rtol=√eps(FT), atol=100eps(FT))

          # Initialise Q1, Q2
          l = length(Q1.data)
          reshape(Q1.data, l) .= rand(FT, l)
          op!(Qrhs, Q1)

          Q2.data .= Q1.data
          fill!(Q1, FT(0))
          fill!(Q2, FT(0))

          # Test that linearsolve! inverts op!
          iters_g, converged_g = linearsolve!(op!, solver_general, Q1, Qrhs; max_iters=Inf)
          iters_s, converged_s = linearsolve!(op!, solver_stack, Q2, Qrhs; max_iters=Inf)

          @test converged_g
          @test converged_s
          @test iters_s <= iters_g
        end
      end
    end
  end
end

nothing
