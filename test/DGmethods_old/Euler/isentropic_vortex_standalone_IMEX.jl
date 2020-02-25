# Standard isentropic vortex test case.  For a more complete description of
# the setup see for Example 3 of:
#
# @article{ZHOU2003159,
#   author = {Y.C. Zhou and G.ρw. Wei},
#   title = {High resolution conjugate filters for the simulation of flows},
#   journal = {Journal of Computational Physics},
#   volume = {189},
#   number = {1},
#   pages = {159--179},
#   year = {2003},
#   doi = {10.1016/S0021-9991(03)00206-7},
#   url = {https://doi.org/10.1016/S0021-9991(03)00206-7},
# }
#
# This version runs the isentropic vortex as a stand alone test (no dependence
# on CLIMA moist thermodynamics)

using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.SpaceMethods
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

const _nstate = 5
const _δρ, _δρu, _δρv, _δρw, _δρe = 1:_nstate
const _δρu⃗ = SVector(_δρu, _δρv, _δρw)
const stateid = (ρid = _δρ, ρuid = _δρu, ρvid = _δρv, ρwid = _δρw, ρeid = _δρe)
const statenames = ("δρ", "δρu", "δρv", "δρw", "δρe")
const _nauxstate = 5
const _a_ρ_ref, _a_ρu_ref, _a_ρv_ref, _a_ρw_ref, _a_ρe_ref = 1:_nauxstate
const γ_exact = 7 // 5

const _Tinf = 1
const _uinf = 2
const _vinf = 1

@inline function velocity_pressure(Q, aux)
  γ::eltype(Q) = γ_exact
  @inbounds begin
    δρ, δρu, δρv, δρw, δρe = Q[_δρ], Q[_δρu], Q[_δρv], Q[_δρw], Q[_δρe]

    ρ_ref, ρe_ref = aux[_a_ρ_ref], aux[_a_ρe_ref]
    ρu_ref, ρv_ref, ρw_ref = aux[_a_ρu_ref], aux[_a_ρv_ref], aux[_a_ρw_ref]

    ρ = ρ_ref + δρ
    ρu = ρu_ref + δρu
    ρv = ρv_ref + δρv
    ρw = ρw_ref + δρw
    ρe = ρe_ref + δρe

    ρinv = 1 / ρ
    u⃗ = SVector(ρinv * ρu, ρinv * ρv, ρinv * ρw)
    ((γ-1)*(ρe - ρinv * (ρu^2 + ρv^2 + ρw^2) / 2), u⃗, ρinv)
  end
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t)
  P, u⃗, ρinv = velocity_pressure(Q, aux)
  n⃗ = SVector(n)
  γ::eltype(Q) = γ_exact
  abs(n⃗' * u⃗) + sqrt(ρinv * γ * P)
end
@inline function wavespeed_linear(n, Q, aux, t)
  ρ_ref, ρe_ref = aux[_a_ρ_ref], aux[_a_ρe_ref]
  ρu⃗_ref = SVector(aux[_a_ρu_ref], aux[_a_ρv_ref], aux[_a_ρw_ref])
  ρinv_ref = 1 / ρ_ref
  n⃗ = SVector(n)
  γ::eltype(Q) = γ_exact
  u⃗_ref = ρinv_ref * ρu⃗_ref
  P_ref = (γ-1)*(ρe_ref - u⃗_ref' * ρu⃗_ref / 2)
  abs(n⃗' * u⃗_ref) + sqrt(ρinv_ref * γ * P_ref)
end

# physical flux function
@inline function eulerflux!(F, Q, QV, aux, t)
  P, u⃗, ρinv = velocity_pressure(Q, aux)
  @inbounds begin
    δρ, δρe = Q[_δρ], Q[_δρe]
    δρu⃗ = SVector(Q[_δρu], Q[_δρv], Q[_δρw])

    ρ_ref, ρe_ref = aux[_a_ρ_ref], aux[_a_ρe_ref]
    ρu⃗_ref = SVector(aux[_a_ρu_ref], aux[_a_ρv_ref], aux[_a_ρw_ref])

    ρ = ρ_ref + δρ
    ρe = ρe_ref + δρe

    ρu⃗ = ρu⃗_ref + δρu⃗

    F[:, _δρ ] = ρu⃗
    F[:, _δρu⃗] = u⃗ * ρu⃗' + P * I
    F[:, _δρe] = u⃗ * (ρe + P)
  end
end

@inline function linearized_eulerflux!(F, Q, QV, aux, t)
  @inbounds begin
    FT = eltype(Q)
    γ::FT = γ_exact

    δρ, δρe = Q[_δρ], Q[_δρe]
    δρu⃗ = SVector(Q[_δρu], Q[_δρv], Q[_δρw])

    ρ_ref, ρe_ref = aux[_a_ρ_ref], aux[_a_ρe_ref]
    ρu⃗_ref = SVector(aux[_a_ρu_ref], aux[_a_ρv_ref], aux[_a_ρw_ref])

    ρinv_ref = 1 / ρ_ref
    u⃗_ref = ρinv_ref * ρu⃗_ref
    e_ref = ρinv_ref * ρe_ref

    P_ref = (γ-1)*(ρe_ref - u⃗_ref' * ρu⃗_ref / 2)
    δP = (γ-1)*(δρe - u⃗_ref' * δρu⃗ + u⃗_ref' * u⃗_ref * δρ)

    p_ref = ρinv_ref * P_ref

    # FIXME: Not sure these pure reference terms are right...
    #=
    F[:, _δρ ] = (ρu⃗_ref + δρu⃗)
    F[:, _δρu⃗] = (ρu⃗_ref * u⃗_ref' - (u⃗_ref * u⃗_ref') * δρ
                  + δρu⃗ * u⃗_ref' + u⃗_ref * δρu⃗'
                  + (P_ref + δP) * I)
    F[:, _δρe] = ((e_ref + p_ref) * ρu⃗_ref + (e_ref + p_ref) * δρu⃗ +
                  (δρe + δP) * u⃗_ref - (e_ref + p_ref) * u⃗_ref * δρ)
    =#
    F[:, _δρ ] = δρu⃗
    F[:, _δρu⃗] = (+ δρu⃗ * u⃗_ref'
                  + u⃗_ref * δρu⃗'
                  - (u⃗_ref * u⃗_ref') * δρ
                  + δP * I)
    F[:, _δρe] = ((e_ref + p_ref) * δρu⃗
                  + u⃗_ref * (δρe + δP)
                  - (e_ref + p_ref) * u⃗_ref * δρ)
  end
end

# initial condition
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    FT = eltype(aux)
    γ::FT = γ_exact

    uinf::FT = _uinf
    vinf::FT = _vinf
    Tinf::FT = _Tinf

    ρ_ref = (Tinf)^(1/(γ-1))
    P_ref = ρ_ref^γ

    u_ref = uinf
    v_ref = vinf
    w_ref = zero(FT)

    ρu_ref = ρ_ref * u_ref
    ρv_ref = ρ_ref * v_ref
    ρw_ref = ρ_ref * w_ref

    ρe_ref = P_ref/(γ-1) + (1//2)*ρ_ref*(u_ref^2 + v_ref^2 + w_ref^2)

    aux[_a_ρ_ref]  = ρ_ref
    aux[_a_ρu_ref] = ρu_ref
    aux[_a_ρv_ref] = ρv_ref
    aux[_a_ρw_ref] = ρw_ref
    aux[_a_ρe_ref] = ρe_ref
  end
end

const halfperiod = 5
function isentropicvortex!(Q, t, x, y, z, aux)
  @inbounds begin
    ρ_ref, ρe_ref = aux[_a_ρ_ref], aux[_a_ρe_ref]
    ρu_ref, ρv_ref, ρw_ref = aux[_a_ρu_ref], aux[_a_ρv_ref], aux[_a_ρw_ref]

    FT = eltype(Q)

    γ::FT    = γ_exact
    uinf::FT = _uinf
    vinf::FT = _vinf
    Tinf::FT = _Tinf
    λ::FT    = 5

    xs = x - uinf*t
    ys = y - vinf*t

    # make the function periodic
    xtn = floor((xs+halfperiod)/(2halfperiod))
    ytn = floor((ys+halfperiod)/(2halfperiod))
    xp = xs - xtn*2*halfperiod
    yp = ys - ytn*2*halfperiod

    rsq = xp^2 + yp^2

    u = uinf - λ*(1//2)*exp(1-rsq)*yp/π
    v = vinf + λ*(1//2)*exp(1-rsq)*xp/π
    w = zero(FT)

    ρ = (Tinf - ((γ-1)*λ^2*exp(2*(1-rsq))/(γ*16*π*π)))^(1/(γ-1))
    P = ρ^γ
    ρu = ρ*u
    ρv = ρ*v
    ρw = ρ*w
    ρe = P/(γ-1) + (1//2)*ρ*(u^2 + v^2 + w^2)

    δρ = ρ - ρ_ref
    δρu = ρu - ρu_ref
    δρv = ρv - ρv_ref
    δρw = ρw - ρw_ref
    δρe = ρe - ρe_ref

    Q[_δρ], Q[_δρu], Q[_δρv], Q[_δρw], Q[_δρe] = δρ, δρu, δρv, δρw, δρe
  end
end

function main(mpicomm, FT, topl::AbstractTopology{dim}, N, timeend,
              ArrayType, dt) where {dim}

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  nonlin_spacedisc = DGBalanceLaw(grid = grid,
                                  length_state_vector = _nstate,
                                  flux! = eulerflux!,
                                  numerical_flux! = (x...) ->
                                  NumericalFluxes.rusanov!(x..., eulerflux!,
                                                           wavespeed),
                                  auxiliary_state_length = _nauxstate,
                                  auxiliary_state_initialization! =
                                  auxiliary_state_initialization!)

  lin_spacedisc = DGBalanceLaw(grid = grid,
                               length_state_vector = _nstate,
                               flux! = linearized_eulerflux!,
                               numerical_flux! = (x...) ->
                               NumericalFluxes.rusanov!(x...,
                                                        linearized_eulerflux!,
                                                        (_...)->0), # central
                                                        # wavespeed_linear), # rusanov!
                               auxiliary_state_length = _nauxstate,
                               auxiliary_state_initialization! =
                               auxiliary_state_initialization!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = isentropicvortex!(Q, FT(0), x...)
  Q = MPIStateArray(nonlin_spacedisc, initialcondition)
  dQ = similar(Q)

  linearsolver = GeneralizedConjugateResidual(3, Q, rtol=1e-10)
  ark = ARK548L2SA2KennedyCarpenter(nonlin_spacedisc, lin_spacedisc,
                                    linearsolver, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf """Update
  simtime = %.16e
  runtime = %s
  norm(Q) = %.16e""" ODESolvers.gettime(ark) Dates.format(convert(Dates.DateTime, Dates.now()-starttime[]), Dates.dateformat"HH:MM:SS") energy
    end
  end

  #= Paraview calculators:
  P = (0.4) * (ρe  - (ρu^2 + ρv^2 + ρw^2) / (2*ρ) - 9.81 * ρ * coordsZ)
  theta = (100000/287.0024093890231) * (P / 100000)^(1/1.4) / ρ
  =#
  step = [0]
  mkpath("vtk")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(100) do (init=false)
    outprefix = @sprintf("vtk/isentropicvortex_%dD_mpirank%04d_step%04d",
                         dim, MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, nonlin_spacedisc, statenames)
    pvtuprefix = @sprintf("isentropicvortex_%dD_step%04d", dim, step[1])
    prefixes = ntuple(i->
                      @sprintf("vtk/isentropicvortex_%dD_mpirank%04d_step%04d",
                               dim, i-1, step[1]),
                      MPI.Comm_size(mpicomm))
    writepvtu(pvtuprefix, prefixes, statenames)
    step[1] += 1
    nothing
  end

  solve!(Q, ark; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = MPIStateArray(nonlin_spacedisc,
                     (Q, x...) -> isentropicvortex!(Q, FT(timeend), x...))
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end

function run(mpicomm, ArrayType, dim, Ne, N, timeend, FT, dt)
  brickrange = ntuple(j->range(FT(-halfperiod); length=Ne[j]+1,
                               stop=halfperiod), dim)
  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))
  main(mpicomm, FT, topl, N, timeend, ArrayType, dt)
end

using Test
let
  CLIMA.init()
  ArrayTypes = (CLIMA.array_type(),)

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  timeend = 1
  numelem = (5, 5, 1)

  polynomialorder = 4

  expected_error = Array{Float64}(undef, 2, 3) # dim-1, lvl
  expected_error[1,1] = 5.7107915113422314e-01
  expected_error[1,2] = 6.9428317829549141e-02
  expected_error[1,3] = 3.2924199180417437e-03
  expected_error[2,1] = 1.8059108422662815e+00
  expected_error[2,2] = 2.1955161886731539e-01
  expected_error[2,3] = 1.0411547208036219e-02
  lvls = integration_testing ? size(expected_error, 2) : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64,) #Float32)
      for dim = 2:3
        err = zeros(FT, lvls)
        for l = 1:lvls
          Ne = ntuple(j->2^(l-1) * numelem[j], dim)
          dt = 3e-1 / Ne[1]
          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps
          @info (ArrayType, FT, dim)
          err[l] = run(mpicomm, ArrayType, dim, Ne, polynomialorder, timeend,
                       FT, dt)
          @test err[l] ≈ FT(expected_error[dim-1, l])
        end
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

isinteractive() || MPI.Finalize()

nothing
