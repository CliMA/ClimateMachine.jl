using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
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

const γ_exact = 7 // 5 # FIXME: Remove this for some moist thermo approach

using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0

const _nstate = 5
const _δρ, _ρu, _ρv, _ρw, _δρe = 1:_nstate
const _ρu⃗ = SVector(_ρu, _ρv, _ρw)
const statenames = ("δρ", "ρu", "ρv", "ρw", "δρe")

const _nauxstate = 6
const _a_ρ0, _a_ρe0, _a_ϕ, _a_ϕ_x, _a_ϕ_y, _a_ϕ_z = 1:_nauxstate
function auxiliary_state_initialization!(aux, x, y, z) #JK, dx, dy, dz)
  @inbounds begin
    ρ0, ρe0 = reference_ρ_ρe(x, y, z)
    aux[_a_ρ0] = ρ0
    aux[_a_ρe0] = ρe0
    aux[_a_ϕ] = y * grav
    aux[_a_ϕ_x] = 0
    aux[_a_ϕ_y] = grav
    aux[_a_ϕ_z] = 0
  end
end

@inline function pressure(Q, aux)
  @inbounds begin
    gravity::eltype(Q) = grav
    γ::eltype(Q) = γ_exact # FIXME: Remove this for some moist thermo approach
    δρ, δρe = Q[_δρ], Q[_δρe]
    ρu⃗ = SVector(Q[_ρu], Q[_ρv], Q[_ρw])
    ρ0, ρe0, ϕ = aux[_a_ρ0], aux[_a_ρe0], aux[_a_ϕ]
    ρ = ρ0 + δρ
    ρe = ρe0 + δρe
    ρinv = 1 / ρ
    (γ-1)*(ρe - ρinv * (ρu⃗' * ρu⃗) / 2 - ϕ * ρ)
  end
end

@inline function wavespeed(n, Q, aux, t)
  γ::eltype(Q) = γ_exact # FIXME: Remove this for some moist thermo approach
  n⃗ = SVector(n)
  @inbounds begin
    P = pressure(Q, aux)
    δρ, δρe = Q[_δρ], Q[_δρe]
    ρu⃗ = SVector(Q[_ρu], Q[_ρv], Q[_ρw])
    ρ0, ρe0 = aux[_a_ρ0], aux[_a_ρe0]
    ρ = ρ0 + δρ
    ρe = ρe0 + δρe
    ρinv = 1 / ρ
    u⃗ = ρinv * ρu⃗
    abs(n⃗' * u⃗) + sqrt(ρinv * γ * P)
  end
end

@inline function euler_flux!(F, Q, _, aux, t)
  P = pressure(Q, aux)
  @inbounds begin
    δρ, δρe = Q[_δρ], Q[_δρe]
    ρu⃗ = SVector(Q[_ρu], Q[_ρv], Q[_ρw])
    ρ0, ρe0 = aux[_a_ρ0], aux[_a_ρe0]
    ρ = ρ0 + δρ
    ρe = ρe0 + δρe
    ρinv = 1 / ρ
    u⃗ = ρinv * ρu⃗

    F[:, _δρ ] = ρu⃗
    F[:, _ρu⃗] = u⃗ * ρu⃗' + P * I
    F[:, _δρe] = u⃗ * (ρe + P)
  end
end

@inline function bcstate!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  if bctype == 1
    nofluxbc!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  else
    error("Unsupported boundary condition")
  end
end

@inline function nofluxbc!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  @inbounds begin
    ρu⃗M = SVector(QM[_ρu], QM[_ρv], QM[_ρw])
    n⃗M = SVector(nM)

    # No flux boundary conditions
    # No shear on walls (free-slip condition)
    ρunM = n⃗M' * ρu⃗M
    QP[_ρu⃗[1]] = ρu⃗M[1] - 2ρunM * n⃗M[1]
    QP[_ρu⃗[2]] = ρu⃗M[2] - 2ρunM * n⃗M[2]
    QP[_ρu⃗[3]] = ρu⃗M[3] - 2ρunM * n⃗M[3]
    QP[_δρ] = QM[_δρ]
    QP[_δρe] = QM[_δρe]
    nothing
  end
end

@inline function source!(S, Q, diffusive, aux, t)
  # Initialise the final block source term
  S .= -zero(eltype(Q))

  source_geopotential!(S, Q, aux, t)
end

@inline function source_geopotential!(S, Q, aux, t)
  @inbounds begin
    δρ = Q[_δρ]
    ρ0 = aux[_a_ρ0]
    ρ = ρ0 + δρ
    ∇ϕ = SVector(aux[_a_ϕ_x], aux[_a_ϕ_y], aux[_a_ϕ_z])
    S[_ρu⃗[1]] -= ρ * ∇ϕ[1]
    S[_ρu⃗[2]] -= ρ * ∇ϕ[2]
    S[_ρu⃗[3]] -= ρ * ∇ϕ[3]
  end
end

function reference_ρ_ρe(x, y, z)
  FT                = eltype(x)
  R_gas::FT         = R_d
  c_p::FT           = cp_d
  c_v::FT           = cv_d
  p0::FT            = MSLP
  gravity::FT       = grav
  # perturbation parameters for rising bubble

  θ0::FT = 303
  π_exner    = 1 - gravity / (c_p * θ0) * y # exner pressure
  ρ          = p0 / (R_gas * θ0) * (π_exner)^ (c_v / R_gas) # density

  P          = p0 * (R_gas * (ρ * θ0) / p0) ^(c_p/c_v) # pressure (absolute)
  T          = P / (ρ * R_gas) # temperature
  u⃗          = SVector(-zero(FT), -zero(FT), -zero(FT))
  # energy definitions
  e_kin = u⃗' * u⃗ / 2
  e_pot = gravity * y
  e_int = cv_d *  T
  ρe    = ρ * (e_int + e_kin + e_pot)
  ρ, ρe
end

# Initial Condition
function rising_bubble!(dim, Q, t, x, y, z, aux)
  FT          = eltype(Q)
  R_gas::FT   = R_d
  c_p::FT     = cp_d
  c_v::FT     = cv_d
  p0::FT      = MSLP
  gravity::FT = grav
  # perturbation parameters for rising bubble

  r⃗ = SVector(x, y, z)
  r⃗_center = SVector(500, 260, 500)
  distance = norm(@view (r⃗ - r⃗_center)[1:dim])

  θ0::FT  = 303
  θ_c::FT = 1 // 2
  Δθ::FT  = -zero(FT)
  a::FT   =  50
  s::FT   = 100
  if distance <= a
    Δθ = θ_c
  elseif distance > a
    Δθ = θ_c * exp(-(distance - a)^2 / s^2)
  end
  θ       = θ0 + Δθ # potential temperature
  π_exner = 1 - gravity / (c_p * θ) * y # exner pressure
  ρ       = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  P       = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T       = P / (ρ * R_gas) # temperature
  u⃗       = SVector(-zero(FT), -zero(FT), -zero(FT))
  # energy definitions
  e_kin = u⃗' * u⃗ / 2
  e_pot = gravity * y
  e_int = cv_d *  T
  ρe    = ρ * (e_int + e_kin + e_pot)
  ρu⃗    = ρ * u⃗

  @inbounds ρ0, ρe0 = aux[_a_ρ0], aux[_a_ρe0]
  @inbounds Q[_δρ], Q[_δρe] = ρ-ρ0, ρe-ρe0
  @inbounds Q[_ρu⃗] = ρu⃗
end

# {{{ Linearization
@inline function lin_eulerflux!(F, Q, _, aux, t)
  F .= -zero(eltype(Q))
  @inbounds begin
    FT = eltype(Q)
    γ::FT = γ_exact # FIXME: Remove this for some moist thermo approach

    δρ, δρe = Q[_δρ], Q[_δρe]
    ρu⃗ = SVector(Q[_ρu], Q[_ρv], Q[_ρw])

    ρ0, ρe0 = aux[_a_ρ0], aux[_a_ρe0]
    ϕ = aux[_a_ϕ]

    ρinv0 = 1 / ρ0
    e0 = ρinv0 * ρe0

    P0 = (γ-1)*(ρe0 - ρ0 * ϕ)
    δP = (γ-1)*(δρe - δρ * ϕ)

    p0 = ρinv0 * P0

    F[:, _δρ ] = ρu⃗
    F[:, _ρu⃗ ] = δP * I + @SMatrix zeros(3,3)
    F[:, _δρe] = (e0 + p0) * ρu⃗
  end
end

@inline function wavespeed_linear(n, Q, aux, t)
  @inbounds begin
    FT = eltype(Q)
    γ::FT = γ_exact # FIXME: Remove this for some moist thermo approach

    ρ0, ρe0, ϕ = aux[_a_ρ0], aux[_a_ρe0], aux[_a_ϕ]
    ρinv0 = 1 / ρ0
    P0 = (γ-1)*(ρe0 - ρ0 * ϕ)

    sqrt(ρinv0 * γ * P0)
  end
end

@inline function lin_source!(S, Q, diffusive, aux, t)
  S .= 0
  lin_source_geopotential!(S, Q, diffusive, aux, t)
end
@inline function lin_source_geopotential!(S, Q, diffusive, aux, t)
  @inbounds begin
    δρ = Q[_δρ]
    ∇ϕ = SVector(aux[_a_ϕ_x], aux[_a_ϕ_y], aux[_a_ϕ_z])
    S[_ρu] -= δρ * ∇ϕ[1]
    S[_ρv] -= δρ * ∇ϕ[2]
    S[_ρw] -= δρ * ∇ϕ[3]
  end
end

@inline function lin_bcstate!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  if bctype == 1
    # this is already a linear boundary condition
    nofluxbc!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  else
    error("Unsupported boundary condition")
  end
end

# }}}

function run(mpicomm, ArrayType, dim, brickrange, periodicity, N, timeend, FT, dt, output_steps)

  topl = StackedBrickTopology(mpicomm, brickrange, periodicity = periodicity)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N)

  numflux!(x...) = NumericalFluxes.rusanov!(x..., euler_flux!, wavespeed)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., euler_flux!,
                                                            bcstate!, wavespeed)

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = euler_flux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!,
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           auxiliary_state_initialization!,
                           source! = source!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = rising_bubble!(dim, Q, FT(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  # {{{ Lineariztion Setup
  lin_numflux!(x...) = NumericalFluxes.rusanov!(x..., lin_eulerflux!,
                                                # (_...)->0) # central
                                                wavespeed_linear)
  lin_numbcflux!(x...) =
  NumericalFluxes.rusanov_boundary_flux!(x..., lin_eulerflux!, lin_bcstate!,
                                         # (_...)->0) # central
                                         wavespeed_linear)
  lin_spacedisc = DGBalanceLaw(grid = grid,
                               length_state_vector = _nstate,
                               flux! = lin_eulerflux!,
                               numerical_flux! = lin_numflux!,
                               numerical_boundary_flux! = lin_numbcflux!,
                               auxiliary_state_length = _nauxstate,
                               auxiliary_state_initialization! =
                               auxiliary_state_initialization!,
                               source! = lin_source!)


  # NOTE: In order to get the same results on the CPU and GPU we force ourselves
  # to take the same number of iterations by setting at really high tolerance
  # specifying the number of restarts
  linearsolver = GeneralizedConjugateResidual(3, Q, rtol=1e-8)

  timestepper = ARK548L2SA2KennedyCarpenter(spacedisc, lin_spacedisc,
                                            linearsolver, Q; dt = dt, t0 = 0)
  # }}}

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      #globmean = global_mean(Q, _δρ)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""",
                     ODESolvers.gettime(timestepper),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy )#, globmean)
    end
  end

  step = [0]
  vtkdir = "vtk_RTB_IMEX"
  mkpath(vtkdir)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(output_steps) do (init=false)
    outprefix = @sprintf("%s/RTB_%dD_mpirank%04d_step%04d", vtkdir, dim,
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, spacedisc, statenames)
    pvtuprefix = @sprintf("RTB_%dD_step%04d", dim, step[1])
    prefixes = ntuple(i->
                      @sprintf("%s/RTB_%dD_mpirank%04d_step%04d", vtkdir,
                               dim, i-1, step[1]),
                      MPI.Comm_size(mpicomm))
    writepvtu(pvtuprefix, prefixes, statenames)
    step[1] += 1
    nothing
  end

  solve!(Q, timestepper; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
  @info @sprintf("""norm(QF) / norm(Q0) = %.16e""", engf / eng0)
  engf / eng0
end

using Test
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
  FT = Float64

  expected_engf_eng0 = Dict()
  expected_engf_eng0[(Float64, 2)] = 1.5850821145834655e+00
  expected_engf_eng0[(Float64, 3)] = 1.4450465596993558e+00

  for dim in (2, 3)
    # Physical domain extents
    domain_start = (0, 0, 0)
    domain_end = (1000, 750, 1000)
    
    # Stable explicit time step
    Δxyz = MVector(25, 25, 25)
    dt = min(Δxyz...) / soundspeed_air(300.0) / polynomialorder
    dt *= dim == 2 ? 40 : 20
  
    output_time = 0.5
    output_steps = ceil(output_time / dt)

    #Get Ne from resolution
    Ls = MVector(domain_end .- domain_start)
    ratios = @. (Ls / Δxyz - 1) / polynomialorder
    Ne = ceil.(Int64, ratios)
    
    brickrange = ntuple(d -> range(domain_start[d], length = Ne[d] + 1, stop = domain_end[d]), dim)
    periodicity = ntuple(d -> false, dim)

    timeend = dim == 2 ? 10 : 1

    # only for printing
    if dim == 2
      Δxyz[3] = 0
      Ne[3] = 0
    end

    @info @sprintf """ ----------------------------------------------------"""
    @info @sprintf """   ______ _      _____ __  ________                  """
    @info @sprintf """  |  ____| |    |_   _|  ...  |  __  |               """
    @info @sprintf """  | |    | |      | | |   .   | |  | |               """
    @info @sprintf """  | |    | |      | | | |   | | |__| |               """
    @info @sprintf """  | |____| |____ _| |_| |   | | |  | |               """
    @info @sprintf """  | _____|______|_____|_|   |_|_|  |_|               """
    @info @sprintf """                                                     """
    @info @sprintf """ ----------------------------------------------------"""
    @info @sprintf """ Rising Bubble in %dD                                """ dim
    @info @sprintf """   Resolution:                                       """
    @info @sprintf """     (Δx, Δy, Δz)   = (%.2e, %.2e, %.2e)             """ Δxyz...
    @info @sprintf """     (Nex, Ney, Nez) = (%d, %d, %d)                  """ Ne...
    @info @sprintf """     dt         = %.2e                               """ dt
    @info @sprintf """ ----------------------------------------------------"""
    
    engf_eng0 = run(mpicomm, ArrayType,
                    dim, brickrange, periodicity, polynomialorder,
                    timeend, FT, dt, output_steps)

    @test engf_eng0 ≈ expected_engf_eng0[FT, dim]
  end
end
nothing
