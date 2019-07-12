using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver
using CLIMA.AdditiveRungeKuttaMethod

const γ_exact = 7 // 5 # FIXME: Remove this for some moist thermo approach

using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0

if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

const _nstate = 5
const _δρ, _ρu, _ρv, _ρw, _δρe = 1:_nstate
const _ρu⃗ = SVector(_ρu, _ρv, _ρw)
const statenames = ("δρ", "ρu", "ρv", "ρw", "δρe")

const numdims = 2
const Δx    = 20
const Δy    = 20
const Δz    = 20
const Npoly = 4

# Physical domain extents
const (xmin, xmax) = (0, 1000)
const (ymin, ymax) = (0, 1500)

# Can be extended to a 3D test case
const (zmin, zmax) = (0, 1000)

#Get Nex, Ney from resolution
const Lx = xmax - xmin
const Ly = ymax - ymin
const Lz = zmax - ymin

const ratiox = (Lx/Δx - 1)/Npoly
const ratioy = (Ly/Δy - 1)/Npoly
const ratioz = (Lz/Δz - 1)/Npoly
const Nex = ceil(Int64, ratiox)
const Ney = ceil(Int64, ratioy)
const Nez = ceil(Int64, ratioz)

const _nauxstate = 6
const _a_ρ0, _a_ρe0, _a_ϕ, _a_ϕ_x, _a_ϕ_y, _a_ϕ_z = 1:_nauxstate
function auxiliary_state_initialization!(aux, x, y, z) #JK, dx, dy, dz)
  @inbounds begin
    ρ0, ρe0 = reference2D_ρ_ρe(x, y, z)
    aux[_a_ρ0] = ρ0
    aux[_a_ρe0] = ρe0
    aux[_a_ϕ] = y * grav
    aux[_a_ϕ_x] = 0
    aux[_a_ϕ_y] = grav
    aux[_a_ϕ_z] = 0
  end
end

function pressure(Q, aux)
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

function wavespeed(n, Q, aux, t)
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

function euler_flux!(F, Q, _, aux, t)
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

function bcstate!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  if bctype == 1
    nofluxbc!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  else
    error("Unsupported boundary condition")
  end
end

function nofluxbc!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
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

function source!(S,Q,aux,t)
  # Initialise the final block source term
  S .= 0

  source_geopotential!(S, Q, aux, t)
end

function source_geopotential!(S, Q, aux, t)
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

function reference2D_ρ_ρe(x, y, z)
  DFloat                = eltype(x)
  R_gas::DFloat         = R_d
  c_p::DFloat           = cp_d
  c_v::DFloat           = cv_d
  p0::DFloat            = MSLP
  gravity::DFloat       = grav
  # perturbation parameters for rising bubble

  θ0::DFloat = 303
  π_exner    = 1 - gravity / (c_p * θ0) * y # exner pressure
  ρ          = p0 / (R_gas * θ0) * (π_exner)^ (c_v / R_gas) # density

  P          = p0 * (R_gas * (ρ * θ0) / p0) ^(c_p/c_v) # pressure (absolute)
  T          = P / (ρ * R_gas) # temperature
  u⃗          = SVector(-zero(DFloat), -zero(DFloat), -zero(DFloat))
  # energy definitions
  e_kin = u⃗' * u⃗ / 2
  e_pot = gravity * y
  e_int = cv_d *  T
  ρe    = ρ * (e_int + e_kin + e_pot)
  ρ, ρe
end

# Initial Condition
function rising_bubble!(dim, Q, t, x, y, z, aux)
  DFloat          = eltype(Q)
  R_gas::DFloat   = R_d
  c_p::DFloat     = cp_d
  c_v::DFloat     = cv_d
  p0::DFloat      = MSLP
  gravity::DFloat = grav
  # perturbation parameters for rising bubble
  rx = 250
  ry = 250
  xc = 500
  yc = 260
  r  = sqrt( (x - xc)^2 + (y - yc)^2 )

  θ0::DFloat  = 303
  θ_c::DFloat = 1 // 2
  Δθ::DFloat  = -zero(DFloat)
  a::DFloat   =  50
  s::DFloat   = 100
  if r <= a
    Δθ = θ_c
  elseif r > a
    Δθ = θ_c * exp(-(r - a)^2 / s^2)
  end
  θ       = θ0 + Δθ # potential temperature
  π_exner = 1 - gravity / (c_p * θ) * y # exner pressure
  ρ       = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  P       = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T       = P / (ρ * R_gas) # temperature
  u⃗       = SVector(-zero(DFloat), -zero(DFloat), -zero(DFloat))
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
function lin_eulerflux!(F, Q, _, aux, t)
  F .= 0
  @inbounds begin
    DFloat = eltype(Q)
    γ::DFloat = γ_exact # FIXME: Remove this for some moist thermo approach

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

function wavespeed_linear(n, Q, aux, t)
  DFloat = eltype(Q)
  γ::DFloat = γ_exact # FIXME: Remove this for some moist thermo approach

  ρ0, ρe0, ϕ = aux[_a_ρ0], aux[_a_ρe0], aux[_a_ϕ]
  ρinv0 = 1 / ρ0
  P0 = (γ-1)*(ρe0 - ρ0 * ϕ)

  sqrt(ρinv0 * γ * P0)
end

function lin_source!(S,Q,aux,t)
  S .= 0
  lin_source_geopotential!(S, Q, aux, t)
end
function lin_source_geopotential!(S, Q, aux, t)
  @inbounds begin
    δρ = Q[_δρ]
    ∇ϕ = SVector(aux[_a_ϕ_x], aux[_a_ϕ_y], aux[_a_ϕ_z])
    S[_ρu] -= δρ * ∇ϕ[1]
    S[_ρv] -= δρ * ∇ϕ[2]
    S[_ρw] -= δρ * ∇ϕ[3]
  end
end

function lin_bcstate!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  if bctype == 1
    # this is already a linear boundary condition
    nofluxbc!(QP, QVP, auxP, nM, QM, QMP, auxM, bctype, t)
  else
    error("Unsupported boundary condition")
  end
end

# }}}

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt, output_steps)

  brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)))

  # User defined periodicity in the topl assignment
  # brickrange defines the domain extents
  topl = StackedBrickTopology(mpicomm, brickrange, periodicity=(false,false))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
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
  initialcondition(Q, x...) = rising_bubble!(Val(dim), Q, DFloat(0), x...)
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
  linearsolver = GeneralizedConjugateResidual(3, Q, 1e-4)

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

  # User defined number of elements
  # User defined timestep estimate
  # User defined simulation end time
  # User defined polynomial order
  numelem = (Nex,Ney)

  # Stable explicit time step
  dt = min(Δx, Δy, Δz) / soundspeed_air(300.0) / Npoly
  dt *= 40

  output_time = 0.5
  output_steps = ceil(output_time / dt)

  @info @sprintf """ ----------------------------------------------------"""
  @info @sprintf """   ______ _      _____ __  ________                  """
  @info @sprintf """  |  ____| |    |_   _|  ...  |  __  |               """
  @info @sprintf """  | |    | |      | | |   .   | |  | |               """
  @info @sprintf """  | |    | |      | | | |   | | |__| |               """
  @info @sprintf """  | |____| |____ _| |_| |   | | |  | |               """
  @info @sprintf """  | _____|______|_____|_|   |_|_|  |_|               """
  @info @sprintf """                                                     """
  @info @sprintf """ ----------------------------------------------------"""
  @info @sprintf """ Rising Bubble                                       """
  @info @sprintf """   Resolution:                                       """
  @info @sprintf """     (Δx, Δy)   = (%.2e, %.2e)                       """ Δx Δy
  @info @sprintf """     (Nex, Ney) = (%d, %d)                           """ Nex Ney
  @info @sprintf """     dt         = %.2e                               """ dt
  @info @sprintf """ ----------------------------------------------------"""

  timeend = 10
  polynomialorder = Npoly
  DFloat = Float64
  expected_engf_eng0 = Dict()
  expected_engf_eng0[Float64] = 1.4389690552059924e+00

  dim = numdims
  engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                  DFloat, dt, output_steps)

  @test engf_eng0 ≈ expected_engf_eng0[DFloat]
end
nothing
