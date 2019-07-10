using MPI
using CLIMA
using CLIMA.Topologies
using CLIMA.Grids
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
using CLIMA.GeneralizedMinimalResidualSolver
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
const _δρ, _U, _V, _W, _δE = 1:_nstate
const _U⃗ = SVector(_U, _V, _V)
const statenames = ("δρ", "U", "V", "W", "δE")

const numdims = 2
const Δx    = 5
const Δy    = 5
const Δz    = 5
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
const _a_ρ0, _a_E0, _a_ϕ, _a_ϕ_x, _a_ϕ_y, _a_ϕ_z = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z) #JK, dx, dy, dz)
  @inbounds begin
    ρ0, E0 = reference2D_ρ_E(x, y, z)
    aux[_a_ρ0] = ρ0
    aux[_a_E0] = E0
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
    δρ, U, V, W, δE = Q[_δρ], Q[_U], Q[_V], Q[_W], Q[_δE]
    ρ0, E0, ϕ = aux[_a_ρ0], aux[_a_E0], aux[_a_ϕ]
    ρ = ρ0 + δρ
    E = E0 + δE
    ρinv = 1 / ρ
    (γ-1)*(E - ρinv * (U^2 + V^2 + W^2) / 2 - ϕ * ρ)
  end
end

@inline function wavespeed(n, Q, aux, t)
  γ::eltype(Q) = γ_exact # FIXME: Remove this for some moist thermo approach
  @inbounds begin
    P = pressure(Q, aux)
    δρ, U, V, W, δE = Q[_δρ], Q[_U], Q[_V], Q[_W], Q[_δE]
    ρ0, E0 = aux[_a_ρ0], aux[_a_E0]
    ρ = ρ0 + δρ
    E = E0 + δE
    ρinv = 1 / ρ
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
  end
end

@inline function euler_flux!(F, Q, _, aux, t)
  P = pressure(Q, aux)
  @inbounds begin
    δρ, U, V, W, δE = Q[_δρ], Q[_U], Q[_V], Q[_W], Q[_δE]
    ρ0, E0 = aux[_a_ρ0], aux[_a_E0]
    ρ = ρ0 + δρ
    E = E0 + δE
    ρinv = 1 / ρ
    u, v, w = ρinv * U, ρinv * V, ρinv * W

    # Inviscid contributions
    F[1, _δρ], F[2, _δρ], F[3, _δρ] = U          , V          , W
    F[1, _U ], F[2, _U ], F[3, _U ] = u * U  + P , v * U      , w * U
    F[1, _V ], F[2, _V ], F[3, _V ] = u * V      , v * V + P  , w * V
    F[1, _W ], F[2, _W ], F[3, _W ] = u * W      , v * W      , w * W + P
    F[1, _δE], F[2, _δE], F[3, _δE] = u * (E + P), v * (E + P), w * (E + P)
  end
end

# -------------------------------------------------------------------------
# generic bc for 2d , 3d

@inline function bcstate!(QP, _, auxP, nM, QM, _, auxM, bctype, t)
  @inbounds begin
    UM, VM, WM = QM[_U], QM[_V], QM[_W]

    # No flux boundary conditions
    # No shear on walls (free-slip condition)
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_V] = VM - 2 * nM[2] * UnM
    QP[_W] = WM - 2 * nM[3] * UnM
    QP[_δρ] = QM[_δρ]
    QP[_δE] = QM[_δE]
    nothing
  end
end

@inline function source!(S,Q,aux,t)
  # Initialise the final block source term
  S .= 0

  # Typically these sources are imported from modules
  @inbounds begin
    source_geopot!(S, Q, aux, t)
  end
end

@inline function source_geopot!(S,Q,aux,t)
  @inbounds begin
    δρ = Q[_δρ]
    ρ0 = aux[_a_ρ0]
    ρ = ρ0 + δρ
    S[_U] -= aux[_a_ϕ_x] * ρ
    S[_V] -= aux[_a_ϕ_y] * ρ
    S[_W] -= aux[_a_ϕ_z] * ρ
  end
end


# ------------------------------------------------------------------
# -------------END DEF SOURCES-------------------------------------#

# initial condition
function reference2D_ρ_E(x, y, z)
  DFloat                = eltype(x)
  R_gas::DFloat         = R_d
  c_p::DFloat           = cp_d
  c_v::DFloat           = cv_d
  p0::DFloat            = MSLP
  gravity::DFloat       = grav
  # initialise with dry domain
  q_tot::DFloat         = 0
  q_liq::DFloat         = 0
  q_ice::DFloat         = 0
  # perturbation parameters for rising bubble
  rx                    = 250
  ry                    = 250
  xc                    = 500
  yc                    = 260
  r                     = sqrt( (x - xc)^2 + (y - yc)^2 )

  θ_ref::DFloat         = 303.0
  θ_c::DFloat           =   0.5
  Δθ::DFloat            =   0.0
  a::DFloat             =  50.0
  s::DFloat             = 100.0
  #=
  if r <= a
  Δθ = θ_c
  elseif r > a
  Δθ = θ_c * exp(-(r - a)^2 / s^2)
  end
  =#
  qvar                  = PhasePartition(q_tot)
  θ                     = θ_ref + Δθ # potential temperature
  π_exner               = 1.0 - gravity / (c_p * θ) * y # exner pressure
  ρ                     = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T                     = P / (ρ * R_gas) # temperature
  U, V, W               = 0.0 , 0.0 , 0.0  # momentum components
  # energy definitions
  e_kin                 = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
  e_pot                 = gravity * y
  e_int                 = cv_d *  T
  E                     = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
  # @inbounds Q[_δρ], Q[_U], Q[_V], Q[_W], Q[_δE], Q[_QT]= ρ, U, V, W, E, ρ * q_tot
  ρ, E
end
function rising_bubble!(dim, Q, t, x, y, z, aux)
  DFloat                = eltype(Q)
  R_gas::DFloat         = R_d
  c_p::DFloat           = cp_d
  c_v::DFloat           = cv_d
  p0::DFloat            = MSLP
  gravity::DFloat       = grav
  # initialise with dry domain
  q_tot::DFloat         = 0
  q_liq::DFloat         = 0
  q_ice::DFloat         = 0
  # perturbation parameters for rising bubble
  rx                    = 250
  ry                    = 250
  xc                    = 500
  yc                    = 260
  r                     = sqrt( (x - xc)^2 + (y - yc)^2 )

  θ_ref::DFloat         = 303.0
  θ_c::DFloat           =   0.5
  Δθ::DFloat            =   0.0
  a::DFloat             =  50.0
  s::DFloat             = 100.0
  if r <= a
    Δθ = θ_c
  elseif r > a
    Δθ = θ_c * exp(-(r - a)^2 / s^2)
  end
  qvar                  = PhasePartition(q_tot)
  θ                     = θ_ref + Δθ # potential temperature
  π_exner               = 1.0 - gravity / (c_p * θ) * y # exner pressure
  ρ                     = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T                     = P / (ρ * R_gas) # temperature
  U, V, W               = 0.0 , 0.0 , 0.0  # momentum components
  # energy definitions
  e_kin                 = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
  e_pot                 = gravity * y
  e_int                 = cv_d *  T
  E                     = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)

  @inbounds ρ0, E0 = aux[_a_ρ0], aux[_a_E0]
  @inbounds Q[_δρ], Q[_U], Q[_V], Q[_W], Q[_δE] = ρ-ρ0, U, V, W, E-E0
end

# {{{ Linearization
@inline function lin_eulerflux!(F, Q, _, aux, t)
  F .= 0
  @inbounds begin
    DFloat = eltype(Q)
    γ::DFloat = γ_exact # FIXME: Remove this for some moist thermo approach

    δρ, δE = Q[_δρ], Q[_δE]
    U⃗ = SVector(Q[_U], Q[_V], Q[_W])

    ρ0, E0 = aux[_a_ρ0], aux[_a_E0]
    ϕ = aux[_a_ϕ]

    ρinv0 = 1 / ρ0
    e0 = ρinv0 * E0

    P0 = (γ-1)*(E0 - ρ0 * ϕ)
    δP = (γ-1)*(δE - δρ * ϕ)

    p0 = ρinv0 * P0

    F[:, _δρ] = U⃗
    F[:, _U⃗ ] = δP * I + @SMatrix zeros(3,3)
    F[:, _δE] = (e0 + p0) * U⃗
  end
end

@inline function wavespeed_linear(n, Q, aux, t)
  DFloat = eltype(Q)
  γ::DFloat = γ_exact # FIXME: Remove this for some moist thermo approach

  ρ0, E0 = aux[_a_ρ0], aux[_a_E0]
  ϕ = aux[_a_ϕ]
  ρinv0 = 1 / ρ0
  P0 = (γ-1)*(E0 - ρ0 * ϕ)
  sqrt(ρinv0 * γ * P0)
end

@inline function lin_source!(S,Q,aux,t)
  # Initialise the final block source term
  S .= 0

  # Typically these sources are imported from modules
  @inbounds begin
    lin_source_geopot!(S, Q, aux, t)
  end
end
@inline function lin_source_geopot!(S,Q,aux,t)
  @inbounds begin
    δρ = Q[_δρ]
    S[_U] -= aux[_a_ϕ_x] * δρ
    S[_V] -= aux[_a_ϕ_y] * δρ
    S[_W] -= aux[_a_ϕ_z] * δρ
  end
end

@inline function lin_bcstate!(QP, _, auxP,
                              nM,
                              QM, _, auxM,
                              bctype, t)
  @inbounds begin
    UM, VM, WM = QM[_U], QM[_V], QM[_W]
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_V] = VM - 2 * nM[2] * UnM
    QP[_W] = WM - 2 * nM[3] * UnM
    nothing
  end
end

function linearoperator!(LQ, Q, rhs_linear!, α)
  rhs_linear!(LQ, Q, 0.0; increment = false)
  @. LQ = Q - α * LQ
end

function solve_linear_problem!(Qtt, Qhat, rhs_linear!, α, gcrk)
  Qtt .= Qhat
  LinearSolvers.linearsolve!((Ax, x) -> linearoperator!(Ax, x, rhs_linear!, α),
                             Qtt, Qhat, gcrk)
  #=
  Qtt .= Qhat
  =#
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


  # linsolver = GeneralizedConjugateResidual(3, Q, 1e-10)
  linsolver = GeneralizedMinimalResidual(30, Q, 1e-10)
  rhs_linear!(x...;increment) = SpaceMethods.odefun!(lin_spacedisc, x...;
                                                     increment=increment)
  # linearoperator!(dQ, Q, rhs_linear!, 1)
  lin_solve!(x...) = solve_linear_problem!(x..., linsolver)
  #=
  timestepper = ARK548L2SA2KennedyCarpenter(spacedisc, lin_spacedisc,
                                            lin_solve!, Q; dt = dt, t0 = 0)
  =#
  timestepper = ARK2GiraldoKellyConstantinescu(spacedisc, lin_spacedisc,
                                               lin_solve!, Q; dt = dt, t0 = 0)
  # }}}

  # timestepper = LSRK144NiegemannDiehlBusch(spacedisc, Q; dt = dt, t0 = 0)

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
  vtkdir = "vtk_RTB_IMEX_GKC"
  mkpath(vtkdir)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(output_steps) do (init=false)
    outprefix = @sprintf("%s/cns_%dD_mpirank%04d_step%04d", vtkdir, dim,
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, spacedisc, statenames)
    #=
    pvtuprefix = @sprintf("vtk/cns_%dD_step%04d", dim, step[1])
    prefixes = ntuple(i->
    @sprintf("vtk/cns_%dD_mpirank%04d_step%04d",
    dim, i-1, step[1]),
    MPI.Comm_size(mpicomm))
    writepvtu(pvtuprefix, prefixes, postnames)
    =#
    step[1] += 1
    nothing
  end

  solve!(Q, timestepper; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
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
  # dt = 0.005

  # Stable explicit time step
  dt = min(Δx, Δy, Δz) / soundspeed_air(300.0) / Npoly

  dt *= 4

  output_time = 0.5
  output_steps = ceil(output_time / dt)
  # dt = output_time / output_steps

  # run with bigger step
  # dt *= 2
  # output_steps /= 2

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

  timeend = 900
  polynomialorder = Npoly
  DFloat = Float64
  dim = numdims
  engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                  DFloat, dt, output_steps)
end
nothing
