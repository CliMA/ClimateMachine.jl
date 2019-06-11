# Load modules that are used in the CliMA project.
# These are general modules not necessarily specific
# to CliMA
using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CUDAnative
using CuArrays
# Load modules specific to CliMA project
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using Dierckx
using DelimitedFiles

# Prognostic equations: ρ, (ρu), (ρv), (ρw), (ρe_tot), (ρq_tot)
# Even for the dry example shown here, we load the moist thermodynamics module 
# and consider the dry equation set to be the same as the moist equations but
# with total specific humidity = 0. 
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0

# For a three dimensional problem 
const _nstate = 6
const _ρ, _U, _V, _W, _E, _QT= 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
const statenames = ("ρ", "U", "V", "W", "E", "QT")

const _nviscstates = 12
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _qx, _qy, _qz, _Tx, _Ty, _Tz = 1:_nviscstates

const _ngradstates = 6
const _states_for_gradient_transform = (_ρ, _U, _V, _W, _E, _QT)

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  using Random
end

const Prandtl = 71 // 100
const k_μ = cp_d / Prandtl
const μ_exact = 2.5
const xmin = 0
const ymin = 0
const zmin = 0
const xmax = 240000
const ymax = 12000
const zmax = 3000
const xc   = xmax / 2
const yc   = ymax / 2
const zc   = zmax / 2
const Nex = 60
const Ney = 20
const Nez = 1
const numdims = 2
const Npoly = 4
# Smagorinsky model requirements
const C_smag = 0.18
const Δx = (xmax-xmin) / ((Nex * Npoly) + 1)
const Δy = (ymax-ymin) / ((Ney * Npoly) + 1)
const Δz = (zmax-zmin) / ((Nez * Npoly) + 1)
if numdims == 2
  Δ = sqrt(Δx * Δy)
  const Δ2 = Δ * Δ
end


# -------------------------------------------------------------------------
# Preflux calculation: This function computes parameters required for the 
# DG RHS (but not explicitly solved for as a prognostic variable)
# In the case of the rising_thermal_bubble example: the saturation
# adjusted temperature and pressure are such examples. Since we define
# the equation and its arguments here the user is afforded a lot of freedom
# around its behaviour. 
# The preflux function interacts with the following  
# Modules: NumericalFluxes.jl 
# functions: wavespeed, cns_flux!, bcstate!
# -------------------------------------------------------------------------
@inline function preflux(Q,VF, aux, _...)
  gravity::eltype(Q) = grav
  R_gas::eltype(Q) = R_d
  @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
  ρinv = 1 / ρ
  x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
  q_tot = QT / ρ
  # Establish the current thermodynamic state using the prognostic variables
  TS = PhaseEquil(e_int, q_tot, ρ)
  T = air_temperature(TS)
  P = air_pressure(TS) # Test with dry atmosphere
  q_liq = PhasePartition(TS).liq
  θ = virtual_pottemp(TS)
  (P, u, v, w, ρinv, q_liq,T,θ)
end

# -------------------------------------------------------------------------
# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv,q_liq, T, θ)
  gravity::eltype(Q) = grav
  @inbounds begin 
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
    q_tot = QT / ρ
    TS = PhaseEquil(e_int, q_tot, ρ)
    (n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(TS)
  end
end

# -------------------------------------------------------------------------
# ### Physical Flux (Required)
#md # Here, we define the physical flux function, i.e. the conservative form
#md # of the equations of motion for the prognostic variables ρ, U, V, W, E, QT
#md # $\frac{\partial Q}{\partial t} + \nabla \cdot \boldsymbol{F} = \boldsymbol {S}$
#md # $\boldsymbol{F}$ contains both the viscous and inviscid flux components
#md # and $\boldsymbol{S}$ contains source terms.
#md # Note that the preflux calculation is splatted at the end of the function call
#md # to cns_flux!
# -------------------------------------------------------------------------
cns_flux!(F, Q, VF, aux, t) = cns_flux!(F, Q, VF, aux, t, preflux(Q,VF, aux)...)
@inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
  @inbounds begin
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    # Inviscid contributions 
    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
    F[1, _QT], F[2, _QT], F[3, _QT] = u * QT  , v * QT     , w * QT 
    # Stress tensor
    τ11, τ22, τ33 = VF[_τ11], VF[_τ22], VF[_τ33]
    τ12 = τ21 = VF[_τ12]
    τ13 = τ31 = VF[_τ13]
    τ23 = τ32 = VF[_τ23]
    vqx, vqy, vqz = VF[_qx], VF[_qy], VF[_qz]
    vTx, vTy, vTz = VF[_Tx], VF[_Ty], VF[_Tz]
    # Viscous contributions
    F[1, _U] -= τ11; F[2, _U] -= τ12; F[3, _U] -= τ13
    F[1, _V] -= τ21; F[2, _V] -= τ22; F[3, _V] -= τ23
    F[1, _W] -= τ31; F[2, _W] -= τ32; F[3, _W] -= τ33
    # Energy dissipation
    F[1, _E] -= u * τ11 + v * τ12 + w * τ13 + k_μ*vTx
    F[2, _E] -= u * τ21 + v * τ22 + w * τ23 + k_μ*vTy
    F[3, _E] -= u * τ31 + v * τ32 + w * τ33 + k_μ*vTz
    # Viscous contributions to mass flux term
    F[1, _ρ] -=  vqx
    F[2, _ρ] -=  vqy
    F[3, _ρ] -=  vqz
    F[1, _QT] -=  vqx
    F[2, _QT] -=  vqy
    F[3, _QT] -=  vqz
  end
end

# -------------------------------------------------------------------------
#md # Here we define a function to extract the velocity components from the 
#md # prognostic equations (i.e. the momentum and density variables). This 
#md # function is not required in general, but provides useful functionality 
#md # in some cases. 
# -------------------------------------------------------------------------
# Compute the velocity from the state
gradient_vars!(vel, Q, aux, t, _...) = gradient_vars!(vel, Q, aux, t, preflux(Q,~,aux)...)
@inline function gradient_vars!(vel, Q, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
  @inbounds begin
    # ordering should match states_for_gradient_transform
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    E, QT = Q[_E], Q[_QT]
    ρinv = 1 / ρ
    vel[1], vel[2], vel[3] = u, v, w
    vel[4], vel[5], vel[6] = ρinv * E, QT, T
    vel[7] = θ
  end
end

# -------------------------------------------------------------------------
#md ### Auxiliary Function (Not required)
#md # In this example the auxiliary function is used to store the spatial
#md # coordinates. This may also be used to store variables for which gradients
#md # are needed, but are not available through teh prognostic variable 
#md # calculations. (An example of this will follow - in the Smagorinsky model, 
#md # where a local Richardson number via potential temperature gradient is required)
# -------------------------------------------------------------------------
const _nauxstate = 4
const _a_x, _a_y, _a_z, _a_qliq = 1:_nauxstate
@inline function coordinates!(aux, x, y, z)
    @inbounds begin
    aux[_a_x] = x
    aux[_a_y] = y
    aux[_a_z] = z
  end
end

# -------------------------------------------------------------------------
#md ### Viscous fluxes. 
#md # The viscous flux function compute_stresses computes the components of 
#md # the velocity gradient tensor, and the corresponding strain rates to
#md # populate the viscous flux array VF. SijSij is calculated in addition
#md # to facilitate implementation of the constant coefficient Smagorinsky model
#md # (pending)
@inline function compute_stresses!(VF, grad_vel,_...)
  μ::eltype(VF) = μ_exact
  gravity::eltype(VF) = grav
  @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]
    # compute gradients of moist vars and temperature
    dqdx, dqdy, dqdz = grad_vel[1, 5], grad_vel[2, 5], grad_vel[3, 5]
    dTdx, dTdy, dTdz = grad_vel[1, 6], grad_vel[2, 6], grad_vel[3, 6]
#   # virtual potential temperature gradient: for richardson calculation
#   dθdx, dθdy, dθdz = grad_vel[1, 7], grad_vel[2, 7], grad_vel[3, 7]
    # strains
    ϵ11 = dudx
    ϵ22 = dvdy
    ϵ33 = dwdz
    ϵ12 = (dudy + dvdx) / 2
    ϵ13 = (dudz + dwdx) / 2
    ϵ23 = (dvdz + dwdy) / 2
    # --------------------------------------------
    # SMAGORINSKY COEFFICIENT COMPONENTS
    # --------------------------------------------
    SijSij = (ϵ11^2 + ϵ22^2 + ϵ33^2
              + 2.0 * ϵ12^2
              + 2.0 * ϵ13^2 
              + 2.0 * ϵ23^2) 
    modSij = sqrt(2.0 * SijSij) 
    #Richardson = (grav/θ) * dθdy / sqrt(2.0 * SijSij) 
    #auxr = max(0.0, 1.0 - Richardson/Prandtl)
    ν_t = C_smag * C_smag * Δ2 * modSij #* sqrt(auxr)
    # --------------------------------------------
    # deviatoric stresses
    VF[_τ11] = 2 * ν_t * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ22] = 2 * ν_t * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ33] = 2 * ν_t * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ12] = 2 * ν_t * ϵ12
    VF[_τ13] = 2 * ν_t * ϵ13
    VF[_τ23] = 2 * ν_t * ϵ23
    VF[_qx],VF[_qy], VF[_qz]  = dqdx, dqdy, dqdz
    VF[_Tx],VF[_Ty],VF[_Tz]   = dTdx, dTdy, dTdz
    # FIXME : shouldnt need to carry the gradient of the virtualpottemp around
#    VF[_θx], VF[_θy], VF[_θz] = dθdx, dθdy, dθdz
  end
end
# -------------------------------------------------------------------------
# generic bc for 2d , 3d
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t, PM, uM, vM, wM, ρinvM, q_liqM, TM, θM)
  @inbounds begin
    x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
    ρM, UM, VM, WM, EM, QTM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E], QM[_QT]
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_V] = VM - 2 * nM[2] * UnM
    QP[_W] = WM - 2 * nM[3] * UnM
    QP[_ρ] = ρM
    QP[_E] = EM
    QP[_QT] = QTM
    VFP .= VFM
    nothing # preflux is computed for the ⁺ state in the NumericalFluxes module 
  end
end
# -------------------------------------------------------------------------

@inline stresses_boundary_penalty!(VF, _...) = VF.=0

@inline function stresses_penalty!(VF, nM, velM, QM, aM, velP, QP, aP, t)
  @inbounds begin
    n_Δvel = similar(VF, Size(3, 3))
    for j = 1:3, i = 1:3
      n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end
    compute_stresses!(VF, n_Δvel)
  end
end
# -------------------------------------------------------------------------
@inline function source!(S,Q,aux,t)
  # Initialise the sum-total of all sources
  S .= 0
  # Typically these sources are imported from modules
  @inbounds begin
    source_sponge!(S, Q, aux, t)
    source_gravity!(S, Q, aux, t)
  end
end

@inline function source_sponge!(S, Q, aux, t)
  y = aux[_a_y]
  x = aux[_a_x]
  V = Q[_V]
  U = Q[_U]
  #Define Sponge Boundaries      
  xc       = (xmax + xmin)/2
  ysponge  = 0.85 * ymax
  xsponger = xmax - 0.15*abs(xmax - xc)
  xspongel = xmin + 0.15*abs(xmin - xc)
  csxl  = 0.0
  csxr  = 0.0
  ctop  = 0.0
  csx   = 1.0
  ct    = 1.0 
  spacefilt = 0.0
  spacefilt2 = 0.0
  spacefilt4 = 0.0
  #x left and right
  #xsl
  if (x <= xspongel)
    spacefilt = sinpi(1/2 * (x - xspongel)/(xmin - xspongel))
    spacefilt2 = spacefilt * spacefilt
    spacefilt4 = spacefilt2 * spacefilt2 
    csxl = csx * spacefilt4
  end
  #xsr
  if (x >= xsponger)
    spacefilt = sinpi(1/2 * (x - xsponger)/(xmax - xsponger))
    spacefilt2 = spacefilt * spacefilt
    spacefilt4 = spacefilt2 * spacefilt2 
    csxr = csx * spacefilt4
  end
  #Vertical sponge:         
  if (y >= ysponge)
    spacefilt = sinpi(1/2 * (x - ysponge)/(xmax - ysponge))
    spacefilt2 = spacefilt * spacefilt
    spacefilt4 = spacefilt2 * spacefilt2 
    ctop = csx * spacefilt4
  end
  beta  = 1.0 - (1.0 - ctop)*(1.0 - csxl)*(1.0 - csxr)
  beta  = min(beta, 1.0)
  alpha = 1.0 - beta
  S[_U] -= beta * U
  S[_V] -= beta * V  
end

@inline function source_gravity!(S,Q,aux,t)
  gravity::eltype(Q) = grav
  @inbounds begin
    ρ, U, V, W, E  = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    S[_V] += - ρ * gravity
  end
end

# ------------------------------------------------------------------
# -------------END DEF SOURCES-------------------------------------# 

function squall_line!(dim, Q, t, x, y, z, _...)
  
  DFloat 	    = eltype(Q)
  p0::DFloat 	    = MSLP
    
  #--------------------------
  function read_sounding()
      #read in the original squal sounding
      fsounding  = open(joinpath(@__DIR__, "./soundings/sounding_JCP2013_with_pressure.dat"))
      sounding = readdlm(fsounding)
      close(fsounding)
      (nzmax, ncols) = size(sounding)
      if nzmax == 0
          error("SOUNDING ERROR: The Sounding file is empty!")
      end
      return (sounding, nzmax, ncols)
  end
  #--------------------------
  # ----------------------------------------------------
  # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
  # This driver accepts data in 6 column format
  # ----------------------------------------------------
  (sounding, _, ncols) = read_sounding()
  
  # WARNING: Not all sounding data is formatted/scaled 
  # the same. Care required in assigning array values
  # height theta qv    u     v     pressure
  zinit, tinit, qinit, uinit, vinit, pinit  = sounding[:, 1],
  sounding[:, 2],
  sounding[:, 3],
  sounding[:, 4],
  sounding[:, 5],
  sounding[:, 6]

  #------------------------------------------------------
  # GET SPLINE FUNCTION
  #------------------------------------------------------
  spl_tinit    = Spline1D(zinit, tinit; k=1)
  spl_qinit    = Spline1D(zinit, qinit; k=1)
  spl_uinit    = Spline1D(zinit, uinit; k=1)
  spl_vinit    = Spline1D(zinit, vinit; k=1)
  spl_pinit    = Spline1D(zinit, pinit; k=1)
  # --------------------------------------------------
  # INITIALISE ARRAYS FOR INTERPOLATED VALUES
  # --------------------------------------------------
  datat          = spl_tinit(y)
  dataq          = spl_qinit(y)
  datau          = spl_uinit(y)
  datav          = spl_vinit(y)
  datap          = spl_pinit(y)
  dataq          = dataq * 1.0e-3
  
  # Using new thermodynamic state identifier to determine the moist variable values
  qvar            = PhasePartition(dataq)
  R_gas::DFloat   = gas_constant_air(qvar)
  c_p::DFloat     = cp_m(qvar)
  c_v::DFloat     = cv_m(qvar)
  cvoverR         = c_v/R_gas
  gravity::DFloat = grav
  
  #TODO Driver constant parameters need references
  rvapor        = 461.0
  levap         = 2.5e6
  es0           = 611.0
  pi0           = 1.0
  p0            = MSLP
  theta0        = 300.4675
  c2            = R_gas / c_p
  c1            = 1.0 / c2
  
  # Convert dataq to kg/kg
  datapi        = (datap / MSLP) ^ (c2)                         # Exner pressure from sounding data
  thetav        = datat * (1.0 + 0.61 * dataq)                  # Liquid potential temperature

  # theta perturbation
  dtheta        = 0.0
  thetac        = 5.0
  rx            = 10000.0
  ry            =  1500.0
  r		= sqrt( ((x-120000)/rx )^2 + ((y - 2000.0)/ry)^2)
  if (r <= 1.0)
      dtheta	  = thetac * (cospi(0.5*r))^2
  end
  θ             = thetav + dtheta
  datarho       = datap / (R_gas * datapi *  θ)
  e             = dataq * datap * rvapor/(dataq * rvapor + R_gas)
  
  q_tot         = dataq
  P             = datap                                         # Assumed known from sounding
  ρ             = datarho
  T             = P / (ρ * R_gas)
  u, v, w       = datau, 0.0, 0.0
  U      	= ρ * u
  V      	= ρ * v
  W      	= ρ * w
  
  # Calculation of energy per unit mass
  e_kin = (u^2 + v^2 + w^2) / 2  
  e_pot = gravity * y
  e_int = internal_energy(T, qvar)
  # Total energy 
  E = ρ * total_energy(e_kin, e_pot, T, qvar)
  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT] = ρ, U, V, W, E, ρ * q_tot
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

  ArrayType = CuArray

  brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)))
  
  # User defined periodicity in the topl assignment
  # brickrange defines the domain extents
  topl = BrickTopology(mpicomm, brickrange, periodicity=(true,false))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N)
  
  numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed, preflux)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!, bcstate!, wavespeed, preflux)

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = cns_flux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!, 
                           number_gradient_states = _ngradstates,
                           states_for_gradient_transform =
                            _states_for_gradient_transform,
                           number_viscous_states = _nviscstates,
                           gradient_transform! = gradient_vars!,
                           viscous_transform! = compute_stresses!,
                           viscous_penalty! = stresses_penalty!,
                           viscous_boundary_penalty! = stresses_boundary_penalty!,
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           coordinates!,
                           source! = source!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = squall_line!(Val(dim), Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(5, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  npoststates = 8
  _P, _u, _v, _w, _ρinv, _q_liq, _T, _θ = 1:npoststates
  postnames = ("P","u", "v", "w", "ρinv", "QL", "T", "THETA")
  postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

  step = [0]
  mkpath("vtk-squall")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(2000) do (init=false)
    DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc,
                                               Q) do R, Q, QV, aux
      @inbounds let
        (R[_P], R[_u], R[_v], R[_w], R[_ρinv], R[_q_liq], R[_T], R[_θ]) = preflux(Q, QV, aux)
      end
    end

    outprefix = @sprintf("vtk-squall/cns_%dD_mpirank%04d_step%04d", dim,
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames,
                                         postprocessarray, postnames)
    step[1] += 1
    nothing
  end
  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


  # Print some end of the simulation information
  engf = norm(Q)
  if integration_testing
    Qe = MPIStateArray(spacedisc,
                       (Q, x...) -> initialcondition!(Val(dim), Q,
                                                      DFloat(timeend), x...))
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf/eng0 engf-eng0 errf errf / engfe
  else
    @info @sprintf """Finished
    norm(Q)            = %.16e
    norm(Q) / norm(Q₀) = %.16e
    norm(Q) - norm(Q₀) = %.16e""" engf engf/eng0 engf-eng0
  end
  integration_testing ? errf : (engf / eng0)
end

using Test
let
  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
  mpicomm = MPI.COMM_WORLD
  if MPI.Comm_rank(mpicomm) == 0
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
    global_logger(ConsoleLogger(stderr, loglevel))
  else
    global_logger(NullLogger())
  end
    # User defined number of elements
    # User defined timestep estimate
    # User defined simulation end time
    # User defined polynomial order 
    numelem = (Nex,Ney,Nez)
    dt = 0.001
    timeend = 3600 * 3
    polynomialorder = 7
    DFloat = Float64
    dim = 2
    engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                        DFloat, dt)
end
isinteractive() || MPI.Finalize()
nothing
