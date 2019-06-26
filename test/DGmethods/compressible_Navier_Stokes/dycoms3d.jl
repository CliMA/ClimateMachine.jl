
# Load Modules 
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
using CLIMA.ParametersType
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using DelimitedFiles
using Dierckx
using Random

using TimerOutputs

const to = TimerOutput()

if haspkg("CuArrays")
    using CUDAdrv
    using CUDAnative
    using CuArrays
    CuArrays.allowscalar(false)
    const ArrayType = CuArray
else
    const ArrayType = Array
end

# Prognostic equations: ρ, (ρu), (ρv), (ρw), (ρe_tot), (ρq_tot)
# For the dry example shown here, we load the moist thermodynamics module 
# and consider the dry equation set to be the same as the moist equations but
# with total specific humidity = 0. 
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0, Omega

# State labels 
const _nstate = 6
const _ρ, _U, _V, _W, _E, _QT = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
const statenames = ("RHO", "U", "V", "W", "E", "QT")

# Viscous state labels
const _nviscstates = 13
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _qx, _qy, _qz, _Tx, _Ty, _Tz, _SijSij = 1:_nviscstates

# Gradient state labels
const _ngradstates = 6
const _states_for_gradient_transform = (_ρ, _U, _V, _W, _E, _QT)

const _nauxstate = 8
const _a_z, _a_sponge, _a_02z, _a_z2inf, _a_T, _a_P, _a_q_liq, _a_soundspeed_air  = 1:_nauxstate

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# Problem constants (TODO: parameters module (?))
@parameter Prandtl_t 1//3 "Prandtl_t"
@parameter cp_over_prandtl cp_d / Prandtl_t "cp_over_prandtl"

# Random number seed
const seed = MersenneTwister(0)

# Problem description 
# --------------------
# 2D thermal perturbation (cold bubble) in a neutrally stratified atmosphere
# No wall-shear, lateral periodic boundaries with no-flux walls at the domain
# top and bottom. 
# Inviscid, Constant viscosity, StandardSmagorinsky, MinimumDissipation
# filters are tested against this benchmark problem
# TODO: link to module SubGridScaleTurbulence

#
# User Input
#
const numdims = 3
const Npoly = 4

#
# Define grid size 
#
Δx    = 30
Δy    = 30
Δz    = 5

#
# OR:
#
# Set Δx < 0 and define  Nex, Ney, Nez:
#
(Nex, Ney, Nez) = (5, 5, 5)

# Physical domain extents 
const (xmin, xmax) = (0, 3820)
const (ymin, ymax) = (0, 3820)
const (zmin, zmax) = (0, 1500)

#Get Nex, Ney from resolution
const Lx = xmax - xmin
const Ly = ymax - ymin
const Lz = zmax - ymin

if ( Δx > 0)
    #
    # User defines the grid size:
    #
    ratiox = (Lx/Δx - 1)/Npoly
    ratioy = (Ly/Δy - 1)/Npoly
    ratioz = (Lz/Δz - 1)/Npoly
    Nex = ceil(Int64, ratiox)
    Ney = ceil(Int64, ratioy)
    Nez = ceil(Int64, ratioz)
    
else
    #
    # User defines the number of elements:
    #
    Δx = Lx / ((Nex * Npoly) + 1)
    Δy = Ly / ((Ney * Npoly) + 1)
    Δz = Lz / ((Nez * Npoly) + 1)
end


DoF = (Nex*Ney*Nez)*(Npoly+1)^numdims*(_nstate)
DoFstorage = (Nex*Ney*Nez)*(Npoly+1)^numdims*(_nstate + _nviscstates + _nauxstate + CLIMA.Grids._nvgeo) +
             (Nex*Ney*Nez)*(Npoly+1)^(numdims-1)*2^numdims*(CLIMA.Grids._nsgeo)


# Smagorinsky model requirements : TODO move to SubgridScaleTurbulence module 
@parameter C_smag 0.15 "C_smag"
# Equivalent grid-scale
Δ = (Δx * Δy * Δz)^(1/3)
const Δsqr = Δ * Δ

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
@inline function preflux(Q, VF, aux, _...)
  @inbounds begin
    ρ, U, V, W = Q[_ρ], Q[_U], Q[_V], Q[_W]
    ρinv = 1 / ρ
    u, v, w = ρinv * U, ρinv * V, ρinv * W
  end
end

#-------------------------------------------------------------------------
#md # Soundspeed computed using the thermodynamic state TS
# max eigenvalue
@inline function wavespeed(n, Q, aux, t, u, v, w)
  @inbounds begin
    (n[1] * u + n[2] * v + n[3] * w) + aux[_a_soundspeed_air]
  end
end


# -------------------------------------------------------------------------
# ### read sounding
#md # 
#md # The sounding file contains the following quantities along a 1D column.
#md # It needs to have the following structure:
#md #
#md # z[m]   theta[K]  q[g/kg]   u[m/s]   v[m/s]   p[Pa]
#md # ...      ...       ...      ...      ...      ...
#md #
#md #
# -------------------------------------------------------------------------
function read_sounding()
  #read in the original squal sounding
  fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_DYCOMS_TEST1.dat"))
  #fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_DYCOMS_from_PyCles.dat"))
  sounding = readdlm(fsounding)
  close(fsounding)
  (nzmax, ncols) = size(sounding)
  if nzmax == 0
    error("SOUNDING ERROR: The Sounding file is empty!")
  end
  return (sounding, nzmax, ncols)
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
@inline function cns_flux!(F, Q, VF, aux, t, u, v, w)
  @inbounds begin
    DFloat = eltype(F)
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    P = aux[_a_P]
    # Inviscid contributions
    F[1, _ρ],  F[2, _ρ],  F[3, _ρ]  = U          , V          , W
    F[1, _U],  F[2, _U],  F[3, _U]  = u * U  + P , v * U      , w * U
    F[1, _V],  F[2, _V],  F[3, _V]  = u * V      , v * V + P  , w * V
    F[1, _W],  F[2, _W],  F[3, _W]  = u * W      , v * W      , w * W + P
    F[1, _E],  F[2, _E],  F[3, _E]  = u * (E + P), v * (E + P), w * (E + P)
    F[1, _QT], F[2, _QT], F[3, _QT] = u * QT     , v * QT     , w * QT

    #Derivative of T and Q:
    vqx, vqy, vqz = VF[_qx], VF[_qy], VF[_qz]
    vTx, vTy, vTz = VF[_Tx], VF[_Ty], VF[_Tz]

    # Radiation contribution
    F_rad = ρ * radiation(aux)

    SijSij = VF[_SijSij]

    #Dynamic eddy viscosity from Smagorinsky:
    ν_e = sqrt(2SijSij) * C_smag^2 * DFloat(Δsqr)
    D_e = ν_e / Prandtl_t

    # Multiply stress tensor by viscosity coefficient:
    τ11, τ22, τ33 = VF[_τ11] * ν_e, VF[_τ22]* ν_e, VF[_τ33] * ν_e
    τ12 = τ21 = VF[_τ12] * ν_e
    τ13 = τ31 = VF[_τ13] * ν_e
    τ23 = τ32 = VF[_τ23] * ν_e

    # Viscous velocity flux (i.e. F^visc_u in Giraldo Restelli 2008)
    F[1, _U] -= τ11; F[2, _U] -= τ12; F[3, _U] -= τ13
    F[1, _V] -= τ21; F[2, _V] -= τ22; F[3, _V] -= τ23
    F[1, _W] -= τ31; F[2, _W] -= τ32; F[3, _W] -= τ33

    # Viscous Energy flux (i.e. F^visc_e in Giraldo Restelli 2008)
    F[1, _E] -= u * τ11 + v * τ12 + w * τ13 + cp_over_prandtl * vTx * ν_e
    F[2, _E] -= u * τ21 + v * τ22 + w * τ23 + cp_over_prandtl * vTy * ν_e
    F[3, _E] -= u * τ31 + v * τ32 + w * τ33 + cp_over_prandtl * vTz * ν_e

    F[3, _E] -= F_rad
    # Viscous contributions to mass flux terms
    F[1, _QT] -=  vqx * D_e
    F[2, _QT] -=  vqy * D_e
    F[3, _QT] -=  vqz * D_e
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
@inline function gradient_vars!(vel, Q, aux, t, u, v, w)
  @inbounds begin
    T = aux[_a_T]
    E, QT = Q[_E], Q[_QT]

    # ordering should match states_for_gradient_transform
    vel[1], vel[2], vel[3] = u, v, w
    vel[4], vel[5], vel[6] = E, QT, T
  end
end

@inline function radiation(aux)
  @inbounds begin
    DFloat = eltype(aux)
    zero_to_z = aux[_a_02z]
    z_to_inf = aux[_a_z2inf]
    z = aux[_a_z]
    z_i = 840  # Start with constant inversion height of 840 meters then build in check based on q_tot
    Δz_i = max(z - z_i, zero(DFloat))
    # Constants
    F_0 = 70
    F_1 = 22
    α_z = 1
    ρ_i = DFloat(1.22)
    D_subsidence = DFloat(3.75e-6)
    term1 = F_0 * exp(-z_to_inf) 
    term2 = F_1 * exp(-zero_to_z)
    term3 = ρ_i * cp_d * D_subsidence * α_z * (DFloat(0.25) * (cbrt(Δz_i))^4 + z_i * cbrt(Δz_i))
    F_rad = term1 + term2 + term3  
  end
end

# -------------------------------------------------------------------------
#md ### Viscous fluxes. 
#md # The viscous flux function compute_stresses computes the components of 
#md # the velocity gradient tensor, and the corresponding strain rates to
#md # populate the viscous flux array VF. SijSij is calculated in addition
#md # to facilitate implementation of the constant coefficient Smagorinsky model
#md # (pending)
@inline function compute_stresses!(VF, grad_vel, _...)
  @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]
    # compute gradients of moist vars and temperature
    dqdx, dqdy, dqdz = grad_vel[1, 5], grad_vel[2, 5], grad_vel[3, 5]
    dTdx, dTdy, dTdz = grad_vel[1, 6], grad_vel[2, 6], grad_vel[3, 6]
    # virtual potential temperature gradient: for richardson calculation
    # strains
    # --------------------------------------------
    # SMAGORINSKY COEFFICIENT COMPONENTS
    # --------------------------------------------
    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = (dudy + dvdx) / 2
    S13 = (dudz + dwdx) / 2
    S23 = (dvdz + dwdy) / 2
    # --------------------------------------------
    # SMAGORINSKY COEFFICIENT COMPONENTS
    # --------------------------------------------
    # FIXME: Grab functions from module SubgridScaleTurbulence 
    SijSij = S11^2 + S22^2 + S33^2 + 2S12^2 + 2S13^2 + 2S23^2

    #--------------------------------------------
    # deviatoric stresses
    # Fix up index magic numbers
    VF[_τ11] = 2 * (S11 - (S11 + S22 + S33) / 3)
    VF[_τ22] = 2 * (S22 - (S11 + S22 + S33) / 3)
    VF[_τ33] = 2 * (S33 - (S11 + S22 + S33) / 3)
    VF[_τ12] = 2 * S12
    VF[_τ13] = 2 * S13
    VF[_τ23] = 2 * S23

    # TODO: Viscous stresse come from SubgridScaleTurbulence module
    VF[_qx], VF[_qy], VF[_qz] = dqdx, dqdy, dqdz
    VF[_Tx], VF[_Ty], VF[_Tz] = dTdx, dTdy, dTdz
    VF[_SijSij] = SijSij
  end
end
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
#md ### Auxiliary Function (Not required)
#md # In this example the auxiliary function is used to store the spatial
#md # coordinates. This may also be used to store variables for which gradients
#md # are needed, but are not available through teh prognostic variable 
#md # calculations. (An example of this will follow - in the Smagorinsky model, 
#md # where a local Richardson number via potential temperature gradient is required)
# -------------------------------------------------------------------------
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    DFloat = eltype(aux)
    aux[_a_z] = z

    #Sponge
    csleft  = zero(DFloat)
    csright = zero(DFloat)
    csfront = zero(DFloat)
    csback  = zero(DFloat)
    ctop    = zero(DFloat)

    cs_left_right = zero(DFloat)
    cs_front_back = zero(DFloat)
    ct            = DFloat(0.75)

    domain_left  = xmin
    domain_right = xmax

    domain_front = ymin
    domain_back  = ymax

    domain_bott  = zmin
    domain_top   = zmax

    #END User modification on domain parameters.

    # Define Sponge Boundaries
    xc       = (domain_right + domain_left) / 2
    yc       = (domain_back  + domain_front) / 2
    zc       = (domain_top   + domain_bott) / 2

    top_sponge  = DFloat(0.85) * domain_top
    xsponger    = domain_right - DFloat(0.15) * (domain_right - xc)
    xspongel    = domain_left  + DFloat(0.15) * (xc - domain_left)
    ysponger    = domain_back  - DFloat(0.15) * (domain_back - yc)
    yspongel    = domain_front + DFloat(0.15) * (yc - domain_front)

    #x left and right
    #xsl
    if x <= xspongel
      csleft = cs_left_right * (sinpi((x - xspongel)/2/(domain_left - xspongel)))^4
    end
    #xsr
    if x >= xsponger
      csright = cs_left_right * (sinpi((x - xsponger)/2/(domain_right - xsponger)))^4
    end
    #y left and right
    #ysl
    if y <= yspongel
      csfront = cs_front_back * (sinpi((y - yspongel)/2/(domain_front - yspongel)))^4
    end
    #ysr
    if y >= ysponger
      csback = cs_front_back * (sinpi((y - ysponger)/2/(domain_back - ysponger)))^4
    end

    #Vertical sponge:
    if z >= top_sponge
      ctop = ct * (sinpi((z - top_sponge)/2/(domain_top - top_sponge)))^4
    end

    beta  = 1 - (1 - ctop) #*(1.0 - csleft)*(1.0 - csright)*(1.0 - csfront)*(1.0 - csback)
    beta  = min(beta, 1)
    aux[_a_sponge] = beta
  end
end

# -------------------------------------------------------------------------
# generic bc for 2d , 3d

@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t, uM, vM, wM)
  @inbounds begin
    UM, VM, WM = QM[_U], QM[_V], QM[_W]
    # No flux boundary conditions
    # No shear on walls (free-slip condition)
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_V] = VM - 2 * nM[2] * UnM
    QP[_W] = WM - 2 * nM[3] * UnM
    VFP .= 0
    nothing
  end
end

# -------------------------------------------------------------------------
@inline function stresses_boundary_penalty!(VF, _...) 
  VF .= 0
end

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
  # Initialise the final block source term 
  S .= 0

  # Typically these sources are imported from modules
  @inbounds begin
    source_geopot!(S, Q, aux, t)
    source_sponge!(S, Q, aux, t)
    source_geostrophic!(S, Q, aux, t)
  end
end

"""
Geostrophic wind forcing
"""
@inline function source_geostrophic!(S,Q,aux,t)
  DFloat = eltype(S)
  f_coriolis = DFloat(7.62e-5)
  U_geostrophic = DFloat(7)
  V_geostrophic = DFloat(-5.5)
  @inbounds begin
    U = Q[_U]
    V = Q[_V]
    S[_U] -= f_coriolis * (U - U_geostrophic)
    S[_V] -= f_coriolis * (V - V_geostrophic)
  end
end

@inline function source_sponge!(S,Q,aux,t)
  @inbounds begin
    U, V, W  = Q[_U], Q[_V], Q[_W]
    beta     = aux[_a_sponge]
    S[_U] -= beta * U
    S[_V] -= beta * V
    S[_W] -= beta * W
  end
end

@inline function source_geopot!(S,Q,aux,t)
  @inbounds S[_W] += - Q[_ρ] * grav
end

# Test integral exactly according to the isentropic vortex example
@inline function integral_knl(val, Q, aux)
  κ = 85
  @inbounds begin
    ρ = Q[_ρ]
    q_liq = aux[_a_q_liq]
    val[1] = ρ * κ * q_liq
  end
end

function preodefun!(disc, Q, t)
  DGBalanceLawDiscretizations.dof_iteration!(disc.auxstate, disc, Q) do R, Q, QV, aux
    @inbounds let
      ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
      z = aux[_a_z]
      e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * grav * z) / ρ
      q_tot = QT / ρ

      TS = PhaseEquil(e_int, q_tot, ρ)
      T = air_temperature(TS)
      P = air_pressure(TS) # Test with dry atmosphere
      q_liq = PhasePartition(TS).liq

      R[_a_T] = T
      R[_a_P] = P
      R[_a_q_liq] = q_liq
      R[_a_soundspeed_air] = soundspeed_air(TS)
    end
  end

  integral_computation(disc, Q, t)
end

function integral_computation(disc, Q, t)
  DGBalanceLawDiscretizations.indefinite_stack_integral!(disc, integral_knl, Q,
                                                         (_a_02z))
  DGBalanceLawDiscretizations.reverse_indefinite_stack_integral!(disc,
                                                                 _a_z2inf,
                                                                 _a_02z)
end

# initial condition
"""
    User-specified. Required.
    This function specifies the initial conditions
    for the dycoms driver.
"""
function dycoms!(dim, Q, t, spl_tinit, spl_qinit, spl_uinit, spl_vinit,
                 spl_pinit, x, y, z, _...)
  DFloat         = eltype(Q)
  # --------------------------------------------------
  # INITIALISE ARRAYS FOR INTERPOLATED VALUES
  # --------------------------------------------------
  xvert          = z

  datat          = spl_tinit(xvert)
  dataq          = spl_qinit(xvert)
  datau          = spl_uinit(xvert)
  datav          = spl_vinit(xvert)
  datap          = spl_pinit(xvert)
  dataq          = dataq / 1000

  randnum1   = rand(seed, DFloat) / 100
  randnum2   = rand(seed, DFloat) / 100

  θ_liq = datat + randnum1 * datat
  q_tot = dataq + randnum2 * dataq
  P     = datap
  T     = air_temperature_from_liquid_ice_pottemp(θ_liq, P, PhasePartition(q_tot))
  ρ     = air_density(T, P)

  # energy definitions
  u, v, w     = datau, datav, zero(DFloat) #geostrophic. TO BE BUILT PROPERLY if Coriolis is considered
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = (u^2 + v^2 + w^2) / 2
  e_pot       = grav * xvert
  e_int       = internal_energy(T, PhasePartition(q_tot))
  E           = ρ * total_energy(e_kin, e_pot, T, PhasePartition(q_tot))

  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT] = ρ, U, V, W, E, ρ * q_tot
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

  brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)),
                range(DFloat(zmin), length=Ne[3]+1, DFloat(zmax)))


  # User defined periodicity in the topl assignment
  # brickrange defines the domain extents
  @timeit to "Topo init" topl = StackedBrickTopology(mpicomm, brickrange, periodicity=(true,true,false))

  @timeit to "Grid init" grid = DiscontinuousSpectralElementGrid(topl,
                                                                 FloatType = DFloat,
                                                                 DeviceArray = ArrayType,
                                                                 polynomialorder = N)

  numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed, preflux)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!, bcstate!, wavespeed, preflux)

  # spacedisc = data needed for evaluating the right-hand side function
  @timeit to "Space Disc init" spacedisc = DGBalanceLaw(grid = grid,
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
                                                        auxiliary_state_initialization! = (x...) ->
                                                        auxiliary_state_initialization!(x...),
                                                        source! = source!,
                                                        preodefun! = preodefun!)

  # This is a actual state/function that lives on the grid
  @timeit to "IC init" begin
    # ----------------------------------------------------
    # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
    # This driver accepts data in 6 column format
    # ----------------------------------------------------
    (sounding, _, ncols) = read_sounding()

    # WARNING: Not all sounding data is formatted/scaled
    # the same. Care required in assigning array values
    # height theta qv    u     v     pressure
    zinit, tinit, qinit, uinit, vinit, pinit  =
      sounding[:, 1], sounding[:, 2], sounding[:, 3], sounding[:, 4], sounding[:, 5], sounding[:, 6]
    #------------------------------------------------------
    # GET SPLINE FUNCTION
    #------------------------------------------------------
    spl_tinit    = Spline1D(zinit, tinit; k=1)
    spl_qinit    = Spline1D(zinit, qinit; k=1)
    spl_uinit    = Spline1D(zinit, uinit; k=1)
    spl_vinit    = Spline1D(zinit, vinit; k=1)
    spl_pinit    = Spline1D(zinit, pinit; k=1)

    initialcondition(Q, x...) = dycoms!(Val(dim), Q, DFloat(0), spl_tinit,
                                        spl_qinit, spl_uinit, spl_vinit,
                                        spl_pinit, x...)
    Q = MPIStateArray(spacedisc, initialcondition)
  end

  @timeit to "Time stepping init" begin
    lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)

    #=eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0
    =#
    # Set up the information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
      if s
        starttime[] = now()
      else
        #energy = norm(Q)
        #globmean = global_mean(Q, _ρ)
        @info @sprintf("""Update
                       simtime = %.16e
                       runtime = %s""",
                       ODESolvers.gettime(lsrk),
                       Dates.format(convert(Dates.DateTime,
                                            Dates.now()-starttime[]),
                                    Dates.dateformat"HH:MM:SS")) #, energy )#, globmean)
      end
    end

    npoststates = 10
    _int1, _int2, _betaout, _P, _u, _v, _w, _q_liq, _T = 1:npoststates
    postnames = ("INT1", "INT2", "BETA", "P", "u", "v", "w", "_q_liq", "T")
    postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

    step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1000) do (init=false)
      DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc, Q) do R, Q, QV, aux
        @inbounds let
          F_rad_out = radiation(aux)
          u, v, w = preflux(Q, QV, aux)
          R[_int1] = aux[_a_02z]
          R[_int2] = aux[_a_z2inf]
          R[_betaout] = F_rad_out
          R[_P] = aux[_a_P]
          R[_u] = u
          R[_v] = v
          R[_w] = w
          R[_q_liq] = aux[_a_q_liq]
          R[_T] = aux[_a_q_T]
        end
      end

      outprefix = @sprintf("cns_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
      @debug "doing VTK output" outprefix
      writevtk(outprefix, Q, spacedisc, statenames,
               postprocessarray, postnames)
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
  end

  @info @sprintf """Starting...
    norm(Q) = %25.16e""" norm(Q)

  # Initialise the integration computation. Kernels calculate this at every timestep?? 
  @timeit to "initial integral" integral_computation(spacedisc, Q, 0) 
  @timeit to "solve" solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


  @info @sprintf """Finished...
    norm(Q) = %25.16e""" norm(Q)

#=
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
=#

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
  dt = 0.0025
  timeend = 10*dt
  # timeend = 14400
  polynomialorder = Npoly
  DFloat = Float64
  dim = numdims

  if MPI.Comm_rank(mpicomm) == 0
    @info @sprintf """ ------------------------------------------------------"""
    @info @sprintf """   ______ _      _____ __  ________                    """     
    @info @sprintf """  |  ____| |    |_   _|  ...  |  __  |                 """  
    @info @sprintf """  | |    | |      | | |   .   | |  | |                 """ 
    @info @sprintf """  | |    | |      | | | |   | | |__| |                 """
    @info @sprintf """  | |____| |____ _| |_| |   | | |  | |                 """
    @info @sprintf """  | _____|______|_____|_|   |_|_|  |_|                 """
    @info @sprintf """                                                       """
    @info @sprintf """ ------------------------------------------------------"""
    @info @sprintf """ Dycoms                                                """
    @info @sprintf """   Resolution:                                         """ 
    @info @sprintf """     (Δx, Δy, Δz)   = (%.2e, %.2e, %.2e)               """ Δx Δy Δz
    @info @sprintf """     (Nex, Ney, Nez) = (%d, %d, %d)                    """ Nex Ney Nez
    @info @sprintf """     DoF = %d                                          """ DoF
    @info @sprintf """     Minimum necessary memory to run this test: %g GBs """ (DoFstorage * sizeof(DFloat))/1000^3
    @info @sprintf """     Time step dt: %.2e                                """ dt
    @info @sprintf """     End time  t : %.2e                                """ timeend
    @info @sprintf """ ------------------------------------------------------"""
  end

  engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                  DFloat, dt)

  show(to)
end

isinteractive() || MPI.Finalize()

nothing
