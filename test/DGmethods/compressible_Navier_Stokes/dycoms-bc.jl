# Load Modules 
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
using DelimitedFiles
using Dierckx
using Random

if haspkg("CuArrays")
    using CUDAdrv
    using CUDAnative
    using CuArrays
    CuArrays.allowscalar(false)
    const ArrayType = CuArray
else
    const ArrayType = Array
end

using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters

# State labels 
const _nstate = 6
const _ρ, _U, _V, _W, _E, _QT = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
const statenames = ("RHO", "U", "V", "W", "E", "QT")

# Viscous state labels
const _nviscstates = 23
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _qtx, _qty, _qtz, _JplusDx, _JplusDy, _JplusDz, _θx, _θy, _θz, _SijSij, _ν_e, _qvx, _qvy, _qvz, _qlx, _qly, _qlz = 1:_nviscstates

const _nauxstate = 22
const _a_x, _a_y, _a_z, _a_sponge, _a_02z, _a_z2inf, _a_rad, _a_ν_e, _a_LWP_02z, _a_LWP_z2inf,_a_q_liq, _a_θ, _a_P,_a_T, _a_soundspeed_air, _a_z_FN, _a_ρ_FN, _a_U_FN, _a_V_FN, _a_W_FN, _a_E_FN, _a_QT_FN = 1:_nauxstate

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
    using Random
end

const μ_sgs           = 100.0
const Prandtl         = 71 // 100
const Prandtl_t       = 1 // 3
const cp_over_prandtl = cp_d / Prandtl_t
const seed = MersenneTwister(0)
# User Input
const numdims = 3
const Npoly = 4

# Define grid size 
Δx    = 35
Δy    = 35
Δz    = 10

const h_first_layer = Δz

# OR:
# Set Δx < 0 and define  Nex, Ney, Nez:
(Nex, Ney, Nez) = (10, 10, 1)

# Physical domain extents 
const (xmin, xmax) = (0, 2000)
const (ymin, ymax) = (0, 2000)
const (zmin, zmax) = (0, 1500)
const zi = 840
#Get Nex, Ney from resolution
const Lx = xmax - xmin
const Ly = ymax - ymin
const Lz = zmax - ymin
if ( Δx > 0)
  # User defines the grid size:
  ratiox = (Lx/Δx - 1)/Npoly
  ratioy = (Ly/Δy - 1)/Npoly
  ratioz = (Lz/Δz - 1)/Npoly
  Nex = ceil(Int64, ratiox)
  Ney = ceil(Int64, ratioy)
  Nez = ceil(Int64, ratioz)
else
  # User defines the number of elements:
  Δx = Lx / ((Nex * Npoly) + 1)
  Δy = Ly / ((Ney * Npoly) + 1)
  Δz = Lz / ((Nez * Npoly) + 1)
end

DoF = (Nex*Ney*Nez)*(Npoly+1)^numdims*(_nstate)
DoFstorage = (Nex*Ney*Nez)*(Npoly+1)^numdims*(_nstate + _nviscstates + _nauxstate + CLIMA.Mesh.Grids._nvgeo) +
    (Nex*Ney*Nez)*(Npoly+1)^(numdims-1)*2^numdims*(CLIMA.Mesh.Grids._nsgeo)

const C_smag = 0.15
# Equivalent grid-scale
Δ = (Δx * Δy * Δz)^(1/3)
const Δsqr = Δ * Δ


# Surface values to calculate surface fluxes:
const SST        = 292.5
const psfc       = 1017.8e2      # Pa
const qtot_sfc   = 13.84e-3      # qs(sst) using Teten's formula
const ρsfc       = 1.22          #kg/m^3
const Cd         = 0.0011        #Drag coefficient
const first_node_level   = 0.0001

const D_subsidence = 3.75e-6

# Random number seed
const seed = MersenneTwister(0)


function global_max(A::MPIStateArray, states=1:size(A, 2))
  host_array = Array ∈ typeof(A).parameters
  h_A = host_array ? A : Array(A)
  locmax = maximum(view(h_A, :, states, A.realelems)) 
  MPI.Allreduce([locmax], MPI.MAX, A.mpicomm)[1]
end

function global_mean(A::MPIStateArray, states=1:size(A,2))
  host_array = Array ∈ typeof(A).parameters
  h_A = host_array ? A : Array(A) 
  (Np, nstate, nelem) = size(A) 
  numpts = (nelem * Np) + 1
  localsum = sum(view(h_A, :, states, A.realelems)) 
  MPI.Allreduce([localsum], MPI.SUM, A.mpicomm)[1] / numpts 
end

# -------------------------------------------------------------------------
# Diagnostics: e.g. thermodynamics properties, preflux no longer used in 
# `bcstate`
# -------------------------------------------------------------------------
@inline function preflux(Q,aux)
  R_gas::eltype(Q) = R_d
  @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
  ρinv = 1 / ρ
  x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * grav * z) / ρ
  q_tot = QT / ρ
  # Establish the current thermodynamic state using the prognostic variables
  TS = PhaseEquil(e_int, q_tot, ρ)
  T = air_temperature(TS)
  Rm = gas_constant_air(TS)
  P = air_pressure(TS) # Test with dry atmosphere
  q_liq = PhasePartition(TS).liq
  θ = virtual_pottemp(TS)
  (u, v, w, T, θ, Rm)
end

#-------------------------------------------------------------------------
#md # Soundspeed computed using the thermodynamic state TS
# max eigenvalue
@inline function wavespeed(n, Q, aux, t)
  @inbounds begin 
    P = aux[_a_P]
    T = aux[_a_T]
    θ = aux[_a_θ]
    (u, v, w) = preflux(Q,aux)
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = E/ρ - (u^2 + v^2+ w^2)/2 - grav * z 
    q_tot = QT / ρ
    TS = PhaseEquil(e_int, q_tot, ρ)
    abs(n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(TS)
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
    #fsounding  = open(joinpath(@__DIR__, "../soundings/SOUNDING_PYCLES_Z_T_P.dat"))
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
# ------------------------------------------------------------------------
function buoyancy_correction(normSij, θv, dθvdz)
  # Brunt-Vaisala frequency
  N2 = grav / θv * dθvdz
  # Richardson number
  Richardson = N2 / (2 * normSij + eps(normSij))
  # Buoyancy correction factor
  buoyancy_factor = N2 <=0 ? 1 : sqrt(max(0.0, 1 - Richardson/Prandtl_t))
  return buoyancy_factor
end


"""
Specify compressible Navier-Stokes and assemble other terms from 
diffusive / viscous terms 
"""
@inline function cns_flux!(F, Q, VF, aux, t)
  @inbounds begin
    P = aux[_a_P]
    T = aux[_a_T]
    θ = aux[_a_θ]
    q_liq = aux[_a_q_liq]
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    xvert = aux[_a_z]
    θ     = aux[_a_θ]
    w -= D_subsidence * xvert
    W  = w*ρ
    # Inviscid contributions
    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
    F[1, _QT], F[2, _QT], F[3, _QT] = u * QT  , v * QT     , w * QT 
    #Derivative of T and Q:
    vqx, vqy, vqz                     = VF[_qtx],  VF[_qty],  VF[_qtz]
    vqvx, vqvy, vqvz                  = VF[_qvx], VF[_qvy], VF[_qvz]
    vqlx, vqly, vqlz                  = VF[_qlx], VF[_qly], VF[_qlz]    
    vJplusDx, vJplusDy, vJplusDz      = VF[_JplusDx], VF[_JplusDy], VF[_JplusDz]
    vθz                               = VF[_θz]
    # Radiation contribution 
    F_rad                     = ρ * radiation(aux)  
    aux[_a_rad]               = F_rad
    SijSij                    = VF[_SijSij]
    f_R                       = buoyancy_correction(SijSij, θ, vθz)
    #Dynamic eddy viscosity from Smagorinsky:
    μ_e                       = ρ * sqrt(2.0 * SijSij) * C_smag^2 * Δsqr
    D_e                       = μ_e / Prandtl_t
    # Multiply stress tensor by viscosity coefficient:
    τ11, τ22, τ33 = VF[_τ11] * μ_e, VF[_τ22]* μ_e, VF[_τ33] * μ_e
    τ12 = τ21 = VF[_τ12] * μ_e 
    τ13 = τ31 = VF[_τ13] * μ_e               
    τ23 = τ32 = VF[_τ23] * μ_e
    # Viscous velocity flux (i.e. F^visc_u in Giraldo Restelli 2008)
    F[1, _U] -= τ11 * f_R ; F[2, _U] -= τ12 * f_R ; F[3, _U] -= τ13 * f_R
    F[1, _V] -= τ21 * f_R ; F[2, _V] -= τ22 * f_R ; F[3, _V] -= τ23 * f_R
    F[1, _W] -= τ31 * f_R ; F[2, _W] -= τ32 * f_R ; F[3, _W] -= τ33 * f_R
    # Viscous Energy flux (i.e. F^visc_e in Giraldo Restelli 2008)
    F[1, _E] -= u * τ11 + v * τ12 + w * τ13 + vJplusDx * D_e  #dTd should not be diffused.
    F[2, _E] -= u * τ21 + v * τ22 + w * τ23 + vJplusDy * D_e
    F[3, _E] -= u * τ31 + v * τ32 + w * τ33 + vJplusDz * D_e
    F[3, _E] += F_rad
    # Viscous contributions to mass flux terms
    F[1, _ρ]  -=  vqx * D_e
    F[2, _ρ]  -=  vqy * D_e
    F[3, _ρ]  -=  vqz * D_e
    F[1, _QT] -=  vqx * D_e
    F[2, _QT] -=  vqy * D_e
    F[3, _QT] -=  vqz * D_e
  end
end

"""
Gradient state labels
"""
const _ngradstates = 7
@inline function gradient_vars!(vel, Q, aux, t)
  @inbounds begin
    (P, u, v, w, ρinv, q_liq,T,θ,Rm) = preflux(Q,aux)
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    vel[1], vel[2], vel[3] = u, v, w
    vel[4], vel[5], vel[6] = E, QT/ρ, E/ρ + Rm*T
    vel[7] = θ
  end
end

"""
Assemble radiation flux from integral 
"""
@inline function radiation(aux)
  zero_to_z = aux[_a_02z]
  z_to_inf = aux[_a_z2inf]
  z = aux[_a_z]
  z_i = 840  # Start with constant inversion height of 840 meters then build in check based on q_tot
  (z - z_i) >=0 ? Δz_i = (z - z_i) : Δz_i = 0 
  # Constants 
  F_0 = 70 
  F_1 = 22
  α_z = 1
  ρ_i = 1.22
  D_subsidence = 3.75e-6
  cloud_top_cooling = F_0 * exp(-z_to_inf) 
  cloud_base_warming = F_1 * exp(-zero_to_z)
  free_troposphere_cooling = ρ_i * cp_d * D_subsidence * α_z * (0.25 * (cbrt(Δz_i))^4 + z_i * cbrt(Δz_i))
  F_rad = cloud_top_cooling + cloud_base_warming + free_troposphere_cooling
  return F_rad
end

"""
Viscous /diffusive terms 
"""
@inline function compute_stresses!(VF, grad_vel, _...)
  @inbounds begin
    # get gradients of velocities 
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]
    # get gradients of moist vars and potential temperature
    dqtdx, dqtdy, dqtdz             = grad_vel[1, 5], grad_vel[2, 5], grad_vel[3, 5]
    dJplusDdx, dJplusDdy, dJplusDdz = grad_vel[1, 6], grad_vel[2, 6], grad_vel[3, 6]
    dθdx, dθdy, dθdz                = grad_vel[1, 7], grad_vel[2, 7], grad_vel[3, 7]
    # --------------------------------------------
    # STRAINRATE TENSOR COMPONENTS
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
    SijSij = (S11^2 + S22^2 + S33^2
              + 2.0 * S12^2
              + 2.0 * S13^2 
              + 2.0 * S23^2) 
    modSij = sqrt(2.0 * SijSij)
    #--------------------------------------------
    # STRESS COMPONENTS
    #--------------------------------------------
    VF[_τ11] = 2 * (S11 - (S11 + S22 + S33) / 3)
    VF[_τ22] = 2 * (S22 - (S11 + S22 + S33) / 3)
    VF[_τ33] = 2 * (S33 - (S11 + S22 + S33) / 3)
    VF[_τ12] = 2 * S12
    VF[_τ13] = 2 * S13
    VF[_τ23] = 2 * S23
    
    VF[_qtx], VF[_qty], VF[_qtz]             = dqtdx,  dqtdy,  dqtdz        
    VF[_JplusDx], VF[_JplusDy], VF[_JplusDz] = dJplusDdx, dJplusDdy, dJplusDdz
    VF[_θx], VF[_θy], VF[_θz]                = dθdx, dθdy, dθdz
    VF[_SijSij]                              = SijSij        
  end
end

"""
Auxiliary state, here, for coordinates and Rayleigh-sponge coefficient 
"""
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    DFloat = eltype(aux)
    xvert = z
    aux[_a_z] = xvert
    #Sponge 
    ctop    = zero(DFloat)
    cs_left_right = zero(DFloat)
    cs_front_back = zero(DFloat)
    ct            = DFloat(0.75)
    domain_bott  = zmin
    domain_top   = zmax
    
    #Vertical sponge:
    sponge_type = 1
    if sponge_type == 1
      ct = 0.9
      bc_zscale  = 600.0
      zd = domain_top - bc_zscale       
      if xvert >= zd
          ctop = ct * (sinpi(0.5*(xvert - zd)/(domain_top - zd)))^4
      end
    else
      aux[_a_x] = x
      aux[_a_y] = y
      aux[_a_z] = z
      #Sponge
      csleft  = 0.0
      csright = 0.0
      csfront = 0.0
      csback  = 0.0
      ctop    = 0.0
      cs_left_right = 0.0
      cs_front_back = 0.0
      ct            = 0.75
      #BEGIN  User modification on domain parameters.
      domain_left  = xmin 
      domain_right = xmax
      domain_front = ymin 
      domain_back  = ymax 
      domain_bott  = zmin 
      domain_top   = zmax 
      #END User modification on domain parameters.
      # Define Sponge Boundaries      
      xc       = 0.5 * (domain_right + domain_left)
      yc       = 0.5 * (domain_back  + domain_front)
      zc       = 0.5 * (domain_top   + domain_bott)
      top_sponge  = 0.85 * domain_top
      xsponger    = domain_right - 0.15 * (domain_right - xc)
      xspongel    = domain_left  + 0.15 * (xc - domain_left)
      ysponger    = domain_back  - 0.15 * (domain_back - yc)
      yspongel    = domain_front + 0.15 * (yc - domain_front)
      #x left and right
      #xsl
      if x <= xspongel
          csleft = cs_left_right * (sinpi(1/2 * (x - xspongel)/(domain_left - xspongel)))^4
      end
      #xsr
      if x >= xsponger
          csright = cs_left_right * (sinpi(1/2 * (x - xsponger)/(domain_right - xsponger)))^4
      end        
      #y left and right
      #ysl
      if y <= yspongel
          csfront = cs_front_back * (sinpi(1/2 * (y - yspongel)/(domain_front - yspongel)))^4
      end
      #ysr
      if y >= ysponger
          csback = cs_front_back * (sinpi(1/2 * (y - ysponger)/(domain_back - ysponger)))^4
      end
      #Vertical sponge:         
      if z >= top_sponge
          ctop = ct * (sinpi(0.5 * (z - top_sponge)/(domain_top - top_sponge)))^4
      end
    end
    beta  = 1.0 - (1.0 - ctop) 
    beta  = min(beta, 1.0)
    aux[_a_sponge] = beta
  end
end

"""
Boundary Conditions
"""
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t)
  @inbounds begin
    ρM, UM, VM, WM, EM, QTM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E], QM[_QT]
    uM, vM, wM  = UM/ρM, VM/ρM, WM/ρM
    q_totM = QTM/ρM
    q_liqM = auxM[_a_q_liq]
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_V] = VM - 2 * nM[2] * UnM
    QP[_W] = WM - 2 * nM[3] * UnM
    QP[_ρ] = ρM
    QP[_QT] = QTM
    VFP .= 0 
    if auxM[_a_z] < 0.001  # TODO specify boundary keyword and get correct bctype for general topography
      # ------------------------------------------------------------------------
      # First node quantities (first-model level here represents the first node)
      # ------------------------------------------------------------------------
      z_FN             = auxM[_a_z_FN]
      ρ_FN             = auxM[_a_ρ_FN]
      U_FN             = auxM[_a_U_FN]
      V_FN             = auxM[_a_V_FN]
      W_FN             = auxM[_a_W_FN]
      E_FN             = auxM[_a_E_FN]
      u_FN, v_FN, w_FN = U_FN/ρ_FN, V_FN/ρ_FN, W_FN/ρ_FN
      windspeed_FN     = sqrt(u_FN^2 + v_FN^2)
      q_tot_FN         = auxM[_a_QT_FN] / ρ_FN
      e_int_FN         = E_FN/ρ_FN - 0.5*windspeed_FN^2 - grav*z_FN
      TS_FN            = PhaseEquil(e_int_FN, q_tot_FN, ρ_FN) 
      T_FN             = air_temperature(TS_FN)
      q_liq_FN         = PhasePartition(TS_FN).liq
      q_vap_FN         = q_tot_FN - PhasePartition(TS_FN).liq
      # -----------------------------------
      # Bottom boundary quantities 
      # -----------------------------------
      zM          = auxM[_a_z]
      q_totM      = QM[_QT]/QM[_ρ]
      q_liqM      = auxM[_a_q_liq]
      windspeed   = sqrt(uM^2 + vM^2)
      e_intM      = EM/ρM - 0.5*windspeed^2 - grav*zM
      TSM         = PhaseEquil(e_intM, q_totM, ρM) 
      q_vapM      = q_totM - PhasePartition(TSM).liq
      TM          = air_temperature(TSM)
      cpmM        = cp_m(TSM)
      # ----------------------------------------------
      # Assigning calculated values to boundary states
      # ----------------------------------------------
      VFP[_τ33] = 0  
      # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
      # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is replaced by the appropriate flux terms
      VFP[_τ13] = -ρM * Cd * windspeed_FN * u_FN 
      VFP[_τ23] = -ρM * Cd * windspeed_FN * v_FN 
      VFP[_qtz]  = 115 /(ρM * LH_v0)
      VFP[_JplusDz] = 130 / ρM 
    end
    nothing
  end
end

@inline function stresses_boundary_penalty!(VF, _...) 
  VF .= 0
end

@inline function stresses_penalty!(VF, nM, velM, QM, aM, velP, QP, aP, t)
  @inbounds begin
    n_Δvel = similar(VF, Size(3, _ngradstates))
    for j = 1:_ngradstates, i = 1:3
        n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end
    compute_stresses!(VF, n_Δvel)
  end
end


"""
Assemble source terms
"""
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
Coriolis force
"""
const f_coriolis = 7.62e-5
const u_geostrophic = 7.0
const v_geostrophic = -5.5 
const Ω = Omega
@inline function source_coriolis!(S,Q,aux,t)
  @inbounds begin
    U, V, W = Q[_U], Q[_V], Q[_W]
    S[_U] -= 0
    S[_V] -= 0
    S[_W] -= 0
  end
end

"""
Geostrophic wind forcing
"""
@inline function source_geostrophic!(S,Q,aux,t)
  @inbounds begin
    ρ = Q[_ρ]
    W = Q[_W]
    U = Q[_U]
    V = Q[_V]
    S[_U] -= f_coriolis * (U/ρ - u_geostrophic)
    S[_V] -= f_coriolis * (V/ρ - v_geostrophic)
  end
end

"""
Sponge source (coefficient from aux)
"""
@inline function source_sponge!(S,Q,aux,t)
  @inbounds begin
    U, V, W  = Q[_U], Q[_V], Q[_W]        
    beta     = aux[_a_sponge]
    S[_U] -= beta * U
    S[_V] -= beta * V
    S[_W] -= beta * W
  end
end

"""
Gravity source 
"""
@inline function source_geopot!(S,Q,aux,t)
  @inbounds begin
    ρ, U, V, W, E  = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    S[_W] -= ρ * grav
  end
end

"""
Defines integrand for DYCOMS radiation
"""
@inline function integral_knl(val, Q, aux)
  κ = 85.0
  @inbounds begin
    @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    ρinv = 1 / ρ
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * grav * z) / ρ
    q_tot = QT / ρ
    # Establish the current thermodynamic state using the prognostic variables
    TS = PhaseEquil(e_int, q_tot, ρ)
    q_liq = PhasePartition(TS).liq
    val[1] = ρ * κ * q_liq 
  end
end

"""
Stores thermodynamic properties and computes other quantities required prior to time update
"""
function preodefun!(disc, Q, t)
  DGBalanceLawDiscretizations.dof_iteration!(disc.auxstate, disc, Q) do R, Q, QV, aux
    @inbounds let
      ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
      xvert = aux[_a_z]
      e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * grav * xvert) / ρ
      q_tot = QT / ρ
      TS = PhaseEquil(e_int, q_tot, ρ)
      T = air_temperature(TS)
      P = air_pressure(TS) # Test with dry atmosphere
      q_liq = PhasePartition(TS).liq
      R[_a_T] = T
      R[_a_P] = P
      R[_a_q_liq] = q_liq
      R[_a_soundspeed_air] = soundspeed_air(TS)
      R[_a_θ] = virtual_pottemp(TS)
    end
  end
  firstnode_info(disc,Q,t)
  integral_computation(disc, Q, t)
end

function firstnode_info(disc,Q,t)
    # User specified kernel to allow access to first-interior points of specific auxiliary and state variables
    DGBalanceLawDiscretizations.aux_firstnode_values!(disc, Q,
                                                      (_a_z_FN), (_a_z))
    DGBalanceLawDiscretizations.state_firstnode_values!(disc, Q,
                                                        (_a_ρ_FN, _a_U_FN, _a_V_FN, _a_W_FN, _a_E_FN, _a_QT_FN), (_ρ, _U, _V, _W, _E, _QT))
end
function integral_computation(disc, Q, t)
  # Kernel to compute vertical integrals
  DGBalanceLawDiscretizations.indefinite_stack_integral!(disc, integral_knl, Q,
                                                         (_a_02z))
  DGBalanceLawDiscretizations.reverse_indefinite_stack_integral!(disc,
                                                                 _a_z2inf,
                                                                 _a_02z)
end

# ------------------------------------------------------------------
# initial condition
"""
    This function specifies the initial conditions
    for the dycoms driver. 
"""
function dycoms!(dim, Q, t, spl_tinit, spl_qinit, spl_uinit, spl_vinit,
                 spl_pinit, x, y, z, _...)
    
    DFloat         = eltype(Q)
    p0::DFloat      = MSLP

    # --------------------------------------------------
    # INITIALISE ARRAYS FOR INTERPOLATED VALUES
    # --------------------------------------------------
    xvert          = z
    datat          = spl_tinit(xvert)
    dataq          = spl_qinit(xvert)
    datau          = spl_uinit(xvert)
    datav          = spl_vinit(xvert)
    datap          = spl_pinit(xvert)
    dataq          = dataq * 1.0e-3
    
    P              = datap
    randnum1   = rand(seed, DFloat) / 100
    randnum2   = rand(seed, DFloat) / 100

    θ_liq = datat
    q_tot = dataq
    if xvert <= 200.0
        θ_liq += randnum1 * datat
        q_tot += randnum2 * dataq
    end

    q_liq = 0.0
    q_ice = 0.0
    if xvert > 600.0 && xvert <= 840.0
        q_liq = (xvert - 600)*0.00045/240.0
    end
    
    
    PhPart = PhasePartition(q_tot, q_liq, q_ice)
    T      = air_temperature_from_liquid_ice_pottemp(θ_liq, P, PhPart)
    ρ      = air_density(T, P, PhPart)
    
    # energy definitions
    u, v, w     = datau, datav, 0.0 #geostrophic. TO BE BUILT PROPERLY if Coriolis is considered
    U           = ρ * u
    V           = ρ * v
    W           = ρ * w
    e_kin       = 0.5 * (u^2 + v^2 + w^2)
    e_pot       = grav * xvert
    E           = ρ * total_energy(e_kin, e_pot, T, PhPart)

    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]= ρ, U, V, W, E, ρ * q_tot
    
end


# ------------------------------------------------------------------
# -------------END DEF SOURCES-------------------------------------# 

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

    brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                  range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)),
                  range(DFloat(zmin), length=Ne[3]+1, DFloat(zmax)))
    
    
    # User defined periodicity in the topl assignment
    # brickrange defines the domain extents
    # TODO: boundary keyword: flag boundary faces 
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity=(true,true,false))

    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = DFloat,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N)
    
    numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed)
    numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!, bcstate!, wavespeed)

    # spacedisc = data needed for evaluating the right-hand side function
    spacedisc = DGBalanceLaw(grid = grid,
                             length_state_vector = _nstate,
                             flux! = cns_flux!,
                             numerical_flux! = numflux!,
                             numerical_boundary_flux! = numbcflux!, 
                             number_gradient_states = _ngradstates,
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
    
    lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
        if s
            starttime[] = now()
        else
            ql_max = global_max(spacedisc.auxstate, _a_q_liq)
            @info @sprintf("""Update
                         simtime = %.16e
                         runtime = %s
                         max(ql) = %.16e""",
                           ODESolvers.gettime(lsrk),
                           Dates.format(convert(Dates.DateTime,
                                                Dates.now()-starttime[]),
                                        Dates.dateformat"HH:MM:SS"), ql_max)
        end
    end

    npoststates = 4
    _o_RAD, _o_q_liq, _o_T, _o_θ = 1:npoststates
    postnames = ("RAD", "_q_liq", "T", "THETA")
    postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

    step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1500) do (init=false)
      DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc, Q) do R, Q, QV, aux
        @inbounds let
            R[_o_RAD]   = aux[_a_z2inf] + aux[_a_02z]
            R[_o_q_liq] = aux[_a_q_liq]
            R[_o_T]     = aux[_a_T]
            R[_o_θ]     = aux[_a_θ]
        end
      end
        
      mkpath("./CLIMA-output-scratch/dycoms-bc/")
      outprefix = @sprintf("./CLIMA-output-scratch/dycoms-bc/dy_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
      @debug "doing VTK output" outprefix
      writevtk(outprefix, Q, spacedisc, statenames,
               postprocessarray, postnames)
      
      step[1] += 1
      nothing
    end
    # Initialise the integration computation. Kernels calculate this at every timestep?? 
    integral_computation(spacedisc, Q, 0) 
    solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

end

using Test
let
    MPI.Initialized() || MPI.Init()
    Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
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

    numelem = (Nex,Ney,Nez)
    dt = 0.005
    timeend = 14400
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
        @info @sprintf """     End time  t : %d                                  """ timeend
        @info @sprintf """ ------------------------------------------------------"""
    end
    
    engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                    DFloat, dt)
end

isinteractive() || MPI.Finalize()

nothing
