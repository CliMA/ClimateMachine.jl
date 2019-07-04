# # Held-Suarez on the Sphere
#
#md # !!! jupyter
#md #     This example is also available as a Jupyter notebook:
#md #     [`held_suarez_balanced.ipynb`](@__NBVIEWER_ROOT_URL__examples/DGmethods/generated/held_suarez_balanced.html)
#
# ## Introduction
#
# In this example we will set up and run the Held-Suarez test case from 
# Held and Suarez (1994) (https://journals.ametsoc.org/doi/pdf/10.1175/1520-0477%281994%29075%3C1825%3AAPFTIO%3E2.0.CO%3B2)

# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref held_suarez_balanced-plain-program).
#
# ## Commented Program

# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref held_suarez_forcing-plain-program).
#
# ## Commented Program

#------------------------------------------------------------------------------
#--------------------------------#
#--------------------------------#
# Can be run with:
# CPU: mpirun -n 1 julia --project=@. held_suarez_forcing.jl
# GPU: mpirun -n 1 julia --project=/home/fxgiraldo/CLIMA/env/gpu held_suarez_forcing.jl
#--------------------------------#
#--------------------------------#

# ### Preliminaries
# Load in modules needed for solving the problem
using MPI
using Logging
using LinearAlgebra
using Dates
using Printf
using CLIMA
using CLIMA.Topologies
using CLIMA.MPIStateArrays
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.Vtk
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.ParametersType
using StaticArrays

# Though not required, here we are explicit about which values we read out the
# `PlanetParameters` and `MoistThermodynamics`
using CLIMA.PlanetParameters: planet_radius, R_d, cp_d, grav, cv_d, MSLP, Omega
using CLIMA.MoistThermodynamics: air_temperature, air_pressure, internal_energy,
                                 soundspeed_air, air_density, gas_constant_air

@parameter C_ss 0.23 "C_ss"
@parameter Prandtl_turb 1//3 "Prandtl_turb"
@parameter Prandtl 71//100 "Prandtl"
@parameter k_μ    cp_d/Prandtl "k_μ"

function Base.sort!(a::MArray{Tuple{3}})
  # Use a (Bose-Nelson Algorithm based) sorting network from
  # <http://pages.ripco.net/~jgamble/nw.html>.
  a[2], a[3] = minmax(a[2], a[3])
  a[1], a[3] = minmax(a[1], a[3])
  a[1], a[2] = minmax(a[1], a[2])
end

function Base.sort(a::SArray{Tuple{3}})
  b = similar(a)
  b .= a
  sort!(b)
  b
end

"""
Smagorinsky model coefficient for anisotropic grids.
Given a description of the grid in terms of Δ1, Δ2, Δ3
and polynomial order Npoly, computes the anisotropic equivalent grid
coefficient such that the Smagorinsky coefficient is modified as follows
Eddy viscosity          ν_e
Smagorinsky coefficient C_ss
Δeq                     Equivalent anisotropic grid
ν_e = (C_ss Δeq)^2 * sqrt(2 * SijSij)

@article{doi:10.1063/1.858537,
author = {Scotti,Alberto  and Meneveau,Charles  and Lilly,Douglas K. },
title = {Generalized Smagorinsky model for anisotropic grids},
  journal = {Physics of Fluids A: Fluid Dynamics},
  volume = {5},
  number = {9},
  pages = {2306-2308},
  year = {1993},
  doi = {10.1063/1.858537},
  URL = {https://doi.org/10.1063/1.858537},
  eprint = {https://doi.org/10.1063/1.858537}
}
In addition, simple alternative methods of computing the geometric average
are also included (in accordance with Deardorff's methods).
"""
function anisotropic_coefficient_sgs3D(Δ1, Δ2, Δ3)
  DFloat = typeof(Δ1)
  # Arguments are the lengthscales in each of the coordinate directions
  # For a cube: this is the edge length
  # For a sphere: the arc length provides one approximation of many
  Δ = cbrt(Δ1 * Δ2 * Δ3)
  Δ_sorted = sort(@SVector [Δ1, Δ2, Δ3])
  # Get smallest two dimensions
  Δ_s1 = Δ_sorted[1]
  Δ_s2 = Δ_sorted[2]
  a1 = Δ_s1 / max(Δ1,Δ2,Δ3)
  a2 = Δ_s2 / max(Δ1,Δ2,Δ3)
  # In 3D we compute a scaling factor for anisotropic grids
  f_anisotropic = 1 + DFloat(2/27) * ((log(a1))^2 - log(a1)*log(a2) + (log(a2))^2)
  Δ = Δ*f_anisotropic
  Δsqr = Δ * Δ
  return Δsqr
end

function anisotropic_coefficient_sgs2D(Δ1, Δ3)
  # Order of arguments does not matter.
  Δ = min(Δ1, Δ3)
  Δsqr = Δ * Δ
  return Δsqr
end

function standard_coefficient_sgs3D(Δ1,Δ2,Δ3)
  Δ = cbrt(Δ1 * Δ2 * Δ3) 
  Δsqr = Δ * Δ
  return Δsqr
end

function standard_coefficient_sgs2D(Δ1, Δ2)
  Δ = sqrt(Δ1 * Δ2)
  Δsqr = Δ * Δ
  return Δsqr
end

"""
Compute components of strain-rate tensor 
Dij = ∇u .................................................. [1]
Sij = 1/2 (∇u + (∇u)ᵀ) .....................................[2]
τij = 2 * ν_e * Sij ........................................[3]
"""
function compute_strainrate_tensor(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
  # Assemble components of the strain-rate tensor 
  S11, = dudx
  S12  = (dudy + dvdx) / 2
  S13  = (dudz + dwdx) / 2
  S22  = dvdy
  S23  = (dvdz + dwdy) / 2
  S33  = dwdz
  SijSij = S11^2 + S22^2 + S33^2 + 2 * (S12^2 + S13^2 + S23^2)
  return (S11, S22, S33, S12, S13, S23, SijSij)
end

"""
Smagorinksy-Lilly SGS Turbulence
--------------------------------
The constant coefficient Standard Smagorinsky Model model for 
(1) eddy viscosity ν_e 
(2) and eddy diffusivity D_e 
The resolved scale stress tensor is calculated as in [3]
where Sij represents the components of the resolved
scale rate of strain tensor. ν_t is the unknown eddy
viscosity which is computed here using the assumption
that subgrid turbulence production and dissipation are 
balanced.
article{doi:10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2,
author = {Smagorinksy, J.},
title = {General circulation experiments with the primitive equations},
journal = {Monthly Weather Review},
volume = {91},
number = {3},
pages = {99-164},
year = {1963},
doi = {10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2},
URL = {https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2},
eprint = {https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2}
}
"""
function standard_smagorinsky(SijSij, Δsqr)
  # Eddy viscosity is a function of the magnitude of the strain-rate tensor
  # This is for use on both spherical and cartesian grids.
  ν_e::eltype(SijSij) = sqrt(2SijSij) * C_ss * C_ss * Δsqr
  D_e::eltype(SijSij) = ν_e / Prandtl_turb
  return (ν_e, D_e)
end

"""
Buoyancy adjusted Smagorinsky coefficient for stratified flows

Ri = N² / (2*SijSij)
Ri = gravity / ρ * ∂ρ∂z / 2 |S_{ij}|
article{doi:10.1111/j.2153-3490.1962.tb00128.x,
author = {LILLY, D. K.},
title = {On the numerical simulation of buoyant convection},
journal = {Tellus},
volume = {14},
number = {2},
pages = {148-172},
doi = {10.1111/j.2153-3490.1962.tb00128.x},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/j.2153-3490.1962.tb00128.x},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.2153-3490.1962.tb00128.x},
year = {1962}
}
"""
function buoyancy_correction(SijSij, ρ, dρdz)
  N2 = grav / ρ * dρdz
  Richardson = N2 / (2SijSij + eps(SijSij))
  buoyancy_factor = N2 <= 0 ? one(SijSij) : sqrt(max(zero(SijSij), 1 - Richardson/Prandtl_turb))
  return buoyancy_factor
end

# Start up MPI if this has not already been done
MPI.Initialized() || MPI.Init()
#md nothing # hide

# If `CuArrays` is in the current environment we will use CUDA, otherwise we
# drop back to the CPU
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const DeviceArrayType = CuArray
else
  const DeviceArrayType = Array
end
#md nothing # hide

# Specify whether to enforce hydrostatic balance at PDE level or not
const PDE_level_hydrostatic_balance = true

# Specify if forcings are ramped up or full forcing are applied from the beginning
const ramp_up_forcings = true
const use_held_suarez_forcings = true
const use_sponge = false
const use_exponential_vertical_warp = false
const use_coriolis = true

# check whether to use default VTK directory or define something else
VTKDIR = get(ENV, "CLIMA_VTK_DIR", "vtk")

# Here we setup constants to for some of the computational parameters; the
# underscore is just syntactic sugar to indicate that these are constants.

# These are parameters related to the Euler state. Here we used the conserved
# variables for the state: perturbation in density, three components of
# momentum, and perturbation to the total energy.
const _nstate = 5
const _dρ, _ρu, _ρv, _ρw, _dρe = 1:_nstate
const _statenames = ("δρ", "ρu", "ρv", "ρw", "δρe")
#md nothing # hide

"""
Viscous state labels
"""
const _nviscstates = 13
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _Tx, _Ty, _Tz, _ρx, _ρy, _ρz, _SijSij = 1:_nviscstates

"""
Number of variables of which gradients are required 
"""
const _ngradstates = 5

"""
Number of states being loaded for gradient computation
"""
const _states_for_gradient_transform = (_dρ, _ρu, _ρv, _ρw, _dρe)


# These will be the auxiliary state which will contain the geopotential,
# gradient of the geopotential, and reference values for density and total
# energy, as well as the coordinates
const _nauxstate = 11
const _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_ρ_ref, _a_ρe_ref, _a_x, _a_y, _a_z, _a_sponge, _a_Δsqr  = 1:_nauxstate
const _auxnames = ("ϕ", "ϕx", "ϕy", "ϕz", "ρ_ref", "ρe_ref", "x", "y", "z", "sponge_coefficient")
#md nothing # hide

# -------------------------------------------------------------------------
#md # Here we define a function to extract the velocity components from the 
#md # prognostic equations (i.e. the momentum and density variables). This 
#md # function is not required in general, but provides useful functionality 
#md # in some cases. 
# -------------------------------------------------------------------------
# Compute the velocity from the state
function gradient_vars!(gradient_list, Q, aux, t)
  @inbounds begin
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    u = ρu/ρ
    v = ρv/ρ
    w = ρw/ρ

    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ
    T = air_temperature(e_int)

    gradient_list[1], gradient_list[2], gradient_list[3] = u, v, w
    gradient_list[4], gradient_list[5] = T, ρ
  end
end

function compute_stresses!(VF, grad_vars, _...)
  gravity::eltype(VF) = grav
  @inbounds begin
    dudx, dudy, dudz = grad_vars[1, 1], grad_vars[2, 1], grad_vars[3, 1]
    dvdx, dvdy, dvdz = grad_vars[1, 2], grad_vars[2, 2], grad_vars[3, 2]
    dwdx, dwdy, dwdz = grad_vars[1, 3], grad_vars[2, 3], grad_vars[3, 3]
    # compute gradients of moist vars and temperature
    dTdx, dTdy, dTdz = grad_vars[1, 4], grad_vars[2, 4], grad_vars[3, 4]
    dρdx, dρdy, dρdz = grad_vars[1, 5], grad_vars[2, 5], grad_vars[3, 5]
    # virtual potential temperature gradient: for richardson calculation
    # strains
    # --------------------------------------------
    (S11,S22,S33,S12,S13,S23,SijSij) = compute_strainrate_tensor(dudx, dudy,
                                                                 dudz, dvdx,
                                                                 dvdy, dvdz,
                                                                 dwdx, dwdy,
                                                                 dwdz)
    #--------------------------------------------
    # deviatoric stresses
    VF[_τ11] = 2 * (S11 - (S11 + S22 + S33) / 3)
    VF[_τ22] = 2 * (S22 - (S11 + S22 + S33) / 3)
    VF[_τ33] = 2 * (S33 - (S11 + S22 + S33) / 3)
    VF[_τ12] = 2 * S12
    VF[_τ13] = 2 * S13
    VF[_τ23] = 2 * S23

    VF[_Tx], VF[_Ty], VF[_Tz] = dTdx, dTdy, dTdz
    VF[_ρx], VF[_ρy], VF[_ρz] = dρdx, dρdy, dρdz
    VF[_SijSij] = SijSij
  end
end

# ### Definition of the physics
#md # Now we define a function which given the state and auxiliary state defines
#md # the physical Euler flux
function eulerflux!(F, Q, VF, aux, t)
  @inbounds begin
    ## extract the states
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    ## compute the velocity
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the pressure
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)

    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## set the actual flux
    F[1, _dρ ], F[2, _dρ ], F[3, _dρ ] = ρu          , ρv          , ρw
    if PDE_level_hydrostatic_balance
      δP = P - P_ref
      F[1, _ρu], F[2, _ρu], F[3, _ρu] = u * ρu  + δP, v * ρu     , w * ρu
      F[1, _ρv], F[2, _ρv], F[3, _ρv] = u * ρv      , v * ρv + δP, w * ρv
      F[1, _ρw], F[2, _ρw], F[3, _ρw] = u * ρw      , v * ρw     , w * ρw + δP
    else
      F[1, _ρu], F[2, _ρu], F[3, _ρu] = u * ρu  + P, v * ρu    , w * ρu
      F[1, _ρv], F[2, _ρv], F[3, _ρv] = u * ρv     , v * ρv + P, w * ρv
      F[1, _ρw], F[2, _ρw], F[3, _ρw] = u * ρw     , v * ρw    , w * ρw + P
    end
    F[1, _dρe], F[2, _dρe], F[3, _dρe] = u * (ρe + P), v * (ρe + P), w * (ρe + P)

    # #Derivative of T and Q:
    # vTx, vTy, vTz = VF[_Tx], VF[_Ty], VF[_Tz]
    # vρx, vρy, vρz = VF[_Tx], VF[_Ty], VF[_Tz]

    # #Richardson contribution:
    # SijSij = VF[_SijSij]

    # #Dynamic eddy viscosity from Smagorinsky:
    # Δsqr = aux[_a_Δsqr]
    # (ν_e, D_e) = standard_smagorinsky(SijSij, Δsqr)
    # # FIXME f_R = buoyancy_correction(SijSij, ρ, vρy)
    # f_R = 1

    # # Multiply stress tensor by viscosity coefficient:
    # τ11, τ22, τ33 = VF[_τ11] * ν_e, VF[_τ22]* ν_e, VF[_τ33] * ν_e
    # τ12 = τ21 = VF[_τ12] * ν_e
    # τ13 = τ31 = VF[_τ13] * ν_e
    # τ23 = τ32 = VF[_τ23] * ν_e

    # # Viscous velocity flux (i.e. F^visc_u in Giraldo Restelli 2008)
    # F[1, _ρu] -= τ11 * f_R ; F[2, _ρu] -= τ12 * f_R ; F[3, _ρu] -= τ13 * f_R
    # F[1, _ρv] -= τ21 * f_R ; F[2, _ρv] -= τ22 * f_R ; F[3, _ρv] -= τ23 * f_R
    # F[1, _ρw] -= τ31 * f_R ; F[2, _ρw] -= τ32 * f_R ; F[3, _ρw] -= τ33 * f_R

    # # Viscous Energy flux (i.e. F^visc_e in Giraldo Restelli 2008)
    # F[1, _dρe] -= u * τ11 + v * τ12 + w * τ13 + ν_e * k_μ * vTx
    # F[2, _dρe] -= u * τ21 + v * τ22 + w * τ23 + ν_e * k_μ * vTy
    # F[3, _dρe] -= u * τ31 + v * τ32 + w * τ33 + ν_e * k_μ * vTz
  end
end
#md nothing # hide

# FXG: Source function => Define the geopotential and Held-Suarez source from the solution and auxiliary variables
function source!(S, Q, aux, t)
  @inbounds begin

    ## Store values
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref = aux[_a_ρ_ref], aux[_a_ρe_ref]
    ϕ, ϕx, ϕy, ϕz = aux[_a_ϕ], aux[_a_ϕx], aux[_a_ϕy], aux[_a_ϕz]
    x, y, z = aux[_a_x], aux[_a_y], aux[_a_z]

    ## Add Geopotential source
    S[_dρ ] = 0
    if PDE_level_hydrostatic_balance
      S[_ρu ] = -dρ * ϕx
      S[_ρv ] = -dρ * ϕy
      S[_ρw ] = -dρ * ϕz
    else
      ρ = ρ_ref + dρ
      S[_ρu ] = -ρ * ϕx
      S[_ρv ] = -ρ * ϕy
      S[_ρw ] = -ρ * ϕz
    end
    S[_dρe] = 0
    
    # Coriolis force
    if use_coriolis
      coriolis_x =  2Omega * ρv
      coriolis_y = -2Omega * ρu
      coriolis_z =  0

      S[_ρu ] += coriolis_x
      S[_ρv ] += coriolis_y
      S[_ρw ] += coriolis_z
    end

    ## Add Held-Suarez Source
    #Extract Temperature
    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)
    
    #Create Held-Suarez forcing
    (kv,kt,T_eq)=held_suarez_forcing(x,y,z,T,P,t) 
    
    #Apply forcing
    if use_held_suarez_forcings
      S[_ρu ]  -= kv*ρu
      S[_ρv ]  -= kv*ρv
      S[_ρw ]  -= kv*ρw
      S[_dρe ] -= ( kt*ρ*cv_d*T + kv*(ρu*ρu + ρv*ρv + ρw*ρw)/ρ )
    end

    if use_sponge
      sponge_coefficient = aux[_a_sponge]
      S[_ρu ] -= sponge_coefficient*ρu
      S[_ρv ] -= sponge_coefficient*ρv
      S[_ρw ] -= sponge_coefficient*ρw
    end
end
end
#md nothing # hide

# FXG: Held-Suarez forcing function
function held_suarez_forcing(x, y, z, T, P, t)
  @inbounds begin
    DFloat = eltype(T)

    #Store Held-Suarez constants
    p0 :: DFloat = MSLP
    θ0 :: DFloat = 315
    N_bv :: DFloat = 0.0158725
    gravity :: DFloat = grav
    ka :: DFloat = 1 / 40 / 86400
    kf :: DFloat = 1 / 86400
    ks :: DFloat = 1 / 4 / 86400
    ΔT_y :: DFloat = 60 
    ΔT_z :: DFloat = 10   
    temp0 :: DFloat = 315
    temp_min :: DFloat = 200
    hd :: DFloat = 7000 #from Smolarkiewicz JAS 2001 paper    
    σ_b :: DFloat = 7 / 10
    
    #Compute Rayleigh Damping terms from Held-Suarez 1994 paper
    (r, λ, φ) = cartesian_to_spherical(x, y, z)
    h = r - DFloat(planet_radius) # height above the planet surface
    σ = exp(-h/hd) #both approx of sigma behave similarly
#    σ = P/p0        #both approx of sigma behave similarly
    Δσ = (σ - σ_b)/(1 - σ_b)
    π = σ^(R_d/cp_d)
    c = max(0, Δσ)
    T_eq = ( temp0 - ΔT_y*sin(φ)^2 - ΔT_z*log(σ)*cos(φ)^2 )*π
    T_eq = max(temp_min, T_eq)
    kt = ka + (ks-ka)*c*cos(φ)^4
    kv = kf*c
    kt = kt*(1 - T_eq/T)

    if ramp_up_forcings
      t_ramp :: DFloat = 86400
      ramp_factor = (1 - exp(-2t / t_ramp)) / (1 + exp(-2t / t_ramp))
      kv *= ramp_factor
      kt *= ramp_factor
    end

    #println("h = ",h,", φ = ",φ,", T_eq = ",T_eq,", T = ",T)

    #Pass Held-Suarez data
    (kv, kt, T_eq)
  end
end
#md nothing # hide

# This defines the local wave speed from the current state (this will be needed
# to define the numerical flux)
function wavespeed(n, Q, aux, _...)
  @inbounds begin
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]

    ## get total energy and density
    ρ = ρ_ref + dρ
    e = (ρe_ref + dρe) / ρ

    ## velocity field
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the temperature
    T = air_temperature(e_int)

    abs(n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(T)
  end
end
#md nothing # hide

# The only boundary condition needed for this test problem is the no flux
# boundary condition, the state for which is defined below. This function
# defines the plus-side (exterior) values from the minus-side (inside) values.
# This plus-side value will then be fed into the numerical flux routine in order
# to enforce the boundary condition.
function nofluxbc!(QP, VFP, _, nM, QM, _, auxM, _...)
  @inbounds begin
    DFloat = eltype(QM)
    ## get the minus values
    dρM, ρuM, ρvM, ρwM, dρeM = QM[_dρ], QM[_ρu], QM[_ρv], QM[_ρw], QM[_dρe]

    ## scalars are preserved
    dρP, dρeP = dρM, dρeM

    ## vectors are reflected
    nx, ny, nz = nM[1], nM[2], nM[3]

    ## reflect velocities
    mag_ρu⃗ = nx * ρuM + ny * ρvM + nz * ρwM
    ρuP = ρuM - 2mag_ρu⃗ * nx
    ρvP = ρvM - 2mag_ρu⃗ * ny
    ρwP = ρwM - 2mag_ρu⃗ * nz

    ## Construct QP state
    QP[_dρ], QP[_ρu], QP[_ρv], QP[_ρw], QP[_dρe] = dρP, ρuP, ρvP, ρwP, dρeP
  end
end
#md nothing # hide

"""
Boundary correction for Neumann boundaries
"""
@inline function stresses_boundary_penalty!(VF, nM, gradient_listM, QM, aM, gradient_listP, QP, aP, bctype, t)
  QP .= 0
  stresses_penalty!(VF, nM, gradient_listM, QM, aM, gradient_listP, QP, aP, t)
end


"""
Gradient term flux correction 
"""
@inline function stresses_penalty!(VF, nM, gradient_listM, QM, aM, gradient_listP, QP, aP, t)
  @inbounds begin
    n_Δgradient_list = similar(VF, Size(3, _ngradstates))
    for j = 1:_ngradstates, i = 1:3
      n_Δgradient_list[i, j] = nM[i] * (gradient_listP[j] - gradient_listM[j]) / 2
    end
    compute_stresses!(VF, n_Δgradient_list)
  end
end
#

#------------------------------------------------------------------------------

# ### Definition of the problem
# Here we define the initial condition as well as the auxiliary state (which
# contains the reference state on which the initial condition depends)

# First it is useful to have a conversion function going between Cartesian and
# spherical coordinates (defined here in terms of radians)
function cartesian_to_spherical(x, y, z)
    r = hypot(x, y, z)
    λ = atan(y, x)
    φ = asin(z / r)
    (r, λ, φ)
end
function spherical_to_cartesian(r, λ, φ)
  x = r*cos(φ)*cos(λ)
  y = r*cos(φ)*sin(λ)
  z = r*sin(φ)
  (x, y, z)
end
#md nothing # hide

# FXG: reference state
# Setup the reference state based on a N=0.0158725 uniformly stratified atmosphere with θ0=315K
function auxiliary_state_initialization!(T0, domain_height, aux, x, y, z, dx, dy, dz)
  @inbounds begin
    DFloat = eltype(aux)
    p0 :: DFloat = MSLP
    θ0 :: DFloat = 315
    gravity :: DFloat = grav
    
    ## Convert to Spherical coordinates
    (r, λ, φ) = cartesian_to_spherical(x, y, z)

    ## Calculate the geopotential ϕ
    h = r - DFloat(planet_radius) # height above the planet surface
    ϕ = gravity * h

    ## Reference Temperature
    T_ref::DFloat = 255

    ## Reference Exner Pressure
    π_ref = exp(-gravity * h / (cp_d * T_ref))

    ## Calculate pressure from exner pressure definition
    P_ref = p0*(π_ref)^(cp_d/R_d)

    ## Density from the ideal gas law
    ρ_ref = air_density(T_ref, P_ref)

    ## Calculate the reference total energy
    e_int = internal_energy(T_ref)
    ρe_ref = e_int * ρ_ref + ρ_ref * ϕ

    ## Sponge coefficient
    ct = DFloat(1 / 2880)
    top_sponge  = planet_radius + 15400

    if r >= top_sponge
      sponge_coefficient = ct * (sinpi((r - top_sponge)/2/(domain_height - top_sponge)))^4
    else
      sponge_coefficient = zero(DFloat)
    end

    ## Fill the auxiliary state array
    aux[_a_ϕ] = ϕ
    ## gradient of the geopotential will be computed numerically below
    aux[_a_ϕx] = 0
    aux[_a_ϕy] = 0
    aux[_a_ϕz] = 0
    aux[_a_ρ_ref]  = ρ_ref
    aux[_a_ρe_ref] = ρe_ref
    aux[_a_x] = x
    aux[_a_y] = y
    aux[_a_z] = z
    aux[_a_sponge] = sponge_coefficient
    aux[_a_Δsqr] = anisotropic_coefficient_sgs3D(dx, dy, dz)
  end
end
#md nothing # hide

# FXG: initial conditions
# The initial condition is the reference state defined previously. Here we define a zero perturbation
function initialcondition!(domain_height, Q, x, y, z, aux, _...)
  @inbounds begin
    DFloat = eltype(Q)
    p0 :: DFloat = MSLP

    (r, λ, φ) = cartesian_to_spherical(x, y, z)
    h = r - DFloat(planet_radius)

    ## Get the reference pressure from the previously defined reference state
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ## Equate total field as reference field
    ρ, ρe = ρ_ref, ρe_ref

    ## Define perturbations from reference field
    dρ = ρ - ρ_ref
    dρe = ρe - ρe_ref
    
    ## Store Initial conditions
    Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe] =  dρ, 0, 0, 0, dρe
  end
end
#md nothing # hide


# This function compute the pressure perturbation for a given state. It will be
# used only in the computation of the pressure perturbation prior to writing the
# VTK output.
function compute_δP!(δP, Q, _, aux)
  @inbounds begin
    ## extract the states
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ## Compute the reference pressure
    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## Compute the fulle states
    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    ## compute the velocity
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the pressure
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)

    ## store the pressure perturbation
    δP[1] = P - P_ref
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

function exponentialverticalwarp(domain_height, x, y, z)
  r, λ, φ = cartesian_to_spherical(x, y, z)

  # vertical grid stretching
  htop = domain_height
  H = 7000 # stretching length scale

  h = r - planet_radius
  h = - H * log(1 - h / htop * (1 - exp(-htop / H)))

  r = planet_radius + h
  spherical_to_cartesian(r, λ, φ)
end

# ### Initialize the DG Method
function setupDG(mpicomm, Ne_vertical, Ne_horizontal, polynomialorder,
                 ArrayType, domain_height, T0, DFloat)

  ## Create the element grid in the vertical direction
  Rrange = range(DFloat(planet_radius), length = Ne_vertical + 1,
                 stop = planet_radius + domain_height)

  ## Set up the mesh topology for the sphere
  topology = StackedCubedSphereTopology(mpicomm, Ne_horizontal, Rrange)

  if use_exponential_vertical_warp
    meshwarp = (x...)->exponentialverticalwarp(domain_height,
                                               Topologies.cubedshellwarp(x...)...)
  else
    meshwarp = Topologies.cubedshellwarp
  end

  ## Set up the grid for the sphere. Note that here we need to pass the
  ## `cubedshellwarp` shell `meshwarp` function so that the degrees of freedom
  ## lay on the sphere (and not just stacked cubes)
  grid = DiscontinuousSpectralElementGrid(topology;
                                          polynomialorder = polynomialorder,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          meshwarp = meshwarp)

  ## Here we use the Rusanov numerical flux which requires the physical flux and
  ## wavespeed
  numflux!(x...) = NumericalFluxes.rusanov!(x..., eulerflux!, wavespeed)

  ## We also use Rusanov to define the numerical boundary flux which also
  ## requires a definition of the state to use for the "plus" side of the
  ## boundary face (calculated here with `nofluxbc!`)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., eulerflux!,
                                                            nofluxbc!,
                                                            wavespeed)

  auxinit!(x...) = auxiliary_state_initialization!(T0, domain_height, x...)
  ## Define the balance law solver
  spatialdiscretization = DGBalanceLaw(grid = grid,
                                       length_state_vector = _nstate,
                                       flux! = eulerflux!,
                                       source! = source!,
                                       numerical_flux! = numflux!,
                                       numerical_boundary_flux! = numbcflux!,
                                       auxiliary_state_length = _nauxstate,
                                       auxiliary_state_initialization! = auxinit!,
                                       # number_gradient_states = _ngradstates,
                                       # states_for_gradient_transform = _states_for_gradient_transform,
                                       # number_viscous_states = _nviscstates,
                                       # gradient_transform! = gradient_vars!,
                                       # viscous_transform! = compute_stresses!,
                                       # viscous_penalty! = stresses_penalty!,
                                       # viscous_boundary_penalty! = stresses_boundary_penalty!
                                      )

  ## Compute Gradient of Geopotential
  DGBalanceLawDiscretizations.grad_auxiliary_state!(spatialdiscretization, _a_ϕ,
                                                    (_a_ϕx, _a_ϕy, _a_ϕz))

  spatialdiscretization
end
#md nothing # hide

# ### Initializing and run the DG method
# Note that the final time and grid size are small so that CI and docs
# generation happens in a reasonable amount of time. Running the simulation to a
# final time of `33` hours allows the wave to propagate all the way around the
# sphere and back. Increasing the numeber of horizontal elements to `~30` is
# required for stable long time simulation.
let
  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)

  ## parameters for defining the cubed sphere.
  Ne_vertical   = 6  # number of vertical elements (small for CI/docs reasons)
  ## Ne_vertical   = 30 # Resolution required for stable long time result
  ## cubed sphere will use Ne_horizontal * Ne_horizontal horizontal elements in
  ## each of the 6 faces
  Ne_horizontal = 3

  polynomialorder = 5

  ## top of the domain and temperature from Tomita and Satoh (2004)
  domain_height = 30e3

  ## isothermal temperature state
  T0 = 315

  ## Floating point type to use in the calculation
  DFloat = Float64

  spatialdiscretization = setupDG(mpicomm, Ne_vertical, Ne_horizontal,
                                  polynomialorder, DeviceArrayType,
                                  domain_height, T0, DFloat)

  Q = MPIStateArray(spatialdiscretization,
                    (x...) -> initialcondition!(domain_height, x...))

  ## Since we are using explicit time stepping the acoustic wave speed will
  ## dominate our CFL restriction along with the vertical element size

  if use_exponential_vertical_warp
    # this assumes the stretching increases radially
    first_elem_top = planet_radius + domain_height / Ne_vertical
    (_, _, first_elem_top) =
      exponentialverticalwarp(domain_height, zero(DFloat), zero(DFloat),
                              first_elem_top)
    element_size = first_elem_top - planet_radius
  else
    element_size = (domain_height / Ne_vertical)
  end


  acoustic_speed = soundspeed_air(DFloat(T0))
  dt = element_size / acoustic_speed / polynomialorder^2

  ## Adjust the time step so we exactly hit 1 hour for VTK output
  #dt = 60 * 60 / ceil(60 * 60 / dt)
  #dt=1
  
  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

  filter = Grids.CutoffFilter(spatialdiscretization.grid, 3)
#  filter = Grids.ExponentialFilter(spatialdiscretization.grid)

  ## Uncomment line below to extend simulation time and output less frequently
  seconds = 1
  minutes = 60
  hours = 3600
  days = 86400
  outputtime = 0.1*days
  finaltime = 20*days
#  outputtime = 0.001*days
#  outputtime =1*days
  
  @show(polynomialorder,Ne_horizontal,Ne_vertical,dt,finaltime,finaltime/dt)

  ## We will use this array for storing the pressure to write out to VTK
  δP = MPIStateArray(spatialdiscretization; nstate = 1)

  ## Define a convenience function for VTK output
  mkpath(VTKDIR)
  function do_output(vtk_step)
    ## name of the file that this MPI rank will write
    filename = @sprintf("%s/held_suarez_forcing_mpirank%04d_step%04d",
                        VTKDIR, MPI.Comm_rank(mpicomm), vtk_step)

    ## fill the `δP` array with the pressure perturbation
    DGBalanceLawDiscretizations.dof_iteration!(compute_δP!, δP,
                                               spatialdiscretization, Q)

    ## write the vtk file for this MPI rank
    writevtk(filename, Q, spatialdiscretization, _statenames, δP, ("δP",))

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
      ## name of the pvtu file
      pvtuprefix = @sprintf("held_suarez_forcing_step%04d", vtk_step)

      ## name of each of the ranks vtk files
      prefixes = ntuple(i->
                        @sprintf("%s/held_suarez_forcing_mpirank%04d_step%04d",
                                 VTKDIR, i-1, vtk_step),
                        MPI.Comm_size(mpicomm))

      ## Write out the pvtu file
      writepvtu(pvtuprefix, prefixes, (_statenames..., "δP",))

      ## write that we have written the file
      with_logger(mpi_logger) do
        @info @sprintf("Done writing VTK: %s", pvtuprefix)
      end
    end
  end

  ## Setup callback for writing VTK every hour of simulation time and dump
  #initial file
  vtk_step = 0
  do_output(vtk_step)
  cb_vtk = GenericCallbacks.EveryXSimulationSteps(floor(outputtime / dt)) do
    vtk_step += 1
    do_output(vtk_step)
    nothing
  end

  cb_filter = GenericCallbacks.EveryXSimulationSteps(1) do
    DGBalanceLawDiscretizations.apply!(Q, 1:_nstate, spatialdiscretization,
                                       filter;
                                       horizontal=true,
                                       vertical=true)
    nothing
  end

  ## Setup a callback to display simulation runtime information
  starttime = Ref(now())
  cb_info = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (init=false)
    if init
      starttime[] = now()
    end
    with_logger(mpi_logger) do
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     norm(Q))
    end
  end

  ## Setup a callback to display simulation runtime information
  starttime = Ref(now())
  cb_info2 = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (init=false)
    if init
      starttime[] = now()
    end
    with_logger(mpi_logger) do
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     norm(Q))
    end
  end

  solve!(Q, lsrk; timeend = finaltime,
         callbacks = (cb_vtk, cb_filter, cb_info, cb_info2))

end
#md nothing # hide

# ### Finalizing MPI (if necessary)
Sys.iswindows() || MPI.finalize_atexit()
Sys.iswindows() && !isinteractive() && MPI.Finalize()
#md nothing # hide

#md # ## [Plain Program](@id held_suarez_forcing-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here:
#md # [ex\_003\_acoustic\_wave.jl](held_suarez_forcing.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
