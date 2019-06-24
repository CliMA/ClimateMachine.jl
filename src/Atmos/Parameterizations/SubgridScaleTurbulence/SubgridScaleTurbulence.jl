module SubgridScaleTurbulence

# Module dependencies
using CLIMA.MoistThermodynamics
using CLIMA.Grids
using CLIMA.PlanetParameters: grav, cp_d, cv_d

# Module exported functions 
export compute_strainrate_tensor
export compute_stress_tensor
export standard_smagorinsky
export dynamic_smagorinsky
export buoyancy_correction!
export anisotropic_minimum_dissipation_viscosity
export anisotropic_minimum_dissipation_diffusivity
export anisotropic_coefficient_sgs

function anisotropic_coefficient_sgs(Δx, Δy, Δz, Npoly)
    Δ = (Δx * Δy * Δz)^(1/3)
    Δ_sorted = sort([Δx, Δy, Δz])  
    Δ_s1 = Δ_sorted[1]
    Δ_s2 = Δ_sorted[2]
    a1 = Δ_s1 / max(Δx,Δy,Δz) / (Npoly + 1)
    a2 = Δ_s2 / max(Δx,Δy,Δz) / (Npoly + 1)
    f_anisotropic = 1 + 2/27 * ((log(a1))^2 - log(a1)*log(a2) + (log(a2))^2 )
    Δ = Δ*f_anisotropic
    Δsqr = Δ * Δ
    return Δsqr
end

const γ = cp_d / cv_d 
const μ_sgs = 100.0
const C_ss = 0.14 # Typical value of the Smagorinsky-Lilly coeff 0.18 for isotropic turb and 0.23 for atmos flows
const Prandtl_turb = 1 // 3
const Prandtl = 71 // 100

"""
compute_velgrad_tensor takes in the 9 velocity gradient terms and assembles them into a tensor
for algebraic manipulation in the subgrid-scale turbulence computations
"""
function compute_velgrad_tensor(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
 @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]
  end
  D11, D12, D13 = dudx, dudy, dudz
  D21, D22, D23 = dvdx, dvdy, dvdz 
  D31, D32, D33 = dwdx, dwdy, dwdz
  return (D11, D12, D13, D21, D22, D23, D31, D32, D33)
end

"""
compute_strainrate_tensor accepts 9 velocity gradient terms as arguments, calls compute_velgrad_tensor
to assemble the gradient tensor, and returns the strain rate tensor 
Dij = ∇u .................................................. [1]
Sij = 1/2 (∇u + (∇u)ᵀ) .....................................[2]
τij = 2 * ν_e * Sij ........................................[3]
"""
function compute_strainrate_tensor(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
  S11, S12, S13 = dudx, (dudy + dvdx) / 2, (dudz + dwdx) / 2
  S22, S23      = dvdy, (dvdz + dwdy) / 2
  S33           = dwdz
  SijSij = S11^2 + S22^2 + S33^2 + 2 * (S12^2 + S13^2 + S23^2)  
  return (S11, S22, S33, S12, S13, S23, SijSij)
end

"""
Smagorinksy-Lilly SGS Turbulence
--------------------------------
The constant coefficient Smagorinsky SGS turbulence model for 
eddy viscosity ν_e 
and eddy diffusivity D_e 
The resolved scale stress tensor is calculated as in [3]
where Sij represents the components of the resolved
scale rate of strain tensor. ν_t is the unknown eddy
viscosity which is computed here using the assumption
that subgrid turbulence production and dissipation are 
balanced. 

The eddy viscosity ν_e and eddy diffusivity D_e
are returned. Inputs to the function are the grid descriptors
(number of elements and polynomial order) and the components
of the resolved scale rate of strain tensor

"""
function standard_smagorinsky(SijSij, Δ2)
  ν_e = sqrt(2.0 * SijSij) * C_ss * C_ss * Δ2
  D_e = ν_e / Prandtl_turb 
  return (ν_e, D_e)
end

"""
Buoyancy adjusted Smagorinsky coefficient for stratified flows
Ri = gravity / θᵥ * ∂θ∂z / 2 |S_{ij}|
"""
function buoyancy_correction!(ν_e, D_e, SijSij, θ, dθdz)
  N2 = grav / θ * dθdz 
  Richardson = N2 / (2 * SijSij)
  buoyancy_factor = N2 <=0 ? 1 : sqrt(max(0.0, 1 - Richardson/Prandtl_turb))
  ν_e *= buoyancy_factor
  D_e *= buoyancy_factor
  return (ν_e, D_e)
end
end
