#### Turbulence closures
using CLIMA.PlanetParameters

abstract type TurbulenceClosure
end

vars_state(::TurbulenceClosure, T) = @vars()
vars_gradient(::TurbulenceClosure, T) = @vars()
vars_diffusive(::TurbulenceClosure, T) = @vars()
vars_aux(::TurbulenceClosure, T) = @vars()

function init_aux!(::TurbulenceClosure, aux::Vars, geom::LocalGeometry)
end

"""
    ConstantViscosityWithDivergence <: TurbulenceClosure

Turbulence with constant dynamic viscosity (`ρν`). Divergence terms are included in the momentum flux tensor.
"""
struct ConstantViscosityWithDivergence <: TurbulenceClosure
  ρν::Float64
end
dynamic_viscosity_tensor(m::ConstantViscosityWithDivergence, S, state::Vars, diffusive::Vars, aux::Vars, t::Real) = m.ρν
function scaled_momentum_flux_tensor(m::ConstantViscosityWithDivergence, ρν, S)
  @inbounds trS = S[1] + S[2] + S[3]
  I = SVector(1,1,1,0,0,0)
  return (-2*ρν) .* S .+ (2*ρν/3)*trS .* I
end

"""
    SmagorinskyLilly <: TurbulenceClosure

  § 1.3.2 in CliMA documentation 

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
struct SmagorinskyLilly{DT} <: TurbulenceClosure
  C_smag::DT
end

vars_aux(::SmagorinskyLilly,T) = @vars(Δ::T, ∂θ∂z::T, f_b::T)
vars_gradient(::SmagorinskyLilly,T) = @vars(θ_v::T)
vars_diffusive(::SmagorinskyLilly,T) = @vars(∂θ∂z::T)
function init_aux!(::SmagorinskyLilly, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end
function gradvariables!(m::SmagorinskyLilly, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.turbulence.θ_v = aux.moisture.θ_v
end
function diffusive!(m::SmagorinskyLilly, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρν::Union{Real,AbstractMatrix})
  diffusive.turbulence.∂θ∂Φ = dot(∇transform.turbulence.θ_v, aux.orientation.∇Φ)
end

"""
  buoyancy_correction(normSij, θᵥ, dθᵥdz)
  return buoyancy_factor, scaling coefficient for Standard Smagorinsky Model
  in stratified flows

Compute the buoyancy adjustment coefficient for stratified flows 
given the strain rate tensor inner product |S| ≡ SijSij ≡ normSij, 
local virtual potential temperature θᵥ and the vertical potential 
temperature gradient dθvdz. 

Brunt-Vaisala frequency N² defined as in equation (1b) in 
  Durran, D.R. and J.B. Klemp, 1982: 
  On the Effects of Moisture on the Brunt-Väisälä Frequency. 
  J. Atmos. Sci., 39, 2152–2158, 
  https://doi.org/10.1175/1520-0469(1982)039<2152:OTEOMO>2.0.CO;2 

Ri = N² / (2*normSij)
Ri = gravity / θᵥ * ∂θᵥ∂z / 2 |S_{ij}|

§1.3.2 in CliMA documentation. 

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
function buoyancy_correction(S, diffusive::Vars, aux::Vars)
  DT = eltype(diffusive)
  Prandtl_t = DT(1//3)
  N² = inv(aux.moisture.θ_v * diffusive.turbulence.∂θ∂Φ)
  normS = sqrt(2*(S[1]^2 + S[2]^2 + S[3]^2 + 2*(S[4]^2 + S[5]^2 + S[6]^2)))
  Richardson = N² / (normS^2 + eps(normS))
  buoyancy_factor = N² <= DT(0) ? DT(1) : sqrt(max(DT(0), DT(1) - Richardson/Prandtl_t))^(DT(1//4))
  return buoyancy_factor
end
function dynamic_viscosity_tensor(m::SmagorinskyLilly, S, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # strain rate tensor norm
  # Notation: normS ≡ norm2S = √(2S:S)
  # ρν = (CₛΔf𝐛)² * √(2S:S)
  T = eltype(state)
  f_b = buoyancy_correction(S, diffusive, aux)
  @inbounds normS = sqrt(2*(S[1]^2 + S[2]^2 + S[3]^2 + 2*(S[4]^2 + S[5]^2 + S[6]^2)))
  # Return Buoyancy-adjusted Smagorinsky Coefficient (ρ scaled)
  return state.ρ * normS * T(m.C_smag * aux.turbulence.Δ * f_b)^2
end
function scaled_momentum_flux_tensor(m::SmagorinskyLilly, ρν, S)
  (-2*ρν) .* S
end
