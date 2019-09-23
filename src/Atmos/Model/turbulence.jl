#### Turbulence closures
using CLIMA.PlanetParameters
using CLIMA.SubgridScaleParameters
export ConstantViscosityWithDivergence, SmagorinskyLilly

export ConstantViscosityWithDivergence, SmagorinskyLilly, Vreman

abstract type TurbulenceClosure
end

vars_state(::TurbulenceClosure, T) = @vars()
vars_gradient(::TurbulenceClosure, T) = @vars()
vars_diffusive(::TurbulenceClosure, T) = @vars()
vars_aux(::TurbulenceClosure, T) = @vars()

function atmos_init_aux!(::TurbulenceClosure, ::AtmosModel, aux::Vars, geom::LocalGeometry)
end
function atmos_nodal_update_aux!(::TurbulenceClosure, ::AtmosModel, state::Vars, aux::Vars, t::Real)
end
function diffusive!(::TurbulenceClosure, diffusive, ∇transform, state, aux, t, ν)
end
function flux_diffusive!(::TurbulenceClosure, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function flux_nondiffusive!(::TurbulenceClosure, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function gradvariables!(::TurbulenceClosure, transform::Vars, state::Vars, aux::Vars, t::Real)
end

"""
    ConstantViscosityWithDivergence <: TurbulenceClosure

Turbulence with constant dynamic viscosity (`ρν`). Divergence terms are included in the momentum flux tensor.
"""
struct ConstantViscosityWithDivergence{T} <: TurbulenceClosure
  ρν::T
end
dynamic_viscosity_tensor(m::ConstantViscosityWithDivergence, S, ∇transform::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = m.ρν
function scaled_momentum_flux_tensor(m::ConstantViscosityWithDivergence, ρν, S)
  @inbounds trS = tr(S)
  return (-2*ρν) * S + (2*ρν/3)*trS * I
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
struct SmagorinskyLilly{T} <: TurbulenceClosure
  C_smag::T
end
vars_aux(::SmagorinskyLilly,T) = @vars(Δ::T)
vars_gradient(::SmagorinskyLilly,T) = @vars(θ_v::T)
vars_diffusive(::SmagorinskyLilly,T) = @vars(∂θ∂Φ::T)
function atmos_init_aux!(::SmagorinskyLilly, ::AtmosModel, aux::Vars, geom::LocalGeometry)
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
function squared_buoyancy_correction(normS, diffusive::Vars, aux::Vars)
  T = eltype(diffusive)
  N² = inv(aux.moisture.θ_v) * diffusive.turbulence.∂θ∂Φ
  Richardson = N² / (normS^2 + eps(normS))
  sqrt(clamp(T(1) - Richardson*inv_Pr_turb, T(0), T(1)))
end

function strain_rate_magnitude(S::SHermitianCompact{3,T,6}) where {T}
  sqrt(2*S[1,1]^2 + 4*S[2,1]^2 + 4*S[3,1]^2 + 2*S[2,2]^2 + 4*S[3,2]^2 + 2*S[3,3]^2)
end

function dynamic_viscosity_tensor(m::SmagorinskyLilly, S, ∇transform::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # strain rate tensor norm
  # Notation: normS ≡ norm2S = √(2S:S)
  # ρν = (Cₛ * Δ * f_b)² * √(2S:S)
  T = eltype(state)
  @inbounds normS = strain_rate_magnitude(S)
  f_b² = squared_buoyancy_correction(normS, diffusive, aux)
  # Return Buoyancy-adjusted Smagorinsky Coefficient (ρ scaled)
  return state.ρ * normS * f_b² * T(m.C_smag * aux.turbulence.Δ)^2
end
function scaled_momentum_flux_tensor(m::SmagorinskyLilly, ρν, S)
  (-2*ρν) * S
end

"""
  Vreman{DT} <: TurbulenceClosure
  
  §1.3.2 in CLIMA documentation 
Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Smagorinsky coefficient
is specified and used to compute the equivalent Vreman coefficient. 

1) ν_e = √(Bᵦ/(αᵢⱼαᵢⱼ)) where αᵢⱼ = ∂uᵢ∂uⱼ with uᵢ the resolved scale velocity component.
2) βij = Δ²αₘᵢαₘⱼ
3) Bᵦ = β₁₁β₂₂ + β₂₂β₃₃ + β₁₁β₃₃ - β₁₂² - β₁₃² - β₂₃²
βᵢⱼ is symmetric, positive-definite. 
If Δᵢ = Δ, then β = Δ²αᵀα

@article{Vreman2004,
  title={An eddy-viscosity subgrid-scale model for turbulent shear flow: Algebraic theory and applications},
  author={Vreman, AW},
  journal={Physics of fluids},
  volume={16},
  number={10},
  pages={3670--3681},
  year={2004},
  publisher={AIP}
}

"""
struct Vreman{DT} <: TurbulenceClosure
  C_smag::DT
end
vars_aux(::Vreman,T) = @vars(Δ::T)
vars_gradient(::Vreman,T) = @vars(θ_v::T)
vars_diffusive(::Vreman,T) = @vars(∂θ∂Φ::T)
function atmos_init_aux!(::Vreman, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end
function dynamic_viscosity_tensor(m::Vreman, S, ∇transform::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  DT = eltype(state)
  ∇u = ∇transform.u
  αijαij = sum(∇u .^ 2)
  @inbounds normS = strain_rate_magnitude(S)
  f_b² = squared_buoyancy_correction(normS, diffusive, aux)
  βij = f_b² * (aux.turbulence.Δ)^2 * (∇u' * ∇u)
  @inbounds Bβ = βij[1,1]*βij[2,2] - βij[1,2]^2 + βij[1,1]*βij[3,3] - βij[1,3]^2 + βij[2,2]*βij[3,3] - βij[2,3]^2 
  return state.ρ * max(0,m.C_smag^2 * 2.5 * sqrt(abs(Bβ/(αijαij+eps(DT))))) 
end
function scaled_momentum_flux_tensor(m::Vreman, ρν, S)
  (-2*ρν) * S
end
