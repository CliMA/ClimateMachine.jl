#### Turbulence closures
using DocStringExtensions
using CLIMA.PlanetParameters
using CLIMA.SubgridScaleParameters
export ConstantViscosityWithDivergence, SmagorinskyLilly, Vreman, AnisoMinDiss

abstract type TurbulenceClosure end

vars_state(::TurbulenceClosure, FT) = @vars()
vars_gradient(::TurbulenceClosure, FT) = @vars()
vars_diffusive(::TurbulenceClosure, FT) = @vars()
vars_aux(::TurbulenceClosure, FT) = @vars()

function atmos_init_aux!(::TurbulenceClosure, ::AtmosModel, aux::Vars, geom::LocalGeometry)
end
function atmos_nodal_update_aux!(::TurbulenceClosure, ::AtmosModel, state::Vars, aux::Vars, t::Real)
end
function diffusive!(::TurbulenceClosure, diffusive, ∇transform, state, aux, t, ν)
end
function gradvariables!(::TurbulenceClosure, transform::Vars, state::Vars, aux::Vars, t::Real)
end

"""
  PrincipalInvariants{FT} 

Calculates principal invariants of a tensor. Returns struct with fields first,second,third 
referring to each of the invariants. 
"""
struct PrincipalInvariants{FT}
  first::FT
  second::FT
  third::FT
end
function compute_principal_invariants(X::StaticArray{Tuple{3,3}})
  first = tr(X)
  second = 1/2 *((tr(X))^2 - tr(X .^ 2))
  third = det(X)
  return PrincipalInvariants{eltype(X)}(first,second,third)
end

"""
    ConstantViscosityWithDivergence <: TurbulenceClosure

Turbulence with constant dynamic viscosity (`ρν`). Divergence terms are included in the momentum flux tensor.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct ConstantViscosityWithDivergence{FT} <: TurbulenceClosure
  "Dynamic Viscosity [kg/m/s]"
  ρν::FT
end
function dynamic_viscosity_tensor(m::ConstantViscosityWithDivergence, S, 
  state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real)
  return m.ρν
end
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

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SmagorinskyLilly{FT} <: TurbulenceClosure
  "Smagorinsky Coefficient [dimensionless]"
  C_smag::FT
end

vars_aux(::SmagorinskyLilly,T) = @vars(Δ::T)
vars_gradient(::SmagorinskyLilly,T) = @vars(θ_v::T)

function atmos_init_aux!(::SmagorinskyLilly, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end

function gradvariables!(m::SmagorinskyLilly, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.turbulence.θ_v = aux.moisture.θ_v
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
function squared_buoyancy_correction(normS, ∇transform::Grad, aux::Vars)
  ∂θ∂Φ = dot(∇transform.turbulence.θ_v, aux.orientation.∇Φ)
  N² = ∂θ∂Φ / aux.moisture.θ_v
  Richardson = N² / (normS^2 + eps(normS))
  sqrt(clamp(1 - Richardson*inv_Pr_turb, 0, 1))
end

function strain_rate_magnitude(S::SHermitianCompact{3,FT,6}) where {FT}
  sqrt(2*S[1,1]^2 + 4*S[2,1]^2 + 4*S[3,1]^2 + 2*S[2,2]^2 + 4*S[3,2]^2 + 2*S[3,3]^2)
end

function dynamic_viscosity_tensor(m::SmagorinskyLilly, S, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real)
  # strain rate tensor norm
  # Notation: normS ≡ norm2S = √(2S:S)
  # ρν = (Cₛ * Δ * f_b)² * √(2S:S)
  FT = eltype(state)
  @inbounds normS = strain_rate_magnitude(S)
  f_b² = squared_buoyancy_correction(normS, ∇transform, aux)
  # Return Buoyancy-adjusted Smagorinsky Coefficient (ρ scaled)
  return state.ρ * normS * f_b² * FT(m.C_smag * aux.turbulence.Δ)^2
end
function scaled_momentum_flux_tensor(m::SmagorinskyLilly, ρν, S)
  (-2*ρν) * S
end

"""
  Vreman{FT} <: TurbulenceClosure
  
  §1.3.2 in CLIMA documentation 
Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Smagorinsky coefficient
is specified and used to compute the equivalent Vreman coefficient. 

1) ν_e = √(Bᵦ/(αᵢⱼαᵢⱼ)) where αᵢⱼ = ∂uⱼ∂uᵢ with uᵢ the resolved scale velocity component.
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

# Fields

$(DocStringExtensions.FIELDS)
"""
struct Vreman{FT} <: TurbulenceClosure
  "Smagorinsky Coefficient [dimensionless]"
  C_smag::FT
end
vars_aux(::Vreman,FT) = @vars(Δ::FT)
vars_gradient(::Vreman,FT) = @vars(θ_v::FT)
function atmos_init_aux!(::Vreman, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end
function gradvariables!(m::Vreman, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.turbulence.θ_v = aux.moisture.θ_v
end
function dynamic_viscosity_tensor(m::Vreman, S, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real)
  FT = eltype(state)
  ∇u = ∇transform.u
  αijαij = sum(∇u .^ 2)
  @inbounds normS = strain_rate_magnitude(S)
  f_b² = squared_buoyancy_correction(normS, ∇transform, aux)
  βij = f_b² * (aux.turbulence.Δ)^2 * (∇u' * ∇u)
  Bβinvariants = compute_principal_invariants(βij)
  @inbounds Bβ = Bβinvariants.second
  return state.ρ * max(0,m.C_smag^2 * 2.5 * sqrt(abs(Bβ/(αijαij+eps(FT))))) 
end
function scaled_momentum_flux_tensor(m::Vreman, ρν, S)
  (-2*ρν) * S
end

"""
  AnisoMinDiss{FT} <: TurbulenceClosure
  
  §1.3.2 in CLIMA documentation 
Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Poincare coefficient
is specified and used to compute the equivalent AnisoMinDiss coefficient (computed as the solution to the 
eigenvalue problem for the Laplacian operator). 

@article{doi:10.1063/1.4928700,
author = {Rozema,Wybe  and Bae,Hyun J.  and Moin,Parviz  and Verstappen,Roel },
title = {Minimum-dissipation models for large-eddy simulation},
journal = {Physics of Fluids},
volume = {27},
number = {8},
pages = {085107},
year = {2015},
doi = {10.1063/1.4928700},
URL = {https://aip.scitation.org/doi/abs/10.1063/1.4928700},
eprint = {https://aip.scitation.org/doi/pdf/10.1063/1.4928700}
}
-------------------------------------------------------------------------------------
# TODO: Future versions will include modifications of Abkar(2016), Verstappen(2018) 
@article{PhysRevFluids.1.041701,
title = {Minimum-dissipation scalar transport model for large-eddy simulation of turbulent flows},
author = {Abkar, Mahdi and Bae, Hyun J. and Moin, Parviz},
journal = {Phys. Rev. Fluids},
volume = {1},
issue = {4},
pages = {041701},
numpages = {10},
year = {2016},
month = {Aug},
publisher = {American Physical Society},
doi = {10.1103/PhysRevFluids.1.041701},
url = {https://link.aps.org/doi/10.1103/PhysRevFluids.1.041701}
}

"""
struct AnisoMinDiss{FT} <: TurbulenceClosure
  C_poincare::FT
end
vars_aux(::AnisoMinDiss,T) = @vars(Δ::T)
vars_gradient(::AnisoMinDiss,T) = @vars(θ_v::T)
function atmos_init_aux!(::AnisoMinDiss, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end
function gradvariables!(m::AnisoMinDiss, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.turbulence.θ_v = aux.moisture.θ_v
end
function dynamic_viscosity_tensor(m::AnisoMinDiss, S, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real)
  FT = eltype(state)
  ∇u = ∇transform.u
  αijαij = dot(∇u,∇u)
  coeff = (aux.turbulence.Δ * m.C_poincare)^2
  βij = -(∇u' * ∇u)
  ν_e = max(0,coeff * (dot(βij, S) / (αijαij + eps(FT))))
  return state.ρ * ν_e
end
function scaled_momentum_flux_tensor(m::AnisoMinDiss, ρν, S)
  (-2*ρν) * S
end
