#### Turbulence closures
using DocStringExtensions
using CLIMA.PlanetParameters
using CLIMA.SubgridScaleParameters
export ConstantViscosityWithDivergence, SmagorinskyLilly, Vreman, AnisoMinDiss
export turbulence_tensors

abstract type TurbulenceClosure end

vars_state(::TurbulenceClosure, FT) = @vars()
vars_aux(::TurbulenceClosure, FT) = @vars()

function atmos_init_aux!(::TurbulenceClosure, ::AtmosModel, aux::Vars, geom::LocalGeometry)
end
function atmos_nodal_update_aux!(::TurbulenceClosure, ::AtmosModel, state::Vars, aux::Vars, t::Real)
end
function gradvariables!(::TurbulenceClosure, transform::Vars, state::Vars, aux::Vars, t::Real)
end
function diffusive!(::TurbulenceClosure, ::Orientation, diffusive, ∇transform, state, aux, t)
end

"""
    ν, τ = turbulence_tensors(::TurbulenceClosure, state::Vars, diffusive::Vars, aux::Vars, t::Real)

Compute the kinematic viscosity tensor (`ν`) and SGS momentum flux tensor (`τ`).
"""
function turbulence_tensors end

"""
    principal_invariants(X)

Calculates principal invariants of a tensor `X`. Returns 3 element tuple containing the invariants.
"""
function principal_invariants(X)
  first = tr(X)
  second = (first^2 - tr(X .^ 2))/2
  third = det(X)
  return (first, second, third)
end

"""
    symmetrize(X)

Compute `(X + X')/2`, returning a `SHermitianCompact` object.
"""
function symmetrize(X::StaticArray{Tuple{3,3}})
  SHermitianCompact(SVector(X[1,1], (X[2,1] + X[1,2])/2, (X[3,1] + X[1,3])/2, X[2,2], (X[3,2] + X[2,3])/2, X[3,3]))
end

"""
    norm2(X)

Compute
```math
\\sum_{i,j} S_{ij}^2
```
"""
function norm2(X::SMatrix{3,3,FT}) where {FT}
  abs2(X[1,1]) + abs2(X[2,1]) + abs2(X[3,1]) +
  abs2(X[1,2]) + abs2(X[2,2]) + abs2(X[3,2]) +
  abs2(X[1,3]) + abs2(X[2,3]) + abs2(X[3,3])
end
function norm2(X::SHermitianCompact{3,FT,6}) where {FT}
  abs2(X[1,1]) + 2*abs2(X[2,1]) + 2*abs2(X[3,1]) +
                   abs2(X[2,2]) + 2*abs2(X[3,2]) +
                                    abs2(X[3,3])
end

"""
    strain_rate_magnitude(S)

Compute
```math
|S| = \\sqrt{2\\sum_{i,j} S_{ij}^2}
```
"""
function strain_rate_magnitude(S::SHermitianCompact{3,FT,6}) where {FT}
  return sqrt(2*norm2(S))
end

"""
    ConstantViscosityWithDivergence <: TurbulenceClosure

Turbulence with constant dynamic viscosity (`ρν`).
Divergence terms are included in the momentum flux tensor.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct ConstantViscosityWithDivergence{FT} <: TurbulenceClosure
  "Dynamic Viscosity [kg/m/s]"
  ρν::FT
end

vars_gradient(::ConstantViscosityWithDivergence,FT) = @vars()
vars_diffusive(::ConstantViscosityWithDivergence, FT) =
  @vars(S::SHermitianCompact{3,FT,6})

function diffusive!(::ConstantViscosityWithDivergence, ::Orientation,
    diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)

  diffusive.turbulence.S = symmetrize(∇transform.u)
end

function turbulence_tensors(m::ConstantViscosityWithDivergence,
    state::Vars, diffusive::Vars, aux::Vars, t::Real)

  S = diffusive.turbulence.S
  ν = m.ρν / state.ρ
  τ = (-2*ν) * S + (2*ν/3)*tr(S) * I
  return ν, τ
end



"""
    SmagorinskyLilly <: TurbulenceClosure

See § 1.3.2 in CliMA documentation

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

Brunt-Vaisala frequency N² defined as in equation (1b) in
  Durran, D.R. and J.B. Klemp, 1982:
  On the Effects of Moisture on the Brunt-Väisälä Frequency.
  J. Atmos. Sci., 39, 2152–2158,
  https://doi.org/10.1175/1520-0469(1982)039<2152:OTEOMO>2.0.CO;2

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SmagorinskyLilly{FT} <: TurbulenceClosure
  "Smagorinsky Coefficient [dimensionless]"
  C_smag::FT
end

vars_aux(::SmagorinskyLilly,FT) = @vars(Δ::FT)
vars_gradient(::SmagorinskyLilly,FT) = @vars(θ_v::FT)
vars_diffusive(::SmagorinskyLilly,FT) = @vars(S::SHermitianCompact{3,FT,6}, N²::FT)


function atmos_init_aux!(::SmagorinskyLilly, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end

function gradvariables!(m::SmagorinskyLilly, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.turbulence.θ_v = aux.moisture.θ_v
end

function diffusive!(::SmagorinskyLilly, orientation::Orientation,
    diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)

  diffusive.turbulence.S = symmetrize(∇transform.u)
  ∇Φ = ∇gravitational_potential(orientation, aux)
  diffusive.turbulence.N² = dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end

function turbulence_tensors(m::SmagorinskyLilly, state::Vars, diffusive::Vars, aux::Vars, t::Real)

  FT = eltype(state)
  S = diffusive.turbulence.S
  normS = strain_rate_magnitude(S)

  # squared buoyancy correction
  Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
  f_b² = sqrt(clamp(1 - Richardson*inv_Pr_turb, 0, 1))
  ν = normS * f_b² * FT(m.C_smag * aux.turbulence.Δ)^2
  τ = (-2*ν) * S
  return ν, τ
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
vars_diffusive(::Vreman,FT) = @vars(∇u::SMatrix{3,3,FT,9}, N²::FT)

function atmos_init_aux!(::Vreman, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end
function gradvariables!(m::Vreman, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.turbulence.θ_v = aux.moisture.θ_v
end
function diffusive!(::Vreman, orientation::Orientation,
                    diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.turbulence.∇u = ∇transform.u
  ∇Φ = ∇gravitational_potential(orientation, aux)
  diffusive.turbulence.N² = dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end

function turbulence_tensors(m::Vreman, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  α = diffusive.turbulence.∇u
  S = symmetrize(α)

  normS = strain_rate_magnitude(S)
  Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
  f_b² = sqrt(clamp(1 - Richardson*inv_Pr_turb, 0, 1))

  β = f_b² * (aux.turbulence.Δ)^2 * (α' * α)
  Bβ = principal_invariants(β)[2]

  ν = max(0, m.C_smag^2 * FT(2.5) * sqrt(abs(Bβ/(norm2(α)+eps(FT)))))
  τ = (-2*ν) * S

  return ν, τ
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
vars_aux(::AnisoMinDiss,FT) = @vars(Δ::FT)
vars_gradient(::AnisoMinDiss,FT) = @vars()
vars_diffusive(::AnisoMinDiss,FT) = @vars(∇u::SMatrix{3,3,FT,9})

function atmos_init_aux!(::AnisoMinDiss, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.turbulence.Δ = lengthscale(geom)
end
function gradvariables!(m::AnisoMinDiss, transform::Vars, state::Vars, aux::Vars, t::Real)
end
function diffusive!(::AnisoMinDiss, ::Orientation,
                    diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.turbulence.∇u = ∇transform.u
end

function turbulence_tensors(m::AnisoMinDiss, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  α = diffusive.turbulence.∇u
  S = symmetrize(α)

  coeff = (aux.turbulence.Δ * m.C_poincare)^2
  βij = -(α' * α)
  ν = max(0, coeff * (dot(βij, S) / (norm2(α) + eps(FT))))
  τ = (-2*ν) * S

  return ν, τ
end
