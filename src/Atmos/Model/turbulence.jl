# ## Turbulence Closures
# In `turbulence.jl` we specify turbulence closures. Currently,
# pointwise models of the eddy viscosity/eddy diffusivity type are
# supported for turbulent shear and tracer diffusivity. Methods currently supported
# are:\
# [`ConstantViscosityWithDivergence`](@ref constant-viscosity)\
# [`SmagorinskyLilly`](@ref smagorinsky-lilly)\
# [`Vreman`](@ref vreman)\
# [`AnisoMinDiss`](@ref aniso-min-diss)\

#md # !!! note
#md #     Usage: This is a quick-ref guide to using turbulence models as a subcomponent
#md #     of `AtmosModel` \
#md #     $\nu$ is the kinematic viscosity, $C_smag$ is the Smagorinsky Model coefficient,
#md #     `turbulence=ConstantViscosityWithDivergence(ν)`\
#md #     `turbulence=SmagorinskyLilly(C_smag)`\
#md #     `turbulence=Vreman(C_smag)`\
#md #     `turbulence=AnisoMinDiss(C_poincare)`

using DocStringExtensions
using CLIMAParameters.Atmos.SubgridScale: inv_Pr_turb
export ConstantViscosityWithDivergence, SmagorinskyLilly, Vreman, AnisoMinDiss
export turbulence_tensors

# ### Abstract Type
# We define a `TurbulenceClosure` abstract type and
# default functions for the generic turbulence closure
# which will be overloaded with model specific functions.
abstract type TurbulenceClosure end

vars_state_conservative(::TurbulenceClosure, FT) = @vars()
vars_state_auxiliary(::TurbulenceClosure, FT) = @vars()


"""
    atmos_init_aux!
Initialise auxiliary variables for turbulence models.
Overload for specific type of turbulence closure.
"""
function atmos_init_aux!(
    ::TurbulenceClosure,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) end

"""
    atmos_nodal_update_auxiliary_state!
Update auxiliary variables for turbulence models.
Overload for specific turbulence closure type
"""
function atmos_nodal_update_auxiliary_state!(
    ::TurbulenceClosure,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end

"""
    compute_gradient_argument!
Assign pre-gradient-transform variables specific to turbulence models.
"""
function compute_gradient_argument!(
    ::TurbulenceClosure,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
"""
    compute_gradient_flux!(::TurbulenceClosure, _...)
Post-gradient-transformed variables specific to turbulence models.
"""
function compute_gradient_flux!(
    ::TurbulenceClosure,
    ::Orientation,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
) end

function turbulence_tensors end

turbulence_tensors(atmos::AtmosModel, args...) =
    turbulence_tensors(atmos.turbulence, atmos, args...)

"""
    ν, D_t, τ = turbulence_tensors(
                    ::TurbulenceClosure,
                    orientation::Orientation,
                    param_set::AbstractParameterSet,
                    state::Vars,
                    diffusive::Vars,
                    aux::Vars,
                    t::Real
                )
Compute the kinematic viscosity (`ν`), the diffusivity (`D_t`) and SGS momentum flux tensor (`τ`)
for a given turbulence closure. Each closure overloads this method with the appropriate calculations
for the returned quantities.

# Arguments

- `::TurbulenceClosure` = Struct identifier for turbulence closure model
- `orientation` = `AtmosModel.orientation`
- `param_set` = `AtmosModel.param_set`
- `state` = Array of prognostic (state) variables. See `vars_state_conservative` in `AtmosModel`
- `diffusive` = Array of diffusive variables
- `aux` = Array of auxiliary variables
- `t` = time
"""
turbulence_tensors(m::TurbulenceClosure, atmos::AtmosModel, args...) =
    turbulence_tensors(m, atmos.orientation, atmos.param_set, args...)

# We also provide generic math functions for use within the turbulence closures,
# commonly used quantities such as the [principal tensor invariants](@ref tensor-invariants), handling of
# [symmetric tensors](@ref symmetric-tensors) and [tensor norms](@ref tensor-norms)are addressed.

# ### [Pricipal Invariants](@id tensor-invariants)
# ```math
# \textit{I}_{1} = \mathrm{tr(X)} \\
# \textit{I}_{2} = (\mathrm{tr(X)}^2 - \mathrm{tr(X^2)}) / 2 \\
# \textit{I}_{3} = \mathrm{det(X)} \\
# ```
"""
    principal_invariants(X)

Calculates principal invariants of a tensor `X`. Returns 3 element tuple containing the invariants.
"""
function principal_invariants(X)
    first = tr(X)
    second = (first^2 - tr(X .^ 2)) / 2
    third = det(X)
    return (first, second, third)
end

# ### [Symmetrize](@id symmetric-tensors)
# ```math
# \frac{\mathrm{X} + \mathrm{X}^{T}}{2} \\
# ```
"""
    symmetrize(X)

Given a (3,3) second rank tensor X, compute `(X + X')/2`, returning a
symmetric `SHermitianCompact` object.
"""
function symmetrize(X::StaticArray{Tuple{3, 3}})
    SHermitianCompact(SVector(
        X[1, 1],
        (X[2, 1] + X[1, 2]) / 2,
        (X[3, 1] + X[1, 3]) / 2,
        X[2, 2],
        (X[3, 2] + X[2, 3]) / 2,
        X[3, 3],
    ))
end

# ### [2-Norm](@id tensor-norms)
# Given a tensor X, return the tensor dot product
# ```math
# \sum_{i,j} S_{ij}^2
# ```
"""
    norm2(X)
Given a tensor `X`, computes its tensor dot product.
"""
function norm2(X::SMatrix{3, 3, FT}) where {FT}
    abs2(X[1, 1]) +
    abs2(X[2, 1]) +
    abs2(X[3, 1]) +
    abs2(X[1, 2]) +
    abs2(X[2, 2]) +
    abs2(X[3, 2]) +
    abs2(X[1, 3]) +
    abs2(X[2, 3]) +
    abs2(X[3, 3])
end
function norm2(X::SHermitianCompact{3, FT, 6}) where {FT}
    abs2(X[1, 1]) +
    2 * abs2(X[2, 1]) +
    2 * abs2(X[3, 1]) +
    abs2(X[2, 2]) +
    2 * abs2(X[3, 2]) +
    abs2(X[3, 3])
end

# ### [Strain-rate Magnitude](@id strain-rate-magnitude)
# By definition, the strain-rate magnitude, as defined in
# standard turbulence modelling is computed such that
# ```math
# |\mathrm{S}| = \sqrt{2 \sum_{i,j} \mathrm{S}_{ij}^2}
# ```
# where
# ```math
# \vec{S}(\vec{u}) = \frac{1}{2}  \left(\nabla\vec{u} +  \left( \nabla\vec{u} \right)^T \right)
# ```
# \mathrm{S} is the rate-of-strain tensor. (Symmetric component of the velocity gradient). Note that the
# skew symmetric component (rate-of-rotation) is not currently computed.
"""
    strain_rate_magnitude(S)
Given the rate-of-strain tensor `S`, computes its magnitude.
"""
function strain_rate_magnitude(S::SHermitianCompact{3, FT, 6}) where {FT}
    return sqrt(2 * norm2(S))
end

# ### [Constant Viscosity Model](@id constant-viscosity)
# `ConstantViscosityWithDivergence` requires a user to specify the constant viscosity (kinematic)
# and appropriately computes the turbulent stress tensor based on this term. Diffusivity can be
# computed using the turbulent Prandtl number for the appropriate problem regime.
# ```math
# \tau = - 2 \nu \mathrm{S}
# ```
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

vars_state_gradient(::ConstantViscosityWithDivergence, FT) = @vars()
vars_state_gradient_flux(::ConstantViscosityWithDivergence, FT) =
    @vars(S::SHermitianCompact{3, FT, 6})

function compute_gradient_flux!(
    ::ConstantViscosityWithDivergence,
    ::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    diffusive.turbulence.S = symmetrize(∇transform.u)
end

function turbulence_tensors(
    m::ConstantViscosityWithDivergence,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)

    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    S = diffusive.turbulence.S
    ν = m.ρν / state.ρ
    D_t = ν * _inv_Pr_turb
    τ = (-2 * ν) * S + (2 * ν / 3) * tr(S) * I
    return ν, D_t, τ
end

# ### [Smagorinsky-Lilly](@id smagorinsky-lilly)
# The Smagorinsky turbulence model, with Lilly's correction to
# stratified atmospheric flows, is included in ClimateMachine.
# The input parameter to this model is the Smagorinsky coefficient.
# For atmospheric flows, the coefficient `C_smag` typically takes values between
# 0.15 and 0.23. Flow dependent `C_smag` are currently not supported (e.g. Germano's
# extension). The Smagorinsky-Lilly model does not contain explicit filtered terms.
# #### Equations
# ```math
# \nu = (C_{s} \mathrm{f}_{b} \Delta)^2 \sqrt{|\mathrm{S}|}
# ```
# with the stratification correction term
# ```math
# f_{b} =
#    \begin{cases}
#    1 & \mathrm{Ri} \leq 0 ,\\
#    \max(0, 1 - \mathrm{Ri} / \mathrm{Pr}_{t})^{1/4} & \mathrm{Ri} > 0 .
#    \end{cases}
# ```
# ```math
# \mathrm{Ri} =  \frac{N^2}{{|S|}^2}
# ```
# ```math
# N = \left( \frac{g}{\theta_v} \frac{\partial \theta_v}{\partial z}\right)^{1/2}
# ```
# Here, $\mathrm{Ri}$ and $\mathrm{Pr}_{t}$ are the Richardson and
# turbulent Prandtl numbers respectively.  $\Delta$ is the mixing length in the
# relevant coordinate direction. We use the DG metric terms to determine the
# local effective resolution (see `src/Mesh/Geometry.jl`), and modify the vertical lengthscale by the
# stratification correction factor $\mathrm{f}_{b}$ so that $\Delta_{vert} = \Delta z f_b$.
"""
    SmagorinskyLilly <: TurbulenceClosure

# Fields

$(DocStringExtensions.FIELDS)

# Smagorinsky Model Reference
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

# Lilly Model Reference
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


# Brunt-Väisälä Frequency Reference
    Brunt-Vaisala frequency N² defined as in equation (1b) in
      Durran, D.R. and J.B. Klemp, 1982:
      On the Effects of Moisture on the Brunt-Väisälä Frequency.
      J. Atmos. Sci., 39, 2152–2158,
      https://doi.org/10.1175/1520-0469(1982)039<2152:OTEOMO>2.0.CO;2

"""
struct SmagorinskyLilly{FT} <: TurbulenceClosure
    "Smagorinsky Coefficient [dimensionless]"
    C_smag::FT
end

vars_state_auxiliary(::SmagorinskyLilly, FT) = @vars(Δ::FT)
vars_state_gradient(::SmagorinskyLilly, FT) = @vars(θ_v::FT)
vars_state_gradient_flux(::SmagorinskyLilly, FT) =
    @vars(S::SHermitianCompact{3, FT, 6}, N²::FT)


function atmos_init_aux!(
    ::SmagorinskyLilly,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.turbulence.Δ = lengthscale(geom)
end

function compute_gradient_argument!(
    m::SmagorinskyLilly,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.θ_v = aux.moisture.θ_v
end

function compute_gradient_flux!(
    ::SmagorinskyLilly,
    orientation::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    diffusive.turbulence.S = symmetrize(∇transform.u)
    ∇Φ = ∇gravitational_potential(orientation, aux)
    diffusive.turbulence.N² =
        dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end

function turbulence_tensors(
    m::SmagorinskyLilly,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)

    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    S = diffusive.turbulence.S
    normS = strain_rate_magnitude(S)
    k̂ = vertical_unit_vector(orientation, param_set, aux)

    # squared buoyancy correction
    Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
    f_b² = sqrt(clamp(FT(1) - Richardson * _inv_Pr_turb, FT(0), FT(1)))
    ν₀ = normS * (m.C_smag * aux.turbulence.Δ)^2 + FT(1e-5)
    ν = SVector{3, FT}(ν₀, ν₀, ν₀)
    ν_v = k̂ .* dot(ν, k̂)
    ν_h = ν₀ .- ν_v
    ν = SDiagonal(ν_h + ν_v .* f_b²)
    D_t = diag(ν) * _inv_Pr_turb
    τ = -2 * ν * S
    return ν, D_t, τ
end

# ### [Vreman Model](@id vreman)
# Vreman's turbulence model for anisotropic flows, which provides a
# less dissipative solution (specifically in the near-wall and transitional regions)
# than the Smagorinsky-Lilly method. This model
# relies of first derivatives of the velocity vector (i.e., the gradient tensor).
# By design, the Vreman model handles transitional as well as fully turbulent flows adequately.
# The input parameter to this model is the Smagorinsky coefficient - the coefficient is modified
# within the model functions to account for differences in model construction.
# #### Equations
# ```math
# \nu_{t} = 2.5 C_{s}^2 \sqrt{\frac{B_{\beta}}{u_{i,j}u_{i,j}}},
# ```
# where ($i,j, m = (1,2,3)$)
# ```math
# \begin{align}
# B_{\beta} &= \beta_{11}\beta_{22} + \beta_{11}\beta_{33} + \beta_{22}\beta_{33} - (\beta_{13}^2 + \beta_{12}^2 + \beta_{23}^2) \\
# \beta_{ij} &= \Delta_{m}^2 u_{i, m} u_{j, m} \\
# u_{i,j} &= \frac{\partial u_{i}}{\partial x_{j}}.
# \end{align}
# ```
"""
    Vreman{FT} <: TurbulenceClosure

Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Smagorinsky coefficient
is specified and used to compute the equivalent Vreman coefficient.

1) ν_e = √(Bᵦ/(αᵢⱼαᵢⱼ)) where αᵢⱼ = ∂uⱼ∂uᵢ with uᵢ the resolved scale velocity component.
2) βij = Δ²αₘᵢαₘⱼ
3) Bᵦ = β₁₁β₂₂ + β₂₂β₃₃ + β₁₁β₃₃ - β₁₂² - β₁₃² - β₂₃²
βᵢⱼ is symmetric, positive-definite.
If Δᵢ = Δ, then β = Δ²αᵀα


# Fields

$(DocStringExtensions.FIELDS)

# Reference

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
struct Vreman{FT} <: TurbulenceClosure
    "Smagorinsky Coefficient [dimensionless]"
    C_smag::FT
end
vars_state_auxiliary(::Vreman, FT) = @vars(Δ::FT)
vars_state_gradient(::Vreman, FT) = @vars(θ_v::FT)
vars_state_gradient_flux(::Vreman, FT) = @vars(∇u::SMatrix{3, 3, FT, 9}, N²::FT)

function atmos_init_aux!(::Vreman, ::AtmosModel, aux::Vars, geom::LocalGeometry)
    aux.turbulence.Δ = lengthscale(geom)
end
function compute_gradient_argument!(
    m::Vreman,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.θ_v = aux.moisture.θ_v
end
function compute_gradient_flux!(
    ::Vreman,
    orientation::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.turbulence.∇u = ∇transform.u
    ∇Φ = ∇gravitational_potential(orientation, aux)
    diffusive.turbulence.N² =
        dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end

function turbulence_tensors(
    m::Vreman,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)
    α = diffusive.turbulence.∇u
    S = symmetrize(α)
    k̂ = vertical_unit_vector(orientation, param_set, aux)

    normS = strain_rate_magnitude(S)
    Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
    f_b² = sqrt(clamp(1 - Richardson * _inv_Pr_turb, 0, 1))

    β = f_b² * (aux.turbulence.Δ)^2 * (α' * α)
    Bβ = principal_invariants(β)[2]

    ν₀ = m.C_smag^2 * FT(2.5) * sqrt(abs(Bβ / (norm2(α) + eps(FT))))

    ν = SVector{3, FT}(ν₀, ν₀, ν₀)
    ν_v = k̂ .* dot(ν, k̂)
    ν_h = ν₀ .- ν_v
    ν = SDiagonal(ν_h + ν_v .* f_b²)
    D_t = diag(ν) * _inv_Pr_turb
    τ = -2 * ν * S
    return ν, D_t, τ
end

# ### [Anisotropic Minimum Dissipation](@id aniso-min-diss)
# This method is based Vreugdenhil and Taylor's minimum-dissipation eddy-viscosity model.
# The principles of the Rayleigh quotient minimizer are applied to the energy dissipation terms in the
# conservation equations, resulting in a maximum dissipation bound, and a model for
# eddy viscosity and eddy diffusivity.
# ```math
# \nu_e = (\mathrm{C}\delta)^2  \mathrm{max}\left[0, - \frac{\hat{\partial}_k \hat{u}_{i} \hat{\partial}_k \hat{u}_{j} \mathrm{\hat{S}}_{ij}}{\hat{\partial}_p \hat{u}_{q}} \right]
# ```
"""
    AnisoMinDiss{FT} <: TurbulenceClosure

Filter width Δ is the local grid resolution calculated from the mesh metric tensor. A Poincare coefficient
is specified and used to compute the equivalent AnisoMinDiss coefficient (computed as the solution to the
eigenvalue problem for the Laplacian operator).

# Fields
$(DocStringExtensions.FIELDS)

# Reference

    @article{
        doi:10.1063/1.5037039,
        author = {Vreugdenhil,Catherine A.  and Taylor,John R. },
        title = {Large-eddy simulations of stratified plane Couette flow using the anisotropic minimum-dissipation model},
        journal = {Physics of Fluids},
        volume = {30},
        number = {8},
        pages = {085104},
        year = {2018},
        doi = {10.1063/1.5037039},
        URL = {https://doi.org/10.1063/1.5037039}
    }

"""
struct AnisoMinDiss{FT} <: TurbulenceClosure
    C_poincare::FT
end
vars_state_auxiliary(::AnisoMinDiss, FT) = @vars(Δ::FT)
vars_state_gradient(::AnisoMinDiss, FT) = @vars(θ_v::FT)
vars_state_gradient_flux(::AnisoMinDiss, FT) =
    @vars(∇u::SMatrix{3, 3, FT, 9}, N²::FT)
function atmos_init_aux!(
    ::AnisoMinDiss,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.turbulence.Δ = lengthscale(geom)
end
function compute_gradient_argument!(
    m::AnisoMinDiss,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.θ_v = aux.moisture.θ_v
end
function compute_gradient_flux!(
    ::AnisoMinDiss,
    orientation::Orientation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Φ = ∇gravitational_potential(orientation, aux)
    diffusive.turbulence.∇u = ∇transform.u
    diffusive.turbulence.N² =
        dot(∇transform.turbulence.θ_v, ∇Φ) / aux.moisture.θ_v
end
function turbulence_tensors(
    m::AnisoMinDiss,
    orientation::Orientation,
    param_set::AbstractParameterSet,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    k̂ = vertical_unit_vector(orientation, param_set, aux)
    _inv_Pr_turb::FT = inv_Pr_turb(param_set)

    ∇u = diffusive.turbulence.∇u
    S = symmetrize(∇u)
    normS = strain_rate_magnitude(S)

    δ = aux.turbulence.Δ
    Richardson = diffusive.turbulence.N² / (normS^2 + eps(normS))
    f_b² = sqrt(clamp(1 - Richardson * _inv_Pr_turb, 0, 1))

    δ_vec = SVector(δ, δ, δ)
    δ_m = δ_vec ./ transpose(δ_vec)
    ∇û = ∇u .* δ_m
    Ŝ = symmetrize(∇û)
    ν₀ =
        (m.C_poincare .* δ_vec) .^ 2 * max(
            FT(1e-5),
            -dot(transpose(∇û) * (∇û), Ŝ) / (dot(∇û, ∇û) .+ eps(normS)),
        )

    ν_v = k̂ .* dot(ν₀, k̂)
    ν_h = ν₀ .- ν_v
    ν = SDiagonal(ν_h + ν_v .* f_b²)
    D_t = diag(ν) * _inv_Pr_turb
    τ = -2 * ν * S
    return ν, D_t, τ
end
