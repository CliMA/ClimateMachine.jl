using StaticArrays
using ClimateMachine.VariableTemplates
import ClimateMachine.DGmethods:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    flux_first_order!,
    flux_second_order!,
    source!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!,
    wavespeed,
    LocalGeometry,
    vars_gradient_laplacian,
    vars_hyperdiffusive,
    transform_post_gradient_laplacian!
using ClimateMachine.DGmethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder

abstract type HyperDiffusionProblem end
struct HyperDiffusion{dim, P} <: BalanceLaw
    problem::P
    function HyperDiffusion{dim}(
        problem::P,
    ) where {dim, P <: HyperDiffusionProblem}
        new{dim, P}(problem)
    end
end

vars_state_auxiliary(::HyperDiffusion, FT) = @vars(D::SMatrix{3, 3, FT, 9})
#
# Density is only state
vars_state_conservative(::HyperDiffusion, FT) = @vars(ρ::FT)

# Take the gradient of density
vars_state_gradient(::HyperDiffusion, FT) = @vars(ρ::FT)
# Take the gradient of laplacian of density
vars_gradient_laplacian(::HyperDiffusion, FT) = @vars(ρ::FT)

vars_state_gradient_flux(::HyperDiffusion, FT) = @vars()
# The hyperdiffusion DG auxiliary variable: D ∇ Δρ
vars_hyperdiffusive(::HyperDiffusion, FT) = @vars(σ::SVector{3, FT})

function flux_first_order!(
    m::HyperDiffusion,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

"""
    flux_second_order!(m::HyperDiffusion, flux::Grad, state::Vars,
                     auxDG::Vars, auxHDG::Vars, aux::Vars, t::Real)

Computes diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (σ) = - ∇ • F
∂t
```
Where

 - `σ` is hyperdiffusion DG auxiliary variable (`σ = D ∇ Δρ` with D being the hyperdiffusion tensor)
"""
function flux_second_order!(
    m::HyperDiffusion,
    flux::Grad,
    state::Vars,
    auxDG::Vars,
    auxHDG::Vars,
    aux::Vars,
    t::Real,
)
    σ = auxHDG.σ
    flux.ρ += σ
end

"""
    compute_gradient_argument!(m::HyperDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function compute_gradient_argument!(
    m::HyperDiffusion,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρ = state.ρ
end

compute_gradient_flux!(m::HyperDiffusion, _...) = nothing
function transform_post_gradient_laplacian!(
    m::HyperDiffusion,
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Δρ = gradvars.ρ
    D = aux.D
    auxHDG.σ = D * ∇Δρ
end

"""
    source!(m::HyperDiffusion, _...)

There is no source in the hyperdiffusion model
"""
source!(m::HyperDiffusion, _...) = nothing

"""
    init_state_auxiliary!(m::HyperDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_state_auxiliary!(
    m::HyperDiffusion,
    aux::Vars,
    geom::LocalGeometry,
)
    init_hyperdiffusion_tensor!(m.problem, aux, geom)
end

function init_state_conservative!(
    m::HyperDiffusion,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    initial_condition!(m.problem, state, aux, coords, t)
end

boundary_state!(nf, ::HyperDiffusion, _...) = nothing
