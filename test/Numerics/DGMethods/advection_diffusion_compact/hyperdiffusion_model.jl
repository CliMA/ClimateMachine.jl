using StaticArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws:
    BalanceLaw,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux,
    GradientHyperFlux,
    GradientLaplacian,
    Hyperdiffusive
import ClimateMachine.BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    compute_gradient_hyperflux!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    boundary_conditions,
    boundary_state!,
    wavespeed,
    transform_post_gradient_laplacian!

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder

using ClimateMachine.Atmos

abstract type HyperDiffusionProblem end
struct HyperDiffusion{dim, P} <: BalanceLaw
    problem::P
    function HyperDiffusion{dim}(
        problem::P,
    ) where {dim, P <: HyperDiffusionProblem}
        new{dim, P}(problem)
    end
end

# Set hyperdiffusion tensor, D, coordinate info, coorc, and c = l^2*(l+1)^2/r^4
vars_state(::HyperDiffusion, ::Auxiliary, FT) = 
    @vars(D::SMatrix{3, 3, FT, 9}, coord::SVector{3, FT}, c::FT, H::SMatrix{3, 3, FT, 9}, P::SMatrix{3, 3, FT, 9})

# Density is only state
vars_state(::HyperDiffusion, ::Prognostic, FT) = 
    @vars(ρ::FT)

# Take the gradient of density
vars_state(::HyperDiffusion, ::Gradient, FT) = 
    @vars(ρ::FT)

# Take the gradient of laplacian of density
vars_state(::HyperDiffusion, ::GradientLaplacian, FT) = 
    @vars(ρ::FT)

#vars_state(::HyperDiffusion, ::GradientFlux, FT) = @vars()

# The DG hyperdiffusion auxiliary variable: P ∇ ρ
vars_state(::HyperDiffusion, ::GradientHyperFlux, FT) =
    @vars(P∇ρ::SVector{3, FT})

# The hyperdiffusion DG auxiliary variable: H ∇ Δρ
vars_state(::HyperDiffusion, ::Hyperdiffusive, FT) = 
    @vars(H∇Δρ::SVector{3, FT})

function flux_first_order!(m::HyperDiffusion, _...) end


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

 - `σ` is hyperdiffusion DG auxiliary variable (`σ = H ∇ Δ ρ` with H being the hyperdiffusion tensor)
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
    H∇Δρ = auxHDG.H∇Δρ
    flux.ρ += H∇Δρ
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

function compute_gradient_hyperflux!(
    ::HyperDiffusion,
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    ) # this is never called
    #@show "yaaayyY!!!!"
    ∇ρ = gradvars.ρ
    P = aux.P
    auxHDG.P∇ρ = P * ∇ρ
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
    H = aux.H
    auxHDG.H∇Δρ = H * ∇Δρ
end


"""
    source!(m::HyperDiffusion, _...)

There is no source in the hyperdiffusion model
"""
source!(m::HyperDiffusion, _...) = nothing

function init_state_prognostic!(
    m::HyperDiffusion,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    initial_condition!(m.problem, state, aux, localgeo, t)
end

boundary_conditions(::HyperDiffusion) = ( AtmosBC(), AtmosBC() )
#boundary_state!(nf, ::HyperDiffusion, _...) = nothing
boundary_state!(
    ::Union{CentralNumericalFluxDivergence, CentralNumericalFluxHigherOrder},
    bc,
    cm::HyperDiffusion,
    _...,
) = nothing

# define variables specific to spherical harmonic testing (generalise later)
function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.coord = geom.coord
    r = norm(aux.coord)
    l = balance_law.problem.l
    aux.c = get_c(l, r)
    aux.D = balance_law.problem.D * SMatrix{3,3,Float64}(I)

    k = aux.coord / norm(aux.coord)
    # horizontal hyperdiffusion tensor
    aux.H = balance_law.problem.D * (I - k * k')
    # horizontal hyperdiffusion projection of gradients
    aux.P = I - k * k'
    # vertical diffusion tensor
    #aux.D = vert_diff_ν * k * k'

end


#@inline function boundary_state!(
#    nf::NumericalFluxFirstOrder,
#    bc,
#    cm::Continuity3dModel,
#)
#return ocean_model_boundary!(cm, bc, nf, args...)
#    args...,
#end