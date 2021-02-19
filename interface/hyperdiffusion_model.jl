using StaticArrays
using LinearAlgebra
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws:
    BalanceLaw,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux,
    GradientLaplacian,
    Hyperdiffusive
import ClimateMachine.BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    boundary_conditions,
    boundary_state!,
    wavespeed,
    transform_post_gradient_laplacian!

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder

# hyperdiffusion-specific files
include("spherical_harmonics_kernels.jl")
include("initial_condition.jl")


"""
HyperDiffusionProblem 
"""
abstract type HyperDiffusionProblem end

# struct 
struct HyperDiffusionCubedSphereProblem{FT} <: HyperDiffusionProblem
    τ::FT
    l::FT
    m::FT
end

# need to modify for box test
struct HyperDiffusionBoxProblem{dir, FT} <: HyperDiffusionProblem
    D::SMatrix{3, 3, FT, 9} # make this a scalar
end

Δt(p::HyperDiffusionProblem, Δ_min) = Δ_min^4 / 25 / sum( D(p, Δ_min) ) 
D(p::HyperDiffusionProblem, Δ ) = (Δ /2)^4/2/ p.τ

"""
HyperDiffusion <: BalanceLaw
"""
struct HyperDiffusion{P} <: BalanceLaw
    problem::P
    function HyperDiffusion(
        problem::P,
    ) where {P <: HyperDiffusionProblem}
        new{P}(problem)
    end
end

# Set hyperdiffusion tensor, D, coordinate info, coorc, and c = l^2*(l+1)^2/r^4
vars_state(::HyperDiffusion, ::Auxiliary, FT) = @vars(D::SMatrix{3, 3, FT, 9}, c::FT)

# Density is only state
#vars_state(::HyperDiffusion, ::Prognostic, FT) = @vars()

# Take the gradient of density
vars_state(::HyperDiffusion, ::Gradient, FT) = @vars(ρ::FT)

# Take the gradient of laplacian of density
vars_state(::HyperDiffusion, ::GradientLaplacian, FT) = @vars(ρ::FT)

#vars_state(::HyperDiffusion, ::GradientFlux, FT) = @vars(ρ::FT)

# The hyperdiffusion DG auxiliary variable: D ∇ Δρ
vars_state(::HyperDiffusion, ::Hyperdiffusive, FT) = @vars(D∇³ρ::SVector{3, FT})

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

    flux.ρ += auxHDG.D∇³ρ
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
    transform.hyperdiffusion.ρ = state.ρ
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
    ∇Δρ = gradvars.hyperdiffusion.ρ
    D = aux.hyperdiffusion.D
    auxHDG.hyperdiffusion.D∇³ρ = D * ∇Δρ
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

boundary_conditions(::HyperDiffusion) = ()
boundary_state!(nf, ::HyperDiffusion, _...) = nothing

# define variables specific to spherical harmonic testing (generalise later)
function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusionCubedSphereProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    
    r = norm(aux.coord)
    l = balance_law.problem.l
    aux.hyperdiffusion.c = get_c(l, r)
    
    Δ_hor = lengthscale_horizontal(geom)
    aux.hyperdiffusion.D = D(balance_law.problem, Δ_hor)  * SMatrix{3,3,Float64}(I)
end
function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusionBoxProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    #aux.coord = geom.coord
    Δ = lengthscale(geom)
    aux.hyperdiffusion.D = D(balance_law.problem, Δ)  * SMatrix{3,3,Float64}(I)
end


"""
    initial condition is given by ρ0 = Y{m,l}(θ, λ)
    test: ∇^4_horz ρ0 = l^2(l+1)^2/r^4 ρ0 where r=a+z
"""
function initial_condition!(
    problem::HyperDiffusionCubedSphereProblem{FT},
    state,
    aux,
    x,
    t,
) where {FT}
    @inbounds begin
        # import planet paraset
        _a::FT = planet_radius(param_set)

        φ = latitude(SphericalOrientation(), aux)
        λ = longitude(SphericalOrientation(), aux)
        r = norm(aux.coord)
        #@show aux.coord
        z = r - _a

        l = Int64(problem.l)
        m = Int64(problem.m)

        c = get_c(l, r)

        #Δ = map(geom -> lengthscale(geom), localgeom)  # LocalGeometry{Np, N}(vgeo, n, e),
        
        
        state.ρ = calc_Ylm(φ, λ, l, m) * exp(- c*t) # - D*c*t
    end
end






