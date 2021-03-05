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

using ClimateMachine.Mesh.Geometry: LocalGeometry, lengthscale, lengthscale_horizontal
using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder

# hyperdiffusion-specific functions
include("spherical_harmonics_kernels.jl")

"""
    HyperDiffusionProblem 
    - collects parameters for hyperdiffusion
"""
abstract type HyperDiffusionProblem end

# struct 
struct HyperDiffusionCubedSphereProblem{FT} <: HyperDiffusionProblem
    τ::FT
    l::FT
    m::FT
end

# TODO: need to modify for box test
struct HyperDiffusionBoxProblem{dir, FT} <: HyperDiffusionProblem
    D::SMatrix{3, 3, FT, 9} # make this a scalar
end

"""
    HyperDiffusion <: BalanceLaw
    - specifies which variable and compute kernels to use to compute the tendency due to hyperdiffusion

    ∂ρ_hyperdiff.
    --            = - ∇ • (D∇³ρ) = - ∇ • F
    ∂t

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
vars_state(::HyperDiffusionProblem, ::Auxiliary, FT) = @vars(c::FT, D::FT)

# variables for gradient computation
vars_state(::HyperDiffusionProblem, ::Gradient, FT) = @vars(ρ::FT)

# variables for gradient of laplacian computation
vars_state(::HyperDiffusionProblem, ::GradientLaplacian, FT) = @vars(ρ::FT)

# variables for Laplacian computation
vars_state(::HyperDiffusionProblem, ::GradientFlux, FT) = @vars()

# The hyperdiffusion DG auxiliary variable: D ∇ Δρ
vars_state(::HyperDiffusionProblem, ::Hyperdiffusive, FT) = @vars(D∇³ρ::SVector{3, FT})

"""
    Compute kernels
    - compute_gradient_argument! - set up the variable to take the gradient of (`ρ` in this case)
    - transform_post_gradient_laplacian! - collect the gradient of the Laplacian (`D∇³ρ`) into hyperdiffusion's aux 
    - flux_second_order! - add the gradient of the Laplacian to the main BL's flux, the gradient of which will be taken after to obtain the tendency
"""
@inline function compute_gradient_argument!(
    ::HyperDiffusionProblem,
    ::BalanceLaw,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.hyperdiffusion.ρ = state.ρ
end
@inline function transform_post_gradient_laplacian!(
    ::HyperDiffusionProblem,
    ::BalanceLaw,
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Δρ = gradvars.hyperdiffusion.ρ
    D = aux.hyperdiffusion.D * SMatrix{3,3,Float64}(I)
    auxHDG.hyperdiffusion.D∇³ρ = D * ∇Δρ
end
@inline function flux_second_order!(
    ::HyperDiffusionProblem,
    ::BalanceLaw,
    flux::Grad,
    state::Vars,
    auxDG::Vars,
    auxHDG::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρ += auxHDG.hyperdiffusion.D∇³ρ
end

"""
    Initialize prognostic (whole state array at once) and auxiliary variables (per each spatial point = node)
"""
@inline function init_state_prognostic!(
    ::HyperDiffusionProblem,
    m::BalanceLaw,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    initial_condition!(m.problem, state, aux, localgeo, t)
end
@inline function nodal_init_state_auxiliary!(
    p::HyperDiffusionCubedSphereProblem,
    m::BalanceLaw,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    
    FT = eltype(aux)
    
    r = norm(aux.coord)
    l = p.l
    aux.hyperdiffusion.c = get_c(l, r)
    
    Δ_hor = lengthscale_horizontal(geom)
    aux.hyperdiffusion.D = D(p, Δ_hor)  
end
@inline function nodal_init_state_auxiliary!(
    p::HyperDiffusionBoxProblem,
    m::BalanceLaw,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    Δ = lengthscale(geom)

    aux.hyperdiffusion.D = D(p, Δ)  
end

"""
    Boundary conditions 
    - nothing if diffusion_direction (or direction) = HorizotalDirection()
"""
@inline boundary_conditions(::HyperDiffusion, ::BalanceLaw) = ()
@inline boundary_state!(nf, ::HyperDiffusion, ::BalanceLaw, _...) = nothing

"""
    Initial conditions
    - initial condition is given by ρ0 = Y{m,l}(θ, λ)
    - test: ∇^4_horz ρ0 = l^2(l+1)^2/r^4 ρ0 where r=a+z
"""
@inline function initial_condition!(
    p::HyperDiffusionCubedSphereProblem{FT},
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
        
        z = r - _a

        l = Int64(p.l)
        m = Int64(p.m)

        c = get_c(l, r)
        D = aux.hyperdiffusion.D

        state.ρ = calc_Ylm(φ, λ, l, m) * exp(- D*c*t) # - D*c*t
    end
end

"""
    Other useful functions
"""
# hyperdiffusion-dependent timestep (only use for hyperdiffusion unit test)
@inline Δt(p::HyperDiffusionProblem, Δ_min) = Δ_min^4 / 25 / sum( D(p, Δ_min) ) 

# lengthscale-dependent hyperdiffusion coefficient
@inline D(p::HyperDiffusionProblem, Δ ) = (Δ /2)^4/2/ p.τ

