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

# diffusion-specific functions
#include("spherical_harmonics_kernels.jl")

"""
    DiffusionProblem 
    - collects parameters for diffusion
"""
abstract type DiffusionProblem <: ProblemType end

# struct 
struct DiffusionCubedSphereProblem{FT} <: DiffusionProblem
    τ::FT
    l::FT
    m::FT
end

# TODO: need to modify for box test
struct DiffusionBoxProblem{dir, FT} <: DiffusionProblem
    D::SMatrix{3, 3, FT, 9} # make this a scalar
end

"""
    DiffusionProblem <: ProblemType
    - specifies which variable and compute kernels to use to compute the tendency due to diffusion

    ∂ρ_diff.
    --            = - ∇ • (D∇ρ) = - ∇ • F
    ∂t

"""

# Set diffusion tensor, D, coordinate info, coorc, and c = l*(l+1)/r^2
vars_state(::DiffusionProblem, ::Auxiliary, FT) = @vars(c::FT, D::FT)

# variables for gradient computation
vars_state(::DiffusionProblem, ::Gradient, FT) = @vars(ρ::FT)

# variables for gradient of laplacian computation
vars_state(::DiffusionProblem, ::GradientLaplacian, FT) = @vars()

# variables for Laplacian computation
vars_state(::DiffusionProblem, ::GradientFlux, FT) = @vars(D∇ρ::SVector{3, FT})

# The hyperdiffusion DG auxiliary variable: D ∇ Δρ
vars_state(::DiffusionProblem, ::Hyperdiffusive, FT) = @vars()

"""
    Compute kernels
    - compute_gradient_argument! - set up the variable to take the gradient of (`ρ` in this case)
    - transform_post_gradient_laplacian! - collect the gradient of the Laplacian (`D∇³ρ`) into hyperdiffusion's aux 
    - flux_second_order! - add the gradient of the Laplacian to the main BL's flux, the gradient of which will be taken after to obtain the tendency
"""
@inline function compute_gradient_argument!(
    ::DiffusionProblem,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbulence.ρ = state.ρ
end

@inline function compute_gradient_flux!(
    ::DiffusionProblem,
    auxDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇ρ = gradvars.turbulence.ρ
    auxDG.D∇ρ = D * ∇ρ
end 

@inline function flux_second_order!(::DiffusionProblem, flux::Grad, auxDG::Vars)
    flux.ρ += auxDG.turbulence.D∇ρ
end


"""
    Initialize prognostic and auxiliary variables (per each spatial point = node)
"""
@inline function init_state_prognostic!(
    problem::DiffusionProblem,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    state.ρ = initial_condition!(problem, state, aux, t)
end
@inline function nodal_init_state_auxiliary!(
    problem::DiffusionCubedSphereProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    
    FT = eltype(aux)
    
    r = norm(aux.coord)
    l = problem.l
    
    aux.turbulence.c = get_c(l, r)
    
    Δ_hor = lengthscale_horizontal(geom)
    aux.turbulence.D = D(problem, Δ_hor)  
end
@inline function nodal_init_state_auxiliary!(
    problem::DiffusionBoxProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    Δ = lengthscale(geom)

    aux.turbulence.D = D(problem, Δ)  
end

"""
    Boundary conditions 
    - nothing if diffusion_direction (or direction) = HorizotalDirection()
"""
@inline boundary_conditions(::DiffusionProblem, ::BalanceLaw) = ()
@inline boundary_state!(nf, ::DiffusionProblem, ::BalanceLaw, _...) = nothing

"""
    Initial conditions
    - initial condition is given by ρ0 = Y{m,l}(θ, λ)
    - test: ∇^2_horz ρ0 = l(l+1)/r^2 ρ0 where r=a+z

    ∇^2 ( ∇^2_horz Y{m,l}(θ, λ) ) =  l(l+1)/r^2 ( l(l+1)/r^2 Y{m,l} )
"""
@inline function initial_condition!(
    problem::DiffusionCubedSphereProblem,
    state,
    aux,
    t,
)
    @inbounds begin
        FT = eltype(aux) 
        # import planet paraset
        _a::FT = planet_radius(param_set)

        φ = latitude(SphericalOrientation(), aux)
        λ = longitude(SphericalOrientation(), aux)
        r = norm(aux.coord)
        
        z = r - _a

        l = Int64(problem.l)
        m = Int64(problem.m)

        c = get_c(l, r)
        D = aux.turbulence.D
        
        return calc_Ylm(φ, λ, l, m) * exp(- D*c*t) # - D*c*t
    end
end

"""
    Other useful functions
"""
# hyperdiffusion-dependent timestep (only use for hyperdiffusion unit test) - may want to generalise for calculate_dt
#@inline Δt(problem::DiffusionProblem, Δ_min) = Δ_min^4 / 25 / sum( D(problem, Δ_min) ) 
@inline Δt(problem::DiffusionProblem, Δx; CFL=0.05) = (Δx /2 )^4 /2 / D(problem, Δx) * CFL 
#dt = CFL_wanted / CFL_max = CFL_wanted / max( D / dx^4 )

# lengthscale-dependent hyperdiffusion coefficient
@inline D(problem::DiffusionProblem, Δx ) = (Δx /2 )^2 /2 / problem.τ

