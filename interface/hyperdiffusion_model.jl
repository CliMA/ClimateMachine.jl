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


"""
    HyperDiffusionProblem 
    - collects parameters for hyperdiffusion
"""
abstract type HyperDiffusionProblem <: ProblemType end

# struct 
struct HyperDiffusionCubedSphereProblem{FT} <: HyperDiffusionProblem
    D::FT
    H::FT
    l::FT
    m::FT
end

# TODO: need to modify for box test
struct HyperDiffusionBoxProblem{dir, FT} <: HyperDiffusionProblem
    H::SMatrix{3, 3, FT, 9} # make this a scalar
end

"""
    HyperDiffusionProblem <: ProblemType
    - specifies which variable and compute kernels to use to compute the tendency due to hyperdiffusion

    ∂ρ_hyperdiff.
    --            = - ∇ • (H∇³ρ) = - ∇ • F
    ∂t

"""

# Set hyperdiffusion tensor, H, coordinate info, coorc, and c = l^2*(l+1)^2/r^4
vars_state(::HyperDiffusionProblem, ::Auxiliary, FT) = @vars(c::FT, H::FT)

# variables for gradient computation
vars_state(::HyperDiffusionProblem, ::Gradient, FT) = @vars(ρ::FT)

# variables for gradient of laplacian computation
vars_state(::HyperDiffusionProblem, ::GradientLaplacian, FT) = @vars(ρ::FT)

# variables for Laplacian computation
vars_state(::HyperDiffusionProblem, ::GradientFlux, FT) = @vars()

# The hyperdiffusion DG auxiliary variable: H ∇ Δρ
vars_state(::HyperDiffusionProblem, ::Hyperdiffusive, FT) = @vars(H∇³ρ::SVector{3, FT})

"""
    Compute kernels
    - compute_gradient_argument! - set up the variable to take the gradient of (`ρ` in this case)
    - transform_post_gradient_laplacian! - collect the gradient of the Laplacian (`H∇³ρ`) into hyperdiffusion's aux 
    - flux_second_order! - add the gradient of the Laplacian to the main BL's flux, the gradient of which will be taken after to obtain the tendency
"""
@inline function compute_gradient_argument!(
    ::HyperDiffusionProblem,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.hyperdiffusion.ρ = state.ρ
end
@inline function transform_post_gradient_laplacian!(
    ::HyperDiffusionProblem,
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇Δρ = gradvars.hyperdiffusion.ρ
    H = aux.hyperdiffusion.H * SMatrix{3,3,Float64}(I)
    auxHDG.hyperdiffusion.H∇³ρ = H * ∇Δρ
end
@inline function flux_second_order!(
    ::HyperDiffusionProblem,
    flux::Grad,
    state::Vars,
    auxDG::Vars,
    auxHDG::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρ += auxHDG.hyperdiffusion.H∇³ρ
end

"""
    Initialize auxiliary variables (per each spatial point = node)
"""
@inline function nodal_init_state_auxiliary!(
    problem::HyperDiffusionCubedSphereProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    
    FT = eltype(aux)
    
    r = norm(aux.coord)
    l = problem.l
    aux.hyperdiffusion.c = ( get_c(l, r) )^2
    
    Δ_hor = lengthscale_horizontal(geom)
    # aux.hyperdiffusion.H = H(problem, Δ_hor) 
    aux.hyperdiffusion.H = problem.H 
    aux.H = aux.hyperdiffusion.H
    aux.cH = aux.hyperdiffusion.c
    nothing  
end
@inline function nodal_init_state_auxiliary!(
    problem::HyperDiffusionBoxProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    Δ = lengthscale(geom)

    # aux.hyperdiffusion.H = H(problem, Δ)
    aux.hyperdiffusion.H = problem.H
    aux.H = aux.hyperdiffusion.H
    nothing  
end

"""
    Boundary conditions 
    - nothing if diffusion_direction (or direction) = HorizotalDirection()
"""
@inline boundary_conditions(::HyperDiffusionProblem, ::BalanceLaw) = ()
@inline boundary_state!(nf, ::HyperDiffusionProblem, ::BalanceLaw, _...) = nothing


"""
    Other useful functions
"""
# hyperdiffusion-dependent timestep - may want to generalise for calculate_dt
@inline Δt(problem::HyperDiffusionProblem, Δx; CFL=0.05) = (Δx /2)^4/2 / problem.H * CFL 

# lengthscale-dependent hyperdiffusion coefficient
# @inline H(problem::HyperDiffusionProblem, Δx ) = (Δx /2)^4/2 / problem.τ

