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
    DiffusionProblem 
    - collects parameters for diffusion
"""
abstract type DiffusionProblem <: ProblemType end

# struct 
struct DiffusionCubedSphereProblem{FT} <: DiffusionProblem
    D::FT
    H::FT
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
    D = aux.turbulence.D * SMatrix{3,3,Float64}(I)
    auxDG.turbulence.D∇ρ = D * ∇ρ
end 

@inline function flux_second_order!(::DiffusionProblem, flux::Grad, auxDG::Vars)
    flux.ρ += auxDG.turbulence.D∇ρ
end


"""
    Initialize auxiliary variables (per each spatial point = node)
"""
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
    aux.turbulence.D = problem.D
    aux.D = aux.turbulence.D
    aux.cD = aux.turbulence.c
    nothing 
end
@inline function nodal_init_state_auxiliary!(
    problem::DiffusionBoxProblem,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    Δ = lengthscale(geom)

    aux.turbulence.D = problem.D
    aux.D = aux.turbulence.D
    nothing   
end

"""
    Boundary conditions 
    - nothing if diffusion_direction (or direction) = HorizotalDirection()
"""
@inline boundary_conditions(::DiffusionProblem, ::BalanceLaw) = ()
@inline boundary_state!(nf, ::DiffusionProblem, ::BalanceLaw, _...) = nothing

"""
    Other useful functions
"""
# diffusion-dependent timestep - may want to generalise for calculate_dt
@inline Δt(problem::DiffusionProblem, Δx; CFL=0.05) = (Δx /2 )^2 /2 / problem.D * CFL 

# lengthscale-dependent hyperdiffusion coefficient
# @inline D(problem::DiffusionProblem, Δx ) = (Δx /2 )^2 /2 / problem.τ

