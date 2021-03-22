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
AdvectionProblem 
    - collects parameters for advection
"""
abstract type AdvectionProblem <: ProblemType end

# struct 
struct AdvectionCubedSphereProblem <: AdvectionProblem end

# TODO: need to modify for box test
struct AdvectionBoxProblem <: AdvectionProblem end

"""
AdvectionProblem <: ProblemType
    - specifies which variable and compute kernels to use to compute the tendency due to advection

    ∂ρ_adv.
    --            = - ∇ • (ρu) = - ∇ • F
    ∂t

"""


"""
    Compute kernels
    - flux_first_order! - add the advection of the state variable to the first order flux, the gradient of which will be taken after to obtain the tendency
"""

@inline function flux_first_order!(
    ::AdvectionProblem,
    flux::Grad,
    state::Vars,
    aux::Vars,
)
    ρ = state.ρ
    u = aux.u
    flux.ρ += u * ρ
end

"""
    Boundary conditions 
    - nothing if diffusion_direction (or direction) = HorizotalDirection()
"""
@inline boundary_conditions(::AdvectionProblem, ::BalanceLaw) = ()
@inline boundary_state!(nf, ::AdvectionProblem, ::BalanceLaw, _...) = nothing


"""
    Other useful functions
"""
@inline Δt(problem::AdvectionProblem, Δx; u = 0, CFL=0.1) = Δx  * CFL  / u 

