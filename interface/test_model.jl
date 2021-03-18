import ClimateMachine.BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    nodal_update_auxiliary_state!

import ClimateMachine.DGMethods: DGModel
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

using ClimateMachine.BalanceLaws

abstract type ProblemType end

include("advection_model.jl") # specific model component 
include("diffusion_model.jl") # specific model component 
include("hyperdiffusion_model.jl") # specific model component 

include("spherical_harmonics_kernels.jl")
"""
    TestEquations <: BalanceLaw
    - A `BalanceLaw` for general testing.
    - specifies which variable and compute kernels to use to compute the tendency due to advection-diffusion-hyperdiffusion

    ```
    ∂ρ
    --  = - ∇ • ( u ρ + D ∇ ρ + H ∇³ρ) = - ∇ • F
    ∂t
    ```
    
    Where
    
     - `ρ` is the solution vector
     - `u` is the initial advection velocity
     - `D` is the diffusion tensor
     - `H` is the hyperdiffusion tensor

"""
abstract type AbstractEquations <: BalanceLaw end
abstract type AbstractEquations3D <: AbstractEquations end

struct TestEquations{FT,D,A,T,HD,C,F,IC,BC,P,PS} <: AbstractEquations3D
    domain::D
    advection::A
    turbulence::T
    hyperdiffusion::HD
    coriolis::C
    forcing::F
    initial_value_problem::IC
    boundary_problem::BC
    params::P
    param_set::PS
end

function TestEquations{FT}(
    domain::D;
    advection::Union{ProblemType, Nothing} = nothing,
    turbulence::Union{ProblemType, Nothing} = nothing,
    hyperdiffusion::Union{ProblemType, Nothing} = nothing,
    coriolis::Union{ProblemType, Nothing} = nothing,
    forcing::Union{ProblemType, Nothing} = nothing,
    initial_value_problem::Union{AbstractInitialValueProblem, Nothing}= nothing,
    boundary_problem::Union{AbstractBoundaryProblem, Nothing} = nothing,
    params::Union{FT, Nothing} = nothing,
    param_set::Union{AbstractEarthParameterSet, Nothing},
) where {FT <: AbstractFloat, D}
    args = (
        domain,
        advection,
        turbulence,
        hyperdiffusion,
        coriolis,
        forcing,
        initial_value_problem,
        boundary_problem,
        params,
        param_set
    )
    return TestEquations{FT, typeof.(args)...}(args...)
end

vars_state(m::TestEquations, st::Prognostic, FT) = @vars(ρ::FT)

function vars_state(m::TestEquations, st::Auxiliary, FT)
    @vars begin
        coord::SVector{3, FT}
        u::SVector{3, FT}
        #ρ_analytical::FT
        
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        advection::vars_state(m.advection, st, FT)
        turbulence::vars_state(m.turbulence, st, FT)
    end
end
function vars_state(m::TestEquations, st::Gradient, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        turbulence::vars_state(m.turbulence, st, FT)
    end
end

function vars_state(m::TestEquations, st::GradientFlux, FT)
@vars begin
    turbulence::vars_state(m.turbulence, st, FT)
    end
end

function vars_state(m::TestEquations, st::GradientLaplacian, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end
function vars_state(m::TestEquations, st::Hyperdiffusive, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end


"""
    Initialize prognostic (whole state array at once) and auxiliary variables (per each spatial point = node)
"""
function init_state_prognostic!(
    m::TestEquations,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    x = aux.coord[1]
    y = aux.coord[2]
    z = aux.coord[3]

    params = m.initial_value_problem.params
    ic = m.initial_value_problem.initial_conditions

    #state.ρ = ic.ρ(params, x, y, z)
    state.ρ = ic.ρ(m.hyperdiffusion, state, aux, t)
end

function nodal_init_state_auxiliary!(
    m::TestEquations,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    # @show m
    # @show m.turbulence
    aux.coord = geom.coord
    
    FT = eltype(aux)

    # From initial conditions
    #aux.u = (FT(1),FT(1),FT(0))
    x = aux.coord[1]
    y = aux.coord[2]
    z = aux.coord[3]
    params = m.initial_value_problem.params
    ic = m.initial_value_problem.initial_conditions
    aux.u = ic.u(params, x, y, z)

    if m.turbulence != nothing
        nodal_init_state_auxiliary!(
            m.turbulence,
            aux,
            tmp,
            geom,
        )
    end
    if m.hyperdiffusion != nothing
        nodal_init_state_auxiliary!(
            m.hyperdiffusion,
            aux,
            tmp,
            geom,
        )
    end
end


"""
    Compute kernels

    ```
    ∂Y
    -- = - ∇ • F_non-diff - ∇ • F_diff + S(Y)  
    ∂t
    ```

    - compute_gradient_argument! - set up the variable to take the gradient computation (∇̢ρ)
    - compute_gradient_flux! - collects gradients of variables defined in compute_gradient_argument!, and sets them up for (∇̢²ρ and ∇̢³ρ) computations
    - transform_post_gradient_laplacian! - collect the gradient of the Laplacian (i.e. hyperdiffusiove flux) into hyperdiffusive's aux 
    - flux_first_order!  - collects non-diffusive fluxes (`F_non-diff`), the gradient of which will be taken after to obtain the tendency
    - flux_second_order! - collects diffusive fluxes (`F_diff`), the gradient of which will be taken after to obtain the tendency
    - source! - adds S(Y) (e.g. Coriolis, linear drag)
    - nodal_update_auxiliary_state - to update auxstate at each node
"""

function compute_gradient_argument!(
    m::TestEquations,
    grad::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_argument!(m.hyperdiffusion, grad, state, aux, t)
    compute_gradient_argument!(m.turbulence, grad, state, aux, t)
end

function compute_gradient_flux!(
    m::TestEquations,
    auxDG::Vars,
    grad::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_flux!(m.turbulence, auxDG, grad, state, aux, t)
end


function transform_post_gradient_laplacian!(
    m::TestEquations,
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform_post_gradient_laplacian!(
    m.hyperdiffusion,
    auxHDG, 
    gradvars,
    state,
    aux,
    t,
    )
end

function flux_first_order!(
    m::TestEquations,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    directions,
)
    flux_first_order!(m.advection, flux, state, aux)
end

function flux_second_order!(
    m::TestEquations,
    flux::Grad,
    state::Vars,
    gradflux::Vars,
    auxMISC::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(m.hyperdiffusion, flux, state, gradflux, auxMISC, aux, t)
    flux_second_order!(m.turbulence, flux, gradflux)
end
@inline function source!(m::TestEquations, _...) end

"""
    Boundary conditions
    
"""
@inline boundary_conditions(::TestEquations) = ()
@inline boundary_state!(nf, ::TestEquations, _...) = nothing


"""
    DGModel constructor - move somewhere general
"""
function DGModel(m::SpatialModel{BL}) where {BL <: BalanceLaw}
    
    numerical_flux_first_order = m.numerics.flux # should be a function

    rhs = DGModel(
        m.balance_law,
        m.grid,
        numerical_flux_first_order,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction=HorizontalDirection(),
        diffusion_direction=HorizontalDirection(),
    )
    
    return rhs
end

"""
    nodal_update_auxiliary_state! 
    - use to update auxstate on the nodal level at each time step 
"""
function nodal_update_auxiliary_state!(
    m::TestEquations,
    state::Vars,
    aux::Vars,
    t::Real,
)
    #get_analytical(m, state, aux, t)
end
function get_analytical(m, state, aux, t)
    aux.ρ_analytical = initial_condition!(m.hyperdiffusion, state, aux, t)
end

# Defaults
vars_state(m::Union{ProblemType, Nothing}, st::Prognostic, FT) = @vars()
vars_state(m::Union{ProblemType, Nothing}, st::Auxiliary, FT) = @vars()
vars_state(m::Union{ProblemType, Nothing}, st::Gradient, FT) = @vars()
vars_state(m::Union{ProblemType, Nothing}, st::GradientFlux, FT) = @vars()
vars_state(m::Union{ProblemType, Nothing}, st::GradientLaplacian, FT) = @vars()
vars_state(m::Union{ProblemType, Nothing}, st::Hyperdiffusive, FT) = @vars()


flux_first_order!(::Nothing, _...) = nothing
flux_second_order!(::Nothing, _...) = nothing
compute_gradient_argument!(::Nothing, _...) = nothing
compute_gradient_flux!(::Nothing, _...) = nothing