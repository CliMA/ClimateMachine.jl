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
    boundary_state!
import ClimateMachine.DGMethods: DGModel
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

using ClimateMachine.BalanceLaws

abstract type ProblemType end

include("hyperdiffusion_model.jl") # specific model component 

"""
    TestEquations <: BalanceLaw
    - A `BalanceLaw` for general testing.
    - specifies which variable and compute kernels to use to compute the tendency due to hyperdiffusion

    ∂ρ
    --  = - ∇ • (... + D∇³ρ) = - ∇ • F
    ∂t

"""
abstract type AbstractEquations <: BalanceLaw end
abstract type AbstractEquations3D <: AbstractEquations end

struct TestEquations{D,FT} <: AbstractEquations3D
    domain::D
    advection::Union{ProblemType, Nothing}
    turbulence::Union{ProblemType, Nothing}
    hyperdiffusion::Union{ProblemType, Nothing}
    coriolis::Union{ProblemType, Nothing}
    forcing::Union{ProblemType, Nothing}
    boundary_conditions::Union{ProblemType, Nothing}
    params::Union{FT, Nothing}
    param_set::Union{AbstractEarthParameterSet, Nothing}
    function TestEquations{FT}(
        domain::D;
        advection = nothing,
        turbulence = nothing,
        hyperdiffusion = nothing,
        coriolis = nothing,
        forcing = nothing,
        boundary_conditions = nothing,
        params = nothing,
        param_set = param_set,
    ) where {FT <: AbstractFloat, D}
        return new{D, FT}(
            domain,
            advection,
            turbulence,
            hyperdiffusion,
            coriolis,
            forcing,
            boundary_conditions,
            params,
            param_set,
        )
    end
end

vars_state(m::TestEquations, st::Prognostic, FT) = @vars(ρ::FT)

function vars_state(m::TestEquations, aux::Auxiliary, FT)
    @vars begin
        coord::SVector{3, FT}
        
        # temporary placement of hyperdiffusion variables
        # hyperdiffusion::vars_state(m.hyperdiffusion, aux, FT)
        hd__c::FT
        hd__D::FT

    end
end
function vars_state(m::TestEquations, grad::Gradient, FT)
    @vars begin
        # hyperdiffusion::vars_state(m.hyperdiffusion, grad, FT)
        hd__ρ::FT
    end
end

vars_state(m::TestEquations, grad::GradientFlux, FT) = @vars()

function vars_state(m::TestEquations, st::GradientLaplacian, FT)
    @vars begin
        #hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        hd__ρ::FT
    end
end
function vars_state(m::TestEquations, st::Hyperdiffusive, FT)
    @vars begin
        #hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
        hd__D∇³ρ::SVector{3, FT}
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
    init_state_prognostic!(
        m.hyperdiffusion,
        state::Vars,
        aux::Vars,
        localgeo,
        t::Real,
    )
end

function nodal_init_state_auxiliary!(
    m::TestEquations,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
    nodal_init_state_auxiliary!(
        m.hyperdiffusion,
        aux,
        tmp,
        geom,
    )
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
"""

function compute_gradient_argument!(
    m::TestEquations,
    grad::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_argument!(m.hyperdiffusion, grad, state, aux, t)
end
function compute_gradient_flux!(m::TestEquations, _...) end
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
function flux_first_order!(m::TestEquations, _...) end

function flux_second_order!(
    m::TestEquations,
    flux::Grad,
    state::Vars,
    gradflux::Vars,
    auxMISC::Vars,
    aux::Vars,
    t::Real,
)
    #@show t
    flux_second_order!(m.hyperdiffusion, flux, state, gradflux, auxMISC, aux, t)
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
