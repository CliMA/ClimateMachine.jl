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
    advection::Union{BalanceLaw, Nothing}
    turbulence::Union{BalanceLaw, Nothing}
    hyperdiffusion::Union{BalanceLaw, Nothing}
    coriolis::Union{BalanceLaw, Nothing}
    forcing::Union{BalanceLaw, Nothing}
    boundary_conditions::Union{BalanceLaw, Nothing}
    params::Union{FT, Nothing}
    function TestEquations{FT}(
        domain::D;
        advection = nothing,
        turbulence = nothing,
        hyperdiffusion = nothing,
        coriolis = nothing,
        forcing = nothing,
        boundary_conditions = nothing,
        params = nothing
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
        )
    end
end

function vars_state(m::TestEquations, st::Prognostic, FT)
    @vars begin
        ρ::FT
    end
end
function vars_state(m::TestEquations, aux::Auxiliary, FT)
    @vars begin
        coord::SVector{3, FT}
        hyperdiffusion::vars_state(m.hyperdiffusion.problem, aux, FT)
    end
end
function vars_state(m::TestEquations, grad::Gradient, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion.problem, grad, FT)
    end
end
function vars_state(m::TestEquations, grad::GradientFlux, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion.problem, grad, FT)
    end
end 
function vars_state(m::TestEquations, st::GradientLaplacian, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion.problem, st, FT)
    end
end
function vars_state(m::TestEquations, st::Hyperdiffusive, FT)
    @vars begin
        hyperdiffusion::vars_state(m.hyperdiffusion.problem, st, FT)
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
        m.hyperdiffusion.problem,
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
        m.hyperdiffusion.problem,
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
    compute_gradient_argument!(m.hyperdiffusion.problem, m.hyperdiffusion, grad, state, aux, t)
end
function compute_gradient_flux!(
    m::TestEquations,
    gradflux::Vars,
    grad::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing# don't need anything for hyperdiffusion here
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
    m.hyperdiffusion.problem,
    m.hyperdiffusion,
    auxHDG, 
    gradvars,
    state,
    aux,
    t,
    )
end
@inline function flux_first_order!(
    m::TestEquations,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    return nothing
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
    #@show t
    flux_second_order!(m.hyperdiffusion.problem, m.hyperdiffusion, flux, state, gradflux, auxMISC, aux, t)
end
@inline function source!(
    m::TestEquations,
    source::Vars,
    state::Vars,
    gradflux::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    #coriolis_force!(model, model.coriolis, source, state, aux, t)
    #forcing_term!(model, model.forcing, source, state, aux, t)
    #linear_drag!(model, model.turbulence, source, state, aux, t)
    return nothing
end

"""
    Boundary conditions
    
"""
boundary_conditions(::TestEquations) = ()
boundary_state!(nf, ::TestEquations, _...) = nothing


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
