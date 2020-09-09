using StaticArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, BoundaryCondition

import ClimateMachine.BalanceLaws:
    vars_state,
    number_states,
    flux_first_order!,
    flux_second_order!,
    source!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_init_state_auxiliary!,
    update_auxiliary_state!,
    init_state_prognostic!,
    boundary_state!,
    boundary_condition,
    wavespeed

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder, NumericalFluxGradient
import ClimateMachine.DGMethods.NumericalFluxes:
    numerical_flux_first_order!, numerical_boundary_flux_second_order!

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

abstract type AdvectionDiffusionProblem end
struct AdvectionDiffusion{dim, P, fluxBC, no_diffusion} <: BalanceLaw
    problem::P
    function AdvectionDiffusion{dim}(
        problem::P,
    ) where {dim, P <: AdvectionDiffusionProblem}
        new{dim, P, false, false}(problem)
    end
    function AdvectionDiffusion{dim, fluxBC}(
        problem::P,
    ) where {dim, P <: AdvectionDiffusionProblem, fluxBC}
        new{dim, P, fluxBC, false}(problem)
    end
    function AdvectionDiffusion{dim, fluxBC, no_diffusion}(
        problem::P,
    ) where {dim, P <: AdvectionDiffusionProblem, fluxBC, no_diffusion}
        new{dim, P, fluxBC, no_diffusion}(problem)
    end
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_state(::AdvectionDiffusion, ::Auxiliary, FT) =
    @vars(coord::SVector{3, FT}, u::SVector{3, FT}, D::SMatrix{3, 3, FT, 9})
function vars_state(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
    ::Auxiliary,
    FT,
) where {dim, P, fluxBC}
    @vars begin
        coord::SVector{3, FT}
        u::SVector{3, FT}
    end
end

# Density is only state
vars_state(::AdvectionDiffusion, ::Prognostic, FT) = @vars(ρ::FT)

# Take the gradient of density
vars_state(::AdvectionDiffusion, ::Gradient, FT) = @vars(ρ::FT)
vars_state(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
    ::Gradient,
    FT,
) where {dim, P, fluxBC} = @vars()

# The DG auxiliary variable: D ∇ρ
vars_state(::AdvectionDiffusion, ::GradientFlux, FT) = @vars(σ::SVector{3, FT})
vars_state(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
    ::GradientFlux,
    FT,
) where {dim, P, fluxBC} = @vars()

"""
    flux_first_order!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                       aux::Vars, t::Real)

Computes non-diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux_first_order!(
    ::AdvectionDiffusion,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    u = aux.u
    flux.ρ += u * ρ
end

"""
flux_second_order!(m::AdvectionDiffusion, flux::Grad, auxDG::Vars)

Computes diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux_second_order!(::AdvectionDiffusion, flux::Grad, auxDG::Vars)
    σ = auxDG.σ
    flux.ρ += -σ
end
flux_second_order!(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
    flux::Grad,
    auxDG::Vars,
) where {dim, P, fluxBC} = nothing
flux_second_order!(
    m::AdvectionDiffusion,
    flux::Grad,
    state::Vars,
    auxDG::Vars,
    auxHDG::Vars,
    aux::Vars,
    t::Real,
) = flux_second_order!(m, flux, auxDG)

"""
    compute_gradient_argument!(m::AdvectionDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function compute_gradient_argument!(
    m::AdvectionDiffusion,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρ = state.ρ
end

"""
    compute_gradient_flux!(m::AdvectionDiffusion, transform::Vars, gradvars::Vars,
               aux::Vars)

Set the variable to take the gradient of (`ρ` in this case)
"""
function compute_gradient_flux!(
    m::AdvectionDiffusion,
    auxDG::Vars,
    gradvars::Grad,
    aux::Vars,
)
    ∇ρ = gradvars.ρ
    D = aux.D
    auxDG.σ = D * ∇ρ
end
compute_gradient_flux!(
    m::AdvectionDiffusion,
    auxDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) = compute_gradient_flux!(m, auxDG, gradvars, aux)

"""
    source!(m::AdvectionDiffusion, _...)

There is no source in the advection-diffusion model
"""
source!(m::AdvectionDiffusion, _...) = nothing

"""
    wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)

Wavespeed with respect to vector `nM`
"""
function wavespeed(
    m::AdvectionDiffusion,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    u = aux.u
    abs(dot(nM, u))
end

function nodal_init_state_auxiliary!(
    m::AdvectionDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
    init_velocity_diffusion!(m.problem, aux, geom)
end

has_variable_coefficients(::AdvectionDiffusionProblem) = false
function update_auxiliary_state!(
    dg::DGModel,
    m::AdvectionDiffusion,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    if has_variable_coefficients(m.problem)
        update_auxiliary_state!(dg, m, Q, t, elems) do m, state, aux, t
            update_velocity_diffusion!(m.problem, m, state, aux, t)
        end
        return true
    end
    return false
end

function init_state_prognostic!(
    m::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    initial_condition!(m.problem, state, aux, coords, t)
end

abstract type Dirichlet <: BoundaryCondition
end
abstract type Neumann <: BoundaryCondition
end

struct DirichletZero <: Dirichlet # 1
end
struct DirichletData <: Dirichlet # 3
end
struct NeumannZero <: Neumann # 4
end
struct NeumannData <: Neumann # 2
end

boundary_condition(::AdvectionDiffusion) = (DirichletZero(), NeumannData(), DirichletData(), NeumannData())

function boundary_state!(nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, bc::Neumann,  m::AdvectionDiffusion, state⁺, args...)
    nothing
end
function boundary_state!(nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, bc::DirichletZero,  m::AdvectionDiffusion, state⁺, args...)
    state⁺.ρ = 0
end
function boundary_state!(nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient}, bc::DirichletData,  m::AdvectionDiffusion, 
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t, args...)

    Dirichlet_data!(m.problem, state⁺, aux⁺, aux⁺.coord, t)
end


function boundary_state!(nf::NumericalFluxSecondOrder, bc::Dirichlet, m::AdvectionDiffusion, args...)
    nothing
end
function boundary_state!(nf::NumericalFluxSecondOrder, bc::Neumann, m::AdvectionDiffusion, 
    state⁺,
    diff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    aux⁻,
    t,
    _...,
)

    FT = eltype(diff⁺)
    ngrad = number_states(m, Gradient())
    ∇state = Grad{vars_state(m, Gradient(), FT)}(similar(
        parent(diff⁺),
        Size(3, ngrad),
    ))
    if bc isa NeumannZero
        # Get analytic gradient
        ∇state.ρ = SVector{3, FT}(0, 0, 0)
    else
        Neumann_data!(m.problem, ∇state, aux⁻, aux⁻.coord, t)
    end
    # convert to auxDG variables
    compute_gradient_flux!(m, diff⁺, ∇state, aux⁻)
    return nothing
end


boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc::Neumann,
    m::AdvectionDiffusion{dim, P, fluxBC, true},
    state⁺,
    diff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    aux⁻,
    t,
    _...,
) where {dim, P, fluxBC} = nothing

function numerical_boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::Dirichlet,
    m::AdvectionDiffusion{dim, P, true},
    F::Grad,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    t,
    _...,
) where {dim, P}

    # Just use the minus side values since Dirichlet
    flux_second_order!(m, F, state⁻, diff⁻, hyperdiff⁻, aux⁻, t)
end

function numerical_boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::NeumannData,
    m::AdvectionDiffusion{dim, P, true},
    F::Grad,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    t,
    _...,
) where {dim, P}

    FT = eltype(diff⁺)
    ngrad = number_states(m, Gradient())
    ∇state = Grad{vars_state(m, Gradient(), FT)}(similar(
        parent(diff⁺),
        Size(3, ngrad),
    ))
    # Get analytic gradient
    Neumann_data!(m.problem, ∇state, aux⁻, aux⁻.coord, t)
    # get the diffusion coefficient
    D = aux⁻.D
    # exact the exact data
    ∇ρ = ∇state.ρ
    # set the flux
    F.ρ = -D * ∇ρ
    return nothing
end

function numerical_boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::NeumannZero,
    m::AdvectionDiffusion{dim, P, true},
    F::Grad,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    t,
    _...,
) where {dim, P}

    FT = eltype(diff⁺)
    F.ρ = SVector{3, FT}(0, 0, 0)
    nothing
end



struct UpwindNumericalFlux <: NumericalFluxFirstOrder end
function numerical_flux_first_order!(
    ::UpwindNumericalFlux,
    ::AdvectionDiffusion,
    fluxᵀn::Vars{S},
    n::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
) where {S, A}
    un⁻ = dot(n, aux⁻.u)
    un⁺ = dot(n, aux⁺.u)
    un = (un⁺ + un⁻) / 2

    fluxᵀn.ρ = un ≥ 0 ? un * state⁻.ρ : un * state⁺.ρ
end
