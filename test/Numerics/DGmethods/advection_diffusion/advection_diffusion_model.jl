using StaticArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.DGmethods: nodal_update_auxiliary_state!
import ClimateMachine.DGmethods:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    flux_first_order!,
    flux_second_order!,
    source!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    init_state_auxiliary!,
    update_auxiliary_state!,
    init_state_conservative!,
    boundary_state!,
    wavespeed,
    LocalGeometry,
    number_state_conservative,
    number_state_gradient
using ClimateMachine.DGmethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder, NumericalFluxGradient
import ClimateMachine.DGmethods.NumericalFluxes:
    numerical_flux_first_order!, boundary_flux_second_order!

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
vars_state_auxiliary(::AdvectionDiffusion, FT) =
    @vars(coord::SVector{3, FT}, u::SVector{3, FT}, D::SMatrix{3, 3, FT, 9})
function vars_state_auxiliary(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
    FT,
) where {dim, P, fluxBC}
    @vars begin
        coord::SVector{3, FT}
        u::SVector{3, FT}
    end
end

# Density is only state
vars_state_conservative(::AdvectionDiffusion, FT) = @vars(ρ::FT)

# Take the gradient of density
vars_state_gradient(::AdvectionDiffusion, FT) = @vars(ρ::FT)
vars_state_gradient(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
    FT,
) where {dim, P, fluxBC} = @vars()

# The DG auxiliary variable: D ∇ρ
vars_state_gradient_flux(::AdvectionDiffusion, FT) = @vars(σ::SVector{3, FT})
vars_state_gradient_flux(
    ::AdvectionDiffusion{dim, P, fluxBC, true},
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
function wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)
    u = aux.u
    abs(dot(nM, u))
end

"""
    init_state_auxiliary!(m::AdvectionDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_state_auxiliary!(
    m::AdvectionDiffusion,
    aux::Vars,
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
        nodal_update_auxiliary_state!(dg, m, Q, t, elems) do m, state, aux, t
            update_velocity_diffusion!(m.problem, m, state, aux, t)
        end
        return true
    end
    return false
end

function init_state_conservative!(
    m::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    initial_condition!(m.problem, state, aux, coords, t)
end

Neumann_data!(problem, ∇state, aux, x, t) = nothing
Dirichlet_data!(problem, state, aux, x, t) = nothing

function boundary_state!(
    nf,
    m::AdvectionDiffusion,
    stateP::Vars,
    auxP::Vars,
    nM,
    stateM::Vars,
    auxM::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # Dirichlet
        Dirichlet_data!(m.problem, stateP, auxP, auxP.coord, t)
    elseif bctype ∈ (2, 4) # Neumann
        stateP.ρ = stateM.ρ
    elseif bctype == 3 # zero Dirichlet
        stateP.ρ = 0
    end
end

function boundary_state!(
    nf::CentralNumericalFluxSecondOrder,
    m::AdvectionDiffusion,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻::SVector,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)

    if bctype ∈ (1, 3) # Dirchlet
        # Just use the minus side values since Dirchlet
        diff⁺.σ = diff⁻.σ
    elseif bctype == 2 # Neumann with data
        FT = eltype(diff⁺)
        ngrad = number_state_gradient(m, FT)
        ∇state = Grad{vars_state_gradient(m, FT)}(similar(
            parent(diff⁺),
            Size(3, ngrad),
        ))
        # Get analytic gradient
        Neumann_data!(m.problem, ∇state, aux⁻, aux⁻.coord, t)
        compute_gradient_flux!(m, diff⁺, ∇state, aux⁻)
        # compute the diffusive flux using the boundary state
    elseif bctype == 4 # zero Neumann
        FT = eltype(diff⁺)
        ngrad = number_state_gradient(m, FT)
        ∇state = Grad{vars_state_gradient(m, FT)}(similar(
            parent(diff⁺),
            Size(3, ngrad),
        ))
        # Get analytic gradient
        ∇state.ρ = SVector{3, FT}(0, 0, 0)
        # convert to auxDG variables
        compute_gradient_flux!(m, diff⁺, ∇state, aux⁻)
    end
    nothing
end
boundary_state!(
    nf::CentralNumericalFluxSecondOrder,
    m::AdvectionDiffusion{dim, P, fluxBC, true},
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻::SVector,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {dim, P, fluxBC} = nothing

function boundary_flux_second_order!(
    nf::CentralNumericalFluxSecondOrder,
    m::AdvectionDiffusion{dim, P, true},
    F,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    bctype,
    t,
    _...,
) where {dim, P}

    # Default initialize flux to minus side
    if bctype ∈ (1, 3) # Dirchlet
        # Just use the minus side values since Dirchlet
        flux_second_order!(m, F, state⁻, diff⁻, hyperdiff⁻, aux⁻, t)
    elseif bctype == 2 # Neumann data
        FT = eltype(diff⁺)
        ngrad = number_state_gradient(m, FT)
        ∇state = Grad{vars_state_gradient(m, FT)}(similar(
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
    elseif bctype == 4 # Zero Neumann
        FT = eltype(diff⁺)
        F.ρ = SVector{3, FT}(0, 0, 0)
    end
    nothing
end
boundary_flux_second_order!(
    ::CentralNumericalFluxSecondOrder,
    ::AdvectionDiffusion{dim, P, true, true},
    _...,
) where {dim, P} = nothing

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
) where {S, A}
    un⁻ = dot(n, aux⁻.u)
    un⁺ = dot(n, aux⁺.u)
    un = (un⁺ + un⁻) / 2

    fluxᵀn.ρ = un ≥ 0 ? un * state⁻.ρ : un * state⁺.ρ
end
