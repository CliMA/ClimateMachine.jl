using StaticArrays
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
    number_states,
    flux_first_order!,
    flux_second_order!,
    source!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_init_state_auxiliary!,
    update_auxiliary_state!,
    init_state_prognostic!,
    boundary_conditions,
    boundary_state!,
    wavespeed,
    transform_post_gradient_laplacian!

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.DGMethods: SpaceDiscretization
using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    NumericalFluxGradient,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxHigherOrder

import ClimateMachine.DGMethods.NumericalFluxes:
    numerical_flux_first_order!, boundary_flux_second_order!

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct Advection{N} <: BalanceLaw end
struct NoAdvection <: BalanceLaw end

struct Diffusion{N} <: BalanceLaw end
struct NoDiffusion <: BalanceLaw end

struct HyperDiffusion{N} <: BalanceLaw end
struct NoHyperDiffusion <: BalanceLaw end

abstract type AdvectionDiffusionProblem end

# Boundary condition types

# boundary condition for operator of order O
# O = 0 -> state BC (Dirichlet)
# O = 1 -> gradient BC (Neumann)
# O = 2 -> laplacian BC
# O = 3 -> gradient laplacian BC
abstract type AbstractBC{O} end
struct HomogeneousBC{O} <: AbstractBC{O} end
struct InhomogeneousBC{O} <: AbstractBC{O} end

any_isa(bcs::AbstractBC, bc) = bcs isa bc
any_isa(bcs::Tuple, bc) = mapreduce(x -> x isa bc, |, bcs)

"""
    AdvectionDiffusion{N} <: BalanceLaw

A balance law describing a system of `N` advection-diffusion-hyperdiffusion
equations:

```
∂ρ
-- = - ∇ • (u ρ - σ + η)
∂t

σ = D ∇ ρ
η = H ∇ Δρ
```
Where

 - `ρ` is the solution vector
 - `u` is the advection velocity
 - `σ` is the DG diffusion auxiliary variable
 - `D` is the diffusion tensor
 - `η` is the DG hyperdiffusion auxiliary variable
 - `H` is the hyperdiffusion tensor
"""
struct AdvectionDiffusion{N, dim, P, fluxBC, A, D, HD, BC} <: BalanceLaw
    problem::P
    advection::A
    diffusion::D
    hyperdiffusion::HD
    boundary_conditions::BC

    function AdvectionDiffusion{dim}(
        problem::P,
        boundary_conditions::BC = ();
        num_equations = 1,
        flux_bc = false,
        advection::Bool = true,
        diffusion::Bool = true,
        hyperdiffusion::Bool = false,
    ) where {dim, P <: AdvectionDiffusionProblem, BC}
        N = num_equations
        adv = advection ? Advection{N}() : NoAdvection()
        A = typeof(adv)
        diff = diffusion ? Diffusion{N}() : NoDiffusion()
        D = typeof(diff)
        hyperdiff = hyperdiffusion ? HyperDiffusion{N}() : NoHyperDiffusion()
        HD = typeof(hyperdiff)
        new{N, dim, P, flux_bc, A, D, HD, BC}(
            problem,
            adv,
            diff,
            hyperdiff,
            boundary_conditions,
        )
    end
end

# Auxiliary variables, always store
# `coord` coordinate points (needed for BCs)
function vars_state(m::AdvectionDiffusion, st::Auxiliary, FT)
    @vars begin
        coord::SVector{3, FT}
        advection::vars_state(m.advection, st, FT)
        diffusion::vars_state(m.diffusion, st, FT)
        hyperdiffusion::vars_state(m.hyperdiffusion, st, FT)
    end
end

#   `u` advection velocity
vars_state(::Advection{1}, ::Auxiliary, FT) = @vars(u::SVector{3, FT})
vars_state(::Advection{N}, ::Auxiliary, FT) where {N} =
    @vars(u::SMatrix{3, N, FT, 3N})
#   `D` diffusion tensor
vars_state(::Diffusion{1}, ::Auxiliary, FT) = @vars(D::SMatrix{3, 3, FT, 9})
vars_state(::Diffusion{N}, ::Auxiliary, FT) where {N} =
    @vars(D::SArray{Tuple{3, 3, N}, FT, 3, 9N})
#   `H` hyperdiffusion tensor
vars_state(::HyperDiffusion{1}, ::Auxiliary, FT) =
    @vars(H::SMatrix{3, 3, FT, 9})
vars_state(::HyperDiffusion{N}, ::Auxiliary, FT) where {N} =
    @vars(H::SArray{Tuple{3, 3, N}, FT, 3, 9N})

# Density `ρ` is the only state
vars_state(::AdvectionDiffusion{1}, ::Prognostic, FT) = @vars(ρ::FT)
vars_state(::AdvectionDiffusion{N}, ::Prognostic, FT) where {N} =
    @vars(ρ::SVector{N, FT})

function vars_state(m::AdvectionDiffusion{N}, ::Gradient, FT) where {N}
    # For pure advection we don't need gradients
    if m.diffusion isa NoDiffusion && m.hyperdiffusion isa NoHyperDiffusion
        return @vars()
    else  # Take the gradient of density
        return N == 1 ? @vars(ρ::FT) : @vars(ρ::SVector{N, FT})
    end
end

# Take the gradient of laplacian of density ρ
vars_state(::HyperDiffusion{1}, ::GradientLaplacian, FT) = @vars(ρ::FT)
vars_state(::HyperDiffusion{N}, ::GradientLaplacian, FT) where {N} =
    @vars(ρ::SVector{N, FT})
vars_state(m::AdvectionDiffusion, st::GradientLaplacian, FT) =
    vars_state(m.hyperdiffusion, st, FT)

# The DG diffusion auxiliary variable: σ = D ∇ρ
vars_state(::Diffusion{1}, ::GradientFlux, FT) = @vars(σ::SVector{3, FT})
vars_state(::Diffusion{N}, ::GradientFlux, FT) where {N} =
    @vars(σ::SMatrix{3, N, FT, 3N})
vars_state(m::AdvectionDiffusion, st::GradientFlux, FT) =
    vars_state(m.diffusion, st, FT)

# The DG hyperdiffusion auxiliary variable: η = H ∇ Δρ
vars_state(::HyperDiffusion{1}, ::Hyperdiffusive, FT) = @vars(η::SVector{3, FT})
vars_state(::HyperDiffusion{N}, ::Hyperdiffusive, FT) where {N} =
    @vars(η::SMatrix{3, N, FT, 3N})
vars_state(m::AdvectionDiffusion, st::Hyperdiffusive, FT) =
    vars_state(m.hyperdiffusion, st, FT)

"""
    flux_first_order!(::Advection, flux::Grad, state::Vars, aux::Vars)

Computes non-diffusive flux `F_adv = u ρ` where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
"""
function flux_first_order!(
    ::Advection{N},
    flux::Grad,
    state::Vars,
    aux::Vars,
) where {N}
    ρ = state.ρ
    u = aux.advection.u
    flux.ρ += u .* ρ'
end
flux_first_order!(::NoAdvection, flux::Grad, state::Vars, aux::Vars) = nothing
flux_first_order!(
    m::AdvectionDiffusion,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
) = flux_first_order!(m.advection, flux, state, aux)

"""
    flux_second_order!(::Diffusion, flux::Grad, auxDG::Vars)

Computes diffusive flux `F_diff = -σ` where:

 - `σ` is DG diffusion auxiliary variable (`σ = D ∇ ρ`
    with `D` being the diffusion tensor)
"""
function flux_second_order!(::Diffusion, flux::Grad, auxDG::Vars)
    σ = auxDG.σ
    flux.ρ += -σ
end
flux_second_order!(::NoDiffusion, flux::Grad, auxDG::Vars) = nothing

"""
    flux_second_order!(::HyperDiffusion, flux::Grad, auxHDG::Vars)

Computes hyperdiffusive flux `F_hyperdiff = η` where:

 - `η` is DG hyperdiffusion auxiliary variable (`η = H ∇ Δρ`
    with `H` being the hyperdiffusion tensor)
"""
function flux_second_order!(::HyperDiffusion, flux::Grad, auxHDG::Vars)
    η = auxHDG.η
    flux.ρ += η
end
flux_second_order!(::NoHyperDiffusion, flux::Grad, auxHDG::Vars) = nothing

function flux_second_order!(
    m::AdvectionDiffusion,
    flux::Grad,
    state::Vars,
    auxDG::Vars,
    auxHDG::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(m.diffusion, flux, auxDG)
    flux_second_order!(m.hyperdiffusion, flux, auxHDG)
end


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
    compute_gradient_flux!(::Diffusion, auxDG::Vars, gradvars::Grad, aux::Vars)

Computes the DG diffusion auxiliary variable `σ = D ∇ ρ` where `D` is
the diffusion tensor.
"""
function compute_gradient_flux!(
    ::Diffusion{N},
    auxDG::Vars,
    gradvars::Grad,
    aux::Vars,
) where {N}
    ∇ρ = gradvars.ρ
    D = aux.diffusion.D
    if N == 1
        auxDG.σ = D * ∇ρ
    else
        auxDG.σ = hcat(ntuple(n -> D[:, :, n] * ∇ρ[:, n], Val(N))...)
    end
end
compute_gradient_flux!(::NoDiffusion, auxDG::Vars, gradvars::Grad, aux::Vars) =
    nothing
compute_gradient_flux!(
    m::AdvectionDiffusion,
    auxDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) = compute_gradient_flux!(m.diffusion, auxDG, gradvars, aux)


"""
    transform_post_gradient_laplacian!(::AdvectionDiffusion, auxHDG::Vars,
        gradvars::Grad, state::Vars, aux::Vars, t::Real)

Computes the DG hyperdiffusion auxiliary variable `η = H ∇ Δρ` where `H` is
the hyperdiffusion tensor.
"""
function transform_post_gradient_laplacian!(
    m::AdvectionDiffusion{N},
    auxHDG::Vars,
    gradvars::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {N}
    ∇Δρ = gradvars.ρ
    H = aux.hyperdiffusion.H
    if N == 1
        auxHDG.η = H * ∇Δρ
    else
        auxHDG.η = hcat(ntuple(n -> H[:, :, n] * ∇Δρ[:, n], Val(N))...)
    end
end

"""
    source!(m::AdvectionDiffusion, _...)

There is no source in the advection-diffusion model
"""
source!(m::AdvectionDiffusion, _...) = nothing

"""
    wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)

Wavespeed with respect to vector `nM`
"""
wavespeed(
    m::AdvectionDiffusion,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
) = wavespeed(m.advection, nM, aux)
function wavespeed(::Advection{N}, nM, aux::Vars) where {N}
    u = aux.advection.u
    if N == 1
        abs(nM' * u)
    else
        SVector(ntuple(n -> abs(nM' * u[:, n]), Val(N)))
    end
end
wavespeed(::NoAdvection, nM, aux::Vars) = 0

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
    spacedisc::SpaceDiscretization,
    m::AdvectionDiffusion,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    if has_variable_coefficients(m.problem)
        update_auxiliary_state!(spacedisc, m, Q, t, elems) do m, state, aux, t
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
    localgeo,
    t::Real,
)
    initial_condition!(m.problem, state, aux, localgeo, t)
end

"""
    inhomogeneous_data!(::Val{O}, problem, data, aux, x, t)

Prescribes `problem` boundary condition data for an operator of order `O`
"""
function inhomogeneous_data! end

boundary_conditions(m::AdvectionDiffusion) = m.boundary_conditions
function boundary_state!(
    nf,
    bcs,
    m::AdvectionDiffusion{N},
    stateP::Vars,
    auxP::Vars,
    nM,
    stateM::Vars,
    auxM::Vars,
    t,
    _...,
) where {N}
    if any_isa(bcs, InhomogeneousBC{0}) # Dirichlet
        inhomogeneous_data!(
            Val(0),
            m.problem,
            stateP,
            auxP,
            (coord = auxP.coord,),
            t,
        )
    elseif any_isa(bcs, AbstractBC{1}) # Neumann
        stateP.ρ = stateM.ρ
    elseif any_isa(bcs, HomogeneousBC{0}) # zero Dirichlet
        stateP.ρ = N == 1 ? 0 : zeros(typeof(stateP.ρ))
    end
end

function boundary_state!(
    nf::CentralNumericalFluxSecondOrder,
    bcs,
    m::AdvectionDiffusion,
    state⁺::Vars,
    diff⁺::Vars,
    hyperdiff⁺::Vars,
    aux⁺::Vars,
    n⁻::SVector,
    state⁻::Vars,
    diff⁻::Vars,
    hyperdiff⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    if m.diffusion isa NoDiffusion && m.hyperdiffusion isa NoHyperDiffusion
        return nothing
    end

    if m.diffusion isa Diffusion
        if any_isa(bcs, AbstractBC{0}) # Dirchlet
            # Just use the minus side values since Dirchlet
            diff⁺.σ = diff⁻.σ
        elseif any_isa(bcs, InhomogeneousBC{1}) # Neumann with data
            FT = eltype(diff⁺)
            ngrad = number_states(m, Gradient())
            ∇state = Grad{vars_state(m, Gradient(), FT)}(similar(
                parent(diff⁺),
                Size(3, ngrad),
            ))
            # Get analytic gradient
            inhomogeneous_data!(Val(1), m.problem, ∇state, aux⁻, aux⁻.coord, t)
            compute_gradient_flux!(m.diffusion, diff⁺, ∇state, aux⁻)
            # compute the diffusive flux using the boundary state
        elseif any_isa(bcs, HomogeneousBC{1}) # zero Neumann
            FT = eltype(diff⁺)
            ngrad = number_states(m, Gradient())
            ∇state = Grad{vars_state(m, Gradient(), FT)}(similar(
                parent(diff⁺),
                Size(3, ngrad),
            ))
            # Get analytic gradient
            ∇state.ρ = zeros(typeof(∇state.ρ))
            # convert to auxDG variables
            compute_gradient_flux!(m.diffusion, diff⁺, ∇state, aux⁻)
        end
    end

    if m.hyperdiffusion isa HyperDiffusion
        if any_isa(bcs, InhomogeneousBC{3})
            FT = eltype(hyperdiff⁺)
            ngradlap = number_states(m, GradientLaplacian())
            ∇Δstate = Grad{vars_state(m, GradientLaplacian(), FT)}(similar(
                parent(hyperdiff⁺),
                Size(3, ngradlap),
            ))
            # Get analytic gradient of laplacian
            inhomogeneous_data!(Val(3), m.problem, ∇Δstate, aux⁻, aux⁻.coord, t)
            transform_post_gradient_laplacian!(
                m,
                hyperdiff⁺,
                ∇Δstate,
                state⁻,
                aux⁻,
                t,
            )
        elseif any_isa(bcs, HomogeneousBC{3})
            FT = eltype(hyperdiff⁺)
            ngradlap = number_states(m, GradientLaplacian())
            ∇Δstate =
                Grad{vars_state(m, GradientLaplacian(), FT)}(zeros(SMatrix{
                    3,
                    ngradlap,
                    FT,
                }))
            transform_post_gradient_laplacian!(
                m,
                hyperdiff⁺,
                ∇Δstate,
                state⁻,
                aux⁻,
                t,
            )
        end
    end
    nothing
end

function boundary_flux_second_order!(
    nf::CentralNumericalFluxSecondOrder,
    bcs,
    m::AdvectionDiffusion{N, dim, P, true},
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
    t,
    _...,
) where {N, dim, P}
    if m.diffusion isa NoDiffusion && m.hyperdiffusion isa NoHyperDiffusion
        return nothing
    end

    # Default initialize flux to minus side
    if any_isa(bcs, AbstractBC{0}) # Dirchlet
        # Just use the minus side values since Dirchlet
        flux_second_order!(m, F, state⁻, diff⁻, hyperdiff⁻, aux⁻, t)
    elseif any_isa(bcs, InhomogeneousBC{1}) # Neumann data
        FT = eltype(diff⁺)
        ngrad = number_states(m, Gradient())
        ∇state = Grad{vars_state(m, Gradient(), FT)}(similar(
            parent(diff⁺),
            Size(3, ngrad),
        ))
        # Get analytic gradient
        inhomogeneous_data!(Val(1), m.problem, ∇state, aux⁻, aux⁻.coord, t)
        # get the diffusion coefficient
        D = aux⁻.diffusion.D
        # exact the exact data
        ∇ρ = ∇state.ρ
        # set the flux
        if N == 1
            F.ρ = -D * ∇ρ
        else
            F.ρ = hcat(ntuple(n -> -D[:, :, n] * ∇ρ[:, n], Val(N))...)
        end
    elseif any_isa(bcs, HomogeneousBC{1}) # Zero Neumann
        F.ρ = zeros(typeof(F.ρ))
    end
    nothing
end

function boundary_state!(
    nf::CentralNumericalFluxDivergence,
    bcs,
    m::AdvectionDiffusion,
    grad⁺::Grad,
    aux⁺::Vars,
    n⁻::SVector,
    grad⁻::Grad,
    aux⁻::Vars,
    t,
)
    if m.hyperdiffusion isa NoHyperDiffusion
        return nothing
    end

    if any_isa(bcs, InhomogeneousBC{1})
        # Get analytic gradient
        inhomogeneous_data!(Val(1), m.problem, grad⁺, aux⁻, aux⁻.coord, t)
    elseif any_isa(bcs, HomogeneousBC{1})
        grad⁺.ρ = zeros(typeof(grad⁺.ρ))
    end
    nothing
end

function boundary_state!(
    ::CentralNumericalFluxHigherOrder,
    bcs,
    m::AdvectionDiffusion{N},
    state⁺::Vars,
    aux⁺::Vars,
    lap⁺::Vars,
    n⁻::SVector,
    state⁻::Vars,
    aux⁻::Vars,
    lap⁻::Vars,
    t,
) where {N}
    if m.hyperdiffusion isa NoHyperDiffusion
        return nothing
    end

    if any_isa(bcs, InhomogeneousBC{2})
        # Get analytic laplacian
        inhomogeneous_data!(Val(2), m.problem, lap⁺, aux⁻, aux⁻.coord, t)
    elseif any_isa(bcs, HomogeneousBC{2})
        lap⁺.ρ = N == 1 ? 0 : zeros(typeof(lap⁺.ρ))
    end
    nothing
end
