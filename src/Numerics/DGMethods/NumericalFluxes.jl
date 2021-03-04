module NumericalFluxes

export NumericalFluxGradient,
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    RusanovNumericalFlux,
    RoeNumericalFlux,
    HLLCNumericalFlux,
    RoeNumericalFluxMoist,
    CentralNumericalFluxGradient,
    CentralNumericalFluxFirstOrder,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxHigherOrder,
    LMARSNumericalFlux


using StaticArrays, LinearAlgebra
using ClimateMachine.VariableTemplates
using KernelAbstractions.Extras: @unroll
using ...BalanceLaws
import ...BalanceLaws:
    vars_state,
    boundary_state!,
    wavespeed,
    flux_first_order!,
    flux_second_order!,
    compute_gradient_flux!,
    compute_gradient_argument!,
    transform_post_gradient_laplacian!,
    boundary_conditions

"""
    NumericalFluxGradient

Any `P <: NumericalFluxGradient` should define methods for:

    numerical_flux_gradient!(
        gnf::P,
        balance_law::BalanceLaw,
        diffF, n⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        t
    )

    numerical_boundary_flux_gradient!(
        gnf::P,
        balance_law::BalanceLaw,
        local_state_gradient_flux,
        n⁻,
        local_transform⁻, local_state_prognostic⁻, local_state_auxiliary⁻,
        local_transform⁺, local_state_prognostic⁺, local_state_auxiliary⁺,
        bctype,
        t
    )

"""
abstract type NumericalFluxGradient end

"""
    CentralNumericalFluxGradient <: NumericalFluxGradient

"""
struct CentralNumericalFluxGradient <: NumericalFluxGradient end

function numerical_flux_gradient!(
    ::CentralNumericalFluxGradient,
    balance_law::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::AbstractArray,
    state_gradient⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_gradient⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)

    transform_gradient .=
        SVector(normal_vector) .* (state_gradient⁺ .+ state_gradient⁻)' ./ 2
end

function numerical_boundary_flux_gradient!(
    numerical_flux::CentralNumericalFluxGradient,
    bctype,
    balance_law::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::SVector,
    state_gradient⁻::Vars{T},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_gradient⁺::Vars{T},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {D, T, S, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        aux1⁻,
    )

    compute_gradient_argument!(
        balance_law,
        state_gradient⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
    transform_gradient .= normal_vector .* parent(state_gradient⁺)'
end

"""
    NumericalFluxFirstOrder

Any `N <: NumericalFluxFirstOrder` should define the a method for

    numerical_flux_first_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qaux⁻,
        Q⁺, Qaux⁺,
        t
    )

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_first_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qaux⁻,
        Q⁺, Qaux⁺,
        bctype, t
    )

"""
abstract type NumericalFluxFirstOrder end

function numerical_flux_first_order! end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}

    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        aux1⁻,
    )

    numerical_flux_first_order!(
        numerical_flux,
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
end


"""
    RusanovNumericalFlux <: NumericalFluxFirstOrder

The RusanovNumericalFlux (aka local Lax-Friedrichs) numerical flux.

# Usage

    RusanovNumericalFlux()

Requires a `flux_first_order!` and `wavespeed` method for the balance law.
"""
struct RusanovNumericalFlux <: NumericalFluxFirstOrder end

update_penalty!(::RusanovNumericalFlux, ::BalanceLaw, _...) = nothing

function numerical_flux_first_order!(
    numerical_flux::RusanovNumericalFlux,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    fluxᵀn = parent(fluxᵀn)
    wavespeed⁻ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    max_wavespeed = max.(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed .* (parent(state_prognostic⁻) - parent(state_prognostic⁺))

    # TODO: should this operate on ΔQ or penalty?
    update_penalty!(
        numerical_flux,
        balance_law,
        normal_vector,
        max_wavespeed,
        Vars{S}(penalty),
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )

    fluxᵀn .+= penalty / 2
end

"""
    CentralNumericalFluxFirstOrder() <: NumericalFluxFirstOrder

The central numerical flux for nondiffusive terms

# Usage

    CentralNumericalFluxFirstOrder()

Requires a `flux_first_order!` method for the balance law.
"""
struct CentralNumericalFluxFirstOrder <: NumericalFluxFirstOrder end

function numerical_flux_first_order!(
    ::CentralNumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    #if balance_law isa AtmosModel
     #  p = (state_auxiliary⁻.moisture.p - state_auxiliary⁻.ref_state.p) * I
    #end
    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector / 2)
end

"""
    RoeNumericalFlux() <: NumericalFluxFirstOrder

A numerical flux based on the approximate Riemann solver of Roe

# Usage

    RoeNumericalFlux()

Requires a custom implementation for the balance law.
"""
struct RoeNumericalFlux <: NumericalFluxFirstOrder end

"""
    HLLCNumericalFlux() <: NumericalFluxFirstOrder

A numerical flux based on the approximate Riemann solver of the
HLLC method. The HLLC flux is a modification of the Harten, Lax, van-Leer
(HLL) flux, where an additional contact property is introduced in order
to restore missing rarefraction waves. The HLLC flux requires
model-specific information, hence it requires a custom implementation
based on the underlying balance law.

# Usage

    HLLCNumericalFlux()

Requires a custom implementation for the balance law.

 - [Toro2013](@cite)
"""
struct HLLCNumericalFlux <: NumericalFluxFirstOrder end


"""
    LMARSNumericalFlux <: NumericalFluxFirstOrder
Low Mach Number Approximate Riemann Solver. Upwind biased
first order flux function. 

- [Chen2013](@cite)
"""
struct LMARSNumericalFlux <: NumericalFluxFirstOrder end

"""
    RoeNumericalFluxMoist <: NumericalFluxFirstOrder

A moist implementation of the numerical flux based on the approximate Riemann solver of Roe

Requires a custom implementation for the balance law.
"""
struct RoeNumericalFluxMoist <: NumericalFluxFirstOrder
    " set to true for low Mach number correction"
    LM::Bool
    " set to true for Hartman Hyman correction"
    HH::Bool
    " set to true for LeVeque correction"
    LV::Bool
    " set to true for Positivity preserving LeVeque Correction"
    LVPP::Bool
end

RoeNumericalFluxMoist(;
    LM::Bool = false,
    HH::Bool = false,
    LV::Bool = false,
    LVPP::Bool = false,
) = RoeNumericalFluxMoist(LM, HH, LV, LVPP)

"""
    NumericalFluxSecondOrder

Any `N <: NumericalFluxSecondOrder` should define the a method for

    numerical_flux_second_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        t
    )

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive diffusive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_second_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        bctype,
        t
    )

"""
abstract type NumericalFluxSecondOrder end

function numerical_flux_second_order! end

function numerical_boundary_flux_second_order! end

"""
    CentralNumericalFluxSecondOrder <: NumericalFluxSecondOrder

The central numerical flux for diffusive terms

# Usage

    CentralNumericalFluxSecondOrder()

Requires a `flux_second_order!` for the balance law.
"""
struct CentralNumericalFluxSecondOrder <: NumericalFluxSecondOrder end

function numerical_flux_second_order!(
    ::CentralNumericalFluxSecondOrder,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector⁻::SVector,
    state_prognostic⁻::Vars{S},
    state_gradient_flux⁻::Vars{D},
    state_hyperdiffusive⁻::Vars{HD},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_gradient_flux⁺::Vars{D},
    state_hyperdiffusive⁺::Vars{HD},
    state_auxiliary⁺::Vars{A},
    t,
) where {S, D, HD, A}

    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_second_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    flux_second_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )
    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector⁻ / 2) #+ p' * (normal_vector⁻ / 2)
end

abstract type DivNumericalPenalty end
struct CentralNumericalFluxDivergence <: DivNumericalPenalty end

function numerical_flux_divergence!(
    ::CentralNumericalFluxDivergence,
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    parent(div_penalty) .=
        (parent(grad⁺) .+ parent(grad⁻))' * (normal_vector / 2)
end

function numerical_boundary_flux_divergence!(
    numerical_flux::CentralNumericalFluxDivergence,
    bctype,
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        grad⁺,
        normal_vector,
        grad⁻,
    )
    numerical_flux_divergence!(
        numerical_flux,
        balance_law,
        div_penalty,
        normal_vector,
        grad⁻,
        grad⁺,
    )
end

abstract type GradNumericalFlux end
struct CentralNumericalFluxHigherOrder <: GradNumericalFlux end

function numerical_flux_higher_order!(
    ::CentralNumericalFluxHigherOrder,
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    G = normal_vector .* (parent(lap⁺) .- parent(lap⁻))' ./ 2
    transform_post_gradient_laplacian!(
        balance_law,
        hyperdiff,
        Grad{GL}(G),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
    )
end

function numerical_boundary_flux_higher_order!(
    numerical_flux::CentralNumericalFluxHigherOrder,
    bctype,
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        lap⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        lap⁻,
        t,
    )
    numerical_flux_higher_order!(
        numerical_flux,
        balance_law,
        hyperdiff,
        normal_vector,
        lap⁻,
        state_prognostic⁻,
        state_auxiliary⁻,
        lap⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
end

numerical_boundary_flux_second_order!(
    numerical_flux::CentralNumericalFluxSecondOrder,
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_gradient_flux⁻::Vars{D},
    state_hyperdiffusive⁻::Vars{HD},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_gradient_flux⁺::Vars{D},
    state_hyperdiffusive⁺::Vars{HD},
    state_auxiliary⁺::Vars{A},
    t,
    state1⁻::Vars{S},
    diff1⁻::Vars{D},
    aux1⁻::Vars{A},
) where {S, D, HD, A} = normal_boundary_flux_second_order!(
    numerical_flux,
    bctype,
    balance_law,
    fluxᵀn,
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
)

function normal_boundary_flux_second_order!(
    numerical_flux,
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
) where {S}
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux, -zero(FT))
    boundary_flux_second_order!(
        numerical_flux,
        bctype,
        balance_law,
        Grad{S}(flux),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        diff1⁻,
        aux1⁻,
    )

    fluxᵀn .+= flux' * normal_vector
end

# This is the function that my be overloaded for flux-based BCs
function boundary_flux_second_order!(
    numerical_flux::NumericalFluxSecondOrder,
    bctype,
    balance_law,
    flux,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
)
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        diff1⁻,
        aux1⁻,
    )
    flux_second_order!(
        balance_law,
        flux,
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )
end

# Array wrappers for numerical Fluxes
function numerical_flux_first_order!(
    numerical_flux_first_order,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
)
    FT = eltype(flux)
    numerical_flux_first_order!(
        numerical_flux_first_order,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
        face_direction,
    )
end

function numerical_boundary_flux_first_order!(
    numerical_flux_first_order,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(flux)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        numerical_boundary_flux_first_order!(
            numerical_flux_first_order,
            bc,
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
            SVector(normal_vector),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            face_direction,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end
end

function numerical_flux_second_order!(
    numerical_flux_second_order,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(flux)
    numerical_flux_second_order!(
        numerical_flux_second_order,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁻),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁺),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
    )
end

function numerical_boundary_flux_second_order!(
    numerical_flux_second_order,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
    state_gradient_flux_bottom1::AbstractArray,
)
    FT = eltype(flux)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        numerical_boundary_flux_second_order!(
            numerical_flux_second_order,
            bc,
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
            SVector(normal_vector),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux⁻,
            ),
            Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                state_hyperdiffusive⁻,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux⁺,
            ),
            Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                state_hyperdiffusive⁺,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end
end

function numerical_boundary_flux_gradient!(
    numerical_flux_gradient,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_gradient⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_gradient⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(flux)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        # Computes G* incorporating boundary conditions
        numerical_boundary_flux_gradient!(
            numerical_flux_gradient,
            bc,
            balance_law,
            flux,
            SVector(normal_vector),
            Vars{vars_state(balance_law, Gradient(), FT)}(state_gradient⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Gradient(), FT)}(state_gradient⁺),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end
end

end
