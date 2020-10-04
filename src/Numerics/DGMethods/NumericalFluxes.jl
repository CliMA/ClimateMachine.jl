module NumericalFluxes

export 
    AbstractNumericalFlux,
    NumericalFluxGradient,
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    RusanovNumericalFlux,
    RoeNumericalFlux,
    HLLCNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxFirstOrder,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxHigherOrder


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
    boundary_condition,
    boundary_flux_second_order!
    
abstract type AbstractNumericalFlux end

# kernels call
#    numerical_boundary_flux_XX(nf::NF, bctag::Integer, bl::BalanceLaw, args...)
# if boundary_condition(nf, bl) isa BoundaryCondition, call
#    numerical_boundary_flux_XX(nf::NF, boundary_condition(nf, bl)::BoundaryCondition, bl::BalanceLaw, args...)
# elseif boundary_condition(nf, bl) isa Tuple, call
#    numerical_boundary_flux_XX(nf::NF, boundary_condition(nf, bl)[bctag]::BoundaryCondition, bl::BalanceLaw, args...)
#    (we unroll the lookup into an if/else sequence to dynamic dispatch)


for fn! in (
    :numerical_boundary_flux_gradient!,
    :numerical_boundary_flux_first_order!,
    :numerical_boundary_flux_second_order!,
    :numerical_boundary_flux_divergence!,
    :numerical_boundary_flux_higher_order!)

    _fn! = Symbol(:_,fn!)
    @eval begin
        function $fn!(
            nf::AbstractNumericalFlux,
            bctag::Integer,
            bl::BalanceLaw,
            args...)

            bc = boundary_condition(nf,bl)
            $_fn!(nf,bctag,bc,bl,args...)
        end

        # intermediate function used for dispatch
        # this shouldn't be extended by users

        # if not a Tuple
        function $_fn!(nf,bctag,bc::BoundaryCondition,bl,args...)
            $fn!(nf,bc,bl,args...)
        end
        # unroll the sequence of if statements to avoid dynamic dispatch
        @generated function $_fn!(nf,bctag,bcs::Tuple,bl,args...)
            N = fieldcount(bcs)
            fn! = $fn!
            quote
                Base.Cartesian.@nif(
                    $(N + 1),
                    i -> bctag == i, # conditionexpr
                    i -> $fn!(
                        nf,
                        bcs[i],
                        bl,
                        args...,
                    ), # expr
                    i -> error("Invalid boundary tag")
                ) # elseexpr
                return nothing
            end
        end
    end
end

# 1. gradient fluxes

"""
    NumericalFluxGradient

Any `P <: NumericalFluxGradient` should define methods for:

   numerical_flux_gradient!(gnf::P, bl::BalanceLaw, diffF, n⁻, Q⁻, Qstate_gradient_flux⁻, Qaux⁻, Q⁺,
                            Qstate_gradient_flux⁺, Qaux⁺, t)
   numerical_boundary_flux_gradient!(gnf::P, bctag, bl::BalanceLaw, local_state_gradient_flux, n⁻, local_transform⁻, local_state_prognostic⁻,
                                     local_state_auxiliary⁻, local_transform⁺, local_state_prognostic⁺, local_state_auxiliary⁺, t)

"""
abstract type NumericalFluxGradient <: AbstractNumericalFlux end

"""
    CentralNumericalFluxGradient <: NumericalFluxGradient

"""
struct CentralNumericalFluxGradient <: NumericalFluxGradient end

function numerical_flux_gradient!(
    ::CentralNumericalFluxGradient,
    bl::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::SVector,
    state_gradient⁻::Vars{T},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_gradient⁺::Vars{T},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {T, S, A}

    transform_gradient .=
        normal_vector .*
        (parent(state_gradient⁺) .+ parent(state_gradient⁻))' ./ 2
end

function numerical_boundary_flux_gradient!(
    nf::CentralNumericalFluxGradient,
    bc::BoundaryCondition,
    bl::BalanceLaw,
    transform_gradient,
    normal_vector,
    state_gradient⁻,
    state_prognostic⁻,
    state_auxiliary⁻,
    state_gradient⁺,
    state_prognostic⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    aux1⁻,
)
    boundary_state!(
        nf,
        bc,
        bl,
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
        bl,
        state_gradient⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
    transform_gradient .= normal_vector .* parent(state_gradient⁺)'
end

# 2. first order fluxes

"""
    NumericalFluxFirstOrder

Any `N <: NumericalFluxFirstOrder` should define the a method for

    numerical_flux_first_order!(numerical_flux::N, bl::BalanceLaw, flux, normal_vector⁻, Q⁻, Qaux⁻, Q⁺,
                                 Qaux⁺, t)

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_first_order!(numerical_flux::N, bctag, bl::BalanceLaw, flux, normal_vector⁻, Q⁻,
                                          Qaux⁻, Q⁺, Qaux⁺, t)

"""
abstract type NumericalFluxFirstOrder <: AbstractNumericalFlux end

function numerical_flux_first_order! end

"""
    numerical_boundary_flux_first_order!

Boundary flux for first order fluxes. To define, any of the following functions can be used:

- `numerical_boundary_flux_first_order!`, with signature:
  - `nf::NumericalFluxFirstOrder`
  - `bc::BC` where `BC<:BoundaryCondition`
  - `bl::BL` where `BL<:BalanceLaw`
  - `fluxᵀn::Vars`
  - `n` normal 
  - `state⁻`
  - `state⁺`
  - `t`
  - `direction`
  - `state1⁻`
"""
function numerical_boundary_flux_first_order!(
    nf::NumericalFluxFirstOrder,
    bc::BoundaryCondition,
    bl::BalanceLaw,
    fluxᵀn::Vars,
    normal_vector,
    state_prognostic⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_auxiliary⁺,
    t,
    direction,
    state1⁻,
    aux1⁻,
)

    boundary_state!(
        nf,
        bc,
        bl,
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
        nf,
        bl,
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
    bl::BalanceLaw,
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
        bl,
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
        bl,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        bl,
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
        bl,
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
    bl::BalanceLaw,
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
    num_state_prognostic = number_states(bl, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_first_order!(
        bl,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    flux_first_order!(
        bl,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

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


# 3. second order fluxes

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

    @book{toro2013riemann,
        title={Riemann solvers and numerical methods for fluid dynamics: a practical introduction},
        author={Toro, Eleuterio F},
        year={2013},
        publisher={Springer Science & Business Media}
    }
"""
struct HLLCNumericalFlux <: NumericalFluxFirstOrder end

"""
    NumericalFluxSecondOrder

Any `N <: NumericalFluxSecondOrder` should define the a method for

    numerical_flux_second_order!(numerical_flux::N, bl::BalanceLaw, flux, normal_vector⁻, Q⁻, Qstate_gradient_flux⁻, Qaux⁻, Q⁺,
                              Qstate_gradient_flux⁺, Qaux⁺, t)

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive diffusive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_second_order!(numerical_flux::N, bctag, bl::BalanceLaw, flux, normal_vector⁻, Q⁻, Qstate_gradient_flux⁻,
                                       Qaux⁻, Q⁺, Qstate_gradient_flux⁺, Qaux⁺, t)

"""
abstract type NumericalFluxSecondOrder <: AbstractNumericalFlux end

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
    bl::BalanceLaw,
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
    num_state_prognostic = number_states(bl, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_second_order!(
        bl,
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
        bl,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )

    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector⁻ / 2)
end


"""
    numerical_boundary_flux_second_order!(nf,bc,bl,fluxᵀn::Vars,...)
    numerical_boundary_flux_second_order!(nf,bc,bl,flux::Grad,  ...)

"""
function numerical_boundary_flux_second_order!(
    numerical_flux,
    bc::BoundaryCondition,
    bl::BalanceLaw,
    fluxᵀn::Vars,
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

    S = template(fluxᵀn)
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(bl, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux, -zero(FT))


    boundary_state!(
        numerical_flux,
        bc,
        bl,
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
        bl,
        Grad{S}(flux),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )

    # additional boundary flux
    boundary_flux_second_order!(numerical_flux,
        bc,
        bl,
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
        aux1⁻,)

    fluxᵀn .+= flux' * normal_vector
end


# 4. divergence flux

abstract type DivNumericalPenalty <: AbstractNumericalFlux end
struct CentralNumericalFluxDivergence <: DivNumericalPenalty end

function numerical_flux_divergence!(
    ::CentralNumericalFluxDivergence,
    bl::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    parent(div_penalty) .=
        (parent(grad⁺) .- parent(grad⁻))' * (normal_vector / 2)
end

function numerical_boundary_flux_divergence!(
    numerical_flux::CentralNumericalFluxDivergence,
    bc::BoundaryCondition,
    bl::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    boundary_state!(
        numerical_flux,
        bc,
        bl,
        grad⁺,
        normal_vector,
        grad⁻,
    )
    numerical_flux_divergence!(
        numerical_flux,
        bl,
        div_penalty,
        normal_vector,
        grad⁻,
        grad⁺,
    )
end

# 5. grad flux
abstract type GradNumericalFlux <: AbstractNumericalFlux end
struct CentralNumericalFluxHigherOrder <: GradNumericalFlux end

function numerical_flux_higher_order!(
    ::CentralNumericalFluxHigherOrder,
    bl::BalanceLaw,
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
    G = normal_vector .* (parent(lap⁻) .+ parent(lap⁺))' ./ 2
    transform_post_gradient_laplacian!(
        bl,
        hyperdiff,
        Grad{GL}(G),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
    )
end

function numerical_boundary_flux_higher_order!(
    numerical_flux::CentralNumericalFluxHigherOrder,
    bc::BoundaryCondition,
    bl::BalanceLaw,
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
        bc,
        bl,
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
        bl,
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

end # module