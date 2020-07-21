import ...BalanceLaws:
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    vars_state_auxiliary,
    init_state_conservative!,
    init_state_auxiliary!,
    update_auxiliary_state!,
    vars_integrals,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!

struct VerticalIntegralModel{M} <: BalanceLaw
    ocean::M
    function VerticalIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

vars_state_gradient(tm::VerticalIntegralModel, FT) = @vars()
vars_state_gradient_flux(tm::VerticalIntegralModel, FT) = @vars()

vars_state_conservative(tm::VerticalIntegralModel, FT) =
    vars_state_conservative(tm.ocean, FT)

function vars_state_auxiliary(m::VerticalIntegralModel, T)
    @vars begin
        ∫x::SVector{2, T}
    end
end

init_state_auxiliary!(tm::VerticalIntegralModel, A::Vars, geom::LocalGeometry) =
    nothing

function vars_integrals(m::VerticalIntegralModel, T)
    @vars begin
        ∫x::SVector{2, T}
    end
end

@inline function integral_load_auxiliary_state!(
    m::VerticalIntegralModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∫x = A.∫x

    return nothing
end

@inline function integral_set_auxiliary_state!(
    m::VerticalIntegralModel,
    A::Vars,
    I::Vars,
)
    A.∫x = I.∫x

    return nothing
end

function update_auxiliary_state!(
    dg::DGModel,
    tm::VerticalIntegralModel,
    x::MPIStateArray,
    t::Real,
)
    A = dg.state_auxiliary

    # copy tendency vector to aux state for integration
    function f!(::VerticalIntegralModel, x, A, t)
        @inbounds begin
            A.∫x = @SVector [x.u[1], x.u[2]]
        end

        return nothing
    end
    nodal_update_auxiliary_state!(f!, dg, tm, x, t)

    # compute integral for Gᵁ
    indefinite_stack_integral!(dg, tm, x, A, t) # bottom -> top

    return true
end
