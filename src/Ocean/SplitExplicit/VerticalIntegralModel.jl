import ...BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    update_auxiliary_state!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!

struct VerticalIntegralModel{M} <: BalanceLaw
    ocean::M
    function VerticalIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

vars_state(tm::VerticalIntegralModel, st::Prognostic, FT) =
    vars_state(tm.ocean, st, FT)

function vars_state(m::VerticalIntegralModel, ::Auxiliary, T)
    @vars begin
        ∫x::SVector{2, T}
    end
end

init_state_auxiliary!(tm::VerticalIntegralModel, A::Vars, geom::LocalGeometry) =
    nothing

function vars_state(m::VerticalIntegralModel, ::UpwardIntegrals, T)
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
