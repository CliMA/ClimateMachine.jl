struct TendencyIntegralModel{M} <: AbstractOceanModel
    ocean::M
    function TendencyIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end
vars_state(tm::TendencyIntegralModel, ::Prognostic, FT) =
    vars_state(tm.ocean, Prognostic(), FT)
vars_state(tm::TendencyIntegralModel, ::GradientFlux, FT) = @vars()

function vars_state(m::TendencyIntegralModel, ::Auxiliary, T)
    @vars begin
        ∫du::SVector{2, T}
    end
end

function vars_state(m::TendencyIntegralModel, ::UpwardIntegrals, T)
    @vars begin
        ∫du::SVector{2, T}
    end
end

@inline function integral_load_auxiliary_state!(
    m::TendencyIntegralModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∫du = A.∫du

    return nothing
end

@inline function integral_set_auxiliary_state!(
    m::TendencyIntegralModel,
    A::Vars,
    I::Vars,
)
    A.∫du = I.∫du

    return nothing
end

init_state_auxiliary!(tm::TendencyIntegralModel, A::Vars, _...) = nothing

function update_auxiliary_state!(
    dg::DGModel,
    tm::TendencyIntegralModel,
    dQ::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    A = dg.state_auxiliary

    # copy tendency vector to aux state for integration
    function f!(::TendencyIntegralModel, dQ, A, t)
        @inbounds begin
            A.∫du = @SVector [dQ.u[1], dQ.u[2]]
        end

        return nothing
    end
    update_auxiliary_state!(f!, dg, tm, dQ, t)

    # compute integral for Gᵁ
    indefinite_stack_integral!(dg, tm, dQ, A, t, elems) # bottom -> top

    return true
end

#-------------------------------------------------------------------------------
struct FlowIntegralModel{M} <: AbstractOceanModel
    ocean::M
    function FlowIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end
vars_state(fm::FlowIntegralModel, ::Prognostic, FT) =
    vars_state(fm.ocean, Prognostic(), FT)
vars_state(fm::FlowIntegralModel, ::GradientFlux, FT) = @vars()

function vars_state(m::FlowIntegralModel, ::Auxiliary, T)
    @vars begin
        ∫u::SVector{2, T}
    end
end

function vars_state(m::FlowIntegralModel, ::UpwardIntegrals, T)
    @vars begin
        ∫u::SVector{2, T}
    end
end

@inline function integral_load_auxiliary_state!(
    m::FlowIntegralModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∫u = Q.u

    return nothing
end

@inline function integral_set_auxiliary_state!(
    m::FlowIntegralModel,
    A::Vars,
    I::Vars,
)
    A.∫u = I.∫u

    return nothing
end

init_state_auxiliary!(fm::FlowIntegralModel, A::Vars, _...) = nothing

function update_auxiliary_state!(
    dg::DGModel,
    fm::FlowIntegralModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    A = dg.state_auxiliary

    # compute vertical integral of u
    indefinite_stack_integral!(dg, fm, Q, A, t, elems) # bottom -> top

    return true
end
