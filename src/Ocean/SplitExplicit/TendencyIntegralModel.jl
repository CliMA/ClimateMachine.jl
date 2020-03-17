struct TendencyIntegralModel{M} <: AbstractOceanModel
    ocean::M
    function TendencyIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end
vars_state(tm::TendencyIntegralModel, FT) = vars_state(lm.ocean,FT)
vars_diffusive(tm::TendencyIntegralModel, FT) = @vars()
vars_integrals(tm::TendencyIntegralModel, FT) = @vars()

function vars_aux(m::TendencyIntegralModel, T)
    @vars begin
        ∫du::SVector{2, T}
    end
end

init_aux!(tm::TendencyIntegralModel, A::Vars, geom::LocalGeometry) = nothing

function update_aux!(dg::DGModel, tm::TendencyIntegralModel, dQ::MPIStateArray, t::Real)
    A  = dg.auxstate

    # copy tendency vector to aux state for integration
    function f!(::TendencyIntegralModel, dQ, A, t)
        @inbounds begin
            A.∫du = @SVector [dQ.u[1], dQ.u[2]]
        end

        return nothing
    end
    nodal_update_aux!(f!, dg, tm, Q, t)

    # compute integral for Gᵁ
    indefinite_stack_integral!(dg, tm, Q, A, t) # bottom -> top

    return true
end
