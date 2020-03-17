struct HorizontalModel{M} <: AbstractOceanModel
    ocean::M
    function HorizontalModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

vars_state(hm::HorizontalModel, T) = vars_state(hm.ocean, T)
vars_gradient(hm::HorizontalModel, T) = vars_gradient(hm.ocean, T)
vars_diffusive(hm::HorizontalModel, T) = vars_diffusive(hm.ocean, T)
vars_aux(hm::HorizontalModel, T) = vars_aux(hm.ocean, T)

@inline function flux_nondiffusive!(m::HorizontalModel, F::Grad, Q::Vars,
                                    A::Vars, t::Real)
    @inbounds begin
        η = Q.η
        Ih = @SMatrix [ 1 -0;
                       -0  1;
                       -0 -0]
        
        # ∇h • (g η)
        F.u += grav * η * Ih
    end
    
    return nothing
end

@inline function flux_diffusive!(m::HorizontalModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
    F.u -= Diagonal([A.ν[1], A.ν[2], -0]) * D.∇u

    return nothing
end

function wavespeed(hm::HorizontalModel, n⁻, _...)
  C = abs(SVector(hm.ocean.cʰ, hm.ocean.cʰ, hm.ocean.cᶻ)' * n⁻)
  return C
end
