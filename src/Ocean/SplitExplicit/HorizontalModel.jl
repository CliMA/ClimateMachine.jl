struct HorizontalModel{M} <: AbstractOceanModel
    ocean::M
    function HorizontalModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

vars_state(m::HorizontalModel, T) = vars_state(m.ocean, T)
vars_gradient(m::HorizontalModel, T) = vars_gradient(m.ocean, T)
vars_diffusive(m::HorizontalModel, T) = vars_diffusive(m.ocean, T)
vars_aux(m::HorizontalModel, T) = vars_aux(m.ocean, T)

@inline function gradvariables!(m::HorizontalModel, G::Vars, Q::Vars, A, t)
    G.u = Q.u

    return nothing
end

@inline function diffusive!(
    m::HorizontalModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    ν = viscosity_tensor(m.ocean)
    D.ν∇u = ν * G.u

    return nothing
end

@inline function flux_nondiffusive!(
    m::HorizontalModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
)
    @inbounds begin
        η = Q.η
        Iʰ = @SMatrix [
            1 -0
            -0 1
            -0 -0
        ]

        # ∇h • (g η)
        F.u += grav(m.ocean.param_set) * η * Iʰ
    end

    return nothing
end

@inline function flux_diffusive!(
    m::HorizontalModel,
    F::Grad,
    Q::Vars,
    D::Vars,
    HD::Vars,
    A::Vars,
    t::Real,
)
    # only do horizontal components
    F.u -= @SVector([1, 0, 0]) * D.ν∇u[1, :]'
    F.u -= @SVector([0, 1, 0]) * D.ν∇u[2, :]'

    return nothing
end

@inline function source!(
    m::HorizontalModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
    direction,
)
    # f × u
    u = Q.u
    f = coriolis_force(m.ocean, A.y)
    S.u -= @SVector [-f * u[2], f * u[1]]

    return nothing
end

@inline update_penalty!(nf::Rusanov, hm::HorizontalModel, args...) =
    update_penalty!(nf, hm.ocean, args...)

@inline wavespeed(hm::HorizontalModel, n⁻, args...) =
    wavespeed(hm.ocean, n⁻, args...)
