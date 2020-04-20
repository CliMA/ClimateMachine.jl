struct OceanModel{PS, P, T} <: AbstractOceanModel
    param_set::PS
    problem::P
    ρₒ::T
    cʰ::T
    cᶻ::T
    αᵀ::T
    νʰ::T
    νᶻ::T
    κʰ::T
    κᶻ::T
    fₒ::T
    β::T
    function OceanModel{FT}(
        param_set::PS,
        problem::P;
        ρₒ = FT(1000),  # kg / m^3
        cʰ = FT(0),     # m/s
        cᶻ = FT(0),     # m/s
        αᵀ = FT(2e-4),  # (m/s)^2 / K
        νʰ = FT(5e3),   # m^2 / s
        νᶻ = FT(5e-3),  # m^2 / s
        κʰ = FT(1e3),   # m^2 / s
        κᶻ = FT(1e-4),  # m^2 / s
        fₒ = FT(1e-4),  # Hz
        β = FT(1e-11), # Hz / m
    ) where {FT <: AbstractFloat, PS, P}
        return new{PS, P, FT}(
            param_set,
            problem,
            ρₒ,
            cʰ,
            cᶻ,
            αᵀ,
            νʰ,
            νᶻ,
            κʰ,
            κᶻ,
            fₒ,
            β,
        )
    end
end

function calculate_dt(grid, model::OceanModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())
    minΔz = min_node_distance(grid, VerticalDirection())

    CFL_gravity = minΔx / model.cʰ
    CFL_diffusive = minΔz^2 / (1000 * model.κᶻ)
    CFL_viscous = minΔz^2 / model.νᶻ

    dt = 1 // 2 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

"""
    OceanDGModel()

helper function to add required filtering
not used in the Driver+Config setup
"""
function OceanDGModel(
    bl::OceanModel,
    grid,
    numfluxnondiff,
    numfluxdiff,
    gradnumflux;
    kwargs...,
)
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)

    tendency_dg = DGModel(
        TendencyIntegralModel(bl),
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
    )

    modeldata = (
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        tendency_dg = tendency_dg,
    )

    return DGModel(
        bl,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux;
        kwargs...,
        modeldata = modeldata,
    )
end

function vars_state(m::OceanModel, T)
    @vars begin
        u::SVector{2, T}
        η::T
        θ::T
    end
end

function init_state!(m::OceanModel, Q::Vars, A::Vars, coords, t)
    return ocean_init_state!(m.problem, Q, A, coords, t)
end

function vars_aux(m::OceanModel, T)
    @vars begin
        y::T              # y-coordinate of the box
        w::T              # vertical velocity
        pkin::T           # ∫(-αᵀ θ)
        wz0::T            # w at z=0
        ∫u::SVector{2, T} # barotropic velocity
        u°::SVector{2, T} # velocity after tendency before reconciliation
        η_barotropic::T   # η from barotropic model
        Δη::T             # difference between explicit and barotropic η
    end
end

function init_aux!(m::OceanModel, A::Vars, geom::LocalGeometry)
    return ocean_init_aux!(m, m.problem, A, geom)
end

function vars_gradient(m::OceanModel, T)
    @vars begin
        u::SVector{2, T}
        θ::T
    end
end

@inline function gradvariables!(m::OceanModel, G::Vars, Q::Vars, A, t)
    G.u = Q.u
    G.θ = Q.θ

    return nothing
end

function vars_diffusive(m::OceanModel, T)
    @vars begin
        ν∇u::SMatrix{3, 2, T, 6}
        κ∇θ::SVector{3, T}
    end
end

@inline function diffusive!(
    m::OceanModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    ν = viscosity_tensor(m)
    D.ν∇u = ν * G.u

    κ = diffusivity_tensor(m, G.θ[3])
    D.κ∇θ = κ * G.θ

    return nothing
end

@inline viscosity_tensor(m::OceanModel) = Diagonal(@SVector [m.νʰ, m.νʰ, m.νᶻ])

@inline function diffusivity_tensor(m::OceanModel, ∂θ∂z)
    ∂θ∂z < 0 ? κ = (@SVector [m.κʰ, m.κʰ, 1000 * m.κᶻ]) : κ =
        (@SVector [m.κʰ, m.κʰ, m.κᶻ])

    return Diagonal(κ)
end

"""
    vars_integral(::OceanModel)

location to store integrands for bottom up integrals
∇hu = the horizontal divegence of u, e.g. dw/dz
"""
function vars_integrals(m::OceanModel, T)
    @vars begin
        ∇hu::T
        αᵀθ::T
        ∫u::SVector{2, T}
    end
end

"""
    integral_load_aux!(::OceanModel)

copy w to var_integral
"""
@inline function integral_load_aux!(m::OceanModel, I::Vars, Q::Vars, A::Vars)
    I.∇hu = A.w # borrow the w value from A...
    I.αᵀθ = -m.αᵀ * Q.θ # integral will be reversed below
    I.∫u = Q.u

    return nothing
end

"""
    integral_set_aux!(::OceanModel)

copy integral results back out to aux
"""
@inline function integral_set_aux!(m::OceanModel, A::Vars, I::Vars)
    A.w = I.∇hu
    A.pkin = I.αᵀθ
    A.∫u = I.∫u

    return nothing
end

"""
    vars_reverse_integral(::OceanModel)

location to store integrands for top down integrals
αᵀθ = density perturbation
"""
function vars_reverse_integrals(m::OceanModel, T)
    @vars begin
        αᵀθ::T
    end
end

"""
    reverse_integral_load_aux!(::OceanModel)

copy αᵀθ to var_reverse_integral
"""
@inline function reverse_integral_load_aux!(
    m::OceanModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.αᵀθ = A.pkin

    return nothing
end

"""
    reverse_integral_set_aux!(::OceanModel)

copy reverse integral results back out to aux
"""
@inline function reverse_integral_set_aux!(m::OceanModel, A::Vars, I::Vars)
    A.pkin = I.αᵀθ

    return nothing
end

@inline function flux_nondiffusive!(
    m::OceanModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
)
    @inbounds begin
        u = Q.u # Horizontal components of velocity
        θ = Q.θ
        w = A.w   # vertical velocity
        pkin = A.pkin
        v = @SVector [u[1], u[2], w]
        Iʰ = @SMatrix [
            1 -0
            -0 1
            -0 -0
        ]

        # ∇ • (u θ)
        F.θ += v * θ

        # ∇h • (- ∫(αᵀ θ))
        F.u += grav(m.param_set) * pkin * Iʰ

        # ∇h • (v ⊗ u)
        # F.u += v * u'
    end

    return nothing
end

@inline function flux_diffusive!(
    m::OceanModel,
    F::Grad,
    Q::Vars,
    D::Vars,
    HD::Vars,
    A::Vars,
    t::Real,
)
    # horizontal viscosity done in horizontal model
    F.u -= @SVector([0, 0, 1]) * D.ν∇u[3, :]'

    F.θ -= D.κ∇θ

    return nothing
end

@inline function source!(
    m::OceanModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
    direction,
)
    S.η += A.wz0

    return nothing
end

@inline coriolis_force(m::OceanModel, y) = m.fₒ + m.β * y

function update_aux!(
    dg::DGModel,
    m::OceanModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    MD = dg.modeldata

    # `update_aux!` gets called twice, once for the real elements and once for
    # the ghost elements.  Only apply the filters to the real elems.
    if elems == dg.grid.topology.realelems
        # required to ensure that after integration velocity field is divergence free
        vert_filter = MD.vert_filter
        index_u = tuple(collect(varsindex(vars_state(m, FT), :u))...)
        apply!(Q, index_u, dg.grid, vert_filter, VerticalDirection())

        exp_filter = MD.exp_filter
        index_θ = tuple(collect(varsindex(vars_state(m, FT), :θ))...)
        apply!(Q, index_θ, dg.grid, exp_filter, VerticalDirection())
    end

    return true
end

function update_aux_diffusive!(
    dg::DGModel,
    m::OceanModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    A = dg.auxstate


    # store ∇ʰu as integrand for w
    # update vertical diffusivity for convective adjustment
    function f!(::OceanModel, Q, A, D, t)
        @inbounds begin
            ν = viscosity_tensor(m)
            ∇u = ν \ D.ν∇u
            A.w = -(∇u[1, 1] + ∇u[2, 2])
        end

        return nothing
    end
    nodal_update_aux!(f!, dg, m, Q, t, elems; diffusive = true)

    # compute integrals for w and pkin
    indefinite_stack_integral!(dg, m, Q, A, t, elems) # bottom -> top
    reverse_indefinite_stack_integral!(dg, m, Q, A, t, elems) # top -> bottom

    # copy down wz0
    index_w = varsindex(vars_aux(m, FT), :w)
    index_wz0 = varsindex(vars_aux(m, FT), :wz0)
    copy_stack_field_down!(dg, m, A, index_w, index_wz0, elems)

    return true
end

@inline wavespeed(m::OceanModel, n⁻, _...) =
    abs(SVector(m.cʰ, m.cʰ, m.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
@inline function update_penalty!(
    ::Rusanov,
    ::OceanModel,
    n⁻,
    λ,
    ΔQ::Vars,
    Q⁻,
    A⁻,
    Q⁺,
    A⁺,
    t,
)
    ΔQ.η = -0

    return nothing
end
