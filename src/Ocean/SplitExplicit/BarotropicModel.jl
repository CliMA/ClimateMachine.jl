abstract type BarotropicSourceTerm end
struct SurfaceStress <: BarotropicSourceTerm end
struct IntegratedTendency <: BarotropicSourceTerm end

struct BarotropicModel{M, S} <: AbstractOceanModel
    baroclinic::M
    source::S
    function BarotropicModel(baroclinic::M, source::S) where {M, S}
        return new{M, S}(baroclinic, source)
    end
end

function vars_state(m::BarotropicModel, T)
    @vars begin
        U::SVector{2, T}
        η::T
    end
end

function init_state!(m::BarotropicModel, Q::Vars, A::Vars, coords, t)
    Q.U = @SVector [-0, -0]
    Q.η = -0
    return nothing
end

function vars_aux(m::BarotropicModel, T)
    @vars begin
        y::T              # y-coordinate of grid
        Gᵁ::SVector{2, T} # integral of baroclinic tendency
        Ū::SVector{2, T}  # running averge of U
        η̄::T              # running averge of η
        Δu::SVector{2, T} # reconciliation adjustment to u, Δu = 1/H * (Ū - ∫u)
        U°::SVector{2, T} # saved value of U to restart with during next stage
        η°::SVector{2, T} # saved value of η to restart with during next stage
    end
end

function init_aux!(m::BarotropicModel, A::Vars, geom::LocalGeometry)
    return ocean_init_aux!(m, m.baroclinic.problem, A, geom)
end

function vars_gradient(m::BarotropicModel, T)
    @vars begin
        U::SVector{2, T}
    end
end

@inline function gradvariables!(m::BarotropicModel, G::Vars, Q::Vars, A, t)
    G.U = Q.U

    return nothing
end

function vars_diffusive(m::BarotropicModel, T)
    @vars begin
        ν∇U::SMatrix{3, 2, T, 6}
    end
end

@inline function diffusive!(
    m::BarotropicModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    ν = viscosity_tensor(m)
    D.ν∇U = ν * G.U

    return nothing
end

@inline function viscosity_tensor(bm::BarotropicModel)
    m = bm.baroclinic
    return Diagonal(@SVector [m.νʰ, m.νʰ, 0])
end

vars_integrals(m::BarotropicModel, T) = @vars()
vars_reverse_integrals(m::BarotropicModel, T) = @vars()

@inline function flux_nondiffusive!(
    m::BarotropicModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
)
    @inbounds begin
        U = @SVector [Q.U[1], Q.U[2], 0]
        η = Q.η
        H = m.baroclinic.problem.H
        Iʰ = @SMatrix [
            1 0
            0 1
            0 0
        ]

        F.η += U
        F.U += grav(m.baroclinic.param_set) * H * η * Iʰ
    end
end

@inline function flux_diffusive!(
    m::BarotropicModel,
    F::Grad,
    Q::Vars,
    D::Vars,
    HD::Vars,
    A::Vars,
    t::Real,
)
    # numerical diffusivity for stability
    F.U -= D.ν∇U

    return nothing
end

@inline function source!(
    m::BarotropicModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
    direction,
)
    # f × u
    U = Q.U
    f = coriolis_force(m.baroclinic, A.y)
    S.U -= @SVector [-f * U[2], f * U[1]]

    source_term(m, m.source, S, Q, A, t)

    return nothing
end

@inline function source_term(
    ::BarotropicModel,
    ::IntegratedTendency,
    S,
    Q,
    A,
    t,
)
    S.U += A.Gᵁ

    return nothing
end

@inline function source_term(bm::BarotropicModel, ::SurfaceStress, S, Q, A, t)
    m = bm.baroclinic
    τ = velocity_flux(m.problem, A.y, m.ρₒ)
    S.U += @SVector [τ, 0]

    return nothing
end

@inline wavespeed(m::BarotropicModel, n⁻, _...) =
    abs(SVector(m.baroclinic.cʰ, m.baroclinic.cʰ, m.baroclinic.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
function update_penalty!(
    ::Rusanov,
    ::BarotropicModel,
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
