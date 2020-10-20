struct BarotropicModel{M} <: AbstractOceanModel
    baroclinic::M
    function BarotropicModel(baroclinic::M) where {M}
        return new{M}(baroclinic)
    end
end

function vars_state(m::BarotropicModel, ::Prognostic, T)
    @vars begin
        U::SVector{2, T}
        η::T
    end
end

function init_state_prognostic!(
    m::BarotropicModel,
    Q::Vars,
    A::Vars,
    localgeo,
    t,
)
    return ocean_init_state!(m, m.baroclinic.problem, Q, A, localgeo, t)
end

function vars_state(m::BarotropicModel, ::Auxiliary, T)
    @vars begin
        Gᵁ::SVector{2, T}  # integral of baroclinic tendency
        U_c::SVector{2, T} # cumulate U value over fast time-steps
        η_c::T             # cumulate η value over fast time-steps
        U_s::SVector{2, T} # starting U field value
        η_s::T             # starting η field value
        Δu::SVector{2, T}  # reconciliation adjustment to u, Δu = 1/H * (U_averaged - ∫u)
        η_diag::T          # η from baroclinic model (for diagnostic)
        Δη::T              # diagnostic difference: η_barotropic - η_baroclinic
        y::T               # y-coordinate of grid
    end
end

function init_state_auxiliary!(
    m::BarotropicModel,
    state_aux::MPIStateArray,
    grid,
    direction,
)
    init_state_auxiliary!(
        m,
        (m, A, tmp, geom) -> ocean_init_aux!(m, m.baroclinic.problem, A, geom),
        state_aux,
        grid,
        direction,
    )
end

function vars_state(m::BarotropicModel, ::Gradient, T)
    @vars begin
        U::SVector{2, T}
    end
end

@inline function compute_gradient_argument!(
    m::BarotropicModel,
    G::Vars,
    Q::Vars,
    A,
    t,
)
    G.U = Q.U
    return nothing
end

function vars_state(m::BarotropicModel, ::GradientFlux, T)
    @vars begin
        ν∇U::SMatrix{3, 2, T, 6}
    end
end

@inline function compute_gradient_flux!(
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

vars_state(m::BarotropicModel, ::UpwardIntegrals, T) = @vars()
vars_state(m::BarotropicModel, ::DownwardIntegrals, T) = @vars()

@inline function flux_first_order!(
    m::BarotropicModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
    direction,
)
    @inbounds begin
        U = @SVector [Q.U[1], Q.U[2], 0]
        η = Q.η
        H = m.baroclinic.problem.H
        g = grav(m.baroclinic.param_set)
        Iʰ = @SMatrix [
            1 0
            0 1
            0 0
        ]

        F.η += U
        F.U += g * H * η * Iʰ
    end
end

@inline function flux_second_order!(
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
    @inbounds begin
        U = Q.U

        # f × u
        f = coriolis_force(m.baroclinic, A.y)
        S.U -= @SVector [-f * U[2], f * U[1]]

        # vertically integrated baroclinic model tendency
        S.U += A.Gᵁ
    end
end

@inline wavespeed(m::BarotropicModel, n⁻, _...) =
    abs(SVector(m.baroclinic.cʰ, m.baroclinic.cʰ, m.baroclinic.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
#   ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
function update_penalty!(
    ::RusanovNumericalFlux,
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

@inline function boundary_state!(nf, bm::BarotropicModel, args...)
    # hack for handling multiple boundaries for now
    # will fix with a future update
    boundary_conditions = (
        bm.baroclinic.problem.boundary_conditions[1],
        bm.baroclinic.problem.boundary_conditions[1],
        bm.baroclinic.problem.boundary_conditions[1],
    )
    return ocean_boundary_state!(nf, boundary_conditions, bm, args...)
end
