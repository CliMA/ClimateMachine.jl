using CLIMAParameters.Planet: R_d, cv_d, T_0, e_int_v0, e_int_i0
import ..DGmethods.NumericalFluxes:
    numerical_flux_first_order!, UpwindNumericalFlux

"""
    linearized_air_pressure(ρ, ρe_tot, ρe_pot, ρq_tot=0, ρq_liq=0, ρq_ice=0)

The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:

 - `ρ` (moist-)air density
 - `ρe_tot` total energy density
 - `ρe_pot` potential energy density
and, optionally,
 - `ρq_tot` total water density
 - `ρq_liq` liquid water density
 - `ρq_ice` ice density
"""
function linearized_air_pressure(
    param_set::AbstractParameterSet,
    ρ::FT,
    ρe_tot::FT,
    ρe_pot::FT,
    ρq_tot::FT = FT(0),
    ρq_liq::FT = FT(0),
    ρq_ice::FT = FT(0),
) where {FT <: Real, PS}
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return ρ * _R_d * _T_0 +
           _R_d / _cv_d * (
        ρe_tot - ρe_pot - (ρq_tot - ρq_liq) * _e_int_v0 +
        ρq_ice * (_e_int_i0 + _e_int_v0)
    )
end

@inline function linearized_pressure(
    moisture::MoistureModel,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    linearized_pressure(
        moisture,
        param_set,
        orientation,
        state.ρ,
        state.ρe,
        aux,
    )
end

# These two versions are used to pick out terms for the upwind flux below
@inline function linearized_pressure(
    ::DryModel,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    ρ::T,
    ρe::T,
    aux::Vars,
) where {T}
    ρe_pot = ρ * gravitational_potential(orientation, aux)
    return linearized_air_pressure(param_set, ρ, ρe, ρe_pot)
end
@inline function linearized_pressure(
    ::EquilMoist,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    ρ::T,
    ρe::T,
    aux::Vars,
) where {T}
    ρe_pot = ρ * gravitational_potential(orientation, aux)
    linearized_air_pressure(param_set, ρ, ρe, ρe_pot, aux.ref_state.ρq_tot)
end

abstract type AtmosLinearModel <: BalanceLaw end

# FIXME: Add moisture back in
function vars_state_conservative(m::AtmosLinearModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end
vars_state_gradient(lm::AtmosLinearModel, FT) = @vars()
vars_state_gradient_flux(lm::AtmosLinearModel, FT) = @vars()
vars_state_auxiliary(lm::AtmosLinearModel, FT) =
    vars_state_auxiliary(lm.atmos, FT)
vars_integrals(lm::AtmosLinearModel, FT) = @vars()
vars_reverse_integrals(lm::AtmosLinearModel, FT) = @vars()


function update_auxiliary_state!(
    dg::DGModel,
    lm::AtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
end
function flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::AtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::AtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing
flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
    ref = aux.ref_state
    return soundspeed_air(lm.atmos.param_set, ref.T)
end

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    atmoslm::AtmosLinearModel,
    args...,
)
    atmos_boundary_state!(nf, AtmosBC(), atmoslm, args...)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    atmoslm::AtmosLinearModel,
    args...,
)
    nothing
end
init_state_auxiliary!(lm::AtmosLinearModel, aux::Vars, geom::LocalGeometry) =
    nothing
init_state_conservative!(
    lm::AtmosLinearModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("AtmosAcousticLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end

function flux_first_order!(
    lm::AtmosAcousticLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ - e_pot) * state.ρu
    nothing
end
source!(::AtmosAcousticLinearModel, _...) = nothing

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticGravityLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("AtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end
function flux_first_order!(
    lm::AtmosAcousticGravityLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu
    nothing
end
function source!(
    lm::AtmosAcousticGravityLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    if direction isa VerticalDirection || direction isa EveryDirection
        ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
        source.ρu -= state.ρ * ∇Φ
    end
    nothing
end


# Note: this assumes that the reference state is continuous across element
# boundaries

# FIXME: Include moisture
function numerical_flux_first_order!(
    ::UpwindNumericalFlux,
    balance_law::Union{
        AtmosAcousticLinearModel,
        AtmosAcousticGravityLinearModel,
    },
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {S, A}

    FT = eltype(state_conservative⁻)

    # Coefficients for the flux matrix
    param_set = balance_law.atmos.param_set

    # Query pressure contribution for each component
    α = linearized_pressure(
        balance_law.atmos.moisture,
        balance_law.atmos.param_set,
        balance_law.atmos.orientation,
        one(FT),
        -zero(FT),
        state_auxiliary⁻,
    )
    β = linearized_pressure(
        balance_law.atmos.moisture,
        balance_law.atmos.param_set,
        balance_law.atmos.orientation,
        -zero(FT),
        one(FT),
        state_auxiliary⁻,
    )

    e_pot =
        gravitational_potential(balance_law.atmos.orientation, state_auxiliary⁻)
    ref = state_auxiliary⁻.ref_state
    γ = balance_law isa AtmosAcousticLinearModel ?
        (ref.ρe + ref.p) / ref.ρ - e_pot : (ref.ρe + ref.p) / ref.ρ

    # wave speed
    λ = sqrt(β * γ + α)

    #=
    #! format: off
    # Matrix for the flux is:
    n = normal_vector
    Tn = [1 0    0     0   0;
          0 n[1] n[2] n[3] 0;
          0 0    0    0    1]

    B = [0 1 0;
         α 0 β;
         0 γ 0]

    An = Tn' * B * Tn

    # The upwinding is based on the following eigenvalue decomposition of B
    V = [-1  β 1;
          λ  0 λ;
         -γ -α γ]

    W = [-α / (2α + 2γ * β)   1 / 2λ  -β / (2α + 2γ * β);
         2γ / (2α + 2γ * β)   0       -2 / (2α + 2γ * β);
          α / (2α + 2γ * β)   1 / 2λ   β / (2α + 2γ * β)]

    @assert B ≈ V * Diagonal([-λ, 0, λ]) * W
    #! format: on
    =#

    # rotated state vector based on outward normal
    ρ⁻ = state_conservative⁻.ρ
    ρu⁻ = dot(normal_vector, state_conservative⁻.ρu)
    ρe⁻ = state_conservative⁻.ρe
    ρ⁺ = state_conservative⁺.ρ
    ρu⁺ = dot(normal_vector, state_conservative⁺.ρu')
    ρe⁺ = state_conservative⁺.ρe

    # Left eigenvector entries
    δ1 = -α / (2α + 2γ * β)
    δ2 = 1 / 2λ
    δ3 = -β / (2α + 2γ * β)

    # incoming wave
    ω⁺ = -λ * (δ1 * ρ⁺ + δ2 * ρu⁺ + δ3 * ρe⁺)

    # outgoing wave
    ω⁻ = +λ * (-δ1 * ρ⁻ + δ2 * ρu⁻ - δ3 * ρe⁻)

    # compute the upwind flux using the right eigenvectors and rotate back based
    # on the outward normal
    fluxᵀn.ρ = ω⁻ - ω⁺
    fluxᵀn.ρu = λ * (ω⁻ + ω⁺) * normal_vector
    fluxᵀn.ρe = γ * (ω⁻ - ω⁺)
end
