### Reference state
using DocStringExtensions
using ..TemperatureProfiles
export ReferenceState, NoReferenceState, HydrostaticState

using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

"""
    ReferenceState

Hydrostatic reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

vars_state(m::ReferenceState, ::AbstractStateType, FT) = @vars()

atmos_init_aux!(
    ::ReferenceState,
    ::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
) = nothing

"""
    NoReferenceState <: ReferenceState

No reference state used
"""
struct NoReferenceState <: ReferenceState end

"""
    HydrostaticState{P,T} <: ReferenceState

A hydrostatic state specified by a virtual
temperature profile and relative humidity.

By default, this is a dry hydrostatic reference
state.
"""
struct HydrostaticState{P, FT} <: ReferenceState
    virtual_temperature_profile::P
    relative_humidity::FT
end
function HydrostaticState(
    virtual_temperature_profile::TemperatureProfile{FT},
) where {FT}
    return HydrostaticState{typeof(virtual_temperature_profile), FT}(
        virtual_temperature_profile,
        FT(0),
    )
end

vars_state(m::HydrostaticState, ::Auxiliary, FT) =
    @vars(ρ::FT, p::FT, T::FT, ρe::FT, ρq_tot::FT)

atmos_init_ref_state_pressure!(m, _...) = nothing
function atmos_init_ref_state_pressure!(
    m::HydrostaticState{P, F},
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) where {P, F}
    z = altitude(atmos, aux)
    _, p = m.virtual_temperature_profile(atmos.param_set, z)
    aux.ref_state.p = p
end

function atmos_init_aux!(
    m::HydrostaticState{P, F},
    atmos::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
) where {P, F}
    z = altitude(atmos, aux)
    T_virt, p = m.virtual_temperature_profile(atmos.param_set, z)
    FT = eltype(aux)
    _R_d::FT = R_d(atmos.param_set)
    k = vertical_unit_vector(atmos, aux)
    ∇Φ = ∇gravitational_potential(atmos, aux)

    # density computation from pressure ρ = -1/g*dpdz
    ρ = -k' * tmp.∇p / (k' * ∇Φ)
    aux.ref_state.ρ = ρ
    RH = m.relative_humidity
    phase_type = PhaseEquil
    (T, q_pt) = temperature_and_humidity_from_virtual_temperature(
        atmos.param_set,
        T_virt,
        ρ,
        RH,
        phase_type,
    )

    # Update temperature to be exactly consistent with
    # p, ρ, and q_pt
    T = air_temperature_from_ideal_gas_law(atmos.param_set, p, ρ, q_pt)
    q_tot = q_pt.tot
    ts = TemperatureSHumEquil(atmos.param_set, T, ρ, q_tot)

    aux.ref_state.ρq_tot = ρ * q_tot
    aux.ref_state.T = T
    e_kin = F(0)
    e_pot = gravitational_potential(atmos.orientation, aux)
    aux.ref_state.ρe = ρ * total_energy(e_kin, e_pot, ts)
end

using ..MPIStateArrays: vars
using ..DGMethods: init_ode_state
using ..DGMethods.NumericalFluxes:
    CentralNumericalFluxFirstOrder,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxGradient


"""
    PressureGradientModel

A mini balance law that is used to take the gradient of reference
pressure. The gradient is computed as ∇ ⋅(pI) and the calculation
uses the balance law interface to be numerically consistent with
the way this gradient is computed in the dynamics.
"""
struct PressureGradientModel <: BalanceLaw end
vars_state(::PressureGradientModel, ::Auxiliary, T) = @vars(p::T)
vars_state(::PressureGradientModel, ::Prognostic, T) = @vars(∇p::SVector{3, T})
vars_state(::PressureGradientModel, ::Gradient, T) = @vars()
vars_state(::PressureGradientModel, ::GradientFlux, T) = @vars()
function init_state_auxiliary!(
    m::PressureGradientModel,
    state_auxiliary::MPIStateArray,
    grid,
) end
function init_state_prognostic!(
    ::PressureGradientModel,
    state::Vars,
    aux::Vars,
    coord,
    t,
) end
function flux_first_order!(
    ::PressureGradientModel,
    flux::Grad,
    state::Vars,
    auxstate::Vars,
    t::Real,
    direction,
)
    flux.∇p -= auxstate.p * I
end
flux_second_order!(::PressureGradientModel, _...) = nothing
source!(::PressureGradientModel, _...) = nothing
boundary_state!(nf, ::PressureGradientModel, _...) = nothing

∇reference_pressure(::NoReferenceState, state_auxiliary, grid) = nothing
function ∇reference_pressure(::ReferenceState, state_auxiliary, grid)
    FT = eltype(state_auxiliary)
    ∇p = similar(state_auxiliary; vars = @vars(∇p::SVector{3, FT}))

    grad_model = PressureGradientModel()
    # Note that the choice of numerical fluxes doesn't matter
    # for taking the gradient of a continuous field
    grad_dg = DGModel(
        grad_model,
        grid,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    # initialize p
    ix_p = varsindex(vars(state_auxiliary), :ref_state, :p)
    grad_dg.state_auxiliary.data .= state_auxiliary.data[:, ix_p, :]

    # FIXME: this isn't used but needs to be passed in
    gradQ = init_ode_state(grad_dg, FT(0))

    grad_dg(∇p, gradQ, nothing, FT(0))
    return ∇p
end
