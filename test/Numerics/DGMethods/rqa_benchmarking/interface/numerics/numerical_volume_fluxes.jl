using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
import ClimateMachine.DGMethods.NumericalFluxes:
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    numerical_flux_first_order!,
    numerical_flux_second_order!,
    numerical_boundary_flux_second_order!
import ClimateMachine.BalanceLaws:
    wavespeed

struct CentralVolumeFlux <: NumericalFluxFirstOrder end
struct KGVolumeFlux <: NumericalFluxFirstOrder end
struct LinearKGVolumeFlux <: NumericalFluxFirstOrder end

function numerical_volume_fluctuation_flux_first_order!(
    ::NumericalFluxFirstOrder,
    model::DryAtmosModel,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)

    sources = model.physics.sources

    ntuple(Val(length(sources))) do s
        Base.@_inline_meta
        calc_fluctuation_component!(D, sources[s], state_1, state_2, aux_1, aux_2)
    end
    #=
    if fluctuation_gravity
        ρ_1, ρ_2 = state_1.ρ, state_2.ρ
        Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ

        α = ave(ρ_1, ρ_2) * 0.5

        D.ρu -= α * (Φ_1 - Φ_2) * I
    end
    =#
end

function numerical_volume_conservative_flux_first_order!(
    ::CentralVolumeFlux,
    m::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    F_1 = similar(F)
    flux_first_order!(m, F_1, state_1, aux_1, FT(0), EveryDirection())

    F_2 = similar(F)
    flux_first_order!(m, F_2, state_2, aux_2, FT(0), EveryDirection())

    parent(F) .= (parent(F_1) .+ parent(F_2)) ./ 2
end

function numerical_volume_conservative_flux_first_order!(
    ::KGVolumeFlux,
    model::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    eos = model.physics.eos
    parameters = model.physics.parameters

    ρ_1 = state_1.ρ
    ρu_1 = state_1.ρu
    ρe_1 = state_1.ρe
    ρq_1 = state_1.ρq
    u_1 = ρu_1 / ρ_1
    e_1 = ρe_1 / ρ_1
    q_1 = ρq_1 / ρ_1
    p_1 = calc_pressure(eos, state_1, aux_1, parameters)

    ρ_2 = state_2.ρ
    ρu_2 = state_2.ρu
    ρe_2 = state_2.ρe
    ρq_2 = state_2.ρq
    u_2 = ρu_2 / ρ_2
    e_2 = ρe_2 / ρ_2
    q_2 = ρq_2 / ρ_2
    p_2 = calc_pressure(eos, state_2, aux_2, parameters)

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    e_avg = ave(e_1, e_2)
    q_avg = ave(q_1, q_2)
    p_avg = ave(p_1, p_2)

    F.ρ  = ρ_avg * u_avg
    F.ρu = p_avg * I + ρ_avg * u_avg .* u_avg'
    F.ρe = ρ_avg * u_avg * e_avg + p_avg * u_avg
    F.ρq = ρ_avg * u_avg * q_avg
end

function numerical_volume_conservative_flux_first_order!(
    ::LinearKGVolumeFlux,
    model::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    eos = model.physics.eos
    parameters = model.physics.parameters
    
    ρu_1 = state_1.ρu
    ρuᵣ = ρu_1 * 0
    p_1 = calc_linear_pressure(eos, state_1, aux_1, parameters)

    # grab reference state
    ρᵣ_1 = aux_1.ref_state.ρ
    pᵣ_1 = aux_1.ref_state.p
    ρeᵣ_1 = aux_1.ref_state.ρe

    # only ρu fluctuates in the non-pressure terms
    u_1 = ρu_1 / ρᵣ_1 
    eᵣ_1 = ρeᵣ_1 / ρᵣ_1
    ρu_2 = state_2.ρu

    ρuᵣ = ρu_2 * 0
    p_2 = calc_linear_pressure(eos, state_2, aux_2, parameters)

    # grab reference state
    ρᵣ_2 = aux_2.ref_state.ρ
    pᵣ_2 = aux_2.ref_state.p
    ρeᵣ_2 = aux_2.ref_state.ρe

    # only ρu fluctuates in the non-pressure terms
    u_2 = ρu_2 / ρᵣ_2 
    eᵣ_2 = ρeᵣ_2 / ρᵣ_2

    # construct averages
    ρᵣ_avg = ave(ρᵣ_1, ρᵣ_2)
    eᵣ_avg = ave(eᵣ_1, eᵣ_2)
    pᵣ_avg = ave(pᵣ_1, pᵣ_2)

    u_avg = ave(u_1, u_2)
    p_avg = ave(p_1, p_2)

    F.ρ = ρᵣ_avg * u_avg 
    F.ρu = p_avg * I + ρuᵣ .* ρuᵣ' # the latter term is needed to determine size of I
    F.ρe = (ρᵣ_avg * eᵣ_avg + pᵣ_avg) * u_avg
end