abstract type AbstractAdvection <: AbstractPhysicsComponent end

struct NonLinearAdvection <: AbstractAdvection end
struct LinearAdvection <: AbstractAdvection end

@inline calc_advective_flux!(flux, ::Nothing, _...) = nothing

@inline function calc_advective_flux!(flux, ::NonLinearAdvection, state, aux, t)
    ρ  = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    #flux.ρ += state.ρu
    flux.ρu += ρu ⊗ ρu / ρ
    flux.ρθ += ρu * ρθ / ρ

    return nothing
end

@inline function calc_flux!(flux, ::LinearAdvection, state, aux, t)
    ρu  = state.ρu
    ρᵣ  = aux.ref_state.ρ
    pᵣ  = aux.ref_state.p
    ρeᵣ = aux.ref_state.ρe

    flux.ρ += ρu
    flux.ρe += (ρeᵣ + pᵣ) / ρᵣ * ρu

    return nothing
end