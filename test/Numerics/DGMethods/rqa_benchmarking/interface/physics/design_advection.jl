abstract type AbstractTerm{𝒯} end

struct NonlinearAdvection{𝒯} <: AbstractTerm{𝒯} end

@inline calc_component!(flux, ::Nothing, _...) = nothing
@inline calc_component!(flux, ::AbstractTerm, _...) = nothing

@inline function calc_component!(flux, ::NonlinearAdvection{(:ρ, :ρu, :ρθ)}, state, aux, physics)
    ρ  = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ
    
    u = ρu / ρ

    flux.ρ  += ρu
    flux.ρu += ρu ⊗ u
    flux.ρθ += ρθ * u

    nothing
end

@inline function calc_component!(flux, ::NonlinearAdvection{(:ρ, :ρu, :ρe)}, state, aux, physics)
    ρ   = state.ρ
    ρu  = state.ρu
    ρe  = state.ρe
    eos = physics.eos

    p = calc_pressure(eos, state, aux)
    u = ρu / ρ

    flux.ρ  += ρu
    flux.ρu += ρu ⊗ u
    flux.ρe += (ρe + p) * u

    nothing
end

@inline function calc_component!(flux, ::LinearAdvection{(:ρ, :ρu, :ρe)}, state, aux, physics)
    ρu  = state.ρu
    ρᵣ  = aux.ref_state.ρ
    pᵣ  = aux.ref_state.p
    ρeᵣ = aux.ref_state.ρe

    flux.ρ  += ρu
    flux.ρe += (ρeᵣ + pᵣ) * ρu / ρᵣ 

    nothing
end