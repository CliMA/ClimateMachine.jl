struct NonlinearAdvection{𝒯} <: AbstractTerm end
struct LinearAdvection{𝒯} <: AbstractTerm end
struct VeryLinearAdvection{𝒯} <: AbstractTerm end

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
    ρq  = state.ρq
    eos = physics.eos
    parameters = physics.parameters

    p = calc_pressure(eos, state, aux, parameters)
    u = ρu / ρ

    flux.ρ  += ρu
    flux.ρu += ρu ⊗ u
    flux.ρe += (ρe + p) * u
    flux.ρq += ρq * u

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

@inline function calc_component!(flux, ::VeryLinearAdvection{(:ρ, :ρu, :ρe)}, state, aux, physics)
    # states
    ρ   = state.ρ
    ρu  = state.ρu
    ρe  = state.ρe
    ρq  = state.ρq

    # thermodynamics
    eos = physics.eos
    parameters = physics.parameters
    p = calc_very_linear_pressure(eos, state, aux, parameters)

    # Reference states
    ρᵣ  = aux.ref_state.ρ
    ρuᵣ = aux.ref_state.ρu
    ρeᵣ = aux.ref_state.ρe
    ρqᵣ = aux.ref_state.ρq
    pᵣ  = aux.ref_state.p

    # derived states
    u = ρu / ρᵣ - ρ * ρuᵣ / (ρᵣ^2)
    q = ρq / ρᵣ - ρ * ρqᵣ / (ρᵣ^2)
    e = ρe / ρᵣ - ρ * ρeᵣ / (ρᵣ^2)

    # derived reference states
    uᵣ = ρuᵣ / ρᵣ
    qᵣ = ρqᵣ / ρᵣ
    eᵣ = ρeᵣ / ρᵣ

    # can be simplified, but written this way to look like the VeryLinearKGVolumeFlux
    
    flux.ρ   += ρᵣ * u + ρ * uᵣ # this is just ρu
    flux.ρu  += p * I + ρᵣ .* (uᵣ .* u' + u .* uᵣ') 
    flux.ρu  += (ρ .* uᵣ) .* uᵣ' 
    flux.ρe  += (ρᵣ * eᵣ + pᵣ) * u
    flux.ρe  += (ρᵣ * e + ρ * eᵣ + p) * uᵣ
    flux.ρq  += ρᵣ * qᵣ * u + (ρᵣ * q + ρ * qᵣ) * uᵣ

    # flux.ρ  += ρu
    # flux.ρu += p * I
    # flux.ρe += (ρeᵣ + pᵣ) * ρu / ρᵣ 

    nothing
end