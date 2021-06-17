struct CompressibleEuler <: AbstractTerm end
struct LinearCompressibleEuler <: AbstractTerm end

@inline calc_component!(flux, ::Nothing, _...) = nothing
@inline calc_component!(flux, ::AbstractTerm, _...) = nothing

@inline function calc_component!(flux, ::CompressibleEuler, state, aux, physics)
    ρ   = state.ρ
    ρu  = state.ρu
    ρe  = state.ρe
    ρq  = state.ρq
    eos = physics.eos
    parameters = physics.parameters

    p = calc_pressure(eos, state, aux, parameters)
    u = ρu / ρ

    flux.ρ  += ρu
    flux.ρu += ρu ⊗ u + p * I
    flux.ρe += (ρe + p) * u
    flux.ρq += ρq * u

    nothing
end

@inline function calc_component!(flux, ::LinearCompressibleEuler, state, aux, physics)
    ρu  = state.ρu
    ρᵣ  = aux.ref_state.ρ
    pᵣ  = aux.ref_state.p
    ρeᵣ = aux.ref_state.ρe
    
    p = calc_linear_pressure(eos, state, aux, parameters)

    flux.ρ  += ρu + p * I
    flux.ρe += (ρeᵣ + pᵣ) * ρu / ρᵣ 

    nothing
end
