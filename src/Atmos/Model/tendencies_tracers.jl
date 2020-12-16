##### Moisture tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Tracers{N}}, m, state, aux, t, ts, direction) where {N}
    u = state.ρu / state.ρ
    return (state.tracers.ρχ .* u')'
end

#####
##### Second order fluxes
#####

function flux(
    ::Diffusion{Tracers{N}},
    m,
    state,
    aux,
    t,
    ts,
    diffusive,
    hyperdiff,
) where {N}
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    d_χ = (-D_t) * aux.tracers.δ_χ' .* diffusive.tracers.∇χ
    return d_χ * state.ρ
end
