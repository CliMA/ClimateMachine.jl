##### Moisture tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Tracers{N}}, atmos, args) where {N}
    @unpack state = args
    u = state.ρu / state.ρ
    return (state.tracers.ρχ .* u')'
end

#####
##### Second order fluxes
#####

function flux(::Diffusion{Tracers{N}}, atmos, args) where {N}
    @unpack state, aux, t, diffusive = args
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    d_χ = (-D_t) * aux.tracers.δ_χ' .* diffusive.tracers.∇χ
    return d_χ * state.ρ
end
