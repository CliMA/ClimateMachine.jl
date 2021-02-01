##### Mass tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Mass}, lm::AtmosLinearModel, args)
    @unpack state = args
    return state.ρu
end


##### Momentum tendencies

#####
##### First order fluxes
#####

struct LinearPressureGradient{PV <: Momentum} <:
       TendencyDef{Flux{FirstOrder}, PV} end

function flux(::LinearPressureGradient{Momentum}, lm::AtmosLinearModel, args)
    @unpack state, aux = args
    s = state.ρu * state.ρu'
    pad = SArray{Tuple{size(s)...}}(ntuple(i -> 0, length(s)))
    pL = linearized_pressure(lm.atmos, state, aux)
    return pad + pL * I
end

#####
##### Sources (Momentum)
#####
function source(s::Gravity{Momentum}, lm::AtmosAcousticGravityLinearModel, args)
    @unpack direction, state, aux = args
    if direction === VerticalDirection || direction === EveryDirection
        ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
        return -state.ρ * ∇Φ
    end
    FT = eltype(state)
    return SVector{3, FT}(0, 0, 0)
end

##### Energy tendencies

#####
##### First order fluxes
#####

struct LinearEnergyFlux{PV <: Energy} <: TendencyDef{Flux{FirstOrder}, PV} end

function flux(::LinearEnergyFlux{Energy}, lm::AtmosAcousticLinearModel, args)
    @unpack state, aux = args
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)
    return ((ref.ρe + ref.p) / ref.ρ - e_pot) * state.ρu
end

function flux(
    ::LinearEnergyFlux{Energy},
    lm::AtmosAcousticGravityLinearModel,
    args,
)
    @unpack state, aux = args
    ref = aux.ref_state
    return ((ref.ρe + ref.p) / ref.ρ) * state.ρu
end
