##### Get prognostic variable list

import ..BalanceLaws: prognostic_vars, get_prog_state

prognostic_vars(::EnergyModel) = (Energy(),)
prognostic_vars(::θModel) = (ρθ_liq_ice(),)

prognostic_vars(::DryModel) = ()
prognostic_vars(::EquilMoist) = (TotalMoisture(),)
prognostic_vars(::NonEquilMoist) =
    (TotalMoisture(), LiquidMoisture(), IceMoisture())

prognostic_vars(::NoPrecipitation) = ()
prognostic_vars(::RainModel) = (Rain(),)
prognostic_vars(::RainSnowModel) = (Rain(), Snow())

prognostic_vars(::NoTracers) = ()
prognostic_vars(::NTracers{N}) where {N} = (Tracers{N}(),)

prognostic_vars(m::AtmosModel) = (
    Mass(),
    Momentum(),
    prognostic_vars(m.energy)...,
    prognostic_vars(m.moisture)...,
    prognostic_vars(m.precipitation)...,
    prognostic_vars(m.tracers)...,
    prognostic_vars(m.turbconv)...,
)

get_prog_state(state, ::Mass) = (state, :ρ)
get_prog_state(state, ::Momentum) = (state, :ρu)
get_prog_state(state, ::Energy) = (state.energy, :ρe)
get_prog_state(state, ::ρθ_liq_ice) = (state.energy, :ρθ_liq_ice)
get_prog_state(state, ::TotalMoisture) = (state.moisture, :ρq_tot)
get_prog_state(state, ::LiquidMoisture) = (state.moisture, :ρq_liq)
get_prog_state(state, ::IceMoisture) = (state.moisture, :ρq_ice)
get_prog_state(state, ::Rain) = (state.precipitation, :ρq_rai)
get_prog_state(state, ::Snow) = (state.precipitation, :ρq_sno)
get_prog_state(state, ::Tracers{N}) where {N} = (state.tracers, :ρχ)

prognostic_vars(m::AtmosLinearModel) = (Mass(), Momentum(), Energy())
