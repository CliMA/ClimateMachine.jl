##### Get prognostic variable list

prognostic_vars(::TotalEnergyModel) = (Energy(),)
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

prognostic_vars(atmos::AtmosModel) = prognostic_vars(atmos.physics)

prognostic_vars(m::AtmosPhysics) = (
    Mass(),
    Momentum(),
    prognostic_vars(energy_model(m))...,
    prognostic_vars(moisture_model(m))...,
    prognostic_vars(precipitation_model(m))...,
    prognostic_vars(tracer_model(m))...,
    prognostic_vars(turbconv_model(m))...,
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

get_specific_state(state, ::Mass) = (state, :ρ)
get_specific_state(state, ::Momentum) = (state, :u)
get_specific_state(state, ::Energy) = (state.energy, :e)
get_specific_state(state, ::ρθ_liq_ice) = (state.energy, :θ_liq_ice)
get_specific_state(state, ::TotalMoisture) = (state.moisture, :q_tot)
get_specific_state(state, ::LiquidMoisture) = (state.moisture, :q_liq)
get_specific_state(state, ::IceMoisture) = (state.moisture, :q_ice)
get_specific_state(state, ::Rain) = (state.precipitation, :q_rai)
get_specific_state(state, ::Snow) = (state.precipitation, :q_sno)
get_specific_state(state, ::Tracers{N}) where {N} = (state.tracers, :χ)

prognostic_vars(m::AtmosLinearModel) = (Mass(), Momentum(), Energy())
