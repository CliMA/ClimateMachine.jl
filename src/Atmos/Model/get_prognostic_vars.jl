##### Get prognostic variable list

import ..BalanceLaws: prognostic_vars

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
