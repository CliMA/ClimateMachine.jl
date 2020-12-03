##### Get prognostic variable list

import ..BalanceLaws: prognostic_vars

prognostic_vars(::DryModel) = ()
prognostic_vars(::EquilMoist) = (TotalMoisture(),)
prognostic_vars(::NonEquilMoist) =
    (TotalMoisture(), LiquidMoisture(), IceMoisture())

prognostic_vars(::NoPrecipitation) = ()
prognostic_vars(::RainModel) = (Rain(),)
prognostic_vars(::NoTracers) = ()
prognostic_vars(::NTracers{N}) where {N} = (Tracers{N}(),)

prognostic_vars(m::AtmosModel) = (
    Mass(),
    Momentum(),
    Energy(),
    prognostic_vars(m.moisture)...,
    prognostic_vars(m.precipitation)...,
    prognostic_vars(m.tracers)...,
)
