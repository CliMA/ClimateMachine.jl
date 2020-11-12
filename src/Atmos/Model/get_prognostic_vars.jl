##### Get prognostic variable list

import ..BalanceLaws: prognostic_vars

prognostic_vars(::DryModel) = ()
prognostic_vars(::EquilMoist) = (TotalMoisture(),)
prognostic_vars(::NonEquilMoist) =
    (TotalMoisture(), LiquidMoisture(), IceMoisture())
prognostic_vars(m::AtmosModel) =
    (Mass(), Momentum(), Energy(), prognostic_vars(m.moisture)...)
