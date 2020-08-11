"""
    Zero-moment bulk microphysics scheme that instantly removes
    moisture above certain threshold.
    This is equivalent to instanteneous conversion of cloud condensate
    into precipitation and precipitation fallout with infinite
    terminal velocity.

"""
module Microphysics_0M

using ClimateMachine.Thermodynamics

using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics

const APS = AbstractParameterSet

export remove_precipitation

"""
    remove_precipitation(param_set::APS, q)

 - `param_set` - abstract set
 - `q` - current PhasePartition

Returns the total water tendency due to precipitation.
All the excess total water specific humidity above user-defined threshold
  is treated as precipitation and removed.
The tendency is obtained assuming a relaxation with a constant timescale
  to a state with precipitable water removed.

"""
function remove_precipitation(
    param_set::APS,
    q::PhasePartition{FT}  # q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
) where {FT <: Real}

    # TODO:
    # τ_rain_removal(param_set), etc - move to ClimaParameters
    # have dt instead of constant timescale
    # threshold based on percentage supersaturation or ql+qi?

    _τ_rain_removal::FT = FT(1000)
    _q_precip_thr::FT = FT(5e-3)

    return max(FT(0), (q.liq + q.ice - _q_precip_thr)) / _τ_rain_removal
end

end #module Microphysics_0M.jl
