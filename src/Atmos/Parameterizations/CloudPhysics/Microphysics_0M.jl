"""
    Zero-moment bulk microphysics scheme that instantly removes
    moisture above certain threshold.

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
All the excess total water above user-defined supersaturation threshold
  is treated as precipitation and removed.
The tendency is obtained assuming a relaxation with a constant timescale
  to a state with precipitable water removed.

"""
function remove_precipitation(
    param_set::APS,
    q::PhasePartition{FT},  #q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
) where {FT <: Real}

    _τ_rain_removal::FT = FT(1000) #τ_rain_removal(param_set) #TODO - move to ClimaParameters
    _q_precip_thr::FT = FT(5e-3)

    return max(FT(0), (q.liq + q.ice - q_tr)) / _τ_cond_evap
end

end #module Microphysics_0M.jl
