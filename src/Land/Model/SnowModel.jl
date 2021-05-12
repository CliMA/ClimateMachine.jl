module SnowModel

using DocStringExtensions
using UnPack
using CLIMAParameters
using CLIMAParameters.Planet: cp_l, cp_i, T_0, LH_f0


"""
    ρc_snow(l::FT,
            ρ_snow::FT,
            param_set::AbstractParamSet
            ) where {FT}

Computes the volumetric heat capacity given the liquid water
fraction l and snow density ρ_snow.
"""
function ρc_snow(l::FT,
                 ρ_snow::FT,
                 param_set::AbstractParamSet
                 ) where {FT}
    _c_i = FT(cp_i(param_set))
    _c_l = FT(cp_l(param_set))
    c_snow = _c_i*(FT(1.0)-l) + l*_c_i
    ρc_snow = ρ_snow*c_snow
    return ρc_snow
end



end
