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

Computes the volumetric heat capacity of snow pack given the liquid water
fraction l and snow density ρ_snow.
"""
function ρc_snow(l::FT,
                 ρ_snow::FT,
                 param_set::AbstractParameterSet
                 ) where {FT}
    _c_i = FT(cp_i(param_set))
    _c_l = FT(cp_l(param_set))
    c_snow = _c_i*(FT(1.0)-l) + l*_c_i
    ρc_snow = ρ_snow*c_snow
    return ρc_snow
end

"""
function l(sn_int::FT,
    ρ_snow::FT,
    param_set::AbstractParameterSet
) where {FT}

Computes the liquid water mass fraction l given volumetric internal energy of snow
sn_int and snow density ρ_snow

Q: elseif sn_int_l0 < sn_int < sn_int_l1

"""
function l(sn_int::FT,
           ρ_snow::FT,
           param_set::AbstractParameterSet
) where {FT}
    _c_i = FT(cp_i(param_set))
    _c_l = FT(cp_l(param_set))
    _T_fr = FT(T_freeze(param_set))
    _T_ref = FT(T_0(param_set))
    _LH_f0 = FO(LH_f0(param_set))
 
    sn_int_l0 =  ρ_snow*(_c_i(_T_fr-_T_ref)-_LH_f0)
    sn_int_l1 =  ρ_snow*_c_l(_T_fr-_T_ref)

    if sn_int < sn_int_l0
        l = 0
    elseif sn_int > sn_int_l0 && sn_int < sn_int_l1
        l = (sn_int/ρ_snow+_LH_f0-_c_i*(_T_fr-_T_ref))/((_c_l-_c_i)*(_T_fr-_T_ref)+_LH_f0)
    else
        l = 1
    end
    return l
end

"""
function sn_T_ave(sn_int::FT,
    ρ_snow::FT,
    param_set::AbstractParameterSet
) where {FT}

Computes the average snow pack temperature given volumetric internal energy of snow
sn_int and volumetric_liquid_fraction l

"""
function sn_T_ave(sn_int::FT,
           l::FT,
           param_set::AbstractParameterSet
) where {FT}
    _T_ref = FT(T_0(param_set))
    _LH_f0 = FO(LH_f0(param_set))
    sn_T_ave = (sn_int+(1-l)*_LH_f0)/ρc_snow + _T_ref
    return sn_T_ave
end
