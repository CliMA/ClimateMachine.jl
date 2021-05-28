module SnowModel

using DocStringExtensions
using UnPack
using CLIMAParameters
using CLIMAParameters.Planet: cp_l, cp_i, T_0, LH_f0

""" define local parameter values
volumetric liquid fraction of water in excess of the residual water in snow. could be a constant or a function of snow density
θ_r = 0.08
dt_runoff: timescale used for snow runoff terms, approximately equals dt
"""

"""
    ρc_snow(l::FT,
            ρ_snow::FT,
            param_set::AbstractParamSet
            ) where {FT}

Computes the volumetric heat capacity of snow pack given the liquid water
fraction l and snow density ρ_snow.
"""
function volumetric_heat_capacity(l::FT,
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
function l(ρe_int::FT,
    ρ_snow::FT,
    param_set::AbstractParameterSet
) where {FT}

Computes the liquid water mass fraction l given volumetric internal energy of snow
ρe_int and snow density ρ_snow

"""
function liquid_fraction(ρe_int::FT,
        ρ_snow::FT,
        param_set::AbstractParameterSet
) where {FT}
    _c_i = FT(cp_i(param_set))
    _c_l = FT(cp_l(param_set))
    _T_fr = FT(T_freeze(param_set))
    _T_ref = FT(T_0(param_set))
    _LH_f0 = FO(LH_f0(param_set))
 
    ρe_int_l0 =  ρ_snow*(_c_i(_T_fr-_T_ref)-_LH_f0)
    ρe_int_l1 =  ρ_snow*_c_l(_T_fr-_T_ref)

    if ρe_int < ρe_int_l0
        l = FT(0)
    elseif ρe_int_l0 < ρe_int < ρe_int_l1
        l = (ρe_int/ρ_snow+_LH_f0-_c_i*(_T_fr-_T_ref))/((_c_l-_c_i)*(_T_fr-_T_ref)+_LH_f0)
    else
        l = FT(1)
    end
    return l
end

"""
function T_snow_ave(ρe_int::FT,
    ρ_snow::FT,
    param_set::AbstractParameterSet
) where {FT}

Computes the average snow pack temperature given volumetric internal energy of snow
ρe_int and volumetric_liquid_fraction l

"""
function T_snow_ave(
           ρe_int::FT,
           l::FT,
           param_set::AbstractParameterSet
) where {FT}
    _T_ref = FT(T_0(param_set))
    _LH_f0 = FO(LH_f0(param_set))
    T_snow_ave = (ρe_int+(1-l)*_LH_f0)/ρc_snow + _T_ref
    return T_snow_ave
end

""" using Q for energy, m for mass, v for volumetric ?
lmax: maximum liquid water mass fraction
ρe_int_max: maximum volumetric internal energy of liquid water
E_runoff: volumetric runoff energy
z_snow: the depth of snow

local pars
θ_r = 0.08: volumetric liquid fraction of water in excess of the residual water in snow. could be a constant or a function of snow density
dt_runoff = dt: timescale used for snow runoff terms, approximately equals dt
H = : a coeffient in the snow runoff equation ?
κ_snow =
"""

function runoff_volumnetric_energy_and_mass(
    ρe_int::FT,
    ρ_snow::FT,
    z_snow::FT,
    param_set::AbstractParameterSet,
    θ_r::FT,
    dt_runoff::FT,
    H::FT
) where {FT}
    _ρ_l = FT(ρ_cloud_liq(param_set))
    _c_i = FT(cp_i(param_set))
    _c_l = FT(cp_l(param_set))
    _LH_f0 = FO(LH_f0(param_set))
    c_snow = _c_i*(FT(1.0)-l) + l*_c_i
    """ missing conditions when ρe_int is between 0 and ρ_snow*_c_l*(_T_fr-_T_ref)?"""
    if ρe_int < FT(0)   """ should it be ρ_snow*_c_l*(_T_fr-_T_ref) instead? """
        lmax = θ_r * _ρ_l/ρ_snow 
    else ρe_int >= ρ_snow*_c_l*(_T_fr-_T_ref)  
        lmax = FT(0)    
        """ all melt """
    end
    ρe_int_max =ρ_snow(c_snow*lmax*(_T_fr-_T_ref)-(1-lmax)*_LH_f0)
    E_runoff = -(-ρe_int_max)/dt_runoff*H*(ρe_int-ρe_int_max)
    m_runoff = E_runoff*z_snow/(ρe_int_l*_ρ_l)
    v_runoff = m_runoff/_ρ_l
    return E_runoff, v_runoff
end

"""
boundary fluxes: code later ...

function evap_volumnetric_mass(
    ρe_int::FT,
    ρ_snow::FT,
    z_snow::FT,
    param_set::AbstractParameterSet,
    θ_r::FT,
    dt_runoff::FT,
    H::FT
) where {FT}

end
"""

"""
Temperature profile
"""
function temperature_profile(
    Q_surf::FT,
    Q_bott::FT,
    z_snow::FT,
    ρ_snow::FT,
    κ_snow::FT,
    T_snow_ave::FT,
    param_set::AbstractParameterSet,
) where {FT}
    _c_i = FT(cp_i(param_set))
    _c_l = FT(cp_l(param_set))
    c_snow = _c_i*(FT(1.0)-l) + l*_c_i
    d = (2*κ_snow*24/(ρ_snow*c_snow))
    h = max(0,z_snow-d)
    T_surf = T_snow_ave+(Q_surf*(h^2-z^2)-Q2*h^2)/(2*z_snow*κ_snow)
    T_bott = T_surf*(h-z_snow)/(h+z_snow) +(2*z_snow*T_snow_ave*κ_snow+Q_bott*h*z_snow)/((h+z_snow)*κ_snow)

    T(z)
    return T_surf, T_bott
end