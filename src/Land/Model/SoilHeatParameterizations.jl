"""
    SoilHeatParameterizations

Functions for volumetric heat capacity, temperature as a function
of volumetric internal energy, saturated thermal conductivity, thermal
conductivity, relative saturation and the Kersten number are included.
Heat capacities denoted by `ρc_` are volumetric, while `cp_` denotes an isobaric
specific heat capacity.
"""
module SoilHeatParameterizations

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0
using CLIMAParameters.Atmos.Microphysics: K_therm
using DocStringExtensions

export volumetric_heat_capacity,
    volumetric_internal_energy,
    saturated_thermal_conductivity,
    thermal_conductivity,
    relative_saturation,
    kersten_number,
    volumetric_internal_energy_liq,
    temperature_from_ρe_int,
    k_solid,
    k_dry,
    ksat_unfrozen,
    ksat_frozen

"""
    function temperature_from_ρe_int(
        ρe_int::FT,
        θ_i::FT,
        ρc_s::FT,
        param_set::AbstractParameterSet
    ) where {FT}

Computes the temperature of soil given `θ_i` and volumetric
internal energy `ρe_int`.
"""
function temperature_from_ρe_int(
    ρe_int::FT,
    θ_i::FT,
    ρc_s::FT,
    param_set::AbstractParameterSet,
) where {FT}
    _ρ_i = FT(ρ_cloud_ice(param_set))
    _T_ref = FT(T_0(param_set))
    _LH_f0 = FT(LH_f0(param_set))
    T = _T_ref + (ρe_int + θ_i * _ρ_i * _LH_f0) / ρc_s
    return T
end

"""
    volumetric_heat_capacity(
        θ_l::FT,
        θ_i::FT,
        ρc_ds::FT,
        param_set::AbstractParameterSet
    ) where {FT}

Compute the expression for volumetric heat capacity.
"""
function volumetric_heat_capacity(
    θ_l::FT,
    θ_i::FT,
    ρc_ds::FT,
    param_set::AbstractParameterSet,
) where {FT}
    _ρ_i = FT(ρ_cloud_ice(param_set))
    ρcp_i = FT(cp_i(param_set) * _ρ_i)

    _ρ_l = FT(ρ_cloud_liq(param_set))
    ρcp_l = FT(cp_l(param_set) * _ρ_l)

    ρc_s = ρc_ds + θ_l * ρcp_l + θ_i * ρcp_i
    return ρc_s
end

"""
    volumetric_internal_energy(
        θ_i::FT,
        ρc_s::FT,
        T::FT,
        param_set::AbstractParameterSet
    ) where {FT}

Compute the expression for volumetric internal energy.
"""
function volumetric_internal_energy(
    θ_i::FT,
    ρc_s::FT,
    T::FT,
    param_set::AbstractParameterSet,
) where {FT}
    _ρ_i = FT(ρ_cloud_ice(param_set))
    _LH_f0 = FT(LH_f0(param_set))
    _T_ref = FT(T_0(param_set))
    ρe_int = ρc_s * (T - _T_ref) - θ_i * _ρ_i * _LH_f0
    return ρe_int
end

"""
    saturated_thermal_conductivity(
        θ_l::FT,
        θ_i::FT,
        κ_sat_unfrozen::FT,
        κ_sat_frozen::FT
    ) where {FT}

Compute the expression for saturated thermal conductivity of soil matrix.
"""
function saturated_thermal_conductivity(
    θ_l::FT,
    θ_i::FT,
    κ_sat_unfrozen::FT,
    κ_sat_frozen::FT,
) where {FT}
    θ_w = θ_l + θ_i
    if θ_w < eps(FT)
        κ_sat = FT(0.0)
    else
        κ_sat = FT(κ_sat_unfrozen^(θ_l / θ_w) * κ_sat_frozen^(θ_i / θ_w))
    end

    return κ_sat
end

"""
    relative_saturation(
            θ_l::FT,
            θ_i::FT,
            porosity::FT
    ) where {FT}

Compute the expression for relative saturation.
"""
function relative_saturation(θ_l::FT, θ_i::FT, porosity::FT) where {FT}
    S_r = (θ_l + θ_i) / porosity
    return S_r
end
"""
    kersten_number(
        θ_i::FT,
        S_r::FT,
        soil_param_functions::PS
    ) where {FT, PS}

Compute the expression for the Kersten number.
"""
function kersten_number(
    θ_i::FT,
    S_r::FT,
    soil_param_functions::PS,
) where {FT, PS}
    a = soil_param_functions.a
    b = soil_param_functions.b
    ν_ss_om = soil_param_functions.ν_ss_om
    ν_ss_quartz = soil_param_functions.ν_ss_quartz
    ν_ss_gravel = soil_param_functions.ν_ss_gravel

    if θ_i < eps(FT)
        K_e =
            S_r^((FT(1) + ν_ss_om - a * ν_ss_quartz - ν_ss_gravel) / FT(2)) *
            (
                (FT(1) + exp(-b * S_r))^(-FT(3)) -
                ((FT(1) - S_r) / FT(2))^FT(3)
            )^(FT(1) - ν_ss_om)
    else
        K_e = S_r^(FT(1) + ν_ss_om)
    end
    return K_e
end

"""
    thermal_conductivity(
        κ_dry::FT,
        K_e::FT,
        κ_sat::FT
    ) where {FT}

Compute the expression for thermal conductivity of soil matrix.
"""
function thermal_conductivity(κ_dry::FT, K_e::FT, κ_sat::FT) where {FT}
    κ = K_e * κ_sat + (FT(1) - K_e) * κ_dry
    return κ
end

"""
    volumetric_internal_energy_liq(
        T::FT,
        T_ref::FT,
    ) where {FT}

Compute the expression for the volumetric internal energy of liquid water.
"""
function volumetric_internal_energy_liq(
    T::FT,
    param_set::AbstractParameterSet,
) where {FT}
    _T_ref = FT(T_0(param_set))
    _ρ_l = FT(ρ_cloud_liq(param_set))
    ρcp_l = FT(cp_l(param_set) * _ρ_l)
    ρe_int_l = ρcp_l * (T - _T_ref)
    return ρe_int_l
end

"""
    function k_solid(
        ν_ss_om::FT,
        ν_ss_quartz::FT,
        κ_quartz::FT,
        κ_minerals::FT,
        κ_om::FT,
    ) where {FT}

Computes the thermal conductivity of the solid material in soil.
The `_ss_` subscript denotes that the volumetric fractions of the soil
components are referred to the soil solid components, not including the pore
space.
"""
function k_solid(
    ν_ss_om::FT,
    ν_ss_quartz::FT,
    κ_quartz::FT,
    κ_minerals::FT,
    κ_om::FT,
) where {FT}
    return κ_om^ν_ss_om *
           κ_quartz^ν_ss_quartz *
           κ_minerals^(FT(1) - ν_ss_om - ν_ss_quartz)
end


"""
    function ksat_frozen(
        κ_solid::FT,
        porosity::FT,
        κ_ice::FT
    ) where {FT}

Computes the thermal conductivity for saturated frozen soil.
"""
function ksat_frozen(κ_solid::FT, porosity::FT, κ_ice::FT) where {FT}
    return κ_solid^(FT(1.0) - porosity) * κ_ice^(porosity)
end

"""
    function ksat_unfrozen(
        κ_solid::FT,
        porosity::FT,
        κ_l::FT
    ) where {FT}

Computes the thermal conductivity for saturated unfrozen soil.
"""
function ksat_unfrozen(κ_solid::FT, porosity::FT, κ_l::FT) where {FT}
    return κ_solid^(FT(1.0) - porosity) * κ_l^porosity
end

"""
    function ρb_ss(porosity::FT, ρp::FT) where {FT}
Computes the dry soil bulk density from the dry soil particle
density.
"""
function ρb_ss(porosity::FT, ρp::FT) where {FT}
    return (FT(1.0) - porosity) * ρp
end

"""
    function k_dry(
        param_set::AbstractParameterSet
        soil_param_functions::PS,
    ) where {PS}

Computes the thermal conductivity of dry soil.
"""
function k_dry(
    param_set::AbstractParameterSet,
    soil_param_functions::PS,
) where {PS}
    κ_dry_parameter = soil_param_functions.κ_dry_parameter
    FT = typeof(κ_dry_parameter)
    porosity = soil_param_functions.porosity
    ρp = soil_param_functions.ρp
    κ_solid = soil_param_functions.κ_solid
    κ_air = FT(K_therm(param_set))
    ρb_val = ρb_ss(porosity, ρp)
    numerator = (κ_dry_parameter * κ_solid - κ_air) * ρb_val + κ_air * ρp
    denom = ρp - (FT(1.0) - κ_dry_parameter) * ρb_val
    return numerator / denom
end

end # Module
