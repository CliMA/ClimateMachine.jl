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
using DocStringExtensions

export volumetric_heat_capacity,
    volumetric_internal_energy,
    saturated_thermal_conductivity,
    thermal_conductivity,
    relative_saturation,
    kersten_number,
    volumetric_internal_energy_liq,
    temperature_from_ρe_int


"""
    function temperature_from_ρe_int(
        ρe_int::FT,
        θ_i::FT,
        ρcs::FT,
        param_set::AbstractParameterSet
    ) where {FT}

Computes the temperature of soil given `θ_i` and volumetric
internal energy `ρe_int`.
"""
function temperature_from_ρe_int(
    ρe_int::FT,
    θ_i::FT,
    ρcs::FT,
    param_set::AbstractParameterSet,
) where {FT}

    _ρ_i = FT(ρ_cloud_ice(param_set))
    _T_ref = FT(T_0(param_set))
    _LH_f0 = FT(LH_f0(param_set))
    T = _T_ref + (ρe_int + θ_i * _ρ_i * _LH_f0) / ρcs
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
    ν_om = soil_param_functions.ν_om
    ν_sand = soil_param_functions.ν_sand
    ν_gravel = soil_param_functions.ν_gravel

    if θ_i < eps(FT)
        K_e =
            S_r^((FT(1) + ν_om - a * ν_sand - ν_gravel) / FT(2)) *
            (
                (FT(1) + exp(-b * S_r))^(-FT(3)) -
                ((FT(1) - S_r) / FT(2))^FT(3)
            )^(FT(1) - ν_om)
    else
        K_e = S_r^(FT(1) + ν_om)
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
        cp_l::FT,
        T::FT,
        T_ref::FT,
    ) where {FT}
Compute the expression for the volumetric internal energy of liquid water.
Here, cp_l is the volumetric heat capacity of liquid water.
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

end # Module
