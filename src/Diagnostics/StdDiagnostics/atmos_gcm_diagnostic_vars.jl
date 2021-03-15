@pointwise_diagnostic(
    AtmosGCMConfigType,
    u,
    "m s^-1",
    "zonal wind",
    "eastward_wind",
    true,
) do (atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.ρu[1] / states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    v,
    "m s^-1",
    "meridional wind",
    "northward_wind",
    true,
) do (atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.ρu[2] / states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    w,
    "m s^-1",
    "vertical wind",
    "upward_air_velocity",
    true,
) do (atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.ρu[3] / states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    rho,
    "kg m^-3",
    "air density",
    "air_density",
) do (atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    et,
    "J kg^-1",
    "total specific energy",
    "specific_dry_energy_of_air",
) do (energy::EnergyModel, atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.energy.ρe / states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    temp,
    "K",
    "air temperature",
    "air_temperature",
) do (atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    air_temperature(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    pres,
    "Pa",
    "air pressure",
    "air_pressure",
) do (atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    air_pressure(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    thd,
    "K",
    "dry potential temperature",
    "air_potential_temperature",
) do (atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    dry_pottemp(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    ei,
    "J kg^-1",
    "specific internal energy",
    "internal_energy",
) do (atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    internal_energy(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    ht,
    "J kg^-1",
    "specific enthalpy based on total energy",
    "",
) do (energy::EnergyModel, atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    e_tot = states.prognostic.energy.ρe / states.prognostic.ρ
    total_specific_enthalpy(ts, e_tot)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    hi,
    "J kg^-1",
    "specific enthalpy based on internal energy",
    "atmosphere_enthalpy_content",
) do (atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    specific_enthalpy(ts)
end

#= TODO
@XXX_diagnostic(
    "vort",
    AtmosGCMConfigType,
    GridInterpolated,
    "s^-1",
    "vertical component of relative velocity",
    "atmosphere_relative_velocity",
) do (atmos::AtmosModel, states::States, curr_time, cache)
end
=#

@pointwise_diagnostic(
    AtmosGCMConfigType,
    qt,
    "kg kg^-1",
    "mass fraction of total water in air (qv+ql+qi)",
    "mass_fraction_of_water_in_air",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    states.prognostic.moisture.ρq_tot / states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    ql,
    "kg kg^-1",
    "mass fraction of liquid water in air",
    "mass_fraction_of_cloud_liquid_water_in_air",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    liquid_specific_humidity(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    qv,
    "kg kg^-1",
    "mass fraction of water vapor in air",
    "specific_humidity",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    vapor_specific_humidity(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    qi,
    "kg kg^-1",
    "mass fraction of ice in air",
    "mass_fraction_of_cloud_ice_in_air",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    ice_specific_humidity(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    thv,
    "K",
    "virtual potential temperature",
    "virtual_potential_temperature",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    virtual_pottemp(ts)
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    thl,
    "K",
    "liquid-ice potential temperature",
    "",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    liquid_ice_pottemp(ts)
end
