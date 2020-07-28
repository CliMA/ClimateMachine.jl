"""
    DiagnosticVariable

Currently only holds information about a diagnostics variable.

Standard names are from:
http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html

TODO: will expand to include definition (code) as well.
"""
struct DiagnosticVariable
    name::String
    attrib::OrderedDict

    DiagnosticVariable(name::String, attrib::OrderedDict = OrderedDict()) =
        new(name, attrib)
end
const Variables = OrderedDict{String, DiagnosticVariable}()

function var_attrib(
    units::String,
    long_name::String,
    standard_name::String,
    fill_value::Union{Nothing, Any} = nothing,
)
    attrib = OrderedDict(
        "units" => units,
        "long_name" => long_name,
        "standard_name" => standard_name,
    )
    if !isnothing(fill_value)
        attrib["_FillValue"] = fill_value
    end
    return attrib
end

"""
    setup_variables()

Called at module initialization to define all currently defined diagnostic
variables.
"""
function setup_variables()
    # les long_name: "x-velocity"
    Variables["u"] = DiagnosticVariable(
        "u",
        var_attrib("m s^-1", "zonal wind", "eastward_wind"),
    )
    # les long_name: "y-velocity"
    Variables["v"] = DiagnosticVariable(
        "v",
        var_attrib("m s^-1", "meridional wind", "northward_wind"),
    )
    # les long_name = "z-velocity"
    Variables["w"] = DiagnosticVariable(
        "w",
        var_attrib("m s^-1", "vertical wind", "upward_air_velocity"),
    )
    Variables["rho"] = DiagnosticVariable(
        "rho",
        var_attrib("kg m^-3", "air density", "air_density"),
    )
    Variables["temp"] = DiagnosticVariable(
        "temp",
        var_attrib("K", "air temperature", "air_temperature"),
    )
    Variables["pres"] = DiagnosticVariable(
        "pres",
        var_attrib("Pa", "air pressure", "air_pressure"),
    )
    Variables["thd"] = DiagnosticVariable(
        "thd",
        var_attrib(
            "K",
            "dry potential temperature",
            "air_potential_temperature",
        ),
    )
    Variables["thv"] = DiagnosticVariable(
        "thv",
        var_attrib(
            "K",
            "virtual potential temperature",
            "virtual_potential_temperature",
        ),
    )
    Variables["et"] = DiagnosticVariable(
        "et",
        var_attrib(
            "J kg^-1",
            "total specific energy",
            "specific_dry_energy_of_air",
        ),
    )
    Variables["ei"] = DiagnosticVariable(
        "ei",
        var_attrib("J kg^-1", "specific internal energy", "internal_energy"),
    )
    Variables["ht"] = DiagnosticVariable(
        "ht",
        var_attrib("J kg^-1", "specific enthalpy based on total energy", ""),
    )
    Variables["hi"] = DiagnosticVariable(
        "hi",
        var_attrib(
            "J kg^-1",
            "specific enthalpy based on internal energy",
            "atmosphere_enthalpy_content",
        ),
    )
    Variables["vort"] = DiagnosticVariable(
        "vort",
        var_attrib(
            "s^-1",
            "vertical component of relative velocity",
            "atmosphere_relative_velocity",
        ),
    )
    Variables["avg_rho"] = DiagnosticVariable(
        "avg_rho",
        var_attrib("kg m^-3", "air density", "air_density"),
    )
    Variables["qt"] = DiagnosticVariable(
        "qt",
        var_attrib(
            "kg kg^-1",
            "mass fraction of total water in air (qv+ql+qi)",
            "mass_fraction_of_water_in_air",
        ),
    )
    Variables["ql"] = DiagnosticVariable(
        "ql",
        var_attrib(
            "kg kg^-1",
            "mass fraction of liquid water in air",
            "mass_fraction_of_cloud_liquid_water_in_air",
        ),
    )
    Variables["qv"] = DiagnosticVariable(
        "qv",
        var_attrib(
            "kg kg^-1",
            "mass fraction of water vapor in air",
            "specific_humidity",
        ),
    )
    Variables["qi"] = DiagnosticVariable(
        "qi",
        var_attrib(
            "kg kg^-1",
            "mass fraction of ice in air",
            "mass_fraction_of_cloud_ice_in_air",
        ),
    )
    Variables["thl"] = DiagnosticVariable(
        "thl",
        var_attrib("K", "liquid-ice potential temperature", ""),
    )
    Variables["cld_frac"] = DiagnosticVariable(
        "cld_frac",
        var_attrib(
            "",
            "cloud fraction",
            "cloud_area_fraction_in_atmosphere_layer",
        ),
    )
    Variables["var_u"] = DiagnosticVariable(
        "var_u",
        var_attrib("m^2 s^-2", "variance of x-velocity", ""),
    )
    Variables["var_v"] = DiagnosticVariable(
        "var_v",
        var_attrib("m^2 s^-2", "variance of y-velocity", ""),
    )
    Variables["var_w"] = DiagnosticVariable(
        "var_w",
        var_attrib("m^2 s^-2", "variance of z-velocity", ""),
    )
    Variables["w3"] = DiagnosticVariable(
        "w3",
        var_attrib("m^3 s^-3", "third moment of z-velocity", ""),
    )
    Variables["tke"] = DiagnosticVariable(
        "tke",
        var_attrib("m^2 s^-2", "turbulent kinetic energy", ""),
    )
    Variables["var_qt"] = DiagnosticVariable(
        "var_qt",
        var_attrib("kg^2 kg^-2", "variance of total specific humidity", ""),
    )
    Variables["var_thl"] = DiagnosticVariable(
        "var_thl",
        var_attrib("K^2", "variance of liquid-ice potential temperature", ""),
    )
    Variables["var_ei"] = DiagnosticVariable(
        "var_ei",
        var_attrib("J^2 kg^-2", "variance of specific internal energy", ""),
    )
    Variables["cov_w_u"] = DiagnosticVariable(
        "cov_w_u",
        var_attrib("m^2 s^-2", "vertical eddy flux of x-velocity", ""),
    )
    Variables["cov_w_v"] = DiagnosticVariable(
        "cov_w_v",
        var_attrib("m^2 s^-2", "vertical eddy flux of y-velocity", ""),
    )
    Variables["cov_w_rho"] = DiagnosticVariable(
        "cov_w_rho",
        var_attrib("kg m^-2 s^-1", "vertical eddy flux of density", ""),
    )
    Variables["cov_w_qt"] = DiagnosticVariable(
        "cov_w_qt",
        var_attrib(
            "kg kg^-1 m s^-1",
            "vertical eddy flux of total specific humidity",
            "",
        ),
    )
    Variables["cov_w_ql"] = DiagnosticVariable(
        "cov_w_ql",
        var_attrib(
            "kg kg^-1 m s^-1",
            "vertical eddy flux of liquid water specific humidity",
            "",
        ),
    )
    Variables["cov_w_qv"] = DiagnosticVariable(
        "cov_w_qv",
        var_attrib(
            "kg kg^-1 m s^-1",
            "vertical eddy flux of water vapor specific humidity",
            "",
        ),
    )
    Variables["cov_w_thd"] = DiagnosticVariable(
        "cov_w_thd",
        var_attrib(
            "K m s^-1",
            "vertical eddy flux of dry potential temperature",
            "",
        ),
    )
    Variables["cov_w_thv"] = DiagnosticVariable(
        "cov_w_thv",
        var_attrib(
            "K m s^-1",
            "vertical eddy flux of virtual potential temperature",
            "",
        ),
    )
    Variables["cov_w_thl"] = DiagnosticVariable(
        "cov_w_thl",
        var_attrib(
            "K m s^-1",
            "vertical eddy flux of liquid-ice potential temperature",
            "",
        ),
    )
    Variables["cov_w_ei"] = DiagnosticVariable(
        "cov_w_ei",
        var_attrib(
            "J kg^-1 m s^-1",
            "vertical eddy flux of specific internal energy",
            "",
        ),
    )
    Variables["cov_qt_thl"] = DiagnosticVariable(
        "cov_qt_thl",
        var_attrib(
            "kg kg^-1 K",
            "covariance of total specific humidity and liquid-ice potential temperature",
            "",
        ),
    )
    Variables["cov_qt_ei"] = DiagnosticVariable(
        "cov_qt_ei",
        var_attrib(
            "kg kg^-1 J kg^-1",
            "covariance of total specific humidity and specific internal energy",
            "",
        ),
    )
    Variables["w_qt_sgs"] = DiagnosticVariable(
        "w_qt_sgs",
        var_attrib(
            "kg kg^-1 m s^-1",
            "vertical sgs flux of total specific humidity",
            "",
        ),
    )
    Variables["w_ht_sgs"] = DiagnosticVariable(
        "w_ht_sgs",
        var_attrib(
            "kg kg^-1 m s^-1",
            "vertical sgs flux of total specific enthalpy",
            "",
        ),
    )
    Variables["cld_cover"] = DiagnosticVariable(
        "cld_cover",
        var_attrib("", "cloud cover", "cloud_area_fraction"),
    )
    Variables["cld_top"] = DiagnosticVariable(
        "cld_top",
        var_attrib("m", "cloud top", "cloud_top_altitude"),
    )
    Variables["cld_base"] = DiagnosticVariable(
        "cld_base",
        var_attrib("m", "cloud base", "cloud_base_altitude"),
    )
    Variables["lwp"] = DiagnosticVariable(
        "lwp",
        var_attrib(
            "kg m^-2",
            "liquid water path",
            "atmosphere_mass_content_of_cloud_condensed_water",
        ),
    )
    Variables["core_frac"] = DiagnosticVariable(
        "core_frac",
        var_attrib("", "cloud core fraction", ""),
    )
    Variables["u_core"] = DiagnosticVariable(
        "u_core",
        var_attrib("m s^-1", "cloud core x-velocity", ""),
    )
    Variables["v_core"] = DiagnosticVariable(
        "v_core",
        var_attrib("m s^-1", "cloud core y-velocity", ""),
    )
    Variables["w_core"] = DiagnosticVariable(
        "w_core",
        var_attrib("m s^-1", "cloud core z-velocity", ""),
    )
    Variables["avg_rho_core"] = DiagnosticVariable(
        "avg_rho_core",
        var_attrib("kg m^-3", "cloud core air density", ""),
    )
    Variables["rho_core"] = DiagnosticVariable(
        "rho_core",
        var_attrib("kg m^-3", "cloud core (density-averaged) air density", ""),
    )
    Variables["qt_core"] = DiagnosticVariable(
        "qt_core",
        var_attrib("kg m^-3", "cloud core total specific humidity", ""),
    )
    Variables["ql_core"] = DiagnosticVariable(
        "ql_core",
        var_attrib("kg m^-3", "cloud core liquid water specific humidity", ""),
    )
    Variables["thv_core"] = DiagnosticVariable(
        "thv_core",
        var_attrib("K", "cloud core virtual potential temperature", ""),
    )
    Variables["thl_core"] = DiagnosticVariable(
        "thl_core",
        var_attrib("K", "cloud core liquid-ice potential temperature", ""),
    )
    Variables["ei_core"] = DiagnosticVariable(
        "ei_core",
        var_attrib("J kg-1", "cloud core specific internal energy", ""),
    )
    Variables["var_u_core"] = DiagnosticVariable(
        "var_u_core",
        var_attrib("m^2 s^-2", "cloud core variance of x-velocity", ""),
    )
    Variables["var_v_core"] = DiagnosticVariable(
        "var_v_core",
        var_attrib("m^2 s^-2", "cloud core variance of y-velocity", ""),
    )
    Variables["var_w_core"] = DiagnosticVariable(
        "var_w_core",
        var_attrib("m^2 s^-2", "cloud core variance of z-velocity", ""),
    )
    Variables["var_qt_core"] = DiagnosticVariable(
        "var_qt_core",
        var_attrib(
            "kg^2 kg^-2",
            "cloud core variance of total specific humidity",
            "",
        ),
    )
    Variables["var_thl_core"] = DiagnosticVariable(
        "var_thl_core",
        var_attrib(
            "K^2",
            "cloud core variance of liquid-ice potential temperature",
            "",
        ),
    )
    Variables["var_ei_core"] = DiagnosticVariable(
        "var_ei_core",
        var_attrib(
            "J^2 kg^-2",
            "cloud core variance of specific internal energy",
            "",
        ),
    )
    Variables["cov_w_rho_core"] = DiagnosticVariable(
        "cov_w_rho_core",
        var_attrib(
            "kg m^-2 s^-1",
            "cloud core vertical eddy flux of density",
            "",
        ),
    )
    Variables["cov_w_qt_core"] = DiagnosticVariable(
        "cov_w_qt_core",
        var_attrib(
            "kg kg^-1 m s^-1",
            "cloud core vertical eddy flux of specific humidity",
            "",
        ),
    )
    Variables["cov_w_thl_core"] = DiagnosticVariable(
        "cov_w_thl_core",
        var_attrib(
            "K m s^-1",
            "cloud core vertical eddy flux of liquid-ice potential temperature",
            "",
        ),
    )
    Variables["cov_w_ei_core"] = DiagnosticVariable(
        "cov_w_ei_core",
        var_attrib(
            "J kg^-1 m^-1 s^-1",
            "cloud core vertical eddy flux of specific internal energy",
            "",
        ),
    )
    Variables["cov_qt_thl_core"] = DiagnosticVariable(
        "cov_qt_thl_core",
        var_attrib(
            "kg kg^-1 K",
            "cloud core covariance of total specific humidity and liquid-ice potential temperature",
            "",
        ),
    )
    Variables["cov_qt_ei_core"] = DiagnosticVariable(
        "cov_qt_ei_core",
        var_attrib(
            "kg kg^-1 J kg^-1",
            "cloud core covariance of total specific humidity and specific internal energy",
            "",
        ),
    )
end
