"""
    DiagnosticVariable

Currently only holds information about a diagnostics variable.

Standard names are from:
http://cfconventions.org/Data/cf-standard-names/71/build/cf-standard-name-table.html

TODO: will expand to include definition (code) as well.
"""
struct DiagnosticVariable
    short::String
    units::String
    long::String
    standard::String
end
const Variables = OrderedDict{String, DiagnosticVariable}()

"""
    setup_variables()

Called at module initialization to define all currently defined diagnostic
variables.
"""
function setup_variables()
    Variables["u"] =
        DiagnosticVariable("u", "m s**-1", "zonal wind", "eastward_wind")
    Variables["v"] =
        DiagnosticVariable("v", "m s**-1", "meridional wind", "northward_wind")
    Variables["w"] = DiagnosticVariable(
        "w",
        "m s**-1",
        "vertical wind",
        "upward_air_velocity",
    )
    Variables["rho"] =
        DiagnosticVariable("rho", "kg m**-3", "air density", "air_density")
    Variables["temp"] =
        DiagnosticVariable("temp", "K", "air temperature", "air_temperature")
    Variables["pres"] =
        DiagnosticVariable("pres", "Pa", "air pressure", "air_pressure")
    Variables["thd"] = DiagnosticVariable(
        "thd",
        "K",
        "dry potential temperature",
        "air_potential_temperature",
    )
    Variables["thv"] = DiagnosticVariable(
        "thv",
        "K",
        "virtual potential temperature",
        "virtual_potential_temperature",
    )
    Variables["et"] = DiagnosticVariable(
        "et",
        "J kg**-1",
        "total specific energy",
        "specific_dry_energy_of_air",
    )
    Variables["ei"] = DiagnosticVariable(
        "ei",
        "J kg**-1",
        "internal specific energy",
        "internal_energy",
    )
    Variables["ht"] = DiagnosticVariable(
        "ht",
        "J kg**-1",
        "specific enthalpy based on total energy",
        "",
    )
    Variables["hi"] = DiagnosticVariable(
        "hi",
        "J kg**-1",
        "specific enthalpy based on internal energy",
        "atmosphere_enthalpy_content",
    )
    Variables["vort"] = DiagnosticVariable(
        "vort",
        "s**-1",
        "vertical component of relative velocity",
        "atmosphere_relative_velocity",
    )
end
