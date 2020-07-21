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
    Variables["u"] = DiagnosticVariable(
        "u",
        var_attrib("m s**-1", "zonal wind", "eastward_wind"),
    )
    Variables["v"] = DiagnosticVariable(
        "v",
        var_attrib("m s**-1", "meridional wind", "northward_wind"),
    )
    Variables["w"] = DiagnosticVariable(
        "w",
        var_attrib("m s**-1", "vertical wind", "upward_air_velocity"),
    )
    Variables["rho"] = DiagnosticVariable(
        "rho",
        var_attrib("kg m**-3", "air density", "air_density"),
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
            "J kg**-1",
            "total specific energy",
            "specific_dry_energy_of_air",
        ),
    )
    Variables["ei"] = DiagnosticVariable(
        "ei",
        var_attrib("J kg**-1", "internal specific energy", "internal_energy"),
    )
    Variables["ht"] = DiagnosticVariable(
        "ht",
        var_attrib("J kg**-1", "specific enthalpy based on total energy", ""),
    )
    Variables["hi"] = DiagnosticVariable(
        "hi",
        var_attrib(
            "J kg**-1",
            "specific enthalpy based on internal energy",
            "atmosphere_enthalpy_content",
        ),
    )
    Variables["vort"] = DiagnosticVariable(
        "vort",
        var_attrib(
            "s**-1",
            "vertical component of relative velocity",
            "atmosphere_relative_velocity",
        ),
    )
end
