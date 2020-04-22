"""
    ConfigTypes

Module containing CLIMA configuration types.
"""
module ConfigTypes

export CLIMAConfigType,
       AtmosLESConfigType,
       AtmosGCMConfigType,
       OceanBoxGCMConfigType

abstract type CLIMAConfigType end
struct AtmosLESConfigType <: CLIMAConfigType end
struct AtmosGCMConfigType <: CLIMAConfigType end
struct OceanBoxGCMConfigType <: CLIMAConfigType end

end
