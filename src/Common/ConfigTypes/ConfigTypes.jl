"""
    ConfigTypes

Module containing ClimateMachine configuration types.
"""
module ConfigTypes

export ClimateMachineConfigType,
    AtmosLESConfigType, AtmosGCMConfigType, OceanBoxGCMConfigType

abstract type ClimateMachineConfigType end
struct AtmosLESConfigType <: ClimateMachineConfigType end
struct AtmosGCMConfigType <: ClimateMachineConfigType end
struct OceanBoxGCMConfigType <: ClimateMachineConfigType end

end
