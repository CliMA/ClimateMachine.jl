"""
    ConfigTypes

Module containing ClimateMachine configuration types.
"""
module ConfigTypes

export ClimateMachineConfigType,
    AtmosLESConfigType,
    AtmosGCMConfigType,
    OceanBoxGCMConfigType,
    SingleStackConfigType

abstract type ClimateMachineConfigType end
struct AtmosLESConfigType <: ClimateMachineConfigType end
struct AtmosGCMConfigType <: ClimateMachineConfigType end
struct OceanBoxGCMConfigType <: ClimateMachineConfigType end
struct SingleStackConfigType <: ClimateMachineConfigType end

end
