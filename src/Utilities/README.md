# CLIMA Utilities
Functions to be shared across model components, e.g., for thermodynamics.

## The MoistThermodynamics Module

The `MoistThermodynamics` module provides all thermodynamic functions needed for the atmosphere. The functions are general for a moist atmosphere; the special case of a dry atmosphere is obtained for zero specific humidities (or simply by omitting the optional specific humidity arguments in functions that are needed for a dry atmosphere)
