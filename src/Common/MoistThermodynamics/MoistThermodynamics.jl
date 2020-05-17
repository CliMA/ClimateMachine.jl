"""
    MoistThermodynamics

Moist thermodynamic functions, e.g., for air pressure (atmosphere equation
of state), latent heats of phase transitions, saturation vapor pressures, and
saturation specific humidities.


## AbstractParameterSet's

Many functions defined in this module rely on CLIMAParameters.jl.
CLIMAParameters.jl defines several functions (e.g., many planet
parameters). For example, to compute the mole-mass ratio:

```julia
using CLIMAParameters.Planet: molmass_ratio
using CLIMAParameters: AbstractEarthParameterSet
struct EarthParameterSet <: AbstractEarthParameterSet end
param_set = EarthParameterSet()
_molmass_ratio = molmass_ratio(param_set)
```

Because these parameters are widely used throughout this module,
`param_set` is an argument for many MoistThermodynamics functions.
"""
module MoistThermodynamics

using DocStringExtensions
using RootSolvers
using RootSolvers: AbstractTolerance

using CLIMAParameters: AbstractParameterSet
using CLIMAParameters.Planet
const APS = AbstractParameterSet

@inline q_pt_0(::Type{FT}) where {FT} = PhasePartition{FT}(FT(0), FT(0), FT(0))

include("states.jl")
include("relations.jl")
include("isentropic.jl")

end #module MoistThermodynamics.jl
