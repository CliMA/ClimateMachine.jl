"""
    Thermodynamics

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
`param_set` is an argument for many Thermodynamics functions.
"""
module Thermodynamics

using DocStringExtensions
using RootSolvers
using RootSolvers: AbstractTolerance
using KernelAbstractions: @print

using CLIMAParameters: AbstractParameterSet
using CLIMAParameters.Planet
const APS = AbstractParameterSet

# Allow users to skip error on non-convergence
# by importing:
# ```julia
# import Thermodynamics
# Thermodynamics.error_on_non_convergence() = false
# ```
# Error on convergence must be the default
# behavior because this can result in printing
# very large logs resulting in CI to seemingly hang.
error_on_non_convergence() = true

# Allow users to skip printing warnings on non-convergence
print_warning() = true

@inline q_pt_0(::Type{FT}) where {FT} = PhasePartition{FT}(FT(0), FT(0), FT(0))

include("states.jl")
include("relations.jl")
include("isentropic.jl")

end #module Thermodynamics.jl
