# Atmospheric EDMF parameterization profiles

```@meta
CurrentModule = ClimateMachine
```

Several EDMF related profiles are to be plotted here

## Usage

Using a profile involves passing two arguments:

 - `param_set` a parameter set, from [CLIMAParameters.jl](https://github.com/CliMA/CLIMAParameters.jl)
 - `max_Grad_Ri` maximum gradient Richarson Number (indepndent, non-dimensional variable for the turbulent Prantl number)

to one of the EDMF profile constructors.

### TurbulentPrantlNumberProfile

```@example
using UnPack
using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using ClimateMachine.Atmos: AtmosModel
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics
using ClimateMachine.BalanceLaws
using ClimateMachine.TurbulenceConvection
include(joinpath(clima_dir, "test", "Atmos", "EDMF", "closures", "turbulence_functions.jl"))
include(joinpath(clima_dir, "test", "Atmos", "EDMF", "edmf_model.jl"))
using Plots
FT = Float64;
Grad_Ri = range(FT(-1), stop = 10 , length = 100);
ml = MixingLengthModel{FT}();
Pr_t = turbulent_Prandtl_number.(Ref(ml.Pr_n), Grad_Ri, Ref(ml.ω_pr))
p1 = plot(Grad_Ri, Pr_t, xlabel=" gradient Richardson number");
plot(p1, title="turbulent Prantl number")
savefig("Pr_t.svg")
```
![](Pr_t.svg)
