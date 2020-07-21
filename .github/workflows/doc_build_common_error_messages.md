# Documenter common warning/error messages:

## General notes
 - Changing order of API/HowToGuides does not fix unresolved path issues

## no doc found for reference
```
┌ Warning: no doc found for reference '[`BatchedGeneralizedMinimalResidual`](@ref)' in src/HowToGuides/Numerics/SystemSolvers/IterativeSolvers.md.
└ @ Documenter.CrossReferences ~/.julia/packages/Documenter/PLD7m/src/CrossReferences.jl:160
```
 - Missing entry in ```@docs ``` in API


## Reference could not be found
```
┌ Warning: reference for 'ClimateMachine.ODESolvers.solve!' could not be found in src\APIs\Driver\index.md.
└ @ Documenter.CrossReferences C:\Users\kawcz\.julia\packages\Documenter\PLD7m\src\CrossReferences.jl:104
```
 - Missing doc string?

## invalid local link: unresolved path
```
┌ Warning: invalid local link: unresolved path in APIs/Atmos/AtmosModel.md
│   link.text =
│    1-element Array{Any,1}:
│     Markdown.Code("", "FlatOrientation")
│   link.url = "@ref"
```
 - Missing entry in ```@docs ``` for FlatOrientation in API OR
 - The "code" in the reference must be to actual code and not arbitrary text

## unable to get the binding
```
┌ Warning: unable to get the binding for 'ClimateMachine.Atmos.AtmosModel.NoOrientation' in `@docs` block in src/APIs/Atmos/AtmosModel.md:9-14 from expression ':(ClimateMachine.Atmos.AtmosModel.NoOrientation)' in module ClimateMachine
│ ```@docs
│ ClimateMachine.Atmos.AtmosModel.NoOrientation
...
│ ```
...
```
 - `ClimateMachine.Atmos.AtmosModel.NoOrientation` should be `ClimateMachine.Atmos.NoOrientation`

