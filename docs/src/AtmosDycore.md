# AtmosDycore

```@meta
CurrentModule = CLIMA.AtmosDycore
```

```@docs
getrhsfunction
```

## `VanillaAtmosDiscretizations`

A discretization adds additional information for the atmosphere problem.

```@docs
VanillaAtmosDiscretizations.VanillaAtmosDiscretization
VanillaAtmosDiscretizations.estimatedt
```

## `AtmosStateArray`

Storage for the state of a discretization.

```@docs
AtmosStateArrays.AtmosStateArray
AtmosStateArrays.postrecvs!
AtmosStateArrays.startexchange!
AtmosStateArrays.finishexchange!
```
