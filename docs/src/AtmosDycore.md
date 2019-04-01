# CLIMAAtmosDycore

```@meta
CurrentModule = CLIMA.CLIMAAtmosDycore
```

```@docs
getrhsfunction
solve!
```

## `Topologies`

Topologies encode the connectivity of the elements, spatial domain interval and MPI
communication.

```@docs
Topologies.BrickTopology
Topologies.StackedBrickTopology
Topologies.CubedShellTopology
Topologies.cubedshellmesh
Topologies.cubedshellwarp
Topologies.StackedCubedSphereTopology
```

## `Grids`

Grids specify the approximation within each element, and any necessary warping.

```@docs
Grids.DiscontinuousSpectralElementGrid
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

## `LSRKmethods`

Methods for time discretization.

```@docs
LSRKmethods.LSRK
LSRKmethods.updatedt!
```
