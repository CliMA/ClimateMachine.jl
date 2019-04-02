# Meshing Stuff

```@meta
CurrentModule = CLIMA
```

## `Topologies`

Topologies encode the connectivity of the elements, spatial domain interval and MPI
communication.

### Types

```@docs
Topologies.AbstractTopology
Topologies.BoxElementTopology
Topologies.BrickTopology
Topologies.StackedBrickTopology
Topologies.CubedShellTopology
Topologies.StackedCubedSphereTopology
```

### Functions

```@docs
Topologies.cubedshellmesh
Topologies.cubedshellwarp
```

## `Grids`

Grids specify the approximation within each element, and any necessary warping.

```@docs
Grids.DiscontinuousSpectralElementGrid
```
