# Meshing Stuff

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
