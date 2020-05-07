# Meshing Stuff

```@meta
CurrentModule = ClimateMachine.Mesh
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

### Constructors

```@docs
Topologies.BrickTopology(mpicomm, Nelems)
Topologies.StackedBrickTopology(mpicomm, elemrange)
Topologies.CubedShellTopology(mpicomm, Neside, T)
Topologies.StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
```

### Functions

```@docs
Topologies.cubedshellmesh
Topologies.cubedshellwarp
Topologies.hasboundary
```

## `Grids`

Grids specify the approximation within each element, and any necessary warping.

```@docs
Grids.DiscontinuousSpectralElementGrid
```

## `Filters`

There are methods used to cleanup state vectors.

```@docs
Filters.CutoffFilter
Filters.ExponentialFilter
Filters.TMARFilter
```

## `Interpolation`

### Types

```@docs
Interpolation.InterpolationBrick
Interpolation.InterpolationCubedSphere
```

### Functions

```@docs
Interpolation.interpolate_local!
Interpolation.project_cubed_sphere!
Interpolation.accumulate_interpolated_data!
```
