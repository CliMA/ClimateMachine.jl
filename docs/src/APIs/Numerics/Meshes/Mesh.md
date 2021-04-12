# Meshing Stuff

```@meta
CurrentModule = ClimateMachine.Mesh
```

## Topologies

Topologies encode the connectivity of the elements, spatial domain interval and MPI
communication.

### Types

```@docs
Topologies.AbstractTopology
Topologies.BoxElementTopology
Topologies.BrickTopology
Topologies.StackedBrickTopology
Topologies.CubedShellTopology
Topologies.AnalyticalTopography
Topologies.NoTopography
Topologies.DCMIPMountain
Topologies.StackedCubedSphereTopology
Topologies.SingleExponentialStretching
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
Topologies.cubed_sphere_warp
Topologies.cubed_sphere_unwarp
Topologies.equiangular_cubed_sphere_warp
Topologies.equiangular_cubed_sphere_unwarp
Topologies.equidistant_cubed_sphere_warp
Topologies.equidistant_cubed_sphere_unwarp
Topologies.hasboundary
Topologies.compute_lat_long
Topologies.cubedshelltopowarp
Topologies.grid1d
```

## Geometry
```@docs
Geometry.LocalGeometry
Geometry.lengthscale
Geometry.resolutionmetric
Geometry.lengthscale_horizontal
```

## Brick Mesh

```@docs
BrickMesh.partition
BrickMesh.brickmesh
BrickMesh.connectmesh
BrickMesh.connectmeshfull
BrickMesh.centroidtocode
```

## Grids

Grids specify the approximation within each element, and any necessary warping.

```@docs
Grids.get_z
Grids.referencepoints
Grids.min_node_distance
Grids.SpectralElementGrid
```

## DSS

Computes the direct stiffness summation of fields in the MPIStateArray.

```@docs
DSS.dss!
DSS.dss_vertex!
DSS.dss_edge!
DSS.dss_face!
```

## Filters

There are methods used to cleanup state vectors.

```@docs
Filters.CutoffFilter
Filters.BoydVandevenFilter
Filters.ExponentialFilter
Filters.TMARFilter
Filters.apply!
Filters.apply_async!
```

## Interpolation

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
