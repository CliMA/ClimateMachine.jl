# Meshing Stuff

```@meta
CurrentModule = CLIMA.Mesh
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

### Constructors

```@docs
Interpolation.InterpolationBrick(grid::DiscontinuousSpectralElementGrid{FT},
                       xbnd::Array{FT,2},
                        x1g::AbstractArray{FT,1},
                        x2g::AbstractArray{FT,1},
                        x3g::AbstractArray{FT,1}) where FT <: AbstractFloat
Interpolation.InterpolationCubedSphere(grid::DiscontinuousSpectralElementGrid, 
                       vert_range::AbstractArray{FT}, 
                             nhor::Int,
                          lat_grd::AbstractArray{FT,1}, 
                         long_grd::AbstractArray{FT,1}, 
                          rad_grd::AbstractArray{FT}) where {FT <: AbstractFloat}
```

### Functions

```@docs
Interpolation.interpolate_local!(intrp_brck::InterpolationBrick{FT}, 
                                         sv::AbstractArray{FT}, 
                                          v::AbstractArray{FT}) where {FT <: AbstractFloat}
Interpolation.interpolate_local!(intrp_cs::InterpolationCubedSphere{FT}, 
                                       sv::AbstractArray{FT}, 
                                        v::AbstractArray{FT}) where {FT <: AbstractFloat}                                          
Interpolation.project_cubed_sphere!(intrp_cs::InterpolationCubedSphere{FT}, 
                                           v::AbstractArray{FT}, 
                                        uvwi::Tuple{Int,Int,Int}) where {FT <: AbstractFloat}                                        
Interpolation.accumulate_interpolated_data!(intrp::InterpolationTopology, 
                                               iv::AbstractArray{FT,2}, 
                                              fiv::AbstractArray{FT,4}) where {FT <: AbstractFloat}                                        
```
