export Dirichlet,
    Neumann, GeneralBoundaryConditions, SurfaceDrivenWaterBoundaryConditions

abstract type AbstractBoundaryFunctions end

"""
    struct Dirichlet{Fs, Fb} <: AbstractBoundaryFunctions

A concrete type to hold the surface state and bottom state variable 
values/functions, if Dirichlet boundary conditions are desired.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct Dirichlet{Fs, Fb} <: AbstractBoundaryFunctions
    "Surface state boundary condition"
    surface_state::Fs = nothing
    "Bottom state boundary condition"
    bottom_state::Fb = nothing
end

"""
    struct Neumann{Fs, Fb} <: AbstractBoundaryFunctions

A concrete type to hold the surface and/or bottom diffusive flux 
values/functions, if Neumann boundary conditions are desired.

Note that these are intended to be scalar values. In the boundary_state!
functions, they are multiplied by the `ẑ` vector (i.e. the normal vector `n̂`
to the domain at the upper boundary, and -`n̂` at the lower boundary. These
normal vectors point out of the domain.)

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct Neumann{Fs, Fb} <: AbstractBoundaryFunctions
    "Surface flux boundary condition"
    surface_flux::Fs = nothing
    "Bottom flux boundary condition"
    bottom_flux::Fb = nothing
end



"""
   AbstractBoundaryConditions 
"""
abstract type AbstractBoundaryConditions end


"""
    GeneralBoundaryConditions{D, N} <: AbstractBoundaryConditions

A concrete example of the abstract type `AbstractBoundaryConditions`; to be used
when the user wishes to supply specific functions of space and time for
boundary conditions, at either the top or bottom of the domain, and either
of type Neumann or Dirichlet.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GeneralBoundaryConditions{D, N} <: AbstractBoundaryConditions
    "Place to store Dirichlet Boundary conditions"
    dirichlet_bc::D
    "Place to store Neumann Boundary conditions"
    neumann_bc::N
end


"""
    GeneralBoundaryConditions(
        dirichlet_bc::AbstractBoundaryFunctions = Dirichlet(),
        neumann_bc::AbstractBoundaryFunctions = Neumann(),
    )

Constructor for the GeneralBoundaryConditions object. The default
is `nothing` boundary functions for the top and bottom of the domain,
for both Dirichlet and Neumann types. The user must change these in order
to apply boundary conditions.
"""
function GeneralBoundaryConditions(
    dirichlet_bc::AbstractBoundaryFunctions = Dirichlet(),
    neumann_bc::AbstractBoundaryFunctions = Neumann(),
)
    args = (dirichlet_bc, neumann_bc)
    return GeneralBoundaryConditions{typeof.(args)...}(args...)
end



"""
    SurfaceDrivenWaterBoundaryConditions{FT, PD, RD} <: AbstractBoundaryConditions

A concrete example of AbstractBoundaryConditions; this is to be used when the user wishes to 
apply physical fluxes of water at the top of the domain (according to precipitation, runoff, and 
evaporation rates) and zero flux at the bottom of the domain.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SurfaceDrivenWaterBoundaryConditions{FT, PD, RD} <:
       AbstractBoundaryConditions where {FT, PD, RD}
    "Precipitation model"
    precip_model::PD
    "Runoff model"
    runoff_model::RD
end


"""
    SurfaceDrivenWaterBoundaryConditions(
        ::Type{FT};
        precip_model::AbstractPrecipModel{FT} = DrivenConstantPrecip{FT}(),
        runoff_model::AbstractSurfaceRunoffModel{FT} = NoRunoff(),
    ) where {FT}

Constructor for the SurfaceDrivenWaterBoundaryConditions object. The default
is a constant precipitation rate on the subgrid scale, and no runoff.
"""
function SurfaceDrivenWaterBoundaryConditions(
    ::Type{FT};
    precip_model::AbstractPrecipModel{FT} = DrivenConstantPrecip{FT}(
        (t) -> (0.0),
    ),
    runoff_model::AbstractSurfaceRunoffModel = NoRunoff(),
) where {FT}
    args = (precip_model, runoff_model)
    return SurfaceDrivenWaterBoundaryConditions{FT, typeof.(args)...}(args...)
end
