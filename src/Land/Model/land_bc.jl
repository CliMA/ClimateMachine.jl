export LandDomainBC,
    LandComponentBC,
    Dirichlet,
    Neumann,
    NoBC,
    SurfaceDrivenWaterBoundaryConditions,
    CATHYWaterBoundaryConditions

"""
   AbstractBoundaryConditions 
"""
abstract type AbstractBoundaryConditions end


"""
    NoBC <: AbstractBoundaryConditions

This type is used for dispatch when no boundary condition needs
to be enforced - for example, if no prognostic variables are included
for a subcomponent, or for lateral faces when a 1D vertical setup
is used.
"""
struct NoBC <: AbstractBoundaryConditions end

"""
    Dirichlet{Fs} <: AbstractBoundaryConditions

A concrete type to hold the state variable 
function, if Dirichlet boundary conditions are desired.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Dirichlet{Fs} <: AbstractBoundaryConditions
    "state boundary condition"
    state_bc::Fs
end

"""
    Neumann{Ff} <: AbstractBoundaryConditions

A concrete type to hold the diffusive flux 
function, if Neumann boundary conditions are desired.

Note that these are intended to be scalar values. In the boundary_state!
functions, they are multiplied by the `ẑ` vector (i.e. the normal vector `n̂`
to the domain at the upper boundary, and -`n̂` at the lower boundary. These
normal vectors point out of the domain.)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Neumann{Ff} <: AbstractBoundaryConditions
    "Scalar flux boundary condition"
    scalar_flux_bc::Ff
end


"""
    SurfaceDrivenWaterBoundaryConditions{FT, PD, RD} <: AbstractBoundaryConditions

Boundary condition type to be used when the user wishes to 
apply physical fluxes of water at the top of the domain (according to precipitation, runoff, and 
evaporation rates).

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
    CATHYWaterBoundaryConditions(
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


"""
    SurfaceDrivenWaterBoundaryConditions{FT, PD, RD} <: AbstractBoundaryConditions

Boundary condition type to be used when the user wishes to 
apply physical fluxes of water at the top of the domain (according to precipitation, runoff, and 
evaporation rates).

# Fields
$(DocStringExtensions.FIELDS)
"""
struct CATHYWaterBoundaryConditions{FT, PD, RD} <:
       AbstractBoundaryConditions where {FT, PD, RD}
    "Precipitation model"
    precip_model::PD
    "Runoff model"
    runoff_model::RD
end

"""
    CATHYWaterBoundaryConditions(
        ::Type{FT};
        precip_model::AbstractPrecipModel{FT} = DrivenConstantPrecip{FT}(),
        runoff_model::AbstractSurfaceRunoffModel{FT} = NoRunoff(),
    ) where {FT}

Constructor for the SurfaceDrivenWaterBoundaryConditions object. The default
is a constant precipitation rate on the subgrid scale, and no runoff.
"""
function CATHYWaterBoundaryConditions(
    ::Type{FT};
    precip_model::AbstractPrecipModel{FT} = DrivenConstantPrecip{FT}(
        (t) -> (0.0),
    ),
    runoff_model::AbstractSurfaceRunoffModel = NoRunoff(),
) where {FT}
    args = (precip_model, runoff_model)
    return CATHYWaterBoundaryConditions{FT, typeof.(args)...}(args...)
end


"""
    LandDomainBC{TBC, BBC, LBC}

A container for the land boundary conditions, with options for surface
boundary conditions, bottom boundary conditions, or lateral face
boundary conditions.

The user should supply an instance of `LandComponentBC` for each piece, as needed.
If none is supplied, the default for is `NoBC` for each subcomponent. At a minimum, both top and bottom boundary conditions should be supplied. Whether or not to include the lateral faces depends on the configuration of the domain.
"""
Base.@kwdef struct LandDomainBC{TBC, BBC, MinXBC, MaxXBC, MinYBC, MaxYBC}
    "surface boundary conditions"
    surface_bc::TBC = LandComponentBC()
    "bottom boundary conditions"
    bottom_bc::BBC = LandComponentBC()
    "lateral boundary conditions"
    minx_bc::MinXBC = LandComponentBC()
    maxx_bc::MaxXBC = LandComponentBC()
    miny_bc::MinYBC = LandComponentBC()
    maxy_bc::MaxYBC = LandComponentBC()
end

"""
    LandComponentBC{SW, SH}

An object that holds the boundary conditions for each of the subcomponents
of the land model. 

The boundary conditions supplied should be of type `AbstractBoundaryConditions`. 
The default is `NoBC` for each component, so that the user only
needs to define the BC for the components they wish to model.
"""
Base.@kwdef struct LandComponentBC{
    SW <: AbstractBoundaryConditions,
    SH <: AbstractBoundaryConditions,
    R <: AbstractBoundaryConditions,
}
    soil_water::SW = NoBC()
    soil_heat::SH = NoBC()
    river::R = NoBC()
end


"""
    function boundary_conditions(land::LandModel)

Unpacks the `boundary_conditions` field of the land model, and
puts into the correct order based on the integers used to identify
faces, as defined in the Driver configuration.
"""
function boundary_conditions(land::LandModel)
    bc = land.boundary_conditions
    mytuple = (bc.bottom_bc, bc.surface_bc, bc.minx_bc, bc.maxx_bc, bc.miny_bc, bc.maxy_bc)
    # faces labeled integer 1,2 are bottom, top, lateral sides are 3, 4, 5, 6
    return mytuple
end



function boundary_state!(
    nf,
    bc::LandComponentBC,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)
    land_boundary_state!(
        nf,
        bc,
        land,
        state⁺,
        aux⁺,
        n,
        state⁻,
        aux⁻,
        t,
        args...,
    )
end

function land_boundary_state!(nf, bc::LandComponentBC, land, args...)
    soil_boundary_state!(nf, bc.soil_water, land.soil.water, land, args...)
    soil_boundary_state!(nf, bc.soil_heat, land.soil.heat, land, args...)
    river_boundary_state!(nf, bc.river, land.river, land, args...)
end


function boundary_state!(
    nf,
    bc::LandComponentBC,
    land::LandModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n,
    state⁻,
    diff⁻,
    aux⁻,
    t,
    args...,
)
    land_boundary_flux!(
        nf,
        bc,
        land,
        state⁺,
        diff⁺,
        aux⁺,
        n,
        state⁻,
        diff⁻,
        aux⁻,
        t,
        args...,
    )
end


function land_boundary_flux!(nf, bc::LandComponentBC, land, args...)
    soil_boundary_flux!(nf, bc.soil_water, land.soil.water, land, args...)
    soil_boundary_flux!(nf, bc.soil_heat, land.soil.heat, land, args...)
    river_boundary_flux!(nf, bc.river, land.river, land, args...)
end
