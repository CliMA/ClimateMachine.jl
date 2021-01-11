module SurfaceFlow

using DocStringExtensions
using UnPack
using ..Land
using ..VariableTemplates
using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    prognostic_vars,
    flux,
    source,
    precompute,
    eq_tends,
    vars_state,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux

using ...DGMethods: LocalGeometry, DGModel
using StaticArrays: SVector

export OverlandFlowModel,
    NoSurfaceFlowModel,
    surface_boundary_flux!,
    surface_boundary_state!,
    calculate_velocity,
    Precip,
    VolumeAdvection,
    SurfaceWaterHeight

"""
    SurfaceWaterHeight <: AbstractPrognosticVariable

The prognostic variable type for the 2d overland flow model. Used only for
dispatching on.
"""
struct SurfaceWaterHeight <: AbstractPrognosticVariable end


"""
    NoSurfaceFlowModel <: BalanceLaw

The default surface flow model, which does not add any prognostic variables
to the land model and therefore does not model surface flow.
"""
struct NoSurfaceFlowModel <: BalanceLaw end

"""
    OverlandFlowModel{Sx,Sy,M} <: BalanceLaw
    
The 2D overland flow model, with a prognostic variable
equal to the height of the surface water.

This model simulates the depth-averaged shallow water equation under the
kinematic approximation, and employs Manning's relationship to relate
velocity to the height of the water.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct OverlandFlowModel{Sx, Sy, M} <: BalanceLaw
    "Slope in x direction; field(x,y), unitless"
    slope_x::Sx
    "Slope in y direction; field(x,y), unitless"
    slope_y::Sy
    "Mannings coefficient; field(x,y), units of s/m^(1/3)"
    mannings::M
end

function OverlandFlowModel(
    slope_x::Function,
    slope_y::Function;
    mannings::Function = (x, y) -> convert(eltype(x), 0.03),
)
    args = (slope_x, slope_y, mannings)
    return OverlandFlowModel{typeof.(args)...}(args...)
end


"""
    calculate_velocity(surface, x::Real, y::Real, h::Real)

Given the surface flow model, calculate the velocity of the flow
based on height, slope, and Manning's coefficient.
"""
function calculate_velocity(surface, x::Real, y::Real, h::Real)
    FT = eltype(h)
    sx = FT(surface.slope_x(x, y))
    sy = FT(surface.slope_y(x, y))
    mannings_coeff = FT(surface.mannings(x, y))
    coeff = h^FT(2 / 3) / mannings_coeff
    #The velocity direction is opposite the slope vector (∂_x z, ∂_y z)
    return SVector(
        -sign(sx) * coeff * sqrt(abs(sx)),
        -sign(sy) * coeff * sqrt(abs(sy)),
        zero(FT),
    )
end

vars_state(surface::OverlandFlowModel, st::Prognostic, FT) = @vars(height::FT)

function Land.land_init_aux!(
    land::LandModel,
    surface::Union{NoSurfaceFlowModel, OverlandFlowModel},
    aux,
    geom::LocalGeometry,
) end

function Land.land_nodal_update_auxiliary_state!(
    land::LandModel,
    surface::Union{NoSurfaceFlowModel, OverlandFlowModel},
    state,
    aux,
    t,
) end

"""
    VolumeAdvection <: TendencyDef{Flux{FirstOrder}}

A first order flux type for the overland flow model.
"""
struct VolumeAdvection <: TendencyDef{Flux{FirstOrder}} end


"""
    flux(::SurfaceWaterHeight, ::VolumeAdvection, land::LandModel, args,)

A first order flux method for the OverlandFlow model, adding in advection of water volume.
"""
function flux(::SurfaceWaterHeight, ::VolumeAdvection, land::LandModel, args)
    @unpack state, aux = args
    x = aux.x
    y = aux.y
    height = max(state.surface.height, eltype(aux)(0.0))
    v = calculate_velocity(land.surface, x, y, height)
    return height * v
end

# Boundary Conditions

# General case - to be used with bc::NoBC
function surface_boundary_flux!(
    nf,
    bc::Land.AbstractBoundaryConditions,
    m::Union{NoSurfaceFlowModel, OverlandFlowModel},
    land::LandModel,
    _...,
) end

function surface_boundary_state!(
    nf,
    bc::Land.AbstractBoundaryConditions,
    m::Union{NoSurfaceFlowModel, OverlandFlowModel},
    land::LandModel,
    _...,
) end

"""
    surface_boundary_flux!(
        nf,
        bc::Land.Dirichlet,
        model::OverlandFlowModel,
        land::LandModel,
        _...,
    )

The surface boundary flux function for the OverlandFlow model, which
does nothing if Dirichlet conditions are chosen in order to not
overconstrain the solution.
"""
function surface_boundary_flux!(
    nf,
    bc::Land.Dirichlet,
    model::OverlandFlowModel,
    land::LandModel,
    _...,
) end

"""
    surface_boundary_state!(
        nf,
        bc::Land.Dirichlet,
        model::OverlandFlowModel,
        land::LandModel,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        t,
        _...,
    )

The surface boundary state function for the OverlandFlow model when
Dirichlet conditions are chosen. This should be equivalent to 
outflow boundary conditions, also referred to as gradient outlet
conditions.
"""
function surface_boundary_state!(
    nf,
    bc::Land.Dirichlet,
    model::OverlandFlowModel,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    bc_function = bc.state_bc
    state⁺.surface.height = bc_function(aux⁻, t)
end

"""
    Precip <: TendencyDef{Source}

A source term for overland flow where a prescribed net precipitation
drives the overland flow.
"""
struct Precip{FT, F} <: TendencyDef{Source}
    precip::F

    function Precip{FT}(precip::F) where {FT, F}
        new{FT, F}(precip)
    end
end

function (p::Precip{FT})(x, y, t) where {FT}
    FT(p.precip(x, y, t))
end


prognostic_vars(::Precip) = (SurfaceWaterHeight(),)


precompute(source_type::Precip, land::LandModel, args, tt::Source) =
    NamedTuple()

function source(::SurfaceWaterHeight, s::Precip, land::LandModel, args)
    @unpack aux, t = args
    return s(aux.x, aux.y, t)
end

end
