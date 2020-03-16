############################
# Basic box problem        #
# Set up dimensions of box #
############################
abstract type AbstractSimpleBoxProblem <: AbstractHydrostaticBoussinesqProblem end

"""
    SimpleBoxProblem <: AbstractSimpleBoxProblem

Stub structure with the dimensions of the box.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
"""
struct SimpleBoxProblem{T} <: AbstractSimpleBoxProblem
  Lˣ::T
  Lʸ::T
  H::T
end

"""
    ocean_init_aux!(::HBModel, ::AbstractSimpleBoxProblem)

save y coordinate for computing coriolis, wind stress, and sea surface temperature

# Arguments
- `m`: model object to dispatch on and get viscosities and diffusivities
- `p`: problem object to dispatch on and get additional parameters
- `A`: auxiliary state vector
- `geom`: geometry stuff
"""
function ocean_init_aux!(m::HBModel, p::AbstractSimpleBoxProblem, A, geom)
  FT = eltype(A)
  @inbounds A.y = geom.coord[2]

  # not sure if this is needed but getting weird intialization stuff
  A.w = 0
  A.pkin = 0
  A.wz0 = 0

  return nothing
end

##########################
# Homogenous wind stress #
# Constant temperature   #
##########################

"""
    HomogeneousBox <: AbstractSimpleBoxProblem

Container structure for a simple box problem with wind-stress and coriolis force.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
τₒ = maximum value of wind-stress (amplitude)
fₒ = first coriolis parameter (constant term)
β  = second coriolis parameter (linear term)
"""
struct HomogeneousBox{T} <: AbstractSimpleBoxProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  function HomogeneousBox{FT}(Lˣ, Lʸ, H;
                                    τₒ = FT(1e-1),  # (m/s)^2
                                    ) where {FT <: AbstractFloat}
    return new{FT}(Lˣ, Lʸ, H, τₒ)
  end
end

"""
    ocean_boundary_state!(::HBModel, ::HomogeneousBox)

dispatches to the correct boundary condition based on bctype
bctype 1 => Coastline => Apply No Slip BC conditions
bctype 2 => OceanFloor => Apply Free Slip BC conditions
bctype 3 => OceanSurface => Apply Windstress but not temperature forcing
"""
@inline function ocean_boundary_state!(m::HBModel, p::HomogeneousBox, bctype, x...)
  if bctype == 1
    return ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    return ocean_boundary_state!(m, OceanFloorFreeSlip(), x...)
  elseif bctype == 3
    return ocean_boundary_state!(m, OceanSurfaceStressNoForcing(), x...)
  end
end

"""
    ocean_init_state!(::HomogeneousBox)

initialize u,v with random values, η with 0, and θ with a constant (20)

# Arguments
- `p`: HomogeneousBox problem object, used to dispatch on
- `Q`: state vector
- `A`: auxiliary state vector, not used
- `coords`: the coordidinates, not used
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(p::HomogeneousBox, Q, A, coords, t)
  Q.u = @SVector [rand(),rand()]
  Q.η = 0
  Q.θ = 20

  return nothing
end

"""
    velocity_flux(::HomogeneousBox)

jet stream like windstress

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
"""
@inline velocity_flux(p::HomogeneousBox, y, ρ) = -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

##########################
# Homogenous wind stress #
# Temperature forcing    #
##########################

"""
    OceanGyre <: AbstractSimpleBoxProblem

Container structure for a simple box problem with wind-stress, coriolis force, and temperature forcing.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
τₒ = maximum value of wind-stress (amplitude)
λʳ = temperature relaxation penetration constant (meters / second)
θᴱ = maximum surface temperature
"""
struct OceanGyre{T} <: AbstractSimpleBoxProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  λʳ::T
  θᴱ::T
  function OceanGyre{FT}(Lˣ, Lʸ, H;
                         τₒ = FT(1e-1),       # (m/s)^2
                         λʳ = FT(4 // 86400), # m / s
                         θᴱ = FT(25),         # K
                         ) where {FT <: AbstractFloat}
    return new{FT}(Lˣ, Lʸ, H, τₒ, λʳ, θᴱ)
  end
end

"""
    ocean_boundary_state(::HBModel, ::OceanGyre)

dispatches to the correct boundary condition based on bctype
bctype 1 => Coastline => Apply No Slip BC conditions
bctype 2 => OceanFloor => Apply Free Slip BC conditions
bctype 3 => OceanSurface => Apply wind-stress and  temperature forcing
"""
@inline function ocean_boundary_state!(m::HBModel, p::OceanGyre, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloorNoSlip(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
  end
end

"""
    ocean_init_state!(::OceanGyre)

initialize u,v,η with 0 and θ linearly distributed between 9 at z=0 and 1 at z=H

# Arguments
- `p`: OceanGyre problem object, used to dispatch on and obtain ocean height H
- `Q`: state vector
- `A`: auxiliary state vector, not used
- `coords`: the coordidinates
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(p::OceanGyre, Q, A, coords, t)
  @inbounds z = coords[3]
  @inbounds H = p.H

  Q.u = @SVector [0,0]
  Q.η = 0
  Q.θ = 9 + 8z/H

  return nothing
end

"""
    velocity_flux(::OceanGyre)

jet stream like windstress

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
"""
@inline velocity_flux(p::OceanGyre, y, ρ) = - (p.τₒ / ρ) * cos(y * π / p.Lʸ)

"""
    temperature_flux(::OceanGyre)

cool-warm north-south linear temperature gradient

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
- `θ`: temperature within element on boundary
"""
@inline function temperature_flux(p::OceanGyre, y, θ)
  θʳ =  p.θᴱ * (1 - y / p.Lʸ)
  return p.λʳ * (θʳ - θ)
end
