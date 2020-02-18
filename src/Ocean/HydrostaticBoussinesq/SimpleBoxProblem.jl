module SimpleBox

export SimpleBoxProblem, HomogeneousBox, OceanGyre

using StaticArrays
using CLIMA.HydrostaticBoussinesq

import CLIMA.HydrostaticBoussinesq: ocean_init_aux!, ocean_init_state!,
                                    ocean_boundary_state!,
                                    CoastlineFreeSlip, CoastlineNoSlip,
                                    OceanFloorFreeSlip, OceanFloorNoSlip,
                                    OceanSurfaceNoStressNoForcing,
                                    OceanSurfaceStressNoForcing,
                                    OceanSurfaceNoStressForcing,
                                    OceanSurfaceStressForcing

HBModel = HydrostaticBoussinesqModel

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
  fₒ::T
  β::T
  function HomogeneousBox{FT}(Lˣ, Lʸ, H;
                                    τₒ = FT(1e-1),  # (m/s)^2
                                    fₒ = FT(1e-4),  # Hz
                                    β  = FT(1e-11), # Hz / m)
                                    ) where {FT <: AbstractFloat}
    return new{FT}(Lˣ, Lʸ, H, τₒ, fₒ, β)
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
end

"""
    ocean_init_aux!(::HBModel, ::HomgoneousBox)

initiaze auxiliary states
jet stream like windstress
northern hemisphere coriolis
cool-warm north-south linear temperature gradient

# Arguments
- `m`: model object to dispatch on and get viscosities and diffusivities
- `p`: problem object to dispatch on and get additional parameters
- `A`: auxiliary state vector
- `geom`: geometry stuff
"""
# aux is Filled afer the state
function ocean_init_aux!(m::HBModel, p::HomogeneousBox, A, geom)
  FT = eltype(A)
  @inbounds y = geom.coord[2]

  Lʸ = p.Lʸ
  τₒ = p.τₒ
  fₒ = p.fₒ
  β  = p.β

  A.τ  = -τₒ * cos(y * π / Lʸ)
  A.f  =  fₒ + β * y

  A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
  A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

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
fₒ = first coriolis parameter (constant term)
β  = second coriolis parameter (linear term)
λʳ = temperature relaxation penetration constant (meters / second)
θᴱ = maximum surface temperature
"""
struct OceanGyre{T} <: AbstractSimpleBoxProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  fₒ::T
  β::T
  λʳ::T
  θᴱ::T
  function OceanGyre{FT}(Lˣ, Lʸ, H;
                         τₒ = FT(1e-1),       # (m/s)^2
                         fₒ = FT(1e-4),       # Hz
                         β  = FT(1e-11),      # Hz / m
                         λʳ = FT(4 // 86400), # m / s
                         θᴱ = FT(25),         # K
                         ) where {FT <: AbstractFloat}
    return new{FT}(Lˣ, Lʸ, H, τₒ, fₒ, β, λʳ, θᴱ)
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
end

"""
    ocean_init_aux!(::HBModel, ::OceanGyre)

initiaze auxiliary states
jet stream like windstress
northern hemisphere coriolis
cool-warm north-south linear temperature gradient

# Arguments
- `m`: model object to dispatch on and get viscosities and diffusivities
- `p`: problem object to dispatch on and get additional parameters
- `A`: auxiliary state vector
- `geom`: geometry stuff
"""
# aux is Filled afer the state
function ocean_init_aux!(m::HBModel, p::OceanGyre, A, geom)
  FT = eltype(A)
  @inbounds y = geom.coord[2]

  Lʸ = p.Lʸ
  τₒ = p.τₒ
  fₒ = p.fₒ
  β  = p.β
  θᴱ = p.θᴱ

  A.τ  = -τₒ * cos(y * π / Lʸ)
  A.f  =  fₒ + β * y
  A.θʳ =  θᴱ * (1 - y / Lʸ)

  A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
  A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

end
