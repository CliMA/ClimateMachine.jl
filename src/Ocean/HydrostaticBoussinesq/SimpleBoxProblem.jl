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
struct SimpleBoxProblem{T} <: AbstractSimpleBoxProblem
  Lˣ::T
  Lʸ::T
  H::T
end

##########################
# Homogenous wind stress #
# Constant temperature   #
##########################

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

@inline function ocean_boundary_state!(m::HBModel, p::HomogeneousBox, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloorFreeSlip(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurfaceStressNoForcing(), x...)
  end
end

# aux is Filled afer the state
function ocean_init_aux!(m::HBModel, P::HomogeneousBox, A, geom)
  FT = eltype(A)
  @inbounds y = geom.coord[2]

  Lʸ = P.Lʸ
  τₒ = P.τₒ
  fₒ = P.fₒ
  β  = P.β

  A.τ  = -τₒ * cos(y * π / Lʸ)
  A.f  =  fₒ + β * y

  A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
  A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

function ocean_init_state!(p::HomogeneousBox, state, aux, coords, t)
  @inbounds z = coords[3]
  @inbounds H = p.H

  state.u = @SVector [rand(),rand()]
  state.η = 0
  state.θ = 20
end

##########################
# Homogenous wind stress #
# Temperature forcing    #
##########################

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

@inline function ocean_boundary_state!(m::HBModel, p::OceanGyre, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloorNoSlip(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
  end
end

# A is Filled afer the state
function ocean_init_aux!(m::HBModel, P::OceanGyre, A, geom)
  FT = eltype(A)
  @inbounds y = geom.coord[2]

  Lʸ = P.Lʸ
  τₒ = P.τₒ
  fₒ = P.fₒ
  β  = P.β
  θᴱ = P.θᴱ

  A.τ  = -τₒ * cos(y * π / Lʸ)
  A.f  =  fₒ + β * y
  A.θʳ =  θᴱ * (1 - y / Lʸ)

  A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
  A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

function ocean_init_state!(P::OceanGyre, Q, A, coords, t)
  @inbounds z = coords[3]
  @inbounds H = P.H

  Q.u = @SVector [0,0]
  Q.η = 0
  Q.θ = 9 + 8z/H
end

end
