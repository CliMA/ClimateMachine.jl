abstract type OceanBoundaryCondition end

struct CoastlineFreeSlip             <: OceanBoundaryCondition end
struct CoastlineNoSlip               <: OceanBoundaryCondition end
struct OceanFloorFreeSlip            <: OceanBoundaryCondition end
struct OceanFloorNoSlip              <: OceanBoundaryCondition end
struct OceanSurfaceNoStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressNoForcing   <: OceanBoundaryCondition end
struct OceanSurfaceNoStressForcing   <: OceanBoundaryCondition end
struct OceanSurfaceStressForcing     <: OceanBoundaryCondition end
"""
COAST LINE FREE SLIP
"""
@inline function ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip,
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end


"""
COAST LINE NO SLIP
"""
@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::Rusanov,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  Q⁺.u = -Q⁻.u

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::CentralNumericalFluxGradient,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  FT = eltype(Q⁺)
  Q⁺.u = SVector(-zero(FT), -zero(FT))

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  Q⁺.u = -Q⁻.u

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

"""
OCEAN FLOOR FREE SLIP
"""
@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::Rusanov,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  A⁺.w = -A⁻.w

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::CentralNumericalFluxGradient,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  FT = eltype(Q⁺)
  A⁺.w = -zero(FT)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  A⁺.w = -A⁻.w
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

"""
OCEAN FLOOR NO SLIP
"""
@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::Rusanov,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  Q⁺.u = -Q⁻.u
  A⁺.w = -A⁻.w

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::CentralNumericalFluxGradient,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  FT = eltype(Q⁺)
  Q⁺.u = SVector(-zero(FT), -zero(FT))
  A⁺.w = -zero(FT)

  return nothing
end


@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)

  Q⁺.u = -Q⁻.u
  A⁺.w = -A⁻.w

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

"""
OCEAN SURFACE
only neumann, no dirichlet
"""
@inline function ocean_boundary_state!(m::HBModel, ::Union{
                                       OceanSurfaceNoStressNoForcing,
                                       OceanSurfaceStressNoForcing,
                                       OceanSurfaceNoStressForcing,
                                       OceanSurfaceStressForcing},
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  return nothing
end

"""
OCEAN SURFACE NO STRESS NO FORCING
"""
@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceNoStressNoForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

"""
OCEAN SURFACE STRESS NO FORCING
"""
@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceStressNoForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  τ = @SMatrix [ -0 -0; -0 -0; A⁺.τ / 1000 -0]
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u + 2 * τ)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

"""
OCEAN SURFACE NO STRESS FORCING
"""
@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceNoStressForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  θ  = Q⁻.θ
  θʳ = A⁺.θʳ
  λʳ = m.problem.λʳ

  σ = @SVector [-0, -0, λʳ * (θʳ - θ)]
  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ + 2 * σ)

  return nothing
end

"""
OCEAN SURFACE STRESS FORCING
"""
@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceStressForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  τ = @SMatrix [ -0 -0; -0 -0; A⁺.τ / 1000 -0]
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u + 2 * τ)

  θ  = Q⁻.θ
  θʳ = A⁺.θʳ
  λʳ = m.problem.λʳ

  σ = @SVector [-0, -0, λʳ * (θʳ - θ)]
  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ + 2 * σ)

  return nothing
end
