using DocStringExtensions
using KernelAbstractions
using Adapt

using ..Mesh.Geometry
using ..VariableTemplates

import ..Mesh.Grids:
    _ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3, ftpxv!
import ..MPIStateArrays: array_device

"""
    VectorGradients{
        FT <: AbstractFloat,
        FTA2D <: AbstractArray{FT, 2},
        FTA3D <: AbstractArray{FT, 3},
    }

This data structure stores the spatial gradients of a velocity field.

# Fields

$(DocStringExtensions.FIELDS)

"""
struct VectorGradients{FT <: AbstractFloat, FTA2D <: AbstractArray{FT, 2}}
    "∂u₁/∂x₁"
    ∂₁u₁::FTA2D
    "∂u₁/∂x₂"
    ∂₂u₁::FTA2D
    "∂u₁/∂x₃"
    ∂₃u₁::FTA2D
    "∂u₂/∂x₁"
    ∂₁u₂::FTA2D
    "∂u₂/∂x₂"
    ∂₂u₂::FTA2D
    "∂u₂/∂x₃"
    ∂₃u₂::FTA2D
    "∂u₃/∂x₁"
    ∂₁u₃::FTA2D
    "∂u₃/∂x₂"
    ∂₂u₃::FTA2D
    "∂u₃/∂x₃"
    ∂₃u₃::FTA2D
end
Adapt.@adapt_structure VectorGradients

function VectorGradients(
    npts::Int,
    nel::Int,
    ::Type{FT},
    ::Type{DA},
) where {FT <: AbstractFloat, DA}

    return VectorGradients{FT, DA{FT, 2}}(
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
    )
end

"""
    VectorGradients(dg::SpaceDiscretization, Q::MPIStateArray)

This constructor computes the spatial gradients of the velocity field.

# Arguments
 - `dg`: SpaceDiscretization
 - `Q`: MPIStateArray containing the prognostic state variables
"""
function VectorGradients(dg::SpaceDiscretization, Q::MPIStateArray)
    bl = dg.balance_law
    FT = eltype(dg.grid)
    DA = (typeof(parent(Q.data)) <: Array) ? Array : CuArray
    Nq = polynomialorders(dg.grid) .+ 1 #N + 1
    Nqmax = maximum(Nq)
    npoints = prod(Nq)
    nrealelem = length(dg.grid.topology.realelems)

    vgrad = VectorGradients(npoints, nrealelem, FT, DA)
    ind = varsindices(vars_state(bl, Prognostic(), FT), ("ρ", "ρu"))
    _ρ, _ρu, _ρv, _ρw = ind[1], ind[2], ind[3], ind[4]

    ρ = view(Q.data, :, _ρ, 1:nrealelem)
    ρu = view(Q.data, :, _ρu, 1:nrealelem)
    ρv = view(Q.data, :, _ρv, 1:nrealelem)
    ρw = view(Q.data, :, _ρw, 1:nrealelem)

    ξ1x1 = view(dg.grid.vgeo, :, _ξ1x1, :)
    ξ1x2 = view(dg.grid.vgeo, :, _ξ1x2, :)
    ξ1x3 = view(dg.grid.vgeo, :, _ξ1x3, :)
    ξ2x1 = view(dg.grid.vgeo, :, _ξ2x1, :)
    ξ2x2 = view(dg.grid.vgeo, :, _ξ2x2, :)
    ξ2x3 = view(dg.grid.vgeo, :, _ξ2x3, :)
    ξ3x1 = view(dg.grid.vgeo, :, _ξ3x1, :)
    ξ3x2 = view(dg.grid.vgeo, :, _ξ3x2, :)
    ξ3x3 = view(dg.grid.vgeo, :, _ξ3x3, :)

    ∂f∂ξ₁ = view(dg.grid.m1_storage, :, 1, :)
    ∂f∂ξ₂ = view(dg.grid.m1_storage, :, 2, :)
    ∂f∂ξ₃ = view(dg.grid.m1_storage, :, 3, :)

    ftpxv!(dg.grid, :m1, :∂ξ₁, false, ρu, ∂f∂ξ₁, vin_den = ρ) # ∂u₁/∂ξ₁
    ftpxv!(dg.grid, :m1, :∂ξ₂, false, ρu, ∂f∂ξ₂, vin_den = ρ) # ∂u₁/∂ξ₂
    ftpxv!(dg.grid, :m1, :∂ξ₃, false, ρu, ∂f∂ξ₃, vin_den = ρ) # ∂u₁/∂ξ₃

    vgrad.∂₁u₁ .= ∂f∂ξ₁ .* ξ1x1 .+ ∂f∂ξ₂ .* ξ2x1 .+ ∂f∂ξ₃ .* ξ3x1 # ∂u₁/∂x₁
    vgrad.∂₂u₁ .= ∂f∂ξ₁ .* ξ1x2 .+ ∂f∂ξ₂ .* ξ2x2 .+ ∂f∂ξ₃ .* ξ3x2 # ∂u₁/∂x₂
    vgrad.∂₃u₁ .= ∂f∂ξ₁ .* ξ1x3 .+ ∂f∂ξ₂ .* ξ2x3 .+ ∂f∂ξ₃ .* ξ3x3 # ∂u₁/∂x₃

    ftpxv!(dg.grid, :m1, :∂ξ₁, false, ρv, ∂f∂ξ₁, vin_den = ρ) # ∂u₂/∂ξ₁
    ftpxv!(dg.grid, :m1, :∂ξ₂, false, ρv, ∂f∂ξ₂, vin_den = ρ) # ∂u₂/∂ξ₂
    ftpxv!(dg.grid, :m1, :∂ξ₃, false, ρv, ∂f∂ξ₃, vin_den = ρ) # ∂u₂/∂ξ₃

    vgrad.∂₁u₂ .= ∂f∂ξ₁ .* ξ1x1 .+ ∂f∂ξ₂ .* ξ2x1 .+ ∂f∂ξ₃ .* ξ3x1 # ∂u₂/∂x₁
    vgrad.∂₂u₂ .= ∂f∂ξ₁ .* ξ1x2 .+ ∂f∂ξ₂ .* ξ2x2 .+ ∂f∂ξ₃ .* ξ3x2 # ∂u₂/∂x₂
    vgrad.∂₃u₂ .= ∂f∂ξ₁ .* ξ1x3 .+ ∂f∂ξ₂ .* ξ2x3 .+ ∂f∂ξ₃ .* ξ3x3 # ∂u₂/∂x₃

    ftpxv!(dg.grid, :m1, :∂ξ₁, false, ρw, ∂f∂ξ₁, vin_den = ρ) # ∂u₃/∂ξ₁
    ftpxv!(dg.grid, :m1, :∂ξ₂, false, ρw, ∂f∂ξ₂, vin_den = ρ) # ∂u₃/∂ξ₂
    ftpxv!(dg.grid, :m1, :∂ξ₃, false, ρw, ∂f∂ξ₃, vin_den = ρ) # ∂u₃/∂ξ₃

    vgrad.∂₁u₃ .= ∂f∂ξ₁ .* ξ1x1 .+ ∂f∂ξ₂ .* ξ2x1 .+ ∂f∂ξ₃ .* ξ3x1 # ∂u₃/∂x₁
    vgrad.∂₂u₃ .= ∂f∂ξ₁ .* ξ1x2 .+ ∂f∂ξ₂ .* ξ2x2 .+ ∂f∂ξ₃ .* ξ3x2 # ∂u₃/∂x₂
    vgrad.∂₃u₃ .= ∂f∂ξ₁ .* ξ1x3 .+ ∂f∂ξ₂ .* ξ2x3 .+ ∂f∂ξ₃ .* ξ3x3 # ∂u₃/∂x₃

    return vgrad
end
#--------------------------------------------------------------------------------------------------

"""
    Vorticity{
        FT <: AbstractFloat,
        FTA2D <: AbstractArray{FT, 2},
        FTA3D <: AbstractArray{FT, 3},
    }

This data structure stores the vorticity of a velocity field.

# Fields

$(DocStringExtensions.FIELDS)

"""
struct Vorticity{FT <: AbstractFloat, FTA2D <: AbstractArray{FT, 2}}
    "x1 component of vorticity"
    Ω₁::FTA2D
    "x2 component of vorticity"
    Ω₂::FTA2D
    "x3 component of vorticity"
    Ω₃::FTA2D
end
Adapt.@adapt_structure Vorticity

function Vorticity(
    npoints::Int,
    nel::Int,
    ::Type{FT},
    ::Type{DA},
) where {FT <: AbstractFloat, DA}
    return Vorticity{FT, DA{FT, 2}}(
        DA{FT, 2}(undef, npoints, nel),
        DA{FT, 2}(undef, npoints, nel),
        DA{FT, 2}(undef, npoints, nel),
    )
end

"""
    Vorticity(
        dg::SpaceDiscretization,
        vgrad::VectorGradients,
    )

This function computes the vorticity of the velocity field.

# Arguments
 - `dg`: SpaceDiscretization
 - `vgrad`: vector gradients
"""
function Vorticity(dg::SpaceDiscretization, vgrad::VectorGradients)
    bl = dg.balance_law
    FT = eltype(dg.grid)
    DA = (typeof(vgrad.∂₁u₁) <: Array) ? Array : CuArray
    npoints = prod(polynomialorders(dg.grid) .+ 1)
    nrealelem = length(dg.grid.topology.realelems)

    vort = Vorticity(npoints, nrealelem, FT, DA)

    Ω₁, Ω₂, Ω₃ = 1, 2, 3

    vort.Ω₁ .= vgrad.∂₂u₃ .- vgrad.∂₃u₂
    vort.Ω₂ .= vgrad.∂₃u₁ .- vgrad.∂₁u₃
    vort.Ω₃ .= vgrad.∂₁u₂ .- vgrad.∂₂u₁

    return vort
end
