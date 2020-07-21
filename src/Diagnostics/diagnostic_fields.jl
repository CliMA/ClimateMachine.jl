using DocStringExtensions
using KernelAbstractions

using ..Mesh.Geometry
using ..VariableTemplates

import ..Mesh.Grids:
    _ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3
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

# Usage

    VectorGradients(data)

# Arguments for the inner constructor
 - `data`: 3-dimensional device array containing the spatial gradients
   (the second dimension must be 9)
"""
struct VectorGradients{
    FT <: AbstractFloat,
    FTA2D <: AbstractArray{FT, 2},
    FTA3D <: AbstractArray{FT, 3},
}
    "Device array storing the spatial gradient data"
    data::FTA3D
    "View of ∂u₁/∂x₁"
    ∂₁u₁::FTA2D
    "View of ∂u₁/∂x₂"
    ∂₂u₁::FTA2D
    "View of ∂u₁/∂x₃"
    ∂₃u₁::FTA2D
    "View of ∂u₂/∂x₁"
    ∂₁u₂::FTA2D
    "View of ∂u₂/∂x₂"
    ∂₂u₂::FTA2D
    "View of ∂u₂/∂x₃"
    ∂₃u₂::FTA2D
    "View of ∂u₃/∂x₁"
    ∂₁u₃::FTA2D
    "View of ∂u₃/∂x₂"
    ∂₂u₃::FTA2D
    "View of ∂u₃/∂x₃"
    ∂₃u₃::FTA2D

    function VectorGradients(
        data::AbstractArray{FT, 3},
    ) where {FT <: AbstractFloat}
        ∂₁u₁, ∂₂u₁, ∂₃u₁ =
            view(data, :, 1, :), view(data, :, 2, :), view(data, :, 3, :)
        ∂₁u₂, ∂₂u₂, ∂₃u₂ =
            view(data, :, 4, :), view(data, :, 5, :), view(data, :, 6, :)
        ∂₁u₃, ∂₂u₃, ∂₃u₃ =
            view(data, :, 7, :), view(data, :, 8, :), view(data, :, 9, :)

        return new{FT, typeof(∂₁u₁), typeof(data)}(
            data,
            ∂₁u₁,
            ∂₂u₁,
            ∂₃u₁,
            ∂₁u₂,
            ∂₂u₂,
            ∂₃u₂,
            ∂₁u₃,
            ∂₂u₃,
            ∂₃u₃,
        )
    end
end

"""
    VectorGradients(dg::DGModel, Q::MPIStateArray)

This constructor computes the spatial gradients of the velocity field.

# Arguments
 - `dg`: DGModel
 - `Q`: MPIStateArray containing the conservative state variables
"""
function VectorGradients(dg::DGModel, Q::MPIStateArray)
    bl = dg.balance_law
    FT = eltype(dg.grid)
    N = polynomialorder(dg.grid)
    Nq = N + 1
    npoints = Nq^3
    nrealelem = length(dg.grid.topology.realelems)

    g = similar(Q.realdata, npoints, nrealelem, 3, 3)
    data = similar(Q.realdata, npoints, 9, nrealelem)

    ind = [
        varsindex(vars_state(bl, Prognostic(), FT), :ρ)
        varsindex(vars_state(bl, Prognostic(), FT), :ρu)
    ]
    _ρ, _ρu, _ρv, _ρw = ind[1], ind[2], ind[3], ind[4]

    device = array_device(Q)
    workgroup = (Nq, Nq)
    ndrange = (nrealelem * Nq, Nq)

    kernel = vector_gradients_kernel!(device, workgroup)
    event = kernel(
        Q.realdata,
        dg.grid.D,
        dg.grid.vgeo,
        g,
        data,
        _ρ,
        _ρu,
        _ρv,
        _ρw,
        Val(Nq),
        ndrange = ndrange,
    )
    wait(event)

    return VectorGradients(data)
end

@kernel function vector_gradients_kernel!(
    sv::AbstractArray{FT},
    D::AbstractArray{FT, 2},
    vgeo::AbstractArray{FT},
    g::AbstractArray{FT, 4},
    vgrad_data::AbstractArray{FT, 3},
    _ρ::Int,
    _ρu::Int,
    _ρv::Int,
    _ρw::Int,
    ::Val{qm1},
) where {qm1, FT <: AbstractFloat}

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    s_D = @localmem FT (qm1, qm1)
    s_U = @localmem FT (qm1, qm1)
    s_V = @localmem FT (qm1, qm1)
    s_W = @localmem FT (qm1, qm1)

    s_D[i, j] = D[i, j]
    @synchronize
    # computing derivatives with respect to ξ1
    for t in 1:qm1, s in 1:qm1
        ijk = j + ((s - 1) + (t - 1) * qm1) * qm1
        s_U[i, j] = s_D[i, j] * (sv[ijk, _ρu, e] / sv[ijk, _ρ, e])
        s_V[i, j] = s_D[i, j] * (sv[ijk, _ρv, e] / sv[ijk, _ρ, e])
        s_W[i, j] = s_D[i, j] * (sv[ijk, _ρw, e] / sv[ijk, _ρ, e])
        @synchronize
        if j == 1
            for r in 2:qm1
                s_U[i, 1] += s_U[i, r]
                s_V[i, 1] += s_V[i, r]
                s_W[i, 1] += s_W[i, r]
            end
        end
        @synchronize
        if j == 1
            g[i + ((s - 1) + (t - 1) * qm1) * qm1, e, 1, 1] = s_U[i, 1] # ∂u₁∂ξ₁
            g[i + ((s - 1) + (t - 1) * qm1) * qm1, e, 2, 1] = s_V[i, 1] # ∂u₂∂ξ₁
            g[i + ((s - 1) + (t - 1) * qm1) * qm1, e, 3, 1] = s_W[i, 1] # ∂u₃∂ξ₁
        end
    end
    @synchronize
    # computing derivatives with respect to ξ2
    for t in 1:qm1, r in 1:qm1
        ijk = r + ((j - 1) + (t - 1) * qm1) * qm1
        s_U[i, j] = s_D[i, j] * (sv[ijk, _ρu, e] / sv[ijk, _ρ, e])
        s_V[i, j] = s_D[i, j] * (sv[ijk, _ρv, e] / sv[ijk, _ρ, e])
        s_W[i, j] = s_D[i, j] * (sv[ijk, _ρw, e] / sv[ijk, _ρ, e])
        @synchronize
        if j == 1
            for s in 2:qm1
                s_U[i, 1] += s_U[i, s]
                s_V[i, 1] += s_V[i, s]
                s_W[i, 1] += s_W[i, s]
            end
        end
        @synchronize
        if j == 1
            g[r + ((i - 1) + (t - 1) * qm1) * qm1, e, 1, 2] = s_U[i, 1] # ∂u₁∂ξ₂
            g[r + ((i - 1) + (t - 1) * qm1) * qm1, e, 2, 2] = s_V[i, 1] # ∂u₂∂ξ₂
            g[r + ((i - 1) + (t - 1) * qm1) * qm1, e, 3, 2] = s_W[i, 1] # ∂u₃∂ξ₂
        end
    end
    @synchronize
    # computing derivatives with respect to ξ3
    for s in 1:qm1, r in 1:qm1
        ijk = r + ((s - 1) + (j - 1) * qm1) * qm1
        s_U[i, j] = s_D[i, j] * (sv[ijk, _ρu, e] / sv[ijk, _ρ, e])
        s_V[i, j] = s_D[i, j] * (sv[ijk, _ρv, e] / sv[ijk, _ρ, e])
        s_W[i, j] = s_D[i, j] * (sv[ijk, _ρw, e] / sv[ijk, _ρ, e])
        @synchronize
        if j == 1
            for t in 2:qm1
                s_U[i, 1] += s_U[i, t]
                s_V[i, 1] += s_V[i, t]
                s_W[i, 1] += s_W[i, t]
            end
        end
        @synchronize
        if j == 1
            g[r + ((s - 1) + (i - 1) * qm1) * qm1, e, 1, 3] = s_U[i, 1] # ∂u₁∂ξ₃
            g[r + ((s - 1) + (i - 1) * qm1) * qm1, e, 2, 3] = s_V[i, 1] # ∂u₂∂ξ₃
            g[r + ((s - 1) + (i - 1) * qm1) * qm1, e, 3, 3] = s_W[i, 1] # ∂u₃∂ξ₃
        end
    end
    @synchronize

    ∂₁u₁, ∂₂u₁, ∂₃u₁ = 1, 2, 3
    ∂₁u₂, ∂₂u₂, ∂₃u₂ = 4, 5, 6
    ∂₁u₃, ∂₂u₃, ∂₃u₃ = 7, 8, 9

    for k in 1:qm1
        ijk = i + ((j - 1) + (k - 1) * qm1) * qm1

        ξ1x1 = vgeo[ijk, _ξ1x1, e]
        ξ1x2 = vgeo[ijk, _ξ1x2, e]
        ξ1x3 = vgeo[ijk, _ξ1x3, e]
        ξ2x1 = vgeo[ijk, _ξ2x1, e]
        ξ2x2 = vgeo[ijk, _ξ2x2, e]
        ξ2x3 = vgeo[ijk, _ξ2x3, e]
        ξ3x1 = vgeo[ijk, _ξ3x1, e]
        ξ3x2 = vgeo[ijk, _ξ3x2, e]
        ξ3x3 = vgeo[ijk, _ξ3x3, e]

        vgrad_data[ijk, ∂₁u₁, e] =
            g[ijk, e, 1, 1] * ξ1x1 +
            g[ijk, e, 1, 2] * ξ2x1 +
            g[ijk, e, 1, 3] * ξ3x1
        vgrad_data[ijk, ∂₂u₁, e] =
            g[ijk, e, 1, 1] * ξ1x2 +
            g[ijk, e, 1, 2] * ξ2x2 +
            g[ijk, e, 1, 3] * ξ3x2
        vgrad_data[ijk, ∂₃u₁, e] =
            g[ijk, e, 1, 1] * ξ1x3 +
            g[ijk, e, 1, 2] * ξ2x3 +
            g[ijk, e, 1, 3] * ξ3x3

        vgrad_data[ijk, ∂₁u₂, e] =
            g[ijk, e, 2, 1] * ξ1x1 +
            g[ijk, e, 2, 2] * ξ2x1 +
            g[ijk, e, 2, 3] * ξ3x1
        vgrad_data[ijk, ∂₂u₂, e] =
            g[ijk, e, 2, 1] * ξ1x2 +
            g[ijk, e, 2, 2] * ξ2x2 +
            g[ijk, e, 2, 3] * ξ3x2
        vgrad_data[ijk, ∂₃u₂, e] =
            g[ijk, e, 2, 1] * ξ1x3 +
            g[ijk, e, 2, 2] * ξ2x3 +
            g[ijk, e, 2, 3] * ξ3x3

        vgrad_data[ijk, ∂₁u₃, e] =
            g[ijk, e, 3, 1] * ξ1x1 +
            g[ijk, e, 3, 2] * ξ2x1 +
            g[ijk, e, 3, 3] * ξ3x1
        vgrad_data[ijk, ∂₂u₃, e] =
            g[ijk, e, 3, 1] * ξ1x2 +
            g[ijk, e, 3, 2] * ξ2x2 +
            g[ijk, e, 3, 3] * ξ3x2
        vgrad_data[ijk, ∂₃u₃, e] =
            g[ijk, e, 3, 1] * ξ1x3 +
            g[ijk, e, 3, 2] * ξ2x3 +
            g[ijk, e, 3, 3] * ξ3x3
    end
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

# Usage

    Vorticity(data)

# Arguments for the inner constructor
 - `data`: 3-dimensional device array containing the vorticity data
   (the second dimension must be 3)
"""
struct Vorticity{
    FT <: AbstractFloat,
    FTA2D <: AbstractArray{FT, 2},
    FTA3D <: AbstractArray{FT, 3},
}
    "Device array storing the vorticity data"
    data::FTA3D
    "View of x1 component of vorticity"
    Ω₁::FTA2D
    "View of x2 component of vorticity"
    Ω₂::FTA2D
    "View of x3 component of vorticity"
    Ω₃::FTA2D
    function Vorticity(data::AbstractArray{FT, 3}) where {FT <: AbstractFloat}
        Ω₁ = view(data, :, 1, :)
        Ω₂ = view(data, :, 2, :)
        Ω₃ = view(data, :, 3, :)
        return new{FT, typeof(Ω₁), typeof(data)}(data, Ω₁, Ω₂, Ω₃)
    end
end

"""
    Vorticity(
        dg::DGModel,
        vgrad::VectorGradients,
    )

This function computes the vorticity of the velocity field.

# Arguments
 - `dg`: DGModel
 - `vgrad`: vector gradients
"""
function Vorticity(dg::DGModel, vgrad::VectorGradients)
    bl = dg.balance_law
    FT = eltype(dg.grid)
    N = polynomialorder(dg.grid)
    Nq = N + 1
    npoints = Nq^3
    nrealelem = length(dg.grid.topology.realelems)

    data = similar(vgrad.data, npoints, 3, nrealelem)

    device = array_device(data)
    workgroup = (Nq, Nq)
    ndrange = (nrealelem * Nq, Nq)

    kernel = vorticity_kernel!(device, workgroup)
    event = kernel(vgrad.data, data, Val(Nq), ndrange = ndrange)
    wait(event)

    return Vorticity(data)
end

@kernel function vorticity_kernel!(
    vgrad_data::AbstractArray{FT, 3},
    vort_data::AbstractArray{FT, 3},
    ::Val{qm1},
) where {qm1, FT <: AbstractFloat}

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    ∂₁u₁, ∂₂u₁, ∂₃u₁ = 1, 2, 3
    ∂₁u₂, ∂₂u₂, ∂₃u₂ = 4, 5, 6
    ∂₁u₃, ∂₂u₃, ∂₃u₃ = 7, 8, 9
    Ω₁, Ω₂, Ω₃ = 1, 2, 3

    for k in 1:qm1
        ijk = i + ((j - 1) + (k - 1) * qm1) * qm1

        vort_data[ijk, Ω₁, e] =
            vgrad_data[ijk, ∂₂u₃, e] - vgrad_data[ijk, ∂₃u₂, e]
        vort_data[ijk, Ω₂, e] =
            vgrad_data[ijk, ∂₃u₁, e] - vgrad_data[ijk, ∂₁u₃, e]
        vort_data[ijk, Ω₃, e] =
            vgrad_data[ijk, ∂₁u₂, e] - vgrad_data[ijk, ∂₂u₁, e]
    end
end
