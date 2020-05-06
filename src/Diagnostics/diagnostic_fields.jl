using DocStringExtensions
using KernelAbstractions
using ClimateMachine.Mesh.Geometry

import ClimateMachine.Mesh.Grids:
    _ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3

using ClimateMachine.VariableTemplates
import ClimateMachine.VariableTemplates.varsindex

"""
    VecGrad{FT <: AbstractFloat,
         FTA2D <: AbstractArray{FT, 2},
         FTA3D <: AbstractArray{FT, 3},}

This data structure stores the spatial gradients of a velocity field.

# Fields

$(DocStringExtensions.FIELDS)

# Usage

    VecGrad(Npl, Nel, ::Type{FT}) where {FT <: AbstractFloat}

# Arguments for the inner constructor    
 - `Npl`: Number of local degrees of freedom in a spectral element
 - `Nel`: Number of spectral elements
 - `FT`: Floating point precision
"""
struct VecGrad{
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

    function VecGrad(Npl, Nel, ::Type{FT}) where {FT <: AbstractFloat}
        DA = ClimateMachine.array_type()
        data = DA{FT}(undef, Npl, 9, Nel)

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
#--------------------------------------------------------------------------------------------------
"""
    Vorticity{FT <: AbstractFloat,
           FTA2D <: AbstractArray{FT, 2},
           FTA3D <: AbstractArray{FT, 3},}

This data structure stores the vorticity of a velocity field.

# Fields

$(DocStringExtensions.FIELDS)

# Usage

    Vorticity(Npl, Nel, ::Type{FT}) where {FT <: AbstractFloat}

# Arguments for the inner constructor    
 - `Npl`: Number of local degrees of freedom in a spectral element
 - `Nel`: Number of spectral elements
 - `FT`: Floating point precision
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
    function Vorticity(Npl, Nel, ::Type{FT}) where {FT <: AbstractFloat}
        DA = ClimateMachine.array_type()
        data = DA{FT}(undef, Npl, 3, Nel)
        Ω₁ = view(data, :, 1, :)
        Ω₂ = view(data, :, 2, :)
        Ω₃ = view(data, :, 3, :)
        return new{FT, typeof(Ω₁), typeof(data)}(data, Ω₁, Ω₂, Ω₃)
    end
end
#--------------------------------------------------------------------------------------------------
"""
    compute_vec_grad(model::BalanceLaw,
                         Q::AbstractArray{FT},
                        dg::DGModel,
                     vgrad::VecGrad{FT},) where {FT <: AbstractFloat}

This function computes the spatial gradients of the velocity field.

# Arguments 
 - `model`: BalanceLaw
 - `Q`: State array
 - `dg`: DGmodel
 - `vgrad`: Vector gradients
"""
function compute_vec_grad(
    model::BalanceLaw,
    Q::AbstractArray{FT},
    dg::DGModel,
) where {FT <: AbstractFloat}
    Nel = size(Q.realdata, 3) # # of spectral elements
    Npl = size(Q.realdata, 1) # # of dof per element
    qm1 = size(dg.grid.D, 2)  # poly order + 1
    DA = ClimateMachine.array_type()

    vgrad = VecGrad(Npl, Nel, FT)
    ind = [
        varsindex(vars_state_conservative(model, FT), :ρ)
        varsindex(vars_state_conservative(model, FT), :ρu)
    ]
    _ρ, _ρu, _ρv, _ρw = ind[1], ind[2], ind[3], ind[4]

    vgrad_data = vgrad.data
    vgeo = dg.grid.vgeo
    D = dg.grid.D
    sv = Q.data

    device = DA <: Array ? CPU() : CUDA()
    g = DA{FT}(undef, Npl, Nel, 3, 3)

    workgroup = (qm1, qm1)
    ndrange = (Nel * qm1, qm1)

    kernel = compute_vec_grad_kernel!(device, workgroup)
    event = kernel(
        sv,
        D,
        vgeo,
        g,
        vgrad_data,
        _ρ,
        _ρu,
        _ρv,
        _ρw,
        Val(qm1),
        ndrange = ndrange,
    )
    wait(event)

    return vgrad
end
#--------------------------------------------------------------------------------------------------
"""
    compute_vorticity(dg::DGModel,
                   vgrad::VecGrad{FT},
                    vort::Vorticity{FT},) where {FT <: AbstractFloat}

This function computes the vorticity of the velocity field.

# Arguments 
 - `dg`: DGmodel
 - `vgrad`: Velocity gradients
 - `vort`: Vorticity
"""
function compute_vorticity(
    dg::DGModel,
    vgrad::VecGrad{FT},
) where {FT <: AbstractFloat}
    Npl = size(vgrad.∂₁u₁, 1)
    Nel = size(vgrad.∂₁u₁, 2)
    qm1 = size(dg.grid.D, 2)  # poly order + 1
    DA = ClimateMachine.array_type()
    device = DA <: Array ? CPU() : CUDA()

    vort = Vorticity(Npl, Nel, FT)

    vgrad_data = vgrad.data
    vort_data = vort.data

    Ω₁, Ω₂, Ω₃ = vort.Ω₁, vort.Ω₂, vort.Ω₃

    workgroup = (qm1, qm1)
    ndrange = (Nel * qm1, qm1)

    kernel = compute_vorticity_kernel!(device, workgroup)
    event = kernel(vgrad_data, vort_data, Val(qm1), ndrange = ndrange)
    wait(event)

    return vort
end
#--------------------------------------------------------------------------------------------------
@kernel function compute_vec_grad_kernel!(
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
@kernel function compute_vorticity_kernel!(
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
#--------------------------------------------------------------------------------------------------
