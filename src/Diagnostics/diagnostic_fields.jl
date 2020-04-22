using DocStringExtensions
using KernelAbstractions
using CLIMA.Mesh.Geometry

import CLIMA.Mesh.Grids:
    _ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3

using CLIMA.VariableTemplates
import CLIMA.VariableTemplates.varsindex

include("diagnostic_fields_kernels.jl")
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
        DA = CLIMA.array_type()
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
        DA = CLIMA.array_type()
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
    DA = CLIMA.array_type()

    vgrad = VecGrad(Npl, Nel, FT)
    ind = [
        varsindex(vars_state(model, FT), :ρ)
        varsindex(vars_state(model, FT), :ρu)
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
    DA = CLIMA.array_type()
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
