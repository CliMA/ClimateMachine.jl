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
            vgrad_data[ijk, ∂₃u₂, e] - vgrad_data[ijk, ∂₂u₃, e]
        vort_data[ijk, Ω₂, e] =
            vgrad_data[ijk, ∂₁u₃, e] - vgrad_data[ijk, ∂₃u₁, e]
        vort_data[ijk, Ω₃, e] =
            vgrad_data[ijk, ∂₂u₁, e] - vgrad_data[ijk, ∂₁u₂, e]
    end
end
#--------------------------------------------------------------------------------------------------
