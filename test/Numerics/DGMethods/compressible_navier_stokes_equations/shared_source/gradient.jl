# "volume" derivatives
function ∂ξ¹∂x⃗!(∇Q, Q, D, nrealelem, N, ξ1x1, ξ1x2, ξ1x3)
    # ξ¹ terms
    for e in 1:nrealelem, i in 1:(N[1]+1), j in 1:(N[2]+1), k in 1:(N[3]+1), n in 1:(N[1]+1)
        DQ = D[i, n] * Q[n,j,k,e]
        ∇Q[i,j,k,e,1] += ξ1x1[i,j,k,e] * DQ
        ∇Q[i,j,k,e,2] += ξ1x2[i,j,k,e] * DQ
        ∇Q[i,j,k,e,3] += ξ1x3[i,j,k,e] * DQ
    end
end

function ∂ξ²∂x⃗!(∇Q, Q, D, nrealelem, N, ξ2x1, ξ2x2, ξ2x3)
    # ξ² terms
    for e in 1:nrealelem, i in 1:(N[1]+1), j in 1:(N[2]+1), k in 1:(N[3]+1), n in 1:(N[2]+1)
        DQ = D[j, n] * Q[i,n,k,e]
        ∇Q[i,j,k,e,1] += ξ2x1[i,j,k,e] * DQ
        ∇Q[i,j,k,e,2] += ξ2x2[i,j,k,e] * DQ
        ∇Q[i,j,k,e,3] += ξ2x3[i,j,k,e] * DQ
    end
    return nothing
end

function ∂ξ³∂x⃗!(∇Q, Q, D, nrealelem, N, ξ3x1, ξ3x2, ξ3x3)
    # ξ³ terms
    for e in 1:nrealelem, i in 1:(N[1]+1), j in 1:(N[2]+1), k in 1:(N[3]+1), n in 1:(N[3]+1)
        DQ = D[k, n] * Q[i,j,n,e]
        ∇Q[i,j,k,e,1] += ξ3x1[i,j,k,e] * DQ
        ∇Q[i,j,k,e,2] += ξ3x2[i,j,k,e] * DQ
        ∇Q[i,j,k,e,3] += ξ3x3[i,j,k,e] * DQ
    end
    return nothing
end

function grad!(∇Q, Q, grid)
    N = polynomialorders(grid)
    nrealelem = Int(length(grid.vgeo[:, grid.x1id, : ,: ]) / prod(N .+ 1))
    r_∇Q = reshape(∇Q, ((N .+ 1)..., nrealelem, 3))
    r_Q  = reshape(Q,  ((N .+ 1)..., nrealelem))
    r_∇Q .= -0.0

    # ∂ξ¹∂x⃗
    r_ξ1x1  = reshape(grid.vgeo[:, grid.ξ1x1id, :, : ], ((N .+1)..., nrealelem))
    r_ξ1x2  = reshape(grid.vgeo[:, grid.ξ1x2id, :, : ], ((N .+1)..., nrealelem))
    r_ξ1x3  = reshape(grid.vgeo[:, grid.ξ1x3id, :, : ], ((N .+1)..., nrealelem))
    ∂ξ¹ = grid.D[1]
    ∂ξ¹∂x⃗!(r_∇Q, r_Q, ∂ξ¹, nrealelem, N, r_ξ1x1, r_ξ1x2, r_ξ1x3)

    # ∂ξ²∂x⃗
    r_ξ2x1  = reshape(grid.vgeo[:, grid.ξ2x1id, :, : ], ((N .+1)..., nrealelem))
    r_ξ2x2  = reshape(grid.vgeo[:, grid.ξ2x2id, :, : ], ((N .+1)..., nrealelem))
    r_ξ2x3  = reshape(grid.vgeo[:, grid.ξ2x3id, :, : ], ((N .+1)..., nrealelem))
    ∂ξ² = grid.D[2]
    ∂ξ²∂x⃗!(r_∇Q, r_Q, ∂ξ², nrealelem, N, r_ξ2x1, r_ξ2x2, r_ξ2x3)

    # ∂ξ³∂x⃗
    r_ξ3x1  = reshape(grid.vgeo[:, grid.ξ3x1id, :, : ], ((N .+1)..., nrealelem))
    r_ξ3x2  = reshape(grid.vgeo[:, grid.ξ3x2id, :, : ], ((N .+1)..., nrealelem))
    r_ξ3x3  = reshape(grid.vgeo[:, grid.ξ3x3id, :, : ], ((N .+1)..., nrealelem))
    ∂ξ³ = grid.D[3]
    ∂ξ³∂x⃗!(r_∇Q, r_Q, ∂ξ³, nrealelem, N, r_ξ3x1, r_ξ3x2, r_ξ3x3)
    return nothing
end

function grad(Q, grid)
    ∇Q = zeros((size(Q)..., 3))
    grad!(∇Q, Q, grid)
    return ∇Q
end

grad(Q, grid::DiscretizedDomain) = grad(Q, grid.numerical)

abstract type AbstractOperation end

struct Nabla{S,T} <: AbstractOperation
    grid::S
    data::T
end

function Nabla(grid::DiscontinuousSpectralElementGrid)
    n_ijk, n_e = size(grid.vgeo[:, grid.ξ3x1id, :, : ])
    ∇Q = zeros(n_ijk, n_e, 3)
    return Nabla(grid, ∇Q)
end

Nabla(grid::DiscretizedDomain) = Nabla(grid.numerical)

function (g::Nabla)(Q) 
    ∇Q = grad(Q, g.grid)
    return ∇Q
end

## div operator
function divergence!(divF⃗, F⃗, grid)
    N = polynomialorders(grid)
    nrealelem = Int(length(grid.vgeo[:, grid.x1id, : ,: ]) / prod(N .+ 1))
    r_F⃗ = reshape(F⃗, ((N .+ 1)..., nrealelem, 3))
    r_divF⃗  = reshape(divF⃗,  ((N .+ 1)..., nrealelem))
    r_divF⃗ .= -0.0

    M = reshape(grid.vgeo[:, grid.Mid, :, : ], ((N .+1)..., nrealelem))
    MI = reshape(grid.vgeo[:, grid.MIid, :, : ], ((N .+1)..., nrealelem))
    ω1 =  reshape(grid.ω[1], (N[1]+1, 1, 1, 1))
    ω2 =  reshape(grid.ω[2], (1, N[2]+1, 1, 1))
    ω3 =  reshape(grid.ω[3], (1, 1, N[3]+1, 1))
    J = M ./ ω1 ./ ω2 ./ ω3
    JI = 1 ./ J

    # ∂ξ¹
    r_ξ1x1  = reshape(grid.vgeo[:, grid.ξ1x1id, :, : ], ((N .+1)..., nrealelem))
    r_ξ1x2  = reshape(grid.vgeo[:, grid.ξ1x2id, :, : ], ((N .+1)..., nrealelem))
    r_ξ1x3  = reshape(grid.vgeo[:, grid.ξ1x3id, :, : ], ((N .+1)..., nrealelem))
    ∂ξ¹ = grid.D[1]
    # ξ¹ terms
    for e in 1:nrealelem, i in 1:(N[1]+1), j in 1:(N[2]+1), k in 1:(N[3]+1), n in 1:(N[1]+1)
        F¹  = r_ξ1x1[n,j,k,e] * r_F⃗[n,j,k,e,1] 
        F¹ += r_ξ1x2[n,j,k,e] * r_F⃗[n,j,k,e,2] 
        F¹ += r_ξ1x3[n,j,k,e] * r_F⃗[n,j,k,e,3] 
        r_divF⃗[i,j,k,e] += ∂ξ¹[i, n] * J[n,j,k,e] * F¹
    end

    # ∂ξ²
    r_ξ2x1  = reshape(grid.vgeo[:, grid.ξ2x1id, :, : ], ((N .+1)..., nrealelem))
    r_ξ2x2  = reshape(grid.vgeo[:, grid.ξ2x2id, :, : ], ((N .+1)..., nrealelem))
    r_ξ2x3  = reshape(grid.vgeo[:, grid.ξ2x3id, :, : ], ((N .+1)..., nrealelem))
    ∂ξ² = grid.D[2]
    # ξ² terms
    for e in 1:nrealelem, i in 1:(N[1]+1), j in 1:(N[2]+1), k in 1:(N[3]+1), n in 1:(N[2]+1)
        F²  = r_ξ2x1[i,n,k,e] * r_F⃗[i,n,k,e,1] 
        F² += r_ξ2x2[i,n,k,e] * r_F⃗[i,n,k,e,2] 
        F² += r_ξ2x3[i,n,k,e] * r_F⃗[i,n,k,e,3] 
        r_divF⃗[i,j,k,e] += ∂ξ²[j, n] * J[i,n,k,e] * F²
    end

    # ∂ξ³∂x⃗
    r_ξ3x1  = reshape(grid.vgeo[:, grid.ξ3x1id, :, : ], ((N .+ 1)..., nrealelem))
    r_ξ3x2  = reshape(grid.vgeo[:, grid.ξ3x2id, :, : ], ((N .+ 1)..., nrealelem))
    r_ξ3x3  = reshape(grid.vgeo[:, grid.ξ3x3id, :, : ], ((N .+ 1)..., nrealelem))
    ∂ξ³ = grid.D[3]
    # ξ³ terms
    for e in 1:nrealelem, i in 1:(N[1]+1), j in 1:(N[2]+1), k in 1:(N[3]+1), n in 1:(N[3]+1)
        F³  = r_ξ3x1[i,j,n,e] * r_F⃗[i,j,n,e,1] 
        F³ += r_ξ3x2[i,j,n,e] * r_F⃗[i,j,n,e,2] 
        F³ += r_ξ3x3[i,j,n,e] * r_F⃗[i,j,n,e,3] 
        r_divF⃗[i,j,k,e] += ∂ξ³[k, n] * J[i,j,n,e] * F³
    end
    r_divF⃗ .= r_divF⃗ .* JI
    return nothing
end

function divergence(F⃗, grid)
    divF⃗ = zeros((size(F⃗)[1:2]))
    divergence!(divF⃗, F⃗, grid)
    return divF⃗
end
