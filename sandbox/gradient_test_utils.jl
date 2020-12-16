# "volume" derivatives

function ∂x!(∇Q, Q, D, nrealelem, N, ξ1x1)
    for e in 1:nrealelem
        for i in 1:(N[1]+1)
            for j in 1:(N[2]+1)
                for k in 1:(N[3]+1)
                    for n in 1:(N[1]+1)
                        ∇Q[i,j,k,e,1] += ξ1x1[i,j,k,e] * D[i, n] * Q[n,j,k,e]
                    end
                end
            end
        end
    end
    return nothing
end

function ∂y!(∇Q, Q, D, nrealelem, N, ξ2x2)
    for e in 1:nrealelem
        for i in 1:(N[1]+1)
            for j in 1:(N[2]+1)
                for k in 1:(N[3]+1)
                    for n in 1:(N[1]+1)
                        ∇Q[i,j,k,e,2] += ξ2x2[i,j,k,e] * D[j, n] * Q[i,n,k,e]
                    end
                end
            end
        end
    end
    return nothing
end

function ∂z!(∇Q, Q, D, nrealelem, N, ξ3x3)
    for e in 1:nrealelem
        for i in 1:(N[1]+1)
            for j in 1:(N[2]+1)
                for k in 1:(N[3]+1)
                    for n in 1:(N[1]+1)
                        ∇Q[i,j,k,e,3] += ξ3x3[i,j,k,e] * D[k, n] * Q[i,j,n,e]
                    end
                end
            end
        end
    end
    return nothing
end

function ∇!(∇Q, Q, grid)
    N = polynomialorders(grid)
    nrealelem = Int(length(grid.vgeo[:, grid.x1id, : ,: ]) / prod(N .+ 1))
    r_∇Q = reshape(∇Q, ((N .+1)..., nrealelem, 3))
    r_Q  = reshape(Q, ((N .+1)..., nrealelem))
    r_∇Q .= -0.0
    r_ξ1x1  = reshape(grid.vgeo[:, grid.ξ1x1id, :, : ], ((N .+1)..., nrealelem))
    r_ξ2x2  = reshape(grid.vgeo[:, grid.ξ2x2id, :, : ], ((N .+1)..., nrealelem))
    r_ξ3x3  = reshape(grid.vgeo[:, grid.ξ3x3id, :, : ], ((N .+1)..., nrealelem))
    D = grid.D[1]
    ∂x!(r_∇Q, r_Q, D, nrealelem, N, r_ξ1x1)
    ∂y!(r_∇Q, r_Q, D, nrealelem, N, r_ξ2x2)
    ∂z!(r_∇Q, r_Q, D, nrealelem, N, r_ξ3x3)
    return nothing
end