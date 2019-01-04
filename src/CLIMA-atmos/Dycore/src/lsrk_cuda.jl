function lowstorageRKupdate!(::Val{dim}, ::Val{N}, rhs::CuArray, Q,
                             vgeo, elems, rka, rkb, dt) where {dim, N}
  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_lowstorageRKupdate!(Val(dim), Val(N), rhs, Q, vgeo,
                                nelem, rka, rkb, dt))
end

function knl_lowstorageRKupdate!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem,
  rka, rkb, dt) where {dim, N}
  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    n = i + (j-1) * Nq + (k-1) * Nq * Nq
    MJI = vgeo[n, _MJI, e]
    for s = 1:_nstate
      Q[n, s, e] += rkb * dt * rhs[n, s, e] * MJI
      rhs[n, s, e] *= rka
    end
  end
  nothing
end
