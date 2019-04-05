function volumerhs!(::Val{dim}, ::Val{N},
                    ::Val{nstate}, ::Val{ngradstate},
                    ::Val{nauxcstate}, ::Val{nauxdstate},
                    flux!,
                    rhs::Array,
                    Q, Qgrad, auxc, auxd, vgeo, t,
                    D, elems) where {dim, N, nstate, ngradstate,
                                    nauxcstate, nauxdstate}
  DFloat = eltype(Q)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, Nqk, nstate, nelem)
  Qgrad = reshape(Qgrad, Nq, Nq, Nqk, ngradstate, nelem)
  rhs = reshape(rhs, Nq, Nq, Nqk, nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, Nqk, _nvgeo, nelem)

  s_F = MArray{Tuple{3, Nq, Nq, Nqk, nstate}, DFloat}(undef)

  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_Qgrad = MArray{Tuple{ngradstate}, DFloat}(undef)
  l_ϕc = MArray{Tuple{nauxcstate}, DFloat}(undef)
  l_ϕd = MArray{Tuple{nauxdstate}, DFloat}(undef)

  l_F = MArray{Tuple{3, nstate}, DFloat}(undef)

  @inbounds for e in elems
    for k = 1:Nqk, j = 1:Nq, i = 1:Nq
      MJ = vgeo[i, j, k, _MJ, e]
      # MJI = vgeo[i, j, k, _MJI, e]
      ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
      ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
      ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

      for s = 1:nstate
        l_Q[s] = Q[i, j, k, s, e]
      end

      for s = 1:ngradstate
        l_Qgrad[s] = Qgrad[i, j, k, s, e]
      end

      for s = 1:nauxcstate
        l_ϕc[s] = ϕc[i, j, k, s, e]
      end

      for s = 1:nauxdstate
        l_ϕd[s] = ϕd[i, j, k, s, e]
      end

      flux!(l_F, l_Q, l_Qgrad, l_ϕc, l_ϕd, t)

      for s = 1:nstate
        s_F[1,i,j,k,s] = MJ * (ξx * l_F[1, s] + ξy * l_F[2, s] + ξz * l_F[3, s])
        s_F[2,i,j,k,s] = MJ * (ηx * l_F[1, s] + ηy * l_F[2, s] + ηz * l_F[3, s])
        s_F[3,i,j,k,s] = MJ * (ζx * l_F[1, s] + ζy * l_F[2, s] + ζz * l_F[3, s])
      end

      # TODO source
    end

    # loop of ξ-grid lines
    for s = 1:nstate, k = 1:Nqk, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, k, _MJI, e]
      for n = 1:Nq
        rhs[i, j, k, s, e] += MJI * D[n, i] * s_F[1, n, j, k, s]
      end
    end
    # loop of η-grid lines
    for s = 1:nstate, k = 1:Nqk, j = 1:Nq, i = 1:Nq
      MJI = vgeo[i, j, k, _MJI, e]
      for n = 1:Nq
        rhs[i, j, k, s, e] += MJI * D[n, j] * s_F[2, i, n, k, s]
      end
    end
    # loop of ζ-grid lines
    if Nqk > 1
      for s = 1:nstate, k = 1:Nqk, j = 1:Nq, i = 1:Nq
        MJI = vgeo[i, j, k, _MJI, e]
        for n = 1:Nqk
          rhs[i, j, k, s, e] += MJI * D[n, k] * s_F[3, i, j, n, s]
        end
      end
    end
  end
end
