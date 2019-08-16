using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ2x1, _ξ3x1 = Grids._ξ2x1, Grids._ξ3x1
const _ξ2x2, _ξ3x2 = Grids._ξ2x2, Grids._ξ3x2
const _ξ2x3, _ξ3x3 = Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

function volumerhs!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{nauxstate},
                    flux!, source!, rhs, Q, auxstate, vgeo, t, ω, D, elems,
                    increment) where {dim, N, nstate, nauxstate}
  DFloat = eltype(Q)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_F = @shmem DFloat (3, Nq, Nq, Nqk, nstate)
  s_ω = @shmem DFloat (Nq, )
  s_half_D = @shmem DFloat (Nq, Nq)
  l_rhs = @scratch DFloat (nstate, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nstate}, DFloat}(undef))
  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)
  l_F = MArray{Tuple{3, nstate}, DFloat}(undef)
  l_M = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ζx1 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ζx2 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ζx3 = @scratch DFloat (Nq, Nq, Nqk) 3

  _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
  _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
  _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      s_ω[j] = ω[j]
      @loop for i in (1:Nq; threadIdx().x)
        s_half_D[i, j] = D[i, j] / 2
      end
    end
  end

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          l_M[i, j, k] = vgeo[ijk, _M, e]
          l_ζx1[i, j, k] = vgeo[ijk, _ζx1, e]
          l_ζx2[i, j, k] = vgeo[ijk, _ζx2, e]
          l_ζx3[i, j, k] = vgeo[ijk, _ζx3, e]

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(DFloat)
          end

          @unroll for s = 1:nstate
            l_Q[s] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s] = auxstate[ijk, s, e]
          end

          flux!(l_F, l_Q, l_aux, t)

          @unroll for s = 1:nstate
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          if source! !== nothing
            source!(l_S, l_Q, l_aux, t)

            @unroll for s = 1:nstate
              l_rhs[s, i, j, k] += l_S[s]
            end
          end
        end
      end
    end
    @synchronize

    # Weak "outside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          @unroll for n = 1:Nq
            @unroll for s = 1:nstate
              if dim == 2
                Dnj = s_half_D[n, j] * s_ω[n] / s_ω[j]
                l_rhs[s, i, j, k] += l_ζx1[i, j, k] * Dnj * s_F[1, i, n, k, s]
                l_rhs[s, i, j, k] += l_ζx2[i, j, k] * Dnj * s_F[2, i, n, k, s]
                l_rhs[s, i, j, k] += l_ζx3[i, j, k] * Dnj * s_F[3, i, n, k, s]
              else
                Dnk = s_half_D[n, k] * s_ω[n] / s_ω[k]
                l_rhs[s, i, j, k] += l_ζx1[i, j, k] * Dnk * s_F[1, i, j, n, s]
                l_rhs[s, i, j, k] += l_ζx2[i, j, k] * Dnk * s_F[2, i, j, n, s]
                l_rhs[s, i, j, k] += l_ζx3[i, j, k] * Dnk * s_F[3, i, j, n, s]
              end
            end
          end
        end
      end
    end
    @synchronize

    # Build "inside metrics" flux
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          @unroll for s = 1:nstate
            F1, F2, F3 = s_F[1,i,j,k,s], s_F[2,i,j,k,s], s_F[3,i,j,k,s]
            s_F[3,i,j,k,s] = l_M[i, j, k] * (l_ζx1[i, j, k] * F1 +
                                             l_ζx2[i, j, k] * F2 +
                                             l_ζx3[i, j, k] * F3)
          end
        end
      end
    end
    @synchronize

    # Weak "inside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          MI = vgeo[ijk, _MI, e]
          @unroll for s = 1:nstate
            @unroll for n = 1:Nq
              if dim == 2
                Dnj = s_half_D[n, j]
                l_rhs[s, i, j, k] += MI * Dnj * s_F[2, i, n, k, s]
              else
                Dnk = s_half_D[n, k]
                l_rhs[s, i, j, k] += MI * Dnk * s_F[3, i, j, n, s]
              end
            end
          end
        end
      end
    end
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:nstate
            rhs[ijk, s, e] = l_rhs[s, i, j, k]
          end
        end
      end
    end
    @synchronize
  end
  nothing
end

function facerhs!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{nauxstate},
                  numerical_flux!, numerical_boundary_flux!, rhs, Q,
                  auxstate, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                  elems) where {dim, N, nstate, nauxstate}

  DFloat = eltype(Q)

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
    faces = 3:4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
    faces = 5:6
  end

  l_QM = MArray{Tuple{nstate}, DFloat}(undef)
  l_auxM = MArray{Tuple{nauxstate}, DFloat}(undef)

  l_QP = MArray{Tuple{nstate}, DFloat}(undef)
  l_auxP = MArray{Tuple{nauxstate}, DFloat}(undef)

  l_F = MArray{Tuple{nstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f = faces
      @loop for n in (1:Nfp; threadIdx().x)
        nM = (sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nstate
          l_QM[s] = Q[vidM, s, eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        # Load plus side data
        @unroll for s = 1:nstate
          l_QP[s] = Q[vidP, s, eP]
        end

        @unroll for s = 1:nauxstate
          l_auxP[s] = auxstate[vidP, s, eP]
        end

        bctype =
            numerical_boundary_flux! === nothing ? 0 : elemtobndy[f, e]
        if bctype == 0
          numerical_flux!(l_F, nM, l_QM, l_auxM, l_QP, l_auxP, t)
        else
          numerical_boundary_flux!(l_F, nM, l_QM, l_auxM, l_QP, l_auxP,
                                   bctype, t)
        end

        #Update RHS
        @unroll for s = 1:nstate
          # FIXME: Should we pretch these?
          rhs[vidM, s, eM] -= vMI * sM * l_F[s]
        end
      end
    end
  end
  nothing
end
