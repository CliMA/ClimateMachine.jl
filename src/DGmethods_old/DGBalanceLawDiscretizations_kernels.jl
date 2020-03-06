using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

"""
    volumerhs!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{nviscstate},
               ::Val{nauxstate}, flux!, source!, rhs, Q, Qvisc, auxstate,
               vgeo, t, D, elems) where {dim, N, nstate, nviscstate,

Computational kernel: Evaluate the volume integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function volumerhs!(::Val{dim}, ::Val{N},
                    ::Val{nstate}, ::Val{nviscstate},
                    ::Val{nauxstate},
                    flux!, source!,
                    rhs, Q, Qvisc, auxstate, vgeo, t,
                    ω, D, elems, increment) where {dim, N, nstate, nviscstate,
                                     nauxstate}
  FT = eltype(Q)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_F = @shmem FT (3, Nq, Nq, Nqk, nstate)
  s_ω = @shmem FT (Nq, )
  s_half_D = @shmem FT (Nq, Nq)
  l_rhs = @scratch FT (nstate, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nstate}, FT}(undef))
  l_Q = MArray{Tuple{nstate}, FT}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)
  l_F = MArray{Tuple{3, nstate}, FT}(undef)
  l_M = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x3 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ2x1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ2x2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ2x3 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ3x1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ3x2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ3x3 = @scratch FT (Nq, Nq, Nqk) 3

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
          l_ξ1x1[i, j, k] = vgeo[ijk, _ξ1x1, e]
          l_ξ1x2[i, j, k] = vgeo[ijk, _ξ1x2, e]
          l_ξ1x3[i, j, k] = vgeo[ijk, _ξ1x3, e]
          l_ξ2x1[i, j, k] = vgeo[ijk, _ξ2x1, e]
          l_ξ2x2[i, j, k] = vgeo[ijk, _ξ2x2, e]
          l_ξ2x3[i, j, k] = vgeo[ijk, _ξ2x3, e]
          l_ξ3x1[i, j, k] = vgeo[ijk, _ξ3x1, e]
          l_ξ3x2[i, j, k] = vgeo[ijk, _ξ3x2, e]
          l_ξ3x3[i, j, k] = vgeo[ijk, _ξ3x3, e]

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(FT)
          end

          @unroll for s = 1:nstate
            l_Q[s] = Q[ijk, s, e]
          end

          @unroll for s = 1:nviscstate
            l_Qvisc[s] = Qvisc[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s] = auxstate[ijk, s, e]
          end

          flux!(l_F, l_Q, l_Qvisc, l_aux, t)

          @unroll for s = 1:nstate
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          if source! !== nothing
            source!(l_S, l_Q, l_Qvisc, l_aux, t)

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
              Dni = s_half_D[n, i] * s_ω[n] / s_ω[i]
              Dnj = s_half_D[n, j] * s_ω[n] / s_ω[j]
              Nqk > 1 && (Dnk = s_half_D[n, k] * s_ω[n] / s_ω[k])

              # ξ1-grid lines
              l_rhs[s, i, j, k] += l_ξ1x1[i, j, k] * Dni * s_F[1, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξ1x2[i, j, k] * Dni * s_F[2, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξ1x3[i, j, k] * Dni * s_F[3, n, j, k, s]

              # ξ2-grid lines
              l_rhs[s, i, j, k] += l_ξ2x1[i, j, k] * Dnj * s_F[1, i, n, k, s]
              l_rhs[s, i, j, k] += l_ξ2x2[i, j, k] * Dnj * s_F[2, i, n, k, s]
              l_rhs[s, i, j, k] += l_ξ2x3[i, j, k] * Dnj * s_F[3, i, n, k, s]

              # ξ3-grid lines
              if Nqk > 1
                l_rhs[s, i, j, k] += l_ξ3x1[i, j, k] * Dnk * s_F[1, i, j, n, s]
                l_rhs[s, i, j, k] += l_ξ3x2[i, j, k] * Dnk * s_F[2, i, j, n, s]
                l_rhs[s, i, j, k] += l_ξ3x3[i, j, k] * Dnk * s_F[3, i, j, n, s]
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
            s_F[1,i,j,k,s] = l_M[i, j, k] * (l_ξ1x1[i, j, k] * F1 +
                                              l_ξ1x2[i, j, k] * F2 +
                                              l_ξ1x3[i, j, k] * F3)
            s_F[2,i,j,k,s] = l_M[i, j, k] * (l_ξ2x1[i, j, k] * F1 +
                                              l_ξ2x2[i, j, k] * F2 +
                                              l_ξ2x3[i, j, k] * F3)
            s_F[3,i,j,k,s] = l_M[i, j, k] * (l_ξ3x1[i, j, k] * F1 +
                                              l_ξ3x2[i, j, k] * F2 +
                                              l_ξ3x3[i, j, k] * F3)
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
              Dni = s_half_D[n, i]
              Dnj = s_half_D[n, j]
              Nqk > 1 && (Dnk = s_half_D[n, k])
              # ξ1-grid lines
              l_rhs[s, i, j, k] += MI * Dni * s_F[1, n, j, k, s]

              # ξ2-grid lines
              l_rhs[s, i, j, k] += MI * Dnj * s_F[2, i, n, k, s]

              # ξ3-grid lines
              Nqk > 1 && (l_rhs[s, i, j, k] += MI * Dnk * s_F[3, i, j, n, s])
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

"""
    facerhs!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{nviscstate},
             ::Val{nauxstate}, numerical_flux!,
             numerical_boundary_flux!, rhs, Q, Qvisc, auxstate,
             vgeo, sgeo, t, vmap⁻, vmap⁺, elemtobndy,
             elems) where {dim, N, nstate, nviscstate, nauxstate}

Computational kernel: Evaluate the surface integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function facerhs!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{nviscstate},
                  ::Val{nauxstate}, numerical_flux!, numerical_boundary_flux!,
                  rhs, Q, Qvisc, auxstate, vgeo, sgeo, t, vmap⁻, vmap⁺,
                  elemtobndy, elems) where {dim, N, nstate, nviscstate,
                                            nauxstate}

  FT = eltype(Q)

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  l_QM = MArray{Tuple{nstate}, FT}(undef)
  l_QviscM = MArray{Tuple{nviscstate}, FT}(undef)
  l_auxM = MArray{Tuple{nauxstate}, FT}(undef)

  l_QP = MArray{Tuple{nstate}, FT}(undef)
  l_QviscP = MArray{Tuple{nviscstate}, FT}(undef)
  l_auxP = MArray{Tuple{nauxstate}, FT}(undef)

  l_F = MArray{Tuple{nstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        nM = (sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmap⁻[n, f, e], vmap⁺[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nstate
          l_QM[s] = Q[vidM, s, eM]
        end

        @unroll for s = 1:nviscstate
          l_QviscM[s] = Qvisc[vidM, s, eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        # Load plus side data
        @unroll for s = 1:nstate
          l_QP[s] = Q[vidP, s, eP]
        end

        @unroll for s = 1:nviscstate
          l_QviscP[s] = Qvisc[vidP, s, eP]
        end

        @unroll for s = 1:nauxstate
          l_auxP[s] = auxstate[vidP, s, eP]
        end

        bctype =
            numerical_boundary_flux! === nothing ? 0 : elemtobndy[f, e]
        if bctype == 0
          numerical_flux!(l_F, nM, l_QM, l_QviscM, l_auxM, l_QP, l_QviscP,
                          l_auxP, t)
        else
          numerical_boundary_flux!(l_F, nM, l_QM, l_QviscM, l_auxM, l_QP,
                                   l_QviscP, l_auxP, bctype, t)
        end

        #Update RHS
        @unroll for s = 1:nstate
          # FIXME: Should we pretch these?
          rhs[vidM, s, eM] -= vMI * sM * l_F[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function volumeviscterms!(::Val{dim}, ::Val{N}, ::Val{nstate},
                          ::Val{ngradstate}, ::Val{nviscstate},
                          ::Val{nauxstate}, viscous_transform!,
                          gradient_transform!, Q, Qvisc, auxstate, vgeo, t, D,
                          elems) where {dim, N, ngradstate,
                                        nviscstate, nstate, nauxstate}
  FT = eltype(Q)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq


  s_G = @shmem FT (Nq, Nq, Nqk, ngradstate)
  s_D = @shmem FT (Nq, Nq)

  l_Q = @scratch FT (nstate, Nq, Nq, Nqk) 3
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3
  l_G = MArray{Tuple{ngradstate}, FT}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_gradG = MArray{Tuple{3, ngradstate}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:nstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          gradient_transform!(l_G, l_Q[:, i, j, k], l_aux[:, i, j, k], t)
          @unroll for s = 1:ngradstate
            s_G[i, j, k, s] = l_G[s]
          end
        end
      end
    end
    @synchronize

    # Compute gradient of each state
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          ξ1x1, ξ1x2, ξ1x3 = vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]
          ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
          ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]

          @unroll for s = 1:ngradstate
            Gξ1 = Gξ2 = Gξ3 = zero(FT)
            @unroll for n = 1:Nq
              Din = s_D[i, n]
              Djn = s_D[j, n]
              Nqk > 1 && (Dkn = s_D[k, n])

              Gξ1 += Din * s_G[n, j, k, s]
              Gξ2 += Djn * s_G[i, n, k, s]
              Nqk > 1 && (Gξ3 += Dkn * s_G[i, j, n, s])
            end
            l_gradG[1, s] = ξ1x1 * Gξ1 + ξ2x1 * Gξ2 + ξ3x1 * Gξ3
            l_gradG[2, s] = ξ1x2 * Gξ1 + ξ2x2 * Gξ2 + ξ3x2 * Gξ3
            l_gradG[3, s] = ξ1x3 * Gξ1 + ξ2x3 * Gξ2 + ξ3x3 * Gξ3
          end

          viscous_transform!(l_Qvisc, l_gradG, l_Q[:, i, j, k],
                             l_aux[:, i, j, k], t)

          @unroll for s = 1:nviscstate
            Qvisc[ijk, s, e] = l_Qvisc[s]
          end
        end
      end
    end
    @synchronize
  end
end

function faceviscterms!(::Val{dim}, ::Val{N}, ::Val{nstate},
                        ::Val{ngradstate}, ::Val{nviscstate},
                        ::Val{nauxstate}, viscous_penalty!,
                        viscous_boundary_penalty!, gradient_transform!,
                        Q, Qvisc, auxstate, vgeo, sgeo, t, vmap⁻, vmap⁺,
                        elemtobndy, elems) where {dim, N, ngradstate,
                                                  nviscstate, nstate, nauxstate}
  FT = eltype(Q)

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  l_QM = MArray{Tuple{nstate}, FT}(undef)
  l_auxM = MArray{Tuple{nauxstate}, FT}(undef)
  l_GM = MArray{Tuple{ngradstate}, FT}(undef)

  l_QP = MArray{Tuple{nstate}, FT}(undef)
  l_auxP = MArray{Tuple{nauxstate}, FT}(undef)
  l_GP = MArray{Tuple{ngradstate}, FT}(undef)

  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        nM = (sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmap⁻[n, f, e], vmap⁺[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nstate
          l_QM[s] = Q[vidM, s, eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        gradient_transform!(l_GM, l_QM, l_auxM, t)

        # Load plus side data
        @unroll for s = 1:nstate
          l_QP[s] = Q[vidP, s, eP]
        end

        @unroll for s = 1:nauxstate
          l_auxP[s] = auxstate[vidP, s, eP]
        end

        gradient_transform!(l_GP, l_QP, l_auxP, t)

        bctype =
            viscous_boundary_penalty! === nothing ? 0 : elemtobndy[f, e]
        if bctype == 0
          viscous_penalty!(l_Qvisc, nM, l_GM, l_QM, l_auxM, l_GP,
                                  l_QP, l_auxP, t)
        else
          viscous_boundary_penalty!(l_Qvisc, nM, l_GM, l_QM, l_auxM,
                                           l_GP, l_QP, l_auxP, bctype, t)
        end

        @unroll for s = 1:nviscstate
          Qvisc[vidM, s, eM] += vMI * sM * l_Qvisc[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

"""
    initstate!(::Val{dim}, ::Val{N}, ::Val{nvar}, ::Val{nauxstate},
               ic!, Q, auxstate, vgeo, elems) where {dim, N, nvar, nauxstate}

Computational kernel: Initialize the state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initstate!(::Val{dim}, ::Val{N}, ::Val{nvar}, ::Val{nauxstate},
                    ic!, Q, auxstate, vgeo, elems) where {dim, N, nvar, nauxstate}

  FT = eltype(Q)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_Qdof = MArray{Tuple{nvar}, FT}(undef)
  l_auxdof = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for i in (1:Np; threadIdx().x)
      x1, x2, x3 = vgeo[i, _x1, e], vgeo[i, _x2, e], vgeo[i, _x3, e]

      @unroll for s = 1:nauxstate
        l_auxdof[s] = auxstate[i, s, e]
      end
      ic!(l_Qdof, x1, x2, x3, l_auxdof)
      @unroll for n = 1:nvar
        Q[i, n, e] = l_Qdof[n]
      end
    end
  end
end

"""
    initauxstate!(::Val{dim}, ::Val{N}, ::Val{nauxstate}, auxstatefun!,
                  auxstate, vgeo, elems) where {dim, N, nauxstate}

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(::Val{dim}, ::Val{N}, ::Val{nauxstate}, auxstatefun!,
                       auxstate, vgeo, elems) where {dim, N, nauxstate}

  FT = eltype(auxstate)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_aux = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      x1, x2, x3 = vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e]
      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      auxstatefun!(l_aux, x1, x2, x3)

      @unroll for s = 1:nauxstate
        auxstate[n, s, e] = l_aux[s]
      end
    end
  end
end

"""
    elem_grad_field!(::Val{dim}, ::Val{N}, ::Val{nstate}, Q, vgeo, D, elems, s,
                     sx, sy, sz) where {dim, N, nstate}

Computational kernel: Compute the element gradient of state `s` of `Q` and store
it in `sx`, `sy`, and `sz` of `Q`.

!!! warning

    This does not compute a DG gradient, but only over the element. If ``Q_s``
    is discontinuous you may want to consider another approach.

"""
function elem_grad_field!(::Val{dim}, ::Val{N}, ::Val{nstate}, Q, vgeo,
                          ω, D, elems, s, sx, sy, sz) where {dim, N, nstate}

  FT = eltype(vgeo)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_f = @shmem FT (3, Nq, Nq, Nqk)
  s_half_D = @shmem FT (Nq, Nq)

  l_f  = @scratch FT (Nq, Nq, Nqk) 3
  l_fd = @scratch FT (3, Nq, Nq, Nqk) 3

  l_J = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1d = @scratch FT (3, Nq, Nq, Nqk) 3
  l_ξ2d = @scratch FT (3, Nq, Nq, Nqk) 3
  l_ξ3d = @scratch FT (3, Nq, Nq, Nqk) 3

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
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
          M = dim == 2 ? ω[i] * ω[j] : ω[i] * ω[j] * ω[k]
          l_J[i, j, k] = vgeo[ijk, _M, e] / M
          l_ξ1d[1, i, j, k] = vgeo[ijk, _ξ1x1, e]
          l_ξ1d[2, i, j, k] = vgeo[ijk, _ξ1x2, e]
          l_ξ1d[3, i, j, k] = vgeo[ijk, _ξ1x3, e]
          l_ξ2d[1, i, j, k] = vgeo[ijk, _ξ2x1, e]
          l_ξ2d[2, i, j, k] = vgeo[ijk, _ξ2x2, e]
          l_ξ2d[3, i, j, k] = vgeo[ijk, _ξ2x3, e]
          l_ξ3d[1, i, j, k] = vgeo[ijk, _ξ3x1, e]
          l_ξ3d[2, i, j, k] = vgeo[ijk, _ξ3x2, e]
          l_ξ3d[3, i, j, k] = vgeo[ijk, _ξ3x3, e]
          l_f[i, j, k] = s_f[1, i, j, k] = Q[ijk, s, e]
        end
      end
    end
    @synchronize

    # reference gradient: outside metrics
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          fξ1 = FT(0)
          fξ2 = FT(0)
          fξ3 = FT(0)
          @unroll for n = 1:Nq
            Din = s_half_D[i, n]
            Djn = s_half_D[j, n]
            Nqk > 1 && (Dkn = s_half_D[k, n])

            # ξ1-grid lines
            fξ1 += Din * s_f[1, n, j, k]

            # ξ2-grid lines
            fξ2 += Djn * s_f[1, i, n, k]

            # ξ3-grid lines
            Nqk > 1 && (fξ3 += Dkn * s_f[1, i, j, n])
          end
          @unroll for d = 1:3
            l_fd[d, i, j, k] = l_ξ1d[d, i, j, k] * fξ1 +
                               l_ξ2d[d, i, j, k] * fξ2 +
                               l_ξ3d[d, i, j, k] * fξ3
          end
        end
      end
    end
    @synchronize

    # Build "inside metrics" flux
    for d = 1:3
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            s_f[1,i,j,k] = l_J[i, j, k] * l_ξ1d[d, i, j, k] * l_f[i, j, k]
            s_f[2,i,j,k] = l_J[i, j, k] * l_ξ2d[d, i, j, k] * l_f[i, j, k]
            s_f[3,i,j,k] = l_J[i, j, k] * l_ξ3d[d, i, j, k] * l_f[i, j, k]
          end
        end
      end
      @synchronize
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            fd = FT(0)
            JI = 1 / l_J[i, j, k]
            @unroll for n = 1:Nq
              Din = s_half_D[i, n]
              Djn = s_half_D[j, n]
              Nqk > 1 && (Dkn = s_half_D[k, n])

              l_fd[d, i, j, k] += JI * Din * s_f[1, n, j, k]
              l_fd[d, i, j, k] += JI * Djn * s_f[2, i, n, k]
              Nqk > 1 && (l_fd[d, i, j, k] += JI * Dkn * s_f[3, i, j, n])
            end
          end
        end
      end
      @synchronize
    end

    # Physical gradient
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          Q[ijk, sx, e] = l_fd[1, i, j, k]
          Q[ijk, sy, e] = l_fd[2, i, j, k]
          Q[ijk, sz, e] = l_fd[3, i, j, k]
        end
      end
    end
    @synchronize
  end
end

"""
    knl_dof_iteration!(::Val{dim}, ::Val{N}, ::Val{nRstate}, ::Val{nstate},
                       ::Val{nviscstate}, ::Val{nauxstate}, dof_fun!, R, Q,
                       QV, auxstate, elems) where {dim, N, nRstate, nstate,
                                                   nviscstate, nauxstate}

Computational kernel: fill postprocessing array

See [`DGBalanceLaw`](@ref) for usage.
"""
function knl_dof_iteration!(::Val{dim}, ::Val{N}, ::Val{nRstate}, ::Val{nstate},
                            ::Val{nviscstate}, ::Val{nauxstate}, dof_fun!, R, Q,
                            QV, auxstate, elems) where {dim, N, nRstate, nstate,
                                                        nviscstate, nauxstate}
  FT = eltype(R)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_R = MArray{Tuple{nRstate}, FT}(undef)
  l_Q = MArray{Tuple{nstate}, FT}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nRstate
        l_R[s] = R[n, s, e]
      end

      @unroll for s = 1:nstate
        l_Q[s] = Q[n, s, e]
      end

      @unroll for s = 1:nviscstate
        l_Qvisc[s] = QV[n, s, e]
      end

      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      dof_fun!(l_R, l_Q, l_Qvisc, l_aux)

      @unroll for s = 1:nRstate
        R[n, s, e] = l_R[s]
      end
    end
  end
end

"""
    knl_indefinite_stack_integral!(::Val{dim}, ::Val{N}, ::Val{nstate},
                                            ::Val{nauxstate}, ::Val{nvertelem},
                                            int_knl!, Q, auxstate, vgeo, Imat,
                                            elems, ::Val{outstate}
                                           ) where {dim, N, nstate, nauxstate,
                                                    outstate, nvertelem}

Computational kernel: compute indefinite integral along the vertical stack

See [`DGBalanceLaw`](@ref) for usage.
"""
function knl_indefinite_stack_integral!(::Val{dim}, ::Val{N}, ::Val{nstate},
                                        ::Val{nauxstate}, ::Val{nvertelem},
                                        int_knl!, P, Q, auxstate, vgeo, Imat,
                                        elems, ::Val{outstate}
                                       ) where {dim, N, nstate, nauxstate,
                                                outstate, nvertelem}
  FT = eltype(Q)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  nout = length(outstate)

  l_Q = MArray{Tuple{nstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)
  l_knl = MArray{Tuple{nout, Nq}, FT}(undef)
  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_int = @scratch FT (nout, Nq, Nq, Nqj) 2

  s_I = @shmem FT (Nq, Nq)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for i in (1:Nq; threadIdx().x)
      @unroll for n = 1:Nq
        s_I[i, n] = Imat[i, n]
      end
    end
  end
  @synchronize

  @inbounds @loop for eh in (elems; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for k in 1:Nq
          @unroll for s = 1:nout
            l_int[s, k, i, j] = 0
          end
        end
      end
    end
    # Loop up the stack of elements
    for ev = 1:nvertelem
      e = ev + (eh - 1) * nvertelem

      # Evaluate the integral kernel at each DOF in the slabk
      @loop for j in (1:Nqj; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          # loop up the pencil
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            Jc = vgeo[ijk, _JcV, e]
            @unroll for s = 1:nstate
              l_Q[s] = Q[ijk, s, e]
            end

            @unroll for s = 1:nauxstate
              l_aux[s] = auxstate[ijk, s, e]
            end

            int_knl!(view(l_knl, :, k), l_Q, l_aux)

            # multiply in the curve jacobian
            @unroll for s = 1:nout
              l_knl[s, k] *= Jc
            end
          end

          # Evaluate the integral up the element
          @unroll for s = 1:nout
            @unroll for k in 1:Nq
              @unroll for n in 1:Nq
                l_int[s, k, i, j] += s_I[k, n] * l_knl[s, n]
              end
            end
          end

          # Store out to memory and reset the background value for next element
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            @unroll for ind_out = 1:nout
              s = outstate[ind_out]
              P[ijk, s, e] = l_int[ind_out, k, i, j]
              l_int[ind_out, k, i, j] = l_int[ind_out, Nq, i, j]
            end
          end
        end
      end
    end
  end
  nothing
end

function knl_reverse_indefinite_stack_integral!(::Val{dim}, ::Val{N},
                                                ::Val{nvertelem}, P, elems,
                                                ::Val{outstate},
                                                ::Val{instate}
                                               ) where {dim, N, outstate,
                                                        instate, nvertelem}
  FT = eltype(P)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  nout = length(outstate)

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_T = MArray{Tuple{nout}, FT}(undef)
  l_V = MArray{Tuple{nout}, FT}(undef)

  @inbounds @loop for eh in (elems; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        ijk = i + Nq * ((j-1) + Nqj * (Nq-1))
        et = nvertelem + (eh - 1) * nvertelem
        @unroll for s = 1:nout
          l_T[s] = P[ijk, instate[s], et]
        end

        # Loop up the stack of elements
        for ev = 1:nvertelem
          e = ev + (eh - 1) * nvertelem
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            @unroll for s = 1:nout
              l_V[s] = P[ijk, instate[s], e]
            end
            @unroll for s = 1:nout
              P[ijk, outstate[s], e] = l_T[s] - l_V[s]
            end
          end
        end
      end
    end
  end
  nothing
end
