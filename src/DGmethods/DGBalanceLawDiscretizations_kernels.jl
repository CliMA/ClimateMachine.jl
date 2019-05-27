using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξx, _ηx, _ζx = Grids._ξx, Grids._ηx, Grids._ζx
const _ξy, _ηy, _ζy = Grids._ξy, Grids._ηy, Grids._ζy
const _ξz, _ηz, _ζz = Grids._ξz, Grids._ηz, Grids._ζz
const _M, _MI = Grids._M, Grids._MI
const _x, _y, _z = Grids._x, Grids._y, Grids._z

const _nx, _ny, _nz = Grids._nx, Grids._ny, Grids._nz
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
                    D, elems) where {dim, N, nstate, nviscstate,
                                     nauxstate}
  DFloat = eltype(Q)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_F = @shmem DFloat (3, Nq, Nq, Nqk, nstate)
  s_D = @shmem DFloat (Nq, Nq)
  l_rhs = @scratch DFloat (nstate, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nstate}, DFloat}(undef))
  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)
  l_F = MArray{Tuple{3, nstate}, DFloat}(undef)

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
          MJ = vgeo[ijk, _M, e]
          ξx, ξy, ξz = vgeo[ijk,_ξx,e], vgeo[ijk,_ξy,e], vgeo[ijk,_ξz,e]
          ηx, ηy, ηz = vgeo[ijk,_ηx,e], vgeo[ijk,_ηy,e], vgeo[ijk,_ηz,e]
          ζx, ζy, ζz = vgeo[ijk,_ζx,e], vgeo[ijk,_ζy,e], vgeo[ijk,_ζz,e]

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] = rhs[ijk, s, e]
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
            s_F[1,i,j,k,s] = MJ * (ξx*l_F[1,s] + ξy*l_F[2,s] + ξz*l_F[3,s])
            s_F[2,i,j,k,s] = MJ * (ηx*l_F[1,s] + ηy*l_F[2,s] + ηz*l_F[3,s])
            s_F[3,i,j,k,s] = MJ * (ζx*l_F[1,s] + ζy*l_F[2,s] + ζz*l_F[3,s])
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

    @unroll for s = 1:nstate
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            ijk = i + Nq * ((j-1) + Nq * (k-1))
            MJI = vgeo[ijk, _MI, e]
            for n = 1:Nq
              Dni = s_D[n, i]
              Dnj = s_D[n, j]
              Nqk > 1 && (Dnk = s_D[n, k])
              # ξ-grid lines
              l_rhs[s, i, j, k] += MJI * Dni * s_F[1, n, j, k, s]

              # η-grid lines
              l_rhs[s, i, j, k] += MJI * Dnj * s_F[2, i, n, k, s]

              # ζ-grid lines
              Nqk > 1 && (l_rhs[s, i, j, k] += MJI * Dnk * s_F[3, i, j, n, s])
            end
          end
        end
      end
    end
    @unroll for s = 1:nstate
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            ijk = i + Nq * ((j-1) + Nq * (k-1))
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
             vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
             elems) where {dim, N, nstate, nviscstate, nauxstate}

Computational kernel: Evaluate the surface integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function facerhs!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{nviscstate},
                  ::Val{nauxstate}, numerical_flux!, numerical_boundary_flux!,
                  rhs, Q, Qvisc, auxstate, vgeo, sgeo, t, vmapM, vmapP,
                  elemtobndy, elems) where {dim, N, nstate, nviscstate,
                                            nauxstate}

  DFloat = eltype(Q)

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

  l_QM = MArray{Tuple{nstate}, DFloat}(undef)
  l_QviscM = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_auxM = MArray{Tuple{nauxstate}, DFloat}(undef)

  l_QP = MArray{Tuple{nstate}, DFloat}(undef)
  l_QviscP = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_auxP = MArray{Tuple{nauxstate}, DFloat}(undef)

  l_F = MArray{Tuple{nstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        nM = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_nz, n, f, e])
        sMJ, vMJI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

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
          rhs[vidM, s, eM] -= vMJI * sMJ * l_F[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function volumeviscterms!(::Val{dim}, ::Val{N}, ::Val{nstate},
                          ::Val{states_grad}, ::Val{ngradstate},
                          ::Val{nviscstate}, ::Val{nauxstate},
                          viscous_transform!, gradient_transform!, Q,
                          Qvisc, auxstate, vgeo, t, D,
                          elems) where {dim, N, states_grad, ngradstate,
                                        nviscstate, nstate, nauxstate}
  DFloat = eltype(Q)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  ngradtransformstate = length(states_grad)

  s_G = @shmem DFloat (Nq, Nq, Nqk, ngradstate)
  s_D = @shmem DFloat (Nq, Nq)

  l_Q = @scratch DFloat (ngradtransformstate, Nq, Nq, Nqk) 3
  l_aux = @scratch DFloat (nauxstate, Nq, Nq, Nqk) 3
  l_G = MArray{Tuple{ngradstate}, DFloat}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_gradG = MArray{Tuple{3, ngradstate}, DFloat}(undef)

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
          @unroll for s = 1:ngradtransformstate
            l_Q[s, i, j, k] = Q[ijk, states_grad[s], e]
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
          ξx, ξy, ξz = vgeo[ijk, _ξx, e], vgeo[ijk, _ξy, e], vgeo[ijk, _ξz, e]
          ηx, ηy, ηz = vgeo[ijk, _ηx, e], vgeo[ijk, _ηy, e], vgeo[ijk, _ηz, e]
          ζx, ζy, ζz = vgeo[ijk, _ζx, e], vgeo[ijk, _ζy, e], vgeo[ijk, _ζz, e]

          @unroll for s = 1:ngradstate
            Gξ = Gη = Gζ = zero(DFloat)
            @unroll for n = 1:Nq
              Din = s_D[i, n]
              Djn = s_D[j, n]
              Nqk > 1 && (Dkn = s_D[k, n])

              Gξ += Din * s_G[n, j, k, s]
              Gη += Djn * s_G[i, n, k, s]
              Nqk > 1 && (Gζ += Dkn * s_G[i, j, n, s])
            end
            l_gradG[1, s] = ξx * Gξ + ηx * Gη + ζx * Gζ
            l_gradG[2, s] = ξy * Gξ + ηy * Gη + ζy * Gζ
            l_gradG[3, s] = ξz * Gξ + ηz * Gη + ζz * Gζ
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

function faceviscterms!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{states_grad},
                        ::Val{ngradstate}, ::Val{nviscstate},
                        ::Val{nauxstate}, viscous_penalty!,
                        viscous_boundary_penalty!, gradient_transform!,
                        Q, Qvisc, auxstate, vgeo, sgeo, t, vmapM, vmapP,
                        elemtobndy, elems) where {dim, N, states_grad,
                                                  ngradstate, nviscstate,
                                                  nstate, nauxstate}
  DFloat = eltype(Q)

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

  ngradtransformstate = length(states_grad)

  l_QM = MArray{Tuple{ngradtransformstate}, DFloat}(undef)
  l_auxM = MArray{Tuple{nauxstate}, DFloat}(undef)
  l_GM = MArray{Tuple{ngradstate}, DFloat}(undef)

  l_QP = MArray{Tuple{ngradtransformstate}, DFloat}(undef)
  l_auxP = MArray{Tuple{nauxstate}, DFloat}(undef)
  l_GP = MArray{Tuple{ngradstate}, DFloat}(undef)

  l_Qvisc = MArray{Tuple{nviscstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        nM = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_nz, n, f, e])
        sMJ, vMJI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:ngradtransformstate
          l_QM[s] = Q[vidM, states_grad[s], eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        gradient_transform!(l_GM, l_QM, l_auxM, t)

        # Load plus side data
        @unroll for s = 1:ngradtransformstate
          l_QP[s] = Q[vidP, states_grad[s], eP]
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
          Qvisc[vidM, s, eM] += vMJI * sMJ * l_Qvisc[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end


"""
    initauxstate!(::Val{dim}, ::Val{N}, ::Val{nauxstate}, auxstatefun!,
                  auxstate, vgeo, elems) where {dim, N, nauxstate}

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(::Val{dim}, ::Val{N}, ::Val{nauxstate}, auxstatefun!,
                       auxstate, vgeo, elems) where {dim, N, nauxstate}

  DFloat = eltype(auxstate)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      x, y, z = vgeo[n, _x, e], vgeo[n, _y, e], vgeo[n, _z, e]
      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      auxstatefun!(l_aux, x, y, z)

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
                          D, elems, s, sx, sy, sz) where {dim, N, nstate}

  DFloat = eltype(vgeo)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_f = @shmem DFloat (Nq, Nq, Nqk)
  s_D = @shmem DFloat (Nq, Nq)

  l_fξ = @scratch DFloat (Nq, Nq, Nqk) 3
  l_fη = @scratch DFloat (Nq, Nq, Nqk) 3
  l_fζ = @scratch DFloat (Nq, Nq, Nqk) 3

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
          s_f[i, j, k] = Q[ijk, s, e]
        end
      end
    end
    @synchronize

    # reference gradient
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          l_fξ[i, j, k] = 0
          l_fη[i, j, k] = 0
          l_fζ[i, j, k] = 0
          @unroll for n = 1:Nq
            Din = s_D[i, n]
            Djn = s_D[j, n]
            Nqk > 1 && (Dkn = s_D[k, n])

            # ξ-grid lines
            l_fξ[i, j, k] += Din * s_f[n, j, k]

            # η-grid lines
            l_fη[i, j, k] += Djn * s_f[i, n, k]

            # ζ-grid lines
            Nqk > 1 && (l_fζ[i, j, k] += Dkn * s_f[i, j, n])
          end
        end
      end
    end

    # Physical gradient
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))

          ξx, ξy, ξz = vgeo[ijk, _ξx, e], vgeo[ijk, _ξy, e], vgeo[ijk, _ξz, e]
          ηx, ηy, ηz = vgeo[ijk, _ηx, e], vgeo[ijk, _ηy, e], vgeo[ijk, _ηz, e]
          ζx, ζy, ζz = vgeo[ijk, _ζx, e], vgeo[ijk, _ζy, e], vgeo[ijk, _ζz, e]

          Q[ijk, sx, e] = ξx * l_fξ[ijk] + ηx * l_fη[ijk] + ζx * l_fζ[ijk]
          Q[ijk, sy, e] = ξy * l_fξ[ijk] + ηy * l_fη[ijk] + ζy * l_fζ[ijk]
          Q[ijk, sz, e] = ξz * l_fξ[ijk] + ηz * l_fη[ijk] + ζz * l_fζ[ijk]
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
  DFloat = eltype(R)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  nelem = size(auxstate)[end]

  l_R = MArray{Tuple{nRstate}, DFloat}(undef)
  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)

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
