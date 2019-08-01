using .NumericalFluxes: GradNumericalFlux, diffusive_boundary_penalty!, diffusive_penalty!,
  DivNumericalFlux, numerical_flux!, numerical_boundary_flux!

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
const _JcV = Grids._JcV

const _nx, _ny, _nz = Grids._nx, Grids._ny, Grids._nz
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

"""
    volumerhs!(bl::BalanceLaw, Val(polyorder), rhs, Q, Qvisc, auxstate,
               vgeo, t, D, elems)

Computational kernel: Evaluate the volume integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function volumerhs!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                    rhs, Q, Qvisc, auxstate, vgeo, t,
                    ω, D, elems, increment) where {dim, polyorder}
  N = polyorder
  DFloat = eltype(Q)
  nstate = num_state(bl,DFloat)
  nviscstate = num_diffusive(bl,DFloat)
  nauxstate = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_F = @shmem DFloat (3, Nq, Nq, Nqk, nstate)
  s_ω = @shmem DFloat (Nq, )
  s_half_D = @shmem DFloat (Nq, Nq)
  l_rhs = @scratch DFloat (nstate, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nstate}, DFloat}(undef))
  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)
  l_F = MArray{Tuple{3, nstate}, DFloat}(undef)
  l_M = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξx = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξy = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξz = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ηx = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ηy = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ηz = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ζx = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ζy = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ζz = @scratch DFloat (Nq, Nq, Nqk) 3

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
          l_ξx[i, j, k] = vgeo[ijk, _ξx, e]
          l_ξy[i, j, k] = vgeo[ijk, _ξy, e]
          l_ξz[i, j, k] = vgeo[ijk, _ξz, e]
          l_ηx[i, j, k] = vgeo[ijk, _ηx, e]
          l_ηy[i, j, k] = vgeo[ijk, _ηy, e]
          l_ηz[i, j, k] = vgeo[ijk, _ηz, e]
          l_ζx[i, j, k] = vgeo[ijk, _ζx, e]
          l_ζy[i, j, k] = vgeo[ijk, _ζy, e]
          l_ζz[i, j, k] = vgeo[ijk, _ζz, e]

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(DFloat)
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

          flux!(bl, Grad{vars_state(bl,DFloat)}(l_F), Vars{vars_state(bl,DFloat)}(l_Q),
                Vars{vars_diffusive(bl,DFloat)}(l_Qvisc), Vars{vars_aux(bl,DFloat)}(l_aux), t)

          @unroll for s = 1:nstate
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          source!(bl, Vars{vars_state(bl,DFloat)}(l_S), Vars{vars_state(bl,DFloat)}(l_Q),
                  Vars{vars_aux(bl,DFloat)}(l_aux), t)

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] += l_S[s]
          end
          # end
        end
      end
    end
    @synchronize

    # Weak "outside metrics" derivative
    @unroll for s = 1:nstate
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            @unroll for n = 1:Nq
              Dni = s_half_D[n, i] * s_ω[n] / s_ω[i]
              Dnj = s_half_D[n, j] * s_ω[n] / s_ω[j]
              Nqk > 1 && (Dnk = s_half_D[n, k] * s_ω[n] / s_ω[k])

              # ξ-grid lines
              l_rhs[s, i, j, k] += l_ξx[i, j, k] * Dni * s_F[1, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξy[i, j, k] * Dni * s_F[2, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξz[i, j, k] * Dni * s_F[3, n, j, k, s]

              # η-grid lines
              l_rhs[s, i, j, k] += l_ηx[i, j, k] * Dnj * s_F[1, i, n, k, s]
              l_rhs[s, i, j, k] += l_ηy[i, j, k] * Dnj * s_F[2, i, n, k, s]
              l_rhs[s, i, j, k] += l_ηz[i, j, k] * Dnj * s_F[3, i, n, k, s]

              # ζ-grid lines
              if Nqk > 1
                l_rhs[s, i, j, k] += l_ζx[i, j, k] * Dnk * s_F[1, i, j, n, s]
                l_rhs[s, i, j, k] += l_ζy[i, j, k] * Dnk * s_F[2, i, j, n, s]
                l_rhs[s, i, j, k] += l_ζz[i, j, k] * Dnk * s_F[3, i, j, n, s]
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
            s_F[1,i,j,k,s] = l_M[i, j, k] * (l_ξx[i, j, k] * F1 +
                                              l_ξy[i, j, k] * F2 +
                                              l_ξz[i, j, k] * F3)
            s_F[2,i,j,k,s] = l_M[i, j, k] * (l_ηx[i, j, k] * F1 +
                                              l_ηy[i, j, k] * F2 +
                                              l_ηz[i, j, k] * F3)
            s_F[3,i,j,k,s] = l_M[i, j, k] * (l_ζx[i, j, k] * F1 +
                                              l_ζy[i, j, k] * F2 +
                                              l_ζz[i, j, k] * F3)
          end
        end
      end
    end
    @synchronize

    # Weak "inside metrics" derivative
    @unroll for s = 1:nstate
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            ijk = i + Nq * ((j-1) + Nq * (k-1))
            MI = vgeo[ijk, _MI, e]
            @unroll for n = 1:Nq
              Dni = s_half_D[n, i]
              Dnj = s_half_D[n, j]
              Nqk > 1 && (Dnk = s_half_D[n, k])
              # ξ-grid lines
              l_rhs[s, i, j, k] += MI * Dni * s_F[1, n, j, k, s]

              # η-grid lines
              l_rhs[s, i, j, k] += MI * Dnj * s_F[2, i, n, k, s]

              # ζ-grid lines
              Nqk > 1 && (l_rhs[s, i, j, k] += MI * Dnk * s_F[3, i, j, n, s])
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
    facerhs!(bl::BalanceLaw, Val(polyorder), divnumflux::DivNumericalFlux, 
             rhs, Q, Qvisc, auxstate,
             vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
             elems)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function facerhs!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, divnumflux::DivNumericalFlux,
                  rhs, Q, Qvisc, auxstate, vgeo, sgeo, t, vmapM, vmapP,
                  elemtobndy, elems) where {dim, polyorder}
  N = polyorder
  DFloat = eltype(Q)
  nstate = num_state(bl,DFloat)
  nviscstate = num_diffusive(bl,DFloat)
  nauxstate = num_aux(bl,DFloat)

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
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
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

        bctype = elemtobndy[f, e]
        if bctype == 0
          numerical_flux!(divnumflux, bl, l_F, nM, l_QM, l_QviscM, l_auxM, l_QP, l_QviscP,
                          l_auxP, t)
        else
          numerical_boundary_flux!(divnumflux, bl, l_F, nM, l_QM, l_QviscM, l_auxM, l_QP,
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

function volumeviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                          Q, Qvisc, auxstate, vgeo, t, D,
                          elems) where {dim, polyorder}  
  N = polyorder
  
  DFloat = eltype(Q)
  nstate = num_state(bl,DFloat)
  ngradstate = num_gradient(bl,DFloat)
  nviscstate = num_diffusive(bl,DFloat)
  nauxstate = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  ngradtransformstate = nstate

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
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          gradvariables!(bl, Vars{vars_gradient(bl,DFloat)}(l_G), Vars{vars_state(bl,DFloat)}(l_Q[:, i, j, k]),
                     Vars{vars_aux(bl,DFloat)}(l_aux[:, i, j, k]), t)
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

          diffusive!(bl, Vars{vars_diffusive(bl,DFloat)}(l_Qvisc), Grad{vars_gradient(bl,DFloat)}(l_gradG),
                     Vars{vars_state(bl,DFloat)}(l_Q[:, i, j, k]), Vars{vars_aux(bl,DFloat)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nviscstate
            Qvisc[ijk, s, e] = l_Qvisc[s]
          end
        end
      end
    end
    @synchronize
  end
end

function faceviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, gradnumflux::GradNumericalFlux,
                        Q, Qvisc, auxstate, vgeo, sgeo, t, vmapM, vmapP,
                        elemtobndy, elems) where {dim, polyorder}  
  N = polyorder
  DFloat = eltype(Q)
  nstate = num_state(bl,DFloat)
  ngradstate = num_gradient(bl,DFloat)
  nviscstate = num_diffusive(bl,DFloat)
  nauxstate = num_aux(bl,DFloat)

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

  ngradtransformstate = nstate

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
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:ngradtransformstate
          l_QM[s] = Q[vidM, s, eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        gradvariables!(bl, Vars{vars_gradient(bl,DFloat)}(l_GM), Vars{vars_state(bl,DFloat)}(l_QM),
                   Vars{vars_aux(bl,DFloat)}(l_auxM), t)

        # Load plus side data
        @unroll for s = 1:ngradtransformstate
          l_QP[s] = Q[vidP, s, eP]
        end

        @unroll for s = 1:nauxstate
          l_auxP[s] = auxstate[vidP, s, eP]
        end

        gradvariables!(bl, Vars{vars_gradient(bl,DFloat)}(l_GP), Vars{vars_state(bl,DFloat)}(l_QP),
                   Vars{vars_aux(bl,DFloat)}(l_auxP), t)

        bctype = elemtobndy[f, e]
        if bctype == 0
          diffusive_penalty!(gradnumflux, bl, l_Qvisc, nM, l_GM, l_QM, l_auxM, l_GP,
                                  l_QP, l_auxP, t)
        else
          diffusive_boundary_penalty!(gradnumflux, bl, l_Qvisc, nM, l_GM, l_QM, l_auxM,
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

  DFloat = eltype(Q)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_Qdof = MArray{Tuple{nvar}, DFloat}(undef)
  l_auxdof = MArray{Tuple{nauxstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for i in (1:Np; threadIdx().x)
      x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]

      @unroll for s = 1:nauxstate
        l_auxdof[s] = auxstate[i, s, e]
      end
      ic!(l_Qdof, x, y, z, l_auxdof)
      @unroll for n = 1:nvar
        Q[i, n, e] = l_Qdof[n]
      end
    end
  end
end

function initstate!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, state, auxstate, vgeo, elems, args...) where {dim, polyorder}
  N = polyorder
  DFloat = eltype(auxstate)
  nauxstate = num_aux(bl,DFloat)
  nstate = num_state(bl,DFloat)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_state = MArray{Tuple{nstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      coords = vgeo[n, _x, e], vgeo[n, _y, e], vgeo[n, _z, e]
      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end
      @unroll for s = 1:nstate
        l_state[s] = state[n, s, e]
      end
      init_state!(bl, Vars{vars_state(bl,DFloat)}(l_state), Vars{vars_aux(bl,DFloat)}(l_aux), coords, args...)
      @unroll for s = 1:nstate
        state[n, s, e] = l_state[s]
      end
    end
  end
end


"""
    initauxstate!(bl::BalanceLaw, Val(polyorder), auxstate, vgeo, elems)

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, auxstate, vgeo, elems) where {dim, polyorder}
  N = polyorder  
  DFloat = eltype(auxstate)
  nauxstate = num_aux(bl,DFloat)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      coords = vgeo[n, _x, e], vgeo[n, _y, e], vgeo[n, _z, e]
      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      init_aux!(bl, Vars{vars_aux(bl,DFloat)}(l_aux), coords)

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

  DFloat = eltype(vgeo)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_f = @shmem DFloat (3, Nq, Nq, Nqk)
  s_half_D = @shmem DFloat (Nq, Nq)

  l_f  = @scratch DFloat (Nq, Nq, Nqk) 3
  l_fd = @scratch DFloat (3, Nq, Nq, Nqk) 3

  l_J = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξd = @scratch DFloat (3, Nq, Nq, Nqk) 3
  l_ηd = @scratch DFloat (3, Nq, Nq, Nqk) 3
  l_ζd = @scratch DFloat (3, Nq, Nq, Nqk) 3

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
          l_ξd[1, i, j, k] = vgeo[ijk, _ξx, e]
          l_ξd[2, i, j, k] = vgeo[ijk, _ξy, e]
          l_ξd[3, i, j, k] = vgeo[ijk, _ξz, e]
          l_ηd[1, i, j, k] = vgeo[ijk, _ηx, e]
          l_ηd[2, i, j, k] = vgeo[ijk, _ηy, e]
          l_ηd[3, i, j, k] = vgeo[ijk, _ηz, e]
          l_ζd[1, i, j, k] = vgeo[ijk, _ζx, e]
          l_ζd[2, i, j, k] = vgeo[ijk, _ζy, e]
          l_ζd[3, i, j, k] = vgeo[ijk, _ζz, e]
          l_f[i, j, k] = s_f[1, i, j, k] = Q[ijk, s, e]
        end
      end
    end
    @synchronize

    # reference gradient: outside metrics
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          fξ = DFloat(0)
          fη = DFloat(0)
          fζ = DFloat(0)
          @unroll for n = 1:Nq
            Din = s_half_D[i, n]
            Djn = s_half_D[j, n]
            Nqk > 1 && (Dkn = s_half_D[k, n])

            # ξ-grid lines
            fξ += Din * s_f[1, n, j, k]

            # η-grid lines
            fη += Djn * s_f[1, i, n, k]

            # ζ-grid lines
            Nqk > 1 && (fζ += Dkn * s_f[1, i, j, n])
          end
          @unroll for d = 1:3
            l_fd[d, i, j, k] = l_ξd[d, i, j, k] * fξ +
                               l_ηd[d, i, j, k] * fη +
                               l_ζd[d, i, j, k] * fζ
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
            s_f[1,i,j,k] = l_J[i, j, k] * l_ξd[d, i, j, k] * l_f[i, j, k]
            s_f[2,i,j,k] = l_J[i, j, k] * l_ηd[d, i, j, k] * l_f[i, j, k]
            s_f[3,i,j,k] = l_J[i, j, k] * l_ζd[d, i, j, k] * l_f[i, j, k]
          end
        end
      end
      @synchronize
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            fd = DFloat(0)
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
function knl_apply_aux!(bl::BalanceLaw, ::Val{dim}, ::Val{N}, f!, Q,
                            QV, auxstate, t, elems) where {dim, N}
  DFloat = eltype(Q)
  nstate = num_state(bl,DFloat)
  nviscstate = num_diffusive(bl,DFloat)
  nauxstate = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nstate
        l_Q[s] = Q[n, s, e]
      end

      @unroll for s = 1:nviscstate
        l_Qvisc[s] = QV[n, s, e]
      end

      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      f!(bl, Vars{vars_state(bl,DFloat)}(l_Q), Vars{vars_diffusive(bl,DFloat)}(l_Qvisc), Vars{vars_aux(bl,DFloat)}(l_aux), t)

      @unroll for s = 1:nauxstate
        auxstate[n, s, e] = l_aux[s]
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
  DFloat = eltype(Q)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  nout = length(outstate)

  l_Q = MArray{Tuple{nstate}, DFloat}(undef)
  l_aux = MArray{Tuple{nauxstate}, DFloat}(undef)
  l_knl = MArray{Tuple{nout, Nq}, DFloat}(undef)
  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_int = @scratch DFloat (nout, Nq, Nq, Nqj) 2

  s_I = @shmem DFloat (Nq, Nq)

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
  DFloat = eltype(P)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  nout = length(outstate)

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_T = MArray{Tuple{nout}, DFloat}(undef)
  l_V = MArray{Tuple{nout}, DFloat}(undef)

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

"""
    knl_apply_filter!(::Val{dim}, ::Val{N}, ::Val{nstate}, ::Val{horizontal},
                      ::Val{vertical}, Q, ::Val{states}, filtermatrix,
                      elems) where {dim, N, nstate, states, horizontal, vertical}

Computational kernel: Applies the `filtermatrix` to the `states` of `Q`.

The arguments `horizontal` and `vertical` are used to control if the filter is
applied in the horizontal and vertical reference directions, respectively.
"""
function knl_apply_filter!(::Val{dim}, ::Val{N}, ::Val{nstate},
                           ::Val{horizontal}, ::Val{vertical}, Q,
                           ::Val{states}, filtermatrix,
                           elems) where {dim, N, nstate, horizontal, vertical,
                                         states}
  DFloat = eltype(Q)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  filterinξ = horizontal
  filterinη = dim == 2 ? vertical : horizontal
  filterinζ = dim == 2 ? false : vertical

  # Return if we are not filtering in any direction
  if !(filterinξ || filterinη || filterinζ)
    return
  end

  nfilterstates = length(states)

  s_filter = @shmem DFloat (Nq, Nq)
  s_Q = @shmem DFloat (Nq, Nq, Nqk, nfilterstates)
  l_Qfiltered = @scratch DFloat (nfilterstates, Nq, Nq, Nqk) 3

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_filter[i, j] = filtermatrix[i, j]
      end
    end
  end

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          @unroll for fs = 1:nfilterstates
            l_Qfiltered[fs, i, j, k] = zero(DFloat)
          end

          ijk = i + Nq * ((j-1) + Nq * (k-1))

          @unroll for fs = 1:nfilterstates
            s_Q[i, j, k, fs] = Q[ijk, states[fs], e]
          end
        end
      end
    end


    if filterinξ
      @synchronize
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            @unroll for n = 1:Nq
              @unroll for fs = 1:nfilterstates
                l_Qfiltered[fs, i, j, k] += s_filter[i, n] * s_Q[n, j, k, fs]
              end
            end
          end
        end
      end

      if filterinη || filterinζ
        @loop for k in (1:Nqk; threadIdx().z)
          @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
              @unroll for fs = 1:nfilterstates
                s_Q[i, j, k, fs] = l_Qfiltered[fs, i, j, k]
                l_Qfiltered[fs, i, j, k] = zero(DFloat)
              end
            end
          end
        end
      end
    end

    if filterinη
      @synchronize
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            @unroll for n = 1:Nq
              @unroll for fs = 1:nfilterstates
                l_Qfiltered[fs, i, j, k] += s_filter[j, n] * s_Q[i, n, k, fs]
              end
            end
          end
        end
      end

      if filterinζ
        @loop for k in (1:Nqk; threadIdx().z)
          @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
              @unroll for fs = 1:nfilterstates
                s_Q[i, j, k, fs] = l_Qfiltered[fs, i, j, k]
                (l_Qfiltered[fs, i, j, k] = zero(DFloat))
              end
            end
          end
        end
      end
    end

    if filterinζ
      @synchronize
      @loop for k in (1:Nqk; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            @unroll for n = 1:Nq
              @unroll for fs = 1:nfilterstates
                l_Qfiltered[fs, i, j, k] += s_filter[k, n] * s_Q[i, j, n, fs]
              end
            end
          end
        end
      end
    end

    # Store result
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for fs = 1:nfilterstates
            Q[ijk, states[fs], e] = l_Qfiltered[fs, i, j, k]
          end
        end
      end
    end

    @synchronize
  end
  nothing
end
