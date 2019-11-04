using .NumericalFluxes: GradNumericalPenalty, diffusive_boundary_penalty!,
                        diffusive_penalty!,
                        NumericalFluxNonDiffusive, NumericalFluxDiffusive,
                        numerical_flux_nondiffusive!,
                        numerical_boundary_flux_nondiffusive!,
                        numerical_flux_diffusive!,
                        numerical_boundary_flux_diffusive!

using ..Mesh.Geometry

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
    volume_tendency!(bl::BalanceLaw, Val(N), rhs, Y, W, A, vgeo, t, D, E)

Computational kernel: Evaluate the volume integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function volume_tendency!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, ::direction,
                    rhs, Y, W, A, vgeo, t, ω, D, E, increment) where {Nd, N, direction}
  FT = eltype(Y)

  nY = num_state(bl,FT)
  nW = num_diffusive(bl,FT)
  nA = num_aux(bl,FT)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

  s_F = @shmem FT (3, Nq, Nq, Nqk, nY)
  s_ω = @shmem FT (Nq, )
  s_half_D = @shmem FT (Nq, Nq)
  l_rhs = @scratch FT (nY, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nY}, FT}(undef))
  l_Y = MArray{Tuple{nY}, FT}(undef)
  l_W = MArray{Tuple{nW}, FT}(undef)
  l_A = MArray{Tuple{nA}, FT}(undef)
  l_F = MArray{Tuple{3, nY}, FT}(undef)
  l_M = @scratch FT (Nq, Nq, Nqk) 3

  l_ξ1x1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x3 = @scratch FT (Nq, Nq, Nqk) 3
  if Nd == 3 || (Nd == 2 && direction == EveryDirection)
    l_ξ2x1 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ2x2 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ2x3 = @scratch FT (Nq, Nq, Nqk) 3
  end
  if Nd == 3 && direction == EveryDirection
    l_ξ3x1 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ3x2 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ3x3 = @scratch FT (Nq, Nq, Nqk) 3
  end

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      s_ω[j] = ω[j]
      @loop for i in (1:Nq; threadIdx().x)
        s_half_D[i, j] = D[i, j] / 2
      end
    end
  end

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          l_M[i, j, k] = vgeo[ijk, _M, e]
          l_ξ1x1[i, j, k] = vgeo[ijk, _ξ1x1, e]
          l_ξ1x2[i, j, k] = vgeo[ijk, _ξ1x2, e]
          l_ξ1x3[i, j, k] = vgeo[ijk, _ξ1x3, e]
          if Nd == 3 || (Nd == 2 && direction == EveryDirection)
            l_ξ2x1[i, j, k] = vgeo[ijk, _ξ2x1, e]
            l_ξ2x2[i, j, k] = vgeo[ijk, _ξ2x2, e]
            l_ξ2x3[i, j, k] = vgeo[ijk, _ξ2x3, e]
          end
          if Nd == 3 && direction == EveryDirection
            l_ξ3x1[i, j, k] = vgeo[ijk, _ξ3x1, e]
            l_ξ3x2[i, j, k] = vgeo[ijk, _ξ3x2, e]
            l_ξ3x3[i, j, k] = vgeo[ijk, _ξ3x3, e]
          end

          @unroll for s = 1:nY
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(FT)
          end

          @unroll for s = 1:nY
            l_Y[s] = Y[ijk, s, e]
          end

          @unroll for s = 1:nW
            l_W[s] = W[ijk, s, e]
          end

          @unroll for s = 1:nA
            l_A[s] = A[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_nondiffusive!(bl,
                             Grad{vars_state(bl,FT)}(l_F),
                             Vars{vars_state(bl,FT)}(l_Y),
                             Vars{vars_aux(bl,FT)}(l_A),
                             t)
          flux_diffusive!(bl,
                          Grad{vars_state(bl,FT)}(l_F),
                          Vars{vars_state(bl,FT)}(l_Y),
                          Vars{vars_diffusive(bl,FT)}(l_W),
                          Vars{vars_aux(bl,FT)}(l_A),
                          t)

          @unroll for s = 1:nY
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          fill!(l_S, -zero(eltype(l_S)))
          source!(bl,
                  Vars{vars_state(bl,FT)}(l_S), Vars{vars_state(bl,FT)}(l_Y),
                  Vars{vars_aux(bl,FT)}(l_A),
                  t)

          @unroll for s = 1:nY
            l_rhs[s, i, j, k] += l_S[s]
          end
          # end
        end
      end
    end
    @synchronize

    # Weak "outside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          @unroll for n = 1:Nq
            @unroll for s = 1:nY
              Dni = s_half_D[n, i] * s_ω[n] / s_ω[i]
              if Nd == 3 || (Nd == 2 && direction == EveryDirection)
                Dnj = s_half_D[n, j] * s_ω[n] / s_ω[j]
              end
              if Nd == 3 && direction == EveryDirection
                Dnk = s_half_D[n, k] * s_ω[n] / s_ω[k]
              end

              # ξ1-grid lines
              l_rhs[s, i, j, k] += l_ξ1x1[i, j, k] * Dni * s_F[1, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξ1x2[i, j, k] * Dni * s_F[2, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξ1x3[i, j, k] * Dni * s_F[3, n, j, k, s]

              # ξ2-grid lines
              if Nd == 3 || (Nd == 2 && direction == EveryDirection)
                l_rhs[s, i, j, k] += l_ξ2x1[i, j, k] * Dnj * s_F[1, i, n, k, s]
                l_rhs[s, i, j, k] += l_ξ2x2[i, j, k] * Dnj * s_F[2, i, n, k, s]
                l_rhs[s, i, j, k] += l_ξ2x3[i, j, k] * Dnj * s_F[3, i, n, k, s]
              end

              # ξ3-grid lines
              if Nd == 3 && direction == EveryDirection
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
          @unroll for s = 1:nY
            F1, F2, F3 = s_F[1,i,j,k,s], s_F[2,i,j,k,s], s_F[3,i,j,k,s]
            s_F[1,i,j,k,s] = l_M[i, j, k] * (l_ξ1x1[i, j, k] * F1 +
                                              l_ξ1x2[i, j, k] * F2 +
                                              l_ξ1x3[i, j, k] * F3)
            if Nd == 3 || (Nd == 2 && direction == EveryDirection)
              s_F[2,i,j,k,s] = l_M[i, j, k] * (l_ξ2x1[i, j, k] * F1 +
                                               l_ξ2x2[i, j, k] * F2 +
                                               l_ξ2x3[i, j, k] * F3)
            end
            if Nd == 3 && direction == EveryDirection
              s_F[3,i,j,k,s] = l_M[i, j, k] * (l_ξ3x1[i, j, k] * F1 +
                                               l_ξ3x2[i, j, k] * F2 +
                                               l_ξ3x3[i, j, k] * F3)
            end
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
          @unroll for s = 1:nY
            @unroll for n = 1:Nq
              # ξ1-grid lines
              l_rhs[s, i, j, k] += MI * s_half_D[n, i] * s_F[1, n, j, k, s]

              # ξ2-grid lines
              if Nd == 3 || (Nd == 2 && direction == EveryDirection)
                l_rhs[s, i, j, k] += MI * s_half_D[n, j] * s_F[2, i, n, k, s]
              end

              # ξ3-grid lines
              if Nd == 3 && direction == EveryDirection
                l_rhs[s, i, j, k] += MI * s_half_D[n, k] * s_F[3, i, j, n, s]
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
          @unroll for s = 1:nY
            rhs[ijk, s, e] = l_rhs[s, i, j, k]
          end
        end
      end
    end
    @synchronize
  end
  nothing
end

function volume_tendency!(bl::BalanceLaw, ::Val{Nd}, ::Val{polyorder},
                    ::VerticalDirection, rhs, Y, Qvisc, auxstate, vgeo, t,
                    ω, D, elems, increment) where {Nd, polyorder}
  N = polyorder
  FT = eltype(Y)
  nstate = num_state(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

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

  l_ζx1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ζx2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ζx3 = @scratch FT (Nq, Nq, Nqk) 3

  _ζx1 = Nd == 2 ? _ξ2x1 : _ξ3x1
  _ζx2 = Nd == 2 ? _ξ2x2 : _ξ3x2
  _ζx3 = Nd == 2 ? _ξ2x3 : _ξ3x3

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
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(FT)
          end

          @unroll for s = 1:nstate
            l_Q[s] = Y[ijk, s, e]
          end

          @unroll for s = 1:nviscstate
            l_Qvisc[s] = Qvisc[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s] = auxstate[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_nondiffusive!(bl, Grad{vars_state(bl,FT)}(l_F),
                             Vars{vars_state(bl,FT)}(l_Q),
                             Vars{vars_aux(bl,FT)}(l_aux), t)
          flux_diffusive!(bl, Grad{vars_state(bl,FT)}(l_F),
                          Vars{vars_state(bl,FT)}(l_Q),
                          Vars{vars_diffusive(bl,FT)}(l_Qvisc),
                          Vars{vars_aux(bl,FT)}(l_aux), t)

          @unroll for s = 1:nstate
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          fill!(l_S, -zero(eltype(l_S)))
          source!(bl, Vars{vars_state(bl,FT)}(l_S), Vars{vars_state(bl,FT)}(l_Q),
                  Vars{vars_aux(bl,FT)}(l_aux), t)

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] += l_S[s]
          end
          # end
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
              if Nd == 2
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
              if Nd == 2
                Dnj = s_half_D[n, j]
                l_rhs[s, i, j, k] += MI * Dnj * s_F[3, i, n, k, s]
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
          @unroll for s = 1:nY
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
    face_tendency!(bl::BalanceLaw, Val(N),
            numfluxnondiff::NumericalFluxNonDiffusive,
            numfluxdiff::NumericalFluxDiffusive,
            rhs, Y, W, A,
            vgeo, sgeo, t, M⁻, M⁺, Mᴮ,
            E)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function face_tendency!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, ::direction,
                  numfluxnondiff::NumericalFluxNonDiffusive,
                  numfluxdiff::NumericalFluxDiffusive,
                  rhs, Y, W, A, vgeo, sgeo, t, M⁻, M⁺, Mᴮ, E) where {Nd, N, direction}
  FT = eltype(Y)

  nY = num_state(bl,FT)
  nW = num_diffusive(bl,FT)
  nA = num_aux(bl,FT)

  if Nd == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif Nd == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif Nd == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  faces = 1:nface
  if direction == VerticalDirection
    faces = nface-1:nface
  elseif direction == HorizontalDirection
    faces = 1:nface-2
  end

  Nq = N + 1
  Nqk = Nd == 2 ? 1 : Nq

  l_Y⁻ = MArray{Tuple{nY}, FT}(undef)
  l_W⁻ = MArray{Tuple{nW}, FT}(undef)
  l_A⁻ = MArray{Tuple{nA}, FT}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify Y⁺
  l_Y⁺₁ = MArray{Tuple{nY}, FT}(undef)
  l_Y⁺₂ = MArray{Tuple{nY}, FT}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify A⁺
  l_A⁺₁ = MArray{Tuple{nA}, FT}(undef)
  l_A⁺₂ = MArray{Tuple{nA}, FT}(undef)

  l_W⁺ = MArray{Tuple{nW}, FT}(undef)

  l_Y_bot1 = MArray{Tuple{nY}, FT}(undef)
  l_W_bot1 = MArray{Tuple{nW}, FT}(undef)
  l_A_bot1 = MArray{Tuple{nA}, FT}(undef)

  l_F = MArray{Tuple{nY}, FT}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        n⁻ = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        id⁻, id⁺ = M⁻[n, f, e], M⁺[n, f, e]

        e⁻, e⁺ = e, ((id⁺ - 1) ÷ Np) + 1
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1,  ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nY
          l_Y⁻[s] = Y[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nW
          l_W⁻[s] = W[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nA
          l_A⁻[s] = A[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s = 1:nY
          l_Y⁺₂[s] = l_Y⁺₁[s] = Y[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nW
          l_W⁺[s] = W[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nA
          l_A⁺₂[s] = l_A⁺₁[s] = A[vid⁺, s, e⁺]
        end

        bctype = Mᴮ[f, e]
        fill!(l_F, -zero(eltype(l_F)))

        if bctype == 0
          numerical_flux_nondiffusive!(numfluxnondiff, bl, l_F,
                                       n⁻, l_Y⁻, l_A⁻, l_Y⁺₁, l_A⁺₁, t)

          numerical_flux_diffusive!(numfluxdiff, bl, l_F,
                                    n⁻, l_Y⁻, l_W⁻, l_A⁻, l_Y⁺₂, l_W⁺, l_A⁺₂, t)
        else
          if (Nd == 2 && f == 3) || (Nd == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nY
              l_Y_bot1[s] = Y[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nW
              l_W_bot1[s] = W[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nA
              l_A_bot1[s] = A[n + Nqk^2,s, e]
            end
          end
          numerical_boundary_flux_nondiffusive!(numfluxnondiff, bl, l_F,
                                                n⁻, l_Y⁻, l_A⁻, l_Y⁺₁, l_A⁺₁, bctype, t,
                                                l_Y_bot1, l_A_bot1)
          numerical_boundary_flux_diffusive!(numfluxdiff, bl, l_F, n⁻,
                                             l_Y⁻, l_W⁻, l_A⁻,
                                             l_Y⁺₂, l_W⁺, l_A⁺₂,
                                             bctype, t,
                                             l_Y_bot1, l_W_bot1, l_A_bot1)
        end

        #Update RHS
        @unroll for s = 1:nY
          # FIXME: Should we pretch these?
          rhs[vid⁻, s, e⁻] -= vMI * sM * l_F[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function volume_diffusive_terms!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                                 ::direction, Y, W, A, vgeo, t, D, E) where {Nd, N, direction}
  FT = eltype(Y)

  nY = num_state(bl,FT)
  nG = num_gradient(bl,FT)
  nW = num_diffusive(bl,FT)
  nA = num_aux(bl,FT)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

  s_G = @shmem FT (Nq, Nq, Nqk, nG)
  s_D = @shmem FT (Nq, Nq)

  l_Y = @scratch FT (nY, Nq, Nq, Nqk) 3
  l_A = @scratch FT (nA, Nq, Nq, Nqk) 3
  l_G = MArray{Tuple{nG}, FT}(undef)
  l_W = MArray{Tuple{nW}, FT}(undef)
  l_∇G = MArray{Tuple{3, nG}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:nY
            l_Y[s, i, j, k] = Y[ijk, s, e]
          end

          @unroll for s = 1:nA
            l_A[s, i, j, k] = A[ijk, s, e]
          end

          fill!(l_G, -zero(eltype(l_G)))
          gradvariables!(bl,
                         Vars{vars_gradient(bl,FT)}(l_G), Vars{vars_state(bl,FT)}(l_Y[:, i, j, k]),
                         Vars{vars_aux(bl,FT)}(l_A[:, i, j, k]),
                         t)

          @unroll for s = 1:nG
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
          if Nd == 3 || (Nd == 2 && direction == EveryDirection)
            ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
          end
          if Nd == 3 && direction == EveryDirection
            ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
          end

          @unroll for s = 1:nG
            Gξ1 = Gξ2 = Gξ3 = zero(FT)
            @unroll for n = 1:Nq
              Gξ1 += s_D[i, n] * s_G[n, j, k, s]
              if Nd == 3 || (Nd == 2 && direction == EveryDirection)
                Gξ2 += s_D[j, n] * s_G[i, n, k, s]
              end
              if Nd == 3 && direction == EveryDirection
                Gξ3 += s_D[k, n] * s_G[i, j, n, s]
              end
            end
            l_∇G[1, s] = ξ1x1 * Gξ1
            l_∇G[2, s] = ξ1x2 * Gξ1
            l_∇G[3, s] = ξ1x3 * Gξ1

            if Nd == 3 || (Nd == 2 && direction == EveryDirection)
              l_∇G[1, s] += ξ2x1 * Gξ2
              l_∇G[2, s] += ξ2x2 * Gξ2
              l_∇G[3, s] += ξ2x3 * Gξ2
            end

            if Nd == 3 && direction == EveryDirection
              l_∇G[1, s] += ξ3x1 * Gξ3
              l_∇G[2, s] += ξ3x2 * Gξ3
              l_∇G[3, s] += ξ3x3 * Gξ3
            end
          end

          fill!(l_W, -zero(eltype(l_W)))
          diffusive!(bl,
                     Vars{vars_diffusive(bl,FT)}(l_W), Grad{vars_gradient(bl,FT)}(l_∇G),
                     Vars{vars_state(bl,FT)}(l_Y[:, i, j, k]), Vars{vars_aux(bl,FT)}(l_A[:, i, j, k]),
                     t)

          @unroll for s = 1:nW
            W[ijk, s, e] = l_W[s]
          end
        end
      end
    end
    @synchronize
  end
end

function face_diffusive_terms!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                        ::direction, gradnumpenalty::GradNumericalPenalty,
                        Y, W, A, vgeo, sgeo, t, M⁻, M⁺,Mᴮ,
                        E) where {Nd, N, direction}
  FT = eltype(Y)

  nY = num_state(bl,FT)
  nG = num_gradient(bl,FT)
  nW = num_diffusive(bl,FT)
  nA = num_aux(bl,FT)

  if Nd == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif Nd == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif Nd == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  faces = 1:nface
  if direction == VerticalDirection
    faces = nface-1:nface
  elseif direction == HorizontalDirection
    faces = 1:nface-2
  end

  Nqk = Nd == 2 ? 1 : N+1

  l_Y⁻ = MArray{Tuple{nY}, FT}(undef)
  l_A⁻ = MArray{Tuple{nA}, FT}(undef)
  l_G⁻ = MArray{Tuple{nG}, FT}(undef)

  l_Y⁺ = MArray{Tuple{nY}, FT}(undef)
  l_A⁺ = MArray{Tuple{nA}, FT}(undef)
  l_G⁺ = MArray{Tuple{nG}, FT}(undef)

  l_W = MArray{Tuple{nW}, FT}(undef)

  l_Y_bot1 = MArray{Tuple{nY}, FT}(undef)
  l_A_bot1 = MArray{Tuple{nA}, FT}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        n⁻ = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        id⁻, id⁺ = M⁻[n, f, e], M⁺[n, f, e]

        e⁻, e⁺ = e, ((id⁺ - 1) ÷ Np) + 1
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1,  ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nY
          l_Y⁻[s] = Y[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nA
          l_A⁻[s] = A[vid⁻, s, e⁻]
        end

        fill!(l_G⁻, -zero(eltype(l_G⁻)))
        gradvariables!(bl,
                       Vars{vars_gradient(bl,FT)}(l_G⁻),
                       Vars{vars_state(bl,FT)}(l_Y⁻),
                       Vars{vars_aux(bl,FT)}(l_A⁻),
                       t)

        # Load plus side data
        @unroll for s = 1:nY
          l_Y⁺[s] = Y[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nA
          l_A⁺[s] = A[vid⁺, s, e⁺]
        end

        fill!(l_G⁺, -zero(eltype(l_G⁺)))
        gradvariables!(bl,
                       Vars{vars_gradient(bl,FT)}(l_G⁺),
                       Vars{vars_state(bl,FT)}(l_Y⁺),
                       Vars{vars_aux(bl,FT)}(l_A⁺),
                       t)

        bctype = Mᴮ[f, e]
        fill!(l_W, -zero(eltype(l_W)))

        if bctype == 0
          diffusive_penalty!(gradnumpenalty, bl, l_W, n⁻, l_G⁻, l_Y⁻,
                             l_A⁻, l_G⁺, l_Y⁺, l_A⁺, t)
        else
          if (Nd == 2 && f == 3) || (Nd == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nY
              l_Y_bot1[s] = Y[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nA
              l_A_bot1[s] = A[n + Nqk^2,s, e]
            end
          end
          diffusive_boundary_penalty!(gradnumpenalty, bl, l_W, n⁻, l_G⁻,
                                      l_Y⁻, l_A⁻, l_G⁺, l_Y⁺, l_A⁺, bctype,
                                      t, l_Y_bot1, l_A_bot1)
        end

        @unroll for s = 1:nW
          W[vid⁻, s, e⁻] += vMI * sM * l_W[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function initstate!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, Y, A, vgeo, E, args...) where {Nd, N}
  FT = eltype(A)

  nA = num_aux(bl,FT)
  nY = num_state(bl,FT)

  Nq = N + 1
  Nqk = Nd == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_Y = MArray{Tuple{nY}, FT}(undef)
  l_A = MArray{Tuple{nA}, FT}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      coords = vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e]
      @unroll for s = 1:nA
        l_A[s] = A[n, s, e]
      end
      @unroll for s = 1:nY
        l_Y[s] = Y[n, s, e]
      end
      init_state!(bl,
                  Vars{vars_state(bl,FT)}(l_Y),
                  Vars{vars_aux(bl,FT)}(l_A),
                  coords, args...)
      @unroll for s = 1:nY
        Y[n, s, e] = l_Y[s]
      end
    end
  end
end


"""
    initauxstate!(bl::BalanceLaw, Val(N), A, vgeo, E)

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, A, vgeo, E) where {Nd, N}
  FT = eltype(A)

  nA = num_aux(bl,FT)

  Nq = N + 1
  Nqk = Nd == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_A = MArray{Tuple{nA}, FT}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nA
        l_A[s] = A[n, s, e]
      end

      init_aux!(bl,
                Vars{vars_aux(bl,FT)}(l_A),
                LocalGeometry(Val(N),vgeo,n,e))

      @unroll for s = 1:nA
        A[n, s, e] = l_A[s]
      end
    end
  end
end

"""
    knl_nodal_update_aux!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, f!, Y, A,
                          t, E) where {Nd, N}

Update the auxiliary state array
"""
function knl_nodal_update_aux!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, f!, Y,
                               A, t, E) where {Nd, N}
  FT = eltype(Y)

  nY = num_state(bl,FT)
  nW = num_diffusive(bl,FT)
  nA = num_aux(bl,FT)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Y = MArray{Tuple{nY}, FT}(undef)
  l_A = MArray{Tuple{nA}, FT}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nY
        l_Y[s] = Y[n, s, e]
      end

      @unroll for s = 1:nA
        l_A[s] = A[n, s, e]
      end

      f!(bl,
         Vars{vars_state(bl,FT)}(l_Y),
         Vars{vars_aux(bl,FT)}(l_A),
         t)

      @unroll for s = 1:nA
        A[n, s, e] = l_A[s]
      end
    end
  end
end

"""
    knl_indefinite_stack_integral!(::Val{Nd}, ::Val{N}, ::Val{nY},
                                            ::Val{nA}, ::Val{nvertelem},
                                            int_knl!, Y, A, vgeo, Imat,
                                            E, ::Val{outstate}
                                           ) where {Nd, N, nY, nA,
                                                    outstate, nvertelem}

Computational kernel: compute indefinite integral along the vertical stack

See [`DGBalanceLaw`](@ref) for usage.
"""
function knl_indefinite_stack_integral!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                                        ::Val{nvertelem},
                                        Y, A, vgeo, Imat, E,
                                        ::Val{nout}
                                        ) where {Nd, N, nvertelem, nout}
  FT = eltype(Y)
  nY = num_state(bl,FT)
  nA = num_aux(bl,FT)

  Nq = N + 1
  Nqj = Nd == 2 ? 1 : Nq

  l_Y = MArray{Tuple{nY}, FT}(undef)
  l_A = MArray{Tuple{nA}, FT}(undef)
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

  @inbounds @loop for eh in (E; blockIdx().x)
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
            @unroll for s = 1:nY
              l_Y[s] = Y[ijk, s, e]
            end

            @unroll for s = 1:nA
              l_A[s] = A[ijk, s, e]
            end

            integrate_aux!(bl,
                           Vars{vars_integrals(bl, FT)}(view(l_knl, :, k)),
                           Vars{vars_state(bl, FT)}(l_Y), Vars{vars_aux(bl,FT)}(l_A))

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
              A[ijk, ind_out, e] = l_int[ind_out, k, i, j]
              l_int[ind_out, k, i, j] = l_int[ind_out, Nq, i, j]
            end
          end
        end
      end
    end
  end
  nothing
end

function knl_reverse_indefinite_stack_integral!(::Val{Nd}, ::Val{N},
                                                ::Val{nvertelem}, A, E,
                                                ::Val{nout}
                                               ) where {Nd, N, nvertelem,
                                                        nout}
  FT = eltype(A)

  Nq = N + 1
  Nqj = Nd == 2 ? 1 : Nq

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_T = MArray{Tuple{nout}, FT}(undef)
  l_V = MArray{Tuple{nout}, FT}(undef)

  @inbounds @loop for eh in (E; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        ijk = i + Nq * ((j-1) + Nqj * (Nq-1))
        et = nvertelem + (eh - 1) * nvertelem
        @unroll for s = 1:nout
          l_T[s] = A[ijk, s, et]
        end

        # Loop up the stack of elements
        for ev = 1:nvertelem
          e = ev + (eh - 1) * nvertelem
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            @unroll for s = 1:nout
              l_V[s] = A[ijk, s, e]
            end
            @unroll for s = 1:nout
              A[ijk, nout+s, e] = l_T[s] - l_V[s]
            end
          end
        end
      end
    end
  end
  nothing
end
