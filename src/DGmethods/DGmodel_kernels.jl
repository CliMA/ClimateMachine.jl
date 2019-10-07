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
    volume_tendency!(bl::BalanceLaw, Val(N), rhs, Y, σ, α, vgeo, t, D, E)

Computational kernel: Evaluate the volume integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function volume_tendency!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                    rhs, Y, σ, α, vgeo, t, ω, D, E, increment) where {Nd, N}
  DFloat = eltype(Y)

  nY = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

  s_F = @shmem DFloat (3, Nq, Nq, Nqk, nY)
  s_ω = @shmem DFloat (Nq, )
  s_half_D = @shmem DFloat (Nq, Nq)
  l_rhs = @scratch DFloat (nY, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nY}, DFloat}(undef))
  l_Y = MArray{Tuple{nY}, DFloat}(undef)
  l_σ = MArray{Tuple{nσ}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)
  l_F = MArray{Tuple{3, nY}, DFloat}(undef)
  l_M = @scratch DFloat (Nq, Nq, Nqk) 3

  l_ξ1x1 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ1x2 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ1x3 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ2x1 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ2x2 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ2x3 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ3x1 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ3x2 = @scratch DFloat (Nq, Nq, Nqk) 3
  l_ξ3x3 = @scratch DFloat (Nq, Nq, Nqk) 3

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
          l_ξ2x1[i, j, k] = vgeo[ijk, _ξ2x1, e]
          l_ξ2x2[i, j, k] = vgeo[ijk, _ξ2x2, e]
          l_ξ2x3[i, j, k] = vgeo[ijk, _ξ2x3, e]
          l_ξ3x1[i, j, k] = vgeo[ijk, _ξ3x1, e]
          l_ξ3x2[i, j, k] = vgeo[ijk, _ξ3x2, e]
          l_ξ3x3[i, j, k] = vgeo[ijk, _ξ3x3, e]

          @unroll for s = 1:nY
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(DFloat)
          end

          @unroll for s = 1:nY
            l_Y[s] = Y[ijk, s, e]
          end

          @unroll for s = 1:nσ
            l_σ[s] = σ[ijk, s, e]
          end

          @unroll for s = 1:nα
            l_α[s] = α[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_nondiffusive!(bl,
                             Grad{vars_state(bl,DFloat)}(l_F),
                             Vars{vars_state(bl,DFloat)}(l_Y),
                             Vars{vars_aux(bl,DFloat)}(l_α),
                             t)
          flux_diffusive!(bl,
                          Grad{vars_state(bl,DFloat)}(l_F),
                          Vars{vars_state(bl,DFloat)}(l_Y),
                          Vars{vars_diffusive(bl,DFloat)}(l_σ),
                          Vars{vars_aux(bl,DFloat)}(l_α),
                          t)

          @unroll for s = 1:nY
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          fill!(l_S, -zero(eltype(l_S)))
          source!(bl,
                  Vars{vars_state(bl,DFloat)}(l_S), Vars{vars_state(bl,DFloat)}(l_Y),
                  Vars{vars_aux(bl,DFloat)}(l_α),
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
          @unroll for s = 1:nY
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
          @unroll for s = 1:nY
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
            rhs, Y, σ, α,
            vgeo, sgeo, t, ι⁻, ι⁺, ιᴮ,
            E)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function face_tendency!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                  numfluxnondiff::NumericalFluxNonDiffusive,
                  numfluxdiff::NumericalFluxDiffusive,
                  rhs, Y, σ, α, vgeo, sgeo, t, ι⁻, ι⁺, ιᴮ, E) where {Nd, N}
  DFloat = eltype(Y)

  nY = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

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

  Nq = N + 1
  Nqk = Nd == 2 ? 1 : Nq

  l_Y⁻ = MArray{Tuple{nY}, DFloat}(undef)
  l_σ⁻ = MArray{Tuple{nσ}, DFloat}(undef)
  l_α⁻ = MArray{Tuple{nα}, DFloat}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify Y⁺
  l_Y⁺₁ = MArray{Tuple{nY}, DFloat}(undef)
  l_Y⁺₂ = MArray{Tuple{nY}, DFloat}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify α⁺
  l_α⁺₁ = MArray{Tuple{nα}, DFloat}(undef)
  l_α⁺₂ = MArray{Tuple{nα}, DFloat}(undef)

  l_σ⁺ = MArray{Tuple{nσ}, DFloat}(undef)

  l_Y_bot1 = MArray{Tuple{nY}, DFloat}(undef)
  l_σ_bot1 = MArray{Tuple{nσ}, DFloat}(undef)
  l_α_bot1 = MArray{Tuple{nα}, DFloat}(undef)

  l_F = MArray{Tuple{nY}, DFloat}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        n⁻ = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        id⁻, id⁺ = ι⁻[n, f, e], ι⁺[n, f, e]

        e⁻, e⁺ = e, ((id⁺ - 1) ÷ Np) + 1
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1,  ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nY
          l_Y⁻[s] = Y[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nσ
          l_σ⁻[s] = σ[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nα
          l_α⁻[s] = α[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s = 1:nY
          l_Y⁺₂[s] = l_Y⁺₁[s] = Y[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nσ
          l_σ⁺[s] = σ[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nα
          l_α⁺₂[s] = l_α⁺₁[s] = α[vid⁺, s, e⁺]
        end

        bctype = ιᴮ[f, e]
        fill!(l_F, -zero(eltype(l_F)))

        if bctype == 0
          numerical_flux_nondiffusive!(numfluxnondiff, bl, l_F,
                                       n⁻, l_Y⁻, l_α⁻, l_Y⁺₁, l_α⁺₁, t)

          numerical_flux_diffusive!(numfluxdiff, bl, l_F,
                                    n⁻, l_Y⁻, l_σ⁻, l_α⁻, l_Y⁺₂, l_σ⁺, l_α⁺₂, t)
        else
          if (Nd == 2 && f == 3) || (Nd == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nY
              l_Y_bot1[s] = Y[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nσ
              l_σ_bot1[s] = σ[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nα
              l_α_bot1[s] = α[n + Nqk^2,s, e]
            end
          end
          numerical_boundary_flux_nondiffusive!(numfluxnondiff, bl, l_F,
                                                n⁻, l_Y⁻, l_α⁻, l_Y⁺₁, l_α⁺₁, bctype, t,
                                                l_Y_bot1, l_α_bot1)
          numerical_boundary_flux_diffusive!(numfluxdiff, bl, l_F, n⁻,
                                             l_Y⁻, l_σ⁻, l_α⁻,
                                             l_Y⁺₂, l_σ⁺, l_α⁺₂,
                                             bctype, t,
                                             l_Y_bot1, l_σ_bot1, l_α_bot1)
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
                          Y, σ, α, vgeo, t, D,
                          E) where {Nd, N}
  DFloat = eltype(Y)

  nY = num_state(bl,DFloat)
  nG = num_gradient(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

  s_G = @shmem DFloat (Nq, Nq, Nqk, nG)
  s_D = @shmem DFloat (Nq, Nq)

  l_Y = @scratch DFloat (nY, Nq, Nq, Nqk) 3
  l_α = @scratch DFloat (nα, Nq, Nq, Nqk) 3
  l_G = MArray{Tuple{nG}, DFloat}(undef)
  l_σ = MArray{Tuple{nσ}, DFloat}(undef)
  l_∇G = MArray{Tuple{3, nG}, DFloat}(undef)

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

          @unroll for s = 1:nα
            l_α[s, i, j, k] = α[ijk, s, e]
          end

          fill!(l_G, -zero(eltype(l_G)))
          gradvariables!(bl,
                         Vars{vars_gradient(bl,DFloat)}(l_G), Vars{vars_state(bl,DFloat)}(l_Y[:, i, j, k]),
                         Vars{vars_aux(bl,DFloat)}(l_α[:, i, j, k]),
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
          ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
          ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]

          @unroll for s = 1:nG
            Gξ1 = Gξ2 = Gξ3 = zero(DFloat)
            @unroll for n = 1:Nq
              Din = s_D[i, n]
              Djn = s_D[j, n]
              Nqk > 1 && (Dkn = s_D[k, n])

              Gξ1 += Din * s_G[n, j, k, s]
              Gξ2 += Djn * s_G[i, n, k, s]
              Nqk > 1 && (Gξ3 += Dkn * s_G[i, j, n, s])
            end
            l_∇G[1, s] = ξ1x1 * Gξ1 + ξ2x1 * Gξ2 + ξ3x1 * Gξ3
            l_∇G[2, s] = ξ1x2 * Gξ1 + ξ2x2 * Gξ2 + ξ3x2 * Gξ3
            l_∇G[3, s] = ξ1x3 * Gξ1 + ξ2x3 * Gξ2 + ξ3x3 * Gξ3
          end

          fill!(l_σ, -zero(eltype(l_σ)))
          diffusive!(bl,
                     Vars{vars_diffusive(bl,DFloat)}(l_σ), Grad{vars_gradient(bl,DFloat)}(l_∇G),
                     Vars{vars_state(bl,DFloat)}(l_Y[:, i, j, k]), Vars{vars_aux(bl,DFloat)}(l_α[:, i, j, k]),
                     t)

          @unroll for s = 1:nσ
            σ[ijk, s, e] = l_σ[s]
          end
        end
      end
    end
    @synchronize
  end
end

function face_diffusive_terms!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                        gradnumpenalty::GradNumericalPenalty,
                        Y, σ, α, vgeo, sgeo, t, ι⁻, ι⁺,
                        ιᴮ, E) where {Nd, N}
  DFloat = eltype(Y)

  nY = num_state(bl,DFloat)
  nG = num_gradient(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

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
  Nqk = Nd == 2 ? 1 : N+1

  l_Y⁻ = MArray{Tuple{nY}, DFloat}(undef)
  l_α⁻ = MArray{Tuple{nα}, DFloat}(undef)
  l_G⁻ = MArray{Tuple{nG}, DFloat}(undef)

  l_Y⁺ = MArray{Tuple{nY}, DFloat}(undef)
  l_α⁺ = MArray{Tuple{nα}, DFloat}(undef)
  l_G⁺ = MArray{Tuple{nG}, DFloat}(undef)

  l_σ = MArray{Tuple{nσ}, DFloat}(undef)

  l_Y_bot1 = MArray{Tuple{nY}, DFloat}(undef)
  l_α_bot1 = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        n⁻ = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        id⁻, id⁺ = ι⁻[n, f, e], ι⁺[n, f, e]

        e⁻, e⁺ = e, ((id⁺ - 1) ÷ Np) + 1
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1,  ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nY
          l_Y⁻[s] = Y[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nα
          l_α⁻[s] = α[vid⁻, s, e⁻]
        end

        fill!(l_G⁻, -zero(eltype(l_G⁻)))
        gradvariables!(bl,
                       Vars{vars_gradient(bl,DFloat)}(l_G⁻),
                       Vars{vars_state(bl,DFloat)}(l_Y⁻),
                       Vars{vars_aux(bl,DFloat)}(l_α⁻),
                       t)

        # Load plus side data
        @unroll for s = 1:nY
          l_Y⁺[s] = Y[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nα
          l_α⁺[s] = α[vid⁺, s, e⁺]
        end

        fill!(l_G⁺, -zero(eltype(l_G⁺)))
        gradvariables!(bl,
                       Vars{vars_gradient(bl,DFloat)}(l_G⁺),
                       Vars{vars_state(bl,DFloat)}(l_Y⁺),
                       Vars{vars_aux(bl,DFloat)}(l_α⁺),
                       t)

        bctype = ιᴮ[f, e]
        fill!(l_σ, -zero(eltype(l_σ)))

        if bctype == 0
          diffusive_penalty!(gradnumpenalty, bl, l_σ, n⁻, l_G⁻, l_Y⁻,
                             l_α⁻, l_G⁺, l_Y⁺, l_α⁺, t)
        else
          if (Nd == 2 && f == 3) || (Nd == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nY
              l_Y_bot1[s] = Y[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nα
              l_α_bot1[s] = α[n + Nqk^2,s, e]
            end
          end
          diffusive_boundary_penalty!(gradnumpenalty, bl, l_σ, n⁻, l_G⁻,
                                      l_Y⁻, l_α⁻, l_G⁺, l_Y⁺, l_α⁺, bctype,
                                      t, l_Y_bot1, l_α_bot1)
        end

        @unroll for s = 1:nσ
          σ[vid⁻, s, e⁻] += vMI * sM * l_σ[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function initstate!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, Y, α, vgeo, E, args...) where {Nd, N}
  DFloat = eltype(α)

  nα = num_aux(bl,DFloat)
  nY = num_state(bl,DFloat)

  Nq = N + 1
  Nqk = Nd == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_Y = MArray{Tuple{nY}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      coords = vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e]
      @unroll for s = 1:nα
        l_α[s] = α[n, s, e]
      end
      @unroll for s = 1:nY
        l_Y[s] = Y[n, s, e]
      end
      init_state!(bl,
                  Vars{vars_state(bl,DFloat)}(l_Y),
                  Vars{vars_aux(bl,DFloat)}(l_α),
                  coords, args...)
      @unroll for s = 1:nY
        Y[n, s, e] = l_Y[s]
      end
    end
  end
end


"""
    initauxstate!(bl::BalanceLaw, Val(N), α, vgeo, E)

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, α, vgeo, E) where {Nd, N}
  DFloat = eltype(α)

  nα = num_aux(bl,DFloat)

  Nq = N + 1
  Nqk = Nd == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_α = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nα
        l_α[s] = α[n, s, e]
      end

      init_aux!(bl,
                Vars{vars_aux(bl,DFloat)}(l_α),
                LocalGeometry(Val(N),vgeo,n,e))

      @unroll for s = 1:nα
        α[n, s, e] = l_α[s]
      end
    end
  end
end

"""
    knl_nodal_update_aux!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, f!, Y, α,
                          t, E) where {Nd, N}

Update the auxiliary state array
"""
function knl_nodal_update_aux!(bl::BalanceLaw, ::Val{Nd}, ::Val{N}, f!, Y,
                               α, t, E) where {Nd, N}
  DFloat = eltype(Y)

  nY = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nd == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Y = MArray{Tuple{nY}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (E; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nY
        l_Y[s] = Y[n, s, e]
      end

      @unroll for s = 1:nα
        l_α[s] = α[n, s, e]
      end

      f!(bl,
         Vars{vars_state(bl,DFloat)}(l_Y),
         Vars{vars_aux(bl,DFloat)}(l_α),
         t)

      @unroll for s = 1:nα
        α[n, s, e] = l_α[s]
      end
    end
  end
end

"""
    knl_indefinite_stack_integral!(::Val{Nd}, ::Val{N}, ::Val{nY},
                                            ::Val{nα}, ::Val{nvertelem},
                                            int_knl!, Y, α, vgeo, Imat,
                                            E, ::Val{outstate}
                                           ) where {Nd, N, nY, nα,
                                                    outstate, nvertelem}

Computational kernel: compute indefinite integral along the vertical stack

See [`DGBalanceLaw`](@ref) for usage.
"""
function knl_indefinite_stack_integral!(bl::BalanceLaw, ::Val{Nd}, ::Val{N},
                                        ::Val{nvertelem},
                                        Y, α, vgeo, Imat, E,
                                        ::Val{nout}
                                        ) where {Nd, N, nvertelem, nout}
  DFloat = eltype(Y)
  nY = num_state(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1
  Nqj = Nd == 2 ? 1 : Nq

  l_Y = MArray{Tuple{nY}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)
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

            @unroll for s = 1:nα
              l_α[s] = α[ijk, s, e]
            end

            integrate_aux!(bl,
                           Vars{vars_integrals(bl, DFloat)}(view(l_knl, :, k)),
                           Vars{vars_state(bl, DFloat)}(l_Y), Vars{vars_aux(bl,DFloat)}(l_α))

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
              α[ijk, ind_out, e] = l_int[ind_out, k, i, j]
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
                                                ::Val{nvertelem}, α, E,
                                                ::Val{nout}
                                               ) where {Nd, N, nvertelem,
                                                        nout}
  DFloat = eltype(α)

  Nq = N + 1
  Nqj = Nd == 2 ? 1 : Nq

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_T = MArray{Tuple{nout}, DFloat}(undef)
  l_V = MArray{Tuple{nout}, DFloat}(undef)

  @inbounds @loop for eh in (E; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        ijk = i + Nq * ((j-1) + Nqj * (Nq-1))
        et = nvertelem + (eh - 1) * nvertelem
        @unroll for s = 1:nout
          l_T[s] = α[ijk, s, et]
        end

        # Loop up the stack of elements
        for ev = 1:nvertelem
          e = ev + (eh - 1) * nvertelem
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            @unroll for s = 1:nout
              l_V[s] = α[ijk, s, e]
            end
            @unroll for s = 1:nout
              α[ijk, nout+s, e] = l_T[s] - l_V[s]
            end
          end
        end
      end
    end
  end
  nothing
end
