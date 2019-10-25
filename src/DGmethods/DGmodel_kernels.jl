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
    volumerhs!(bl::BalanceLaw, Val(N), rhs, Q, σ, α, vgeo, t, D, Ω)

Computational kernel: Evaluate the volume integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function volumerhs!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N},
                    rhs, Q, σ, α, vgeo, t, ω, D, Ω, increment) where {Nᵈ, N}
  DFloat = eltype(Q)

  nQ = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nᵈ == 2 ? 1 : Nq

  s_F = @shmem DFloat (3, Nq, Nq, Nqk, nQ)
  s_ω = @shmem DFloat (Nq, )
  s_half_D = @shmem DFloat (Nq, Nq)
  l_rhs = @scratch DFloat (nQ, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nQ}, DFloat}(undef))
  l_Q = MArray{Tuple{nQ}, DFloat}(undef)
  l_σ = MArray{Tuple{nσ}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)
  l_F = MArray{Tuple{3, nQ}, DFloat}(undef)
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

  @inbounds @loop for e in (Ω; blockIdx().x)
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

          @unroll for s = 1:nQ
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(DFloat)
          end

          @unroll for s = 1:nQ
            l_Q[s] = Q[ijk, s, e]
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
                             Vars{vars_state(bl,DFloat)}(l_Q),
                             Vars{vars_aux(bl,DFloat)}(l_α),
                             t)
          flux_diffusive!(bl,
                          Grad{vars_state(bl,DFloat)}(l_F),
                          Vars{vars_state(bl,DFloat)}(l_Q),
                          Vars{vars_diffusive(bl,DFloat)}(l_σ),
                          Vars{vars_aux(bl,DFloat)}(l_α),
                          t)

          @unroll for s = 1:nQ
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          fill!(l_S, -zeros(eltype(l_S)))
          source!(bl,
                  Vars{vars_state(bl,DFloat)}(l_S), Vars{vars_state(bl,DFloat)}(l_Q),
                  Vars{vars_aux(bl,DFloat)}(l_α),
                  Vars{vars_diffusive(bl,DFloat)}(l_σ),
                  t)

          @unroll for s = 1:nQ
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
            @unroll for s = 1:nQ
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
          @unroll for s = 1:nQ
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
          @unroll for s = 1:nQ
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
          @unroll for s = 1:nQ
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
    facerhs!(bl::BalanceLaw, Val(N),
            numfluxnondiff::NumericalFluxNonDiffusive,
            numfluxdiff::NumericalFluxDiffusive,
            rhs, Q, σ, α,
            vgeo, sgeo, t, ι⁻, ι⁺, ιᴮ,
            Ω)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function facerhs!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N},
                  numfluxnondiff::NumericalFluxNonDiffusive,
                  numfluxdiff::NumericalFluxDiffusive,
                  rhs, Q, σ, α, vgeo, sgeo, t, ι⁻, ι⁺, ιᴮ, Ω) where {Nᵈ, N}
  DFloat = eltype(Q)

  nQ = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  if Nᵈ == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif Nᵈ == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif Nᵈ == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  Nq = N + 1
  Nqk = Nᵈ == 2 ? 1 : Nq

  l_Q⁻ = MArray{Tuple{nQ}, DFloat}(undef)
  l_σ⁻ = MArray{Tuple{nσ}, DFloat}(undef)
  l_α⁻ = MArray{Tuple{nα}, DFloat}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify Q⁺
  l_Q⁺₁ = MArray{Tuple{nQ}, DFloat}(undef)
  l_Q⁺₂ = MArray{Tuple{nQ}, DFloat}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify α⁺
  l_α⁺₁ = MArray{Tuple{nα}, DFloat}(undef)
  l_α⁺₂ = MArray{Tuple{nα}, DFloat}(undef)

  l_σ⁺ = MArray{Tuple{nσ}, DFloat}(undef)

  l_Q_bot1 = MArray{Tuple{nQ}, DFloat}(undef)
  l_σ_bot1 = MArray{Tuple{nσ}, DFloat}(undef)
  l_α_bot1 = MArray{Tuple{nα}, DFloat}(undef)

  l_F = MArray{Tuple{nQ}, DFloat}(undef)

  @inbounds @loop for e in (Ω; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        n⁻ = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        id⁻, id⁺ = ι⁻[n, f, e], ι⁺[n, f, e]

        e⁻, e⁺ = e, ((id⁺ - 1) ÷ Np) + 1
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1,  ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nQ
          l_Q⁻[s] = Q[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nσ
          l_σ⁻[s] = σ[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nα
          l_α⁻[s] = α[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s = 1:nQ
          l_Q⁺₂[s] = l_Q⁺₁[s] = Q[vid⁺, s, e⁺]
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
                                       n⁻, l_Q⁻, l_α⁻, l_Q⁺₁, l_α⁺₁, t)

          numerical_flux_diffusive!(numfluxdiff, bl, l_F,
                                    n⁻, l_Q⁻, l_σ⁻, l_α⁻, l_Q⁺₂, l_σ⁺, l_α⁺₂, t)
        else
          if (Nᵈ == 2 && f == 3) || (Nᵈ == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nQ
              l_Q_bot1[s] = Q[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nσ
              l_σ_bot1[s] = σ[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nα
              l_α_bot1[s] = α[n + Nqk^2,s, e]
            end
          end
          numerical_boundary_flux_nondiffusive!(numfluxnondiff, bl, l_F,
                                                n⁻, l_Q⁻, l_α⁻, l_Q⁺₁, l_α⁺₁, bctype, t,
                                                l_Q_bot1, l_α_bot1)
          numerical_boundary_flux_diffusive!(numfluxdiff, bl, l_F, n⁻,
                                             l_Q⁻, l_σ⁻, l_α⁻,
                                             l_Q⁺₂, l_σ⁺, l_α⁺₂,
                                             bctype, t,
                                             l_Q_bot1, l_σ_bot1, l_α_bot1)
          surface_flux!(bl,
              Vars{vars_state(bl,DFloat)}(l_F),
              f,
              Vars{vars_state(bl,DFloat)}(l_Q⁻),
              Vars{vars_diffusive(bl,DFloat)}(l_σ⁻),
              Vars{vars_aux(bl,DFloat)}(l_α⁻),
              t)
        end

        #Update RHS
        @unroll for s = 1:nQ
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

function volumeviscterms!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N},
                          Q, σ, α, vgeo, t, D,
                          Ω) where {Nᵈ, N}
  DFloat = eltype(Q)

  nQ = num_state(bl,DFloat)
  nG = num_gradient(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nᵈ == 2 ? 1 : Nq

  s_G = @shmem DFloat (Nq, Nq, Nqk, nG)
  s_D = @shmem DFloat (Nq, Nq)

  l_Q = @scratch DFloat (nQ, Nq, Nq, Nqk) 3
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

  @inbounds @loop for e in (Ω; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:nQ
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nα
            l_α[s, i, j, k] = α[ijk, s, e]
          end

          fill!(l_G, -zero(eltype(l_G)))
          gradvariables!(bl,
                         Vars{vars_gradient(bl,DFloat)}(l_G), Vars{vars_state(bl,DFloat)}(l_Q[:, i, j, k]),
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
                     Vars{vars_state(bl,DFloat)}(l_Q[:, i, j, k]), Vars{vars_aux(bl,DFloat)}(l_α[:, i, j, k]),
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

function faceviscterms!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N},
                        gradnumpenalty::GradNumericalPenalty,
                        Q, σ, α, vgeo, sgeo, t, ι⁻, ι⁺,
                        ιᴮ, Ω) where {Nᵈ, N}
  DFloat = eltype(Q)

  nQ = num_state(bl,DFloat)
  nG = num_gradient(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  if Nᵈ == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif Nᵈ == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif Nᵈ == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end
  Nqk = Nᵈ == 2 ? 1 : N+1

  l_Q⁻ = MArray{Tuple{nQ}, DFloat}(undef)
  l_α⁻ = MArray{Tuple{nα}, DFloat}(undef)
  l_G⁻ = MArray{Tuple{nG}, DFloat}(undef)

  l_Q⁺ = MArray{Tuple{nQ}, DFloat}(undef)
  l_α⁺ = MArray{Tuple{nα}, DFloat}(undef)
  l_G⁺ = MArray{Tuple{nG}, DFloat}(undef)

  l_σ = MArray{Tuple{nσ}, DFloat}(undef)

  l_Q_bot1 = MArray{Tuple{nQ}, DFloat}(undef)
  l_α_bot1 = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (Ω; blockIdx().x)
    for f = 1:nface
      @loop for n in (1:Nfp; threadIdx().x)
        n⁻ = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        id⁻, id⁺ = ι⁻[n, f, e], ι⁺[n, f, e]

        e⁻, e⁺ = e, ((id⁺ - 1) ÷ Np) + 1
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1,  ((id⁺ - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nQ
          l_Q⁻[s] = Q[vid⁻, s, e⁻]
        end

        @unroll for s = 1:nα
          l_α⁻[s] = α[vid⁻, s, e⁻]
        end

        fill!(l_G⁻, -zero(eltype(l_G⁻)))
        gradvariables!(bl,
                       Vars{vars_gradient(bl,DFloat)}(l_G⁻),
                       Vars{vars_state(bl,DFloat)}(l_Q⁻),
                       Vars{vars_aux(bl,DFloat)}(l_α⁻),
                       t)

        # Load plus side data
        @unroll for s = 1:nQ
          l_Q⁺[s] = Q[vid⁺, s, e⁺]
        end

        @unroll for s = 1:nα
          l_α⁺[s] = α[vid⁺, s, e⁺]
        end

        fill!(l_G⁺, -zero(eltype(l_G⁺)))
        gradvariables!(bl,
                       Vars{vars_gradient(bl,DFloat)}(l_G⁺),
                       Vars{vars_state(bl,DFloat)}(l_Q⁺),
                       Vars{vars_aux(bl,DFloat)}(l_α⁺),
                       t)

        bctype = ιᴮ[f, e]
        fill!(l_σ, -zero(eltype(l_σ)))

        if bctype == 0
          diffusive_penalty!(gradnumpenalty, bl, l_σ, n⁻, l_G⁻, l_Q⁻,
                             l_α⁻, l_G⁺, l_Q⁺, l_α⁺, t)
        else
          if (Nᵈ == 2 && f == 3) || (Nᵈ == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nQ
              l_Q_bot1[s] = Q[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nα
              l_α_bot1[s] = α[n + Nqk^2,s, e]
            end
          end
          diffusive_boundary_penalty!(gradnumpenalty, bl, l_σ, n⁻, l_G⁻,
                                      l_Q⁻, l_α⁻, l_G⁺, l_Q⁺, l_α⁺, bctype,
                                      t, l_Q_bot1, l_α_bot1)
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

function initstate!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N}, Q, α, vgeo, Ω, args...) where {Nᵈ, N}
  DFloat = eltype(α)

  nα = num_aux(bl,DFloat)
  nQ = num_state(bl,DFloat)

  Nq = N + 1
  Nqk = Nᵈ == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_Q = MArray{Tuple{nQ}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (Ω; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      coords = vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e]
      @unroll for s = 1:nα
        l_α[s] = α[n, s, e]
      end
      @unroll for s = 1:nQ
        l_Q[s] = Q[n, s, e]
      end
      init_state!(bl,
                  Vars{vars_state(bl,DFloat)}(l_Q),
                  Vars{vars_aux(bl,DFloat)}(l_α),
                  coords, args...)
      @unroll for s = 1:nQ
        Q[n, s, e] = l_Q[s]
      end
    end
  end
end


"""
    initauxstate!(bl::BalanceLaw, Val(N), α, vgeo, Ω)

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N}, α, vgeo, Ω) where {Nᵈ, N}
  DFloat = eltype(α)

  nα = num_aux(bl,DFloat)

  Nq = N + 1
  Nqk = Nᵈ == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_α = MArray{Tuple{nα}, DFloat}(undef)

  @inbounds @loop for e in (Ω; blockIdx().x)
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
    knl_nodal_update_aux!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N}, f!, Q, α,
                          t, Ω) where {Nᵈ, N}

Update the auxiliary state array
"""
function knl_nodal_update_aux!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N}, f!, Q,
                               α, σ, t, Ω) where {Nᵈ, N}
  DFloat = eltype(Q)

  nQ = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nᵈ == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Q = MArray{Tuple{nQ}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)
  l_σ = MArray{Tuple{nσ}, DFloat}(undef)

  @inbounds @loop for e in (Ω; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nQ
        l_Q[s] = Q[n, s, e]
      end

      @unroll for s = 1:nα
        l_α[s] = α[n, s, e]
      end

      @unroll for s = 1:nσ
        l_σ[s] = σ[n, s, e]
      end

      f!(bl,
         Vars{vars_state(bl,DFloat)}(l_Q),
         Vars{vars_aux(bl,DFloat)}(l_α),
         Vars{vars_diffusive(bl,DFloat)}(l_σ),
         t)

      @unroll for s = 1:nα
        α[n, s, e] = l_α[s]
      end
    end
  end
end

"""
    knl_nodal_update_state!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N}, f!, Q, α,
                          t, Ω) where {Nᵈ, N}

Update the state array
"""
function knl_nodal_update_state!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N}, f!, Q,
                               α, σ, t, Ω) where {Nᵈ, N}
  DFloat = eltype(Q)

  nQ = num_state(bl,DFloat)
  nσ = num_diffusive(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1

  Nqk = Nᵈ == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Q = MArray{Tuple{nQ}, DFloat}(undef)
  l_α = MArray{Tuple{nα}, DFloat}(undef)
  l_σ = MArray{Tuple{nσ}, DFloat}(undef)

  @inbounds @loop for e in (Ω; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nQ
        l_Q[s] = Q[n, s, e]
      end

      @unroll for s = 1:nα
        l_α[s] = α[n, s, e]
      end

      @unroll for s = 1:nσ
        l_σ[s] = σ[n, s, e]
      end

      f!(bl,
         Vars{vars_state(bl,DFloat)}(l_Q),
         Vars{vars_aux(bl,DFloat)}(l_α),
         Vars{vars_diffusive(bl,DFloat)}(l_σ),
         t)

      @unroll for s = 1:nQ
        Q[n, s, e] = l_Q[s]
      end
    end
  end
end

"""
    knl_indefinite_stack_integral!(::Val{Nᵈ}, ::Val{N}, ::Val{nQ},
                                            ::Val{nα}, ::Val{nvertelem},
                                            int_knl!, Q, α, vgeo, Imat,
                                            Ω, ::Val{outstate}
                                           ) where {Nᵈ, N, nQ, nα,
                                                    outstate, nvertelem}

Computational kernel: compute indefinite integral along the vertical stack

See [`DGBalanceLaw`](@ref) for usage.
"""
function knl_indefinite_stack_integral!(bl::BalanceLaw, ::Val{Nᵈ}, ::Val{N},
                                        ::Val{nvertelem},
                                        Q, α, vgeo, Imat, Ω,
                                        ::Val{nout}
                                        ) where {Nᵈ, N, nvertelem, nout}
  DFloat = eltype(Q)
  nQ = num_state(bl,DFloat)
  nα = num_aux(bl,DFloat)

  Nq = N + 1
  Nqj = Nᵈ == 2 ? 1 : Nq

  l_Q = MArray{Tuple{nQ}, DFloat}(undef)
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

  @inbounds @loop for eh in (Ω; blockIdx().x)
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
            @unroll for s = 1:nQ
              l_Q[s] = Q[ijk, s, e]
            end

            @unroll for s = 1:nα
              l_α[s] = α[ijk, s, e]
            end

            integrate_aux!(bl,
                           Vars{vars_integrals(bl, DFloat)}(view(l_knl, :, k)),
                           Vars{vars_state(bl, DFloat)}(l_Q), Vars{vars_aux(bl,DFloat)}(l_α))

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

function knl_reverse_indefinite_stack_integral!(::Val{Nᵈ}, ::Val{N},
                                                ::Val{nvertelem}, α, Ω,
                                                ::Val{nout}
                                               ) where {Nᵈ, N, nvertelem,
                                                        nout}
  DFloat = eltype(α)

  Nq = N + 1
  Nqj = Nᵈ == 2 ? 1 : Nq

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_T = MArray{Tuple{nout}, DFloat}(undef)
  l_V = MArray{Tuple{nout}, DFloat}(undef)

  @inbounds @loop for eh in (Ω; blockIdx().x)
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

# TODO: Generalize to more than one field?
function knl_copy_stack_field_down!(::Val{dim}, ::Val{N}, ::Val{nvertelem},
                                    auxstate, elems, ::Val{fldin},
                                    ::Val{fldout}) where {dim, N, nvertelem,
                                                          fldin, fldout}
  DFloat = eltype(auxstate)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  @inbounds @loop for eh in (elems; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        ijk = i + Nq * ((j-1) + Nqj * (Nq-1))
        et = nvertelem + (eh - 1) * nvertelem
        val = auxstate[ijk, fldin, et]

        # Loop up the stack of elements
        for ev = 1:nvertelem
          e = ev + (eh - 1) * nvertelem
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            auxstate[ijk, fldout, e] = val
          end
        end
      end
    end
  end
  nothing
end
