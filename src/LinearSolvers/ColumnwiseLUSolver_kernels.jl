using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end
using StaticArrays

"""
    band_lu_knl!(A, Val(Nq), Val(Nqi), Val(Nqj), Val(nstate), Val(nvertelem),
                 Val(nhorzelem), Val(eband))

This performs Band Gaussian Elimination (Algorithm 4.3.1 of Golub and Van
Loan).  The array `A` contains a band matrix for each vertical column.  For
example, `A[i, j, :, :, h]`, is the band matrix associated with the `(i, j)`th
degree of freedom in the horizontal element `h`.

Each `n` by `n` band matrix is assumed to have upper bandwidth `q` and lower
bandwidth `p` where `n = nstate * Nq * nvertelem` and `p = q = nstate * Nq *
eband - 1`.

Each band matrix is stored in the LAPACK band storage
<https://www.netlib.org/lapack/lug/node124.html>.  For example the band matrix

    B = [b₁₁ b₁₂ 0   0   0
         b₂₁ b₂₂ b₂₃ 0   0
         b₃₁ b₃₂ b₃₃ b₃₄ 0
         0   b₄₂ b₄₃ b₄₄ b₄₅
         0   0   b₅₃ b₅₄ b₅₅]

is stored as

    B = [0   b₁₂ b₂₃ b₃₄ b₄₅
         b₁₁ b₂₂ b₃₃ b₄₄ b₅₅
         b₂₁ b₃₂ b₄₃ b₅₄ 0
         b₃₁ b₄₂ b₅₃ 0   0]

### Reference
    @book{GolubVanLoan,
      title = {Matrix Computations},
      author = {Gene H. Golub and Charles F. Van Loan},
      edition = {4th},
      isbn = {9781421407944},
      publisher = {Johns Hopkins University Press},
      address = {Baltimore, MD, USA},
      url = {http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm},
      year = 2013
    }

"""
function band_lu_knl!(A, ::Val{Nq}, ::Val{Nqi}, ::Val{Nqj}, ::Val{nstate},
                      ::Val{nvertelem}, ::Val{nhorzelem},
                      ::Val{eband}) where {Nq, Nqi, Nqj, nstate, nvertelem,
                                           nhorzelem, eband}
  FT = eltype(A)
  n = nstate * Nq * nvertelem
  p = q = nstate * Nq * eband - 1

  @inbounds @loop for h in (1:nhorzelem; blockIdx().x)
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nqi; threadIdx().x)
        for v = 1:nvertelem
          for k = 1:Nq
            for s = 1:nstate
              kk = s + (k - 1) * nstate + (v - 1) * nstate * Nq

              Aq = A[i, j, q + 1, kk, h]
              for ii = 1:p
                A[i, j, q + ii + 1, kk, h] /= Aq
              end

              for jj = 1:q
                if jj + kk ≤ n
                  Ajj = A[i, j, q - jj + 1, jj + kk, h]
                  for ii = 1:p
                    A[i, j, q + ii - jj + 1, jj + kk, h] -=
                        A[i, j, q + ii + 1, kk, h] * Ajj
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end

"""
    band_forward_knl!(b, LU, Val(Nq), Val(Nqj), Val(nstate), Val(nvertelem),
                      Val(nhorzelem), Val(eband))

This performs Band Forward Substitution (Algorithm 4.3.2 of Golub and Van
Loan), i.e., the right-hand side `b` is replaced with the solution of `L*x=b`.

The array `b` is of the size `(Nq * Nqj * Nq, nstate, nvertelem * nhorzelem)`.

The LU-factorization array `LU` contains a single band matrix or one
for each vertical column, see [`band_lu!`](@ref).

Each `n` by `n` band matrix is assumed to have upper bandwidth `q` and lower
bandwidth `p` where `n = nstate * Nq * nvertelem` and `p = q = nstate * Nq *
eband - 1`.

### Reference
    @book{GolubVanLoan,
      title = {Matrix Computations},
      author = {Gene H. Golub and Charles F. Van Loan},
      edition = {4th},
      isbn = {9781421407944},
      publisher = {Johns Hopkins University Press},
      address = {Baltimore, MD, USA},
      url = {http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm},
      year = 2013
    }

"""
function band_forward_knl!(b, LU::AbstractArray{T,N}, ::Val{Nq}, ::Val{Nqj},
                           ::Val{nstate}, ::Val{nvertelem}, ::Val{nhorzelem},
                           ::Val{eband}) where {T, N, Nq, Nqj, nstate,
                                                nvertelem, nhorzelem, eband}
  FT = eltype(b)
  n = nstate * Nq * nvertelem
  p = q = eband * nstate * Nq - 1

  l_b = MArray{Tuple{p+1}, FT}(undef)

  @inbounds @loop for h in (1:nhorzelem; blockIdx().x)
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for v = 1:eband
          @unroll for k = 1:Nq
            @unroll for s = 1:nstate
              ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
              ee = v + nvertelem * (h - 1)
              ii  = s + (k - 1) * nstate + (v - 1) * nstate * Nq
              l_b[ii] =  nvertelem ≥ v ? b[ijk, s, ee] : zero(FT)
            end
          end
        end

        for v = 1:nvertelem
          @unroll for k = 1:Nq
            @unroll for s = 1:nstate
              jj = s + (k - 1) * nstate + (v - 1) * nstate * Nq

              @unroll for ii = 2:p+1
                Lii = N == 2 ? LU[ii+q, jj] : LU[i, j, ii+q, jj, h]
                l_b[ii] -= Lii * l_b[1]
              end

              ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
              ee = v + nvertelem * (h - 1)

              b[ijk, s, ee] = l_b[1]

              @unroll for ii = 1:p
                l_b[ii] = l_b[ii + 1]
              end

              if jj + p < n
                (idx, si) = fldmod1(jj + p + 1, nstate)
                (vi, ki) = fldmod1(idx, Nq)

                ijk = i + Nqj * (j-1) + Nq * Nqj * (ki-1)
                ee = vi + nvertelem * (h - 1)

                l_b[p + 1] = b[ijk, si, ee]
              end
            end
          end
        end
      end
    end
  end

  nothing
end

"""
    band_back_knl!(b, LU, Val(Nq), Val(Nqj), Val(nstate), Val(nvertelem),
                   Val(nhorzelem), Val(eband))

This performs Band Back Substitution (Algorithm 4.3.3 of Golub and Van
Loan), i.e., the right-hand side `b` is replaced with the solution of `U*x=b`.

The array `b` is of the size `(Nq * Nqj * Nq, nstate, nvertelem * nhorzelem)`.

The LU-factorization array `LU` contains a single band matrix or one
for each vertical column, see [`band_lu!`](@ref).

Each `n` by `n` band matrix is assumed to have upper bandwidth `q` and lower
bandwidth `p` where `n = nstate * Nq * nvertelem` and `p = q = nstate * Nq *
eband - 1`.

### Reference
    @book{GolubVanLoan,
      title = {Matrix Computations},
      author = {Gene H. Golub and Charles F. Van Loan},
      edition = {4th},
      isbn = {9781421407944},
      publisher = {Johns Hopkins University Press},
      address = {Baltimore, MD, USA},
      url = {http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm},
      year = 2013
    }

"""
function band_back_knl!(b, LU::AbstractArray{T, N}, ::Val{Nq}, ::Val{Nqj},
                        ::Val{nstate}, ::Val{nvertelem}, ::Val{nhorzelem},
                        ::Val{eband}) where {T, N, Nq, Nqj, nstate, nvertelem,
                                             nhorzelem, eband}
  FT = eltype(b)
  n = nstate * Nq * nvertelem
  q = nstate * Nq * eband - 1

  l_b = MArray{Tuple{q+1}, FT}(undef)

  @inbounds @loop for h in (1:nhorzelem; blockIdx().x)
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for v = nvertelem:-1:(nvertelem - eband + 1)
          @unroll for k = Nq:-1:1
            @unroll for s = nstate:-1:1
              vi = eband - nvertelem + v
              ii = s + (k - 1) * nstate + (vi - 1) * nstate * Nq

              ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
              ee = v + nvertelem * (h - 1)

              l_b[ii] =  b[ijk, s, ee]
            end
          end
        end

        for v = nvertelem:-1:1
          @unroll for k = Nq:-1:1
            @unroll for s = nstate:-1:1
              jj = s + (k - 1) * nstate + (v - 1) * nstate * Nq

              l_b[q + 1] /= N == 2 ? LU[q + 1, jj] : LU[i, j, q + 1, jj, h]

              @unroll for ii = 1:q
                Uii = N == 2 ? LU[ii, jj] : LU[i, j, ii, jj, h]
                l_b[ii] -= Uii * l_b[q + 1]
              end

              ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
              ee = v + nvertelem * (h - 1)

              b[ijk, s, ee] = l_b[q + 1]

              @unroll for ii = q:-1:1
                l_b[ii+1] = l_b[ii]
              end

              if jj - q  > 1
                (idx, si) = fldmod1(jj - q - 1, nstate)
                (vi, ki) = fldmod1(idx, Nq)

                ijk = i + Nqj * (j-1) + Nq * Nqj * (ki-1)
                ee = vi + nvertelem * (h - 1)

                l_b[1] = b[ijk, si, ee]
              end
            end
          end
        end
      end
    end
  end

  nothing
end


function knl_set_banded_data!(bl::BalanceLaw, ::Val{dim}, ::Val{N},
                              ::Val{nvertelem}, Q, kin, sin, evin, helems,
                              velems) where {dim, N, nvertelem}
  FT = eltype(Q)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq
  nstate = num_state(bl,FT)

  @inbounds @loop for eh in (helems; blockIdx().y)
    @loop for ev in (velems; blockIdx().x)
      e = ev + (eh - 1) * nvertelem
      @loop for k in (1:Nq; threadIdx().z)
        @loop for j in (1:Nqj; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
            @unroll for s = 1:nstate
              if k == kin && s == sin && evin == ev
                Q[ijk, s, e] = 1
              else
                Q[ijk, s, e] = 0
              end
            end
          end
        end
      end
    end
  end
  nothing
end

function knl_set_banded_matrix!(bl::BalanceLaw, ::Val{dim}, ::Val{N},
                                ::Val{nvertelem}, ::Val{p}, ::Val{q},
                                ::Val{eshift}, A::AbstractArray{FT, AN}, dQ,
                                kin, sin, evin, helems, vpelems
                               ) where {dim, N, nvertelem, p, q, eshift,
                                        FT, AN}
  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq
  nstate = num_state(bl,FT)

  # sin, kin, evin are the state, vertical fod, and vert element we are
  # handling

  # column index of matrix
  jj = sin + (kin - 1) * nstate + (evin - 1) * nstate * Nq

  # one thread is launch for dof that might contribute to column jj's band
  @inbounds @loop for eh in (helems; blockIdx().y)
    @loop for ep in (vpelems; blockIdx().x - eshift)
      # ep is the shift we need to add to evin to get the element we need to
      # consider
      ev = ep + evin
      if 1 ≤ ev ≤ nvertelem
        e = ev + (eh - 1) * nvertelem
        @loop for k in (1:Nq; threadIdx().z)
          @loop for j in (1:Nqj; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
              ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
              @unroll for s = 1:nstate
                # row index of matrix
                ii = s + (k - 1) * nstate + (ev - 1) * nstate * Nq
                # row band index
                bb = ii - jj
                # make sure we're in the bandwidth
                if -q ≤ bb ≤ p
                  if AN === 5
                    A[i, j, bb + q+1, jj, eh] = dQ[ijk, s, e]
                  elseif AN === 2
                    if (i, j, eh) == (1, 1, 1)
                      A[bb + q+1, jj] = dQ[ijk, s, e]
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end
  nothing
end

function knl_banded_matrix_vector_product!(bl::BalanceLaw, ::Val{dim}, ::Val{N},
                                           ::Val{nvertelem}, ::Val{p}, ::Val{q},
                                           dQ, A::AbstractArray{FT, AN}, Q,
                                           helems, velems
                                          ) where {dim, N, nvertelem, p, q, FT,
                                                   AN}

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq
  nstate = num_state(bl,FT)

  elo = div(q, Nq * nstate-1)
  eup = div(p, Nq * nstate-1)

  # matrix row loops
  @loop for eh in (helems; blockIdx().y)
    @loop for ev in (velems; blockIdx().x)
      e = ev + nvertelem * (eh - 1)
      @loop for k in (1:Nq; threadIdx().z)
        @loop for j in (1:Nqj; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            @unroll for s = 1:nstate
              Ax = -zero(FT)
              ii = s + (k - 1) * nstate + (ev - 1) * nstate * Nq

              # banded matrix column loops
              @unroll for evv = max(1, ev-elo):min(nvertelem, ev+eup)
                ee = evv + nvertelem * (eh - 1)
                @unroll for kk = 1:Nq
                  ijk = i + Nqj * (j-1) + Nq * Nqj * (kk-1)
                  @unroll for ss = 1:nstate
                    jj = ss + (kk - 1) * nstate + (evv - 1) * nstate * Nq
                    bb = ii - jj
                    if -q ≤ bb ≤ p
                      if AN === 5
                        Ax += A[i, j, bb + q + 1, jj, eh] * Q[ijk, ss, ee]
                      elseif AN === 2
                        Ax += A[bb + q + 1, jj] * Q[ijk, ss, ee]
                      end
                    end
                  end
                end
              end
              ijk = i + Nqj * (j-1) + Nq * Nqj * (k-1)
              dQ[ijk, s, e] = Ax
            end
          end
        end
      end
    end
  end
  nothing
end
