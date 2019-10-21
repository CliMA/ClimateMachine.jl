using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

const _M = Grids._M

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
  FT = eltype(Q)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  filterinξ1 = horizontal
  filterinξ2 = dim == 2 ? vertical : horizontal
  filterinξ3 = dim == 2 ? false : vertical

  # Return if we are not filtering in any direction
  if !(filterinξ1 || filterinξ2 || filterinξ3)
    return
  end

  nfilterstates = length(states)

  s_filter = @shmem FT (Nq, Nq)
  s_Q = @shmem FT (Nq, Nq, Nqk, nfilterstates)
  l_Qfiltered = @scratch FT (nfilterstates, Nq, Nq, Nqk) 3

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
            l_Qfiltered[fs, i, j, k] = zero(FT)
          end

          ijk = i + Nq * ((j-1) + Nq * (k-1))

          @unroll for fs = 1:nfilterstates
            s_Q[i, j, k, fs] = Q[ijk, states[fs], e]
          end
        end
      end
    end


    if filterinξ1
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

      if filterinξ2 || filterinξ3
        @loop for k in (1:Nqk; threadIdx().z)
          @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
              @unroll for fs = 1:nfilterstates
                s_Q[i, j, k, fs] = l_Qfiltered[fs, i, j, k]
                l_Qfiltered[fs, i, j, k] = zero(FT)
              end
            end
          end
        end
      end
    end

    if filterinξ2
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

      if filterinξ3
        @loop for k in (1:Nqk; threadIdx().z)
          @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
              @unroll for fs = 1:nfilterstates
                s_Q[i, j, k, fs] = l_Qfiltered[fs, i, j, k]
                (l_Qfiltered[fs, i, j, k] = zero(FT))
              end
            end
          end
        end
      end
    end

    if filterinξ3
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

function knl_apply_TMAR_filter!(::Val{nreduce}, ::Val{dim}, ::Val{N}, Q,
                                ::Val{filterstates}, vgeo,
                                elems) where {nreduce, dim, N, filterstates}
  FT = eltype(Q)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  nfilterstates = length(filterstates)

  # note that k is the second not 4th index (since this is scratch memory and
  # k needs to be persistent across threads)
  l_Q = @scratch FT (nfilterstates, Nq, Nq, Nqj) 2

  l_MJ = @scratch FT (Nq, Nq, Nqj) 2

  s_MJQ = @shmem FT (Nq * Nqj, nfilterstates)
  s_MJQclipped = @shmem FT (Nq * Nqj, nfilterstates)

  nelemperblock = 1

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        # loop up the pencil and load Q and MJ
        @unroll for k in 1:Nq
          ijk = i + Nq * ((j-1) + Nqj * (k-1))

          @unroll for sf = 1:nfilterstates
            s = filterstates[sf]
            l_Q[sf, k, i, j] = Q[ijk, s, e]
          end

          l_MJ[k, i, j] = vgeo[ijk, _M, e]
        end
      end
    end

    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for sf = 1:nfilterstates
          MJQ, MJQclipped = zero(FT), zero(FT)

          @unroll for k in 1:Nq
            MJ = l_MJ[k, i, j]
            Qs = l_Q[sf, k, i, j]
            Qsclipped = Qs ≥ 0 ? Qs : zero(Qs)

            MJQ += MJ * Qs
            MJQclipped += MJ * Qsclipped
          end

          ij = i + Nq * (j-1)

          s_MJQ[ij, sf] = MJQ
          s_MJQclipped[ij, sf] = MJQclipped
        end
      end
    end

    @synchronize

    @unroll for n = 11:-1:1
      if nreduce ≥ 2^n
        @loop for j in (1:Nqj; threadIdx().y)
          @loop for i in (1:Nq; threadIdx().x)
            ij = i + Nq * (j-1)
            ijshift = ij + 2^(n-1)
            if ij ≤ 2^(n-1) && ijshift ≤ Nq * Nqj
              @unroll for sf = 1:nfilterstates
                s_MJQ[ij, sf] += s_MJQ[ijshift, sf]
                s_MJQclipped[ij, sf] += s_MJQclipped[ijshift, sf]
              end
            end
          end
        end
        @synchronize
      end
    end

    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for sf = 1:nfilterstates
          qs_average = s_MJQ[1, sf]
          qs_clipped_average = s_MJQclipped[1, sf]

          r = qs_average > 0 ? qs_average / qs_clipped_average : zero(FT)

          s = filterstates[sf]
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))

            Qs = l_Q[sf, k, i, j]
            Q[ijk, s, e] = Qs ≥ 0 ? r*Qs : zero(Qs)
          end
        end
      end
    end
  end

  nothing
end
