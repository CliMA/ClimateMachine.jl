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
