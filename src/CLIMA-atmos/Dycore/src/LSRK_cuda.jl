function update!(::Val{nstates}, ::Val{Np}, rhs::CuArray, Q, elems, rka, rkb,
                 dt) where {nstates, Np}
  nelem = length(elems)
  @cuda(threads=Np, blocks=nelem,
        knl_update!(Val(nstates), Val(Np), rhs, Q, nelem, rka, rkb, dt))
end

function knl_update!(::Val{nstates}, ::Val{Np}, rhs, Q, nelem, rka, rkb,
                     dt) where {nstates, Np}
  n = threadIdx().x
  e = blockIdx().x

  @inbounds if n <= Np && e <= nelem
    for s = 1:nstates
      Q[n, s, e] += rkb * dt * rhs[n, s, e]
      rhs[n, s, e] *= rka
    end
  end
  nothing
end
