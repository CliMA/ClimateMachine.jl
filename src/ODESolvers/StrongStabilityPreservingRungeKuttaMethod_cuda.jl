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
        Q[n, s, e] = rka[1]*Q0[n, s, e] + rka[2]*Q[n, s, e] + dt*rkb*rhs[n, s, e]
    end
  end
  nothing
end
