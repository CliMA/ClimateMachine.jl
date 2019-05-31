function update!(::Val{nstates}, ::Val{Np}, rhs::CuArray, Q0, Q, elems, rka1, rka2, rkb, dt) where {nstates, Np}
  nelem = length(elems)
  @cuda(threads=Np, blocks=nelem,
        knl_update!(Val(nstates), Val(Np), rhs, Q0, Q, nelem, rka1, rka2, rkb, dt))
end

function knl_update!(::Val{nstates}, ::Val{Np}, rhs, Q0, Q, nelem, rka1, rka2, rkb, dt) where {nstates, Np}
  n = threadIdx().x
  e = blockIdx().x

  @inbounds if n <= Np && e <= nelem
    for s = 1:nstates
    	#FXG: might want to store Q as Q[s,n,e]
        Q[n, s, e] = rka1*Q0[n, s, e] + rka2*Q[n, s, e] + dt*rkb*rhs[n, s, e]
    end
  end
  nothing
end
