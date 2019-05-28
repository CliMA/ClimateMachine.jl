# FIXME: Add link to https://github.com/paranumal/libparanumal here and in
# advection (also update the license)

# {{{ Fill sendbuf on device with buf (for all dimensions)
function knl_fillsendbuf!(::Val{Np}, ::Val{nvar}, sendbuf, buf,
                          sendelems) where {Np, nvar}
  n = threadIdx().x
  e = blockIdx().x

  @inbounds if n <= Np && e <= length(sendelems)
    re = sendelems[e]
    for s = 1:nvar
      sendbuf[n, s, e] = buf[n, s, re]
    end
  end
  nothing
end
# }}}

# {{{ Fill buf on device with recvbuf (for all dimensions)
function knl_transferrecvbuf!(::Val{Np}, ::Val{nvar}, buf, recvbuf, nelem,
                              nrealelem) where {Np, nvar}
  n = threadIdx().x
  e = blockIdx().x

  @inbounds if n <= Np && e <= nelem
    for s = 1:nvar
      buf[n, s, nrealelem + e] = recvbuf[n, s, e]
    end
  end
  nothing
end
# }}}

# {{{ MPI Buffer handling
function fillsendbuf!(sendbuf, d_sendbuf::CuArray, d_buf::CuArray, d_sendelems)
  nsendelem = length(d_sendelems)
  Np = size(d_buf, 1)
  nvar = size(d_buf, 2)
  if nsendelem > 0
    @cuda(threads=Np, blocks=nsendelem,
          knl_fillsendbuf!(Val(Np), Val(nvar), d_sendbuf, d_buf, d_sendelems))
    copyto!(sendbuf, d_sendbuf)
  end
end

function transferrecvbuf!(d_recvbuf::CuArray, recvbuf, d_buf::CuArray, nrealelem)
  nrecvelem = size(recvbuf)[end]
  Np = size(d_buf, 1)
  nvar = size(d_buf, 2)
  if nrecvelem > 0
    copyto!(d_recvbuf, recvbuf)
    @cuda(threads=Np, blocks=nrecvelem,
          knl_transferrecvbuf!(Val(Np), Val(nvar), d_buf, d_recvbuf,
                               nrecvelem, nrealelem))
  end
end
# }}}
