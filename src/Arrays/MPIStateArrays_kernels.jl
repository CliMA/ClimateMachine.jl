function knl_fillsendbuf!(::Val{Np}, ::Val{nvar}, sendbuf, buf,
                          vmapsend, nvmapsend) where {Np, nvar}

  @inbounds @loop for i in (1:nvmapsend;
                            threadIdx().x + blockDim().x * (blockIdx().x-1))
    e, n = fldmod1(vmapsend[i], Np)
    @unroll for s = 1:nvar
      sendbuf[s, i] = buf[n, s, e]
    end
  end
  nothing
end

function knl_transferrecvbuf!(::Val{Np}, ::Val{nvar}, buf, recvbuf,
                              vmaprecv, nvmaprecv) where {Np, nvar}

  @inbounds @loop for i in (1:nvmaprecv;
                            threadIdx().x + blockDim().x * (blockIdx().x-1))
    e, n = fldmod1(vmaprecv[i], Np)
    @unroll for s = 1:nvar
      buf[n, s, e] = recvbuf[s, i]
    end
  end
  nothing
end
