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

# {{{ Kernel wrappers
function volumegrad!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                     d_grad::CuArray, d_Q, d_vgeo, gravity, d_D,
                     elems) where {dim, N, nmoist, ntrace}
  ngrad = _nstategrad + 3nmoist
  Qshape    = (ntuple(j->N+1, dim)..., size(d_Q, 2), size(d_Q, 3))
  gradshape = (ntuple(j->N+1, dim)..., ngrad, size(d_Q, 3))
  vgeoshape = (ntuple(j->N+1, dim)..., size(d_vgeo, 2), size(d_Q, 3))

  d_gradC = reshape(d_grad, gradshape)
  d_QC = reshape(d_Q, Qshape)
  d_vgeoC = reshape(d_vgeo, vgeoshape)

  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_volumegrad!(Val(dim), Val(N), Val(nmoist), Val(ntrace), d_gradC,
                        d_QC, d_vgeoC, gravity, d_D, nelem))
end

function facegrad!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                   d_grad::CuArray, d_Q, d_vgeo, d_sgeo, gravity, elems,
                   d_vmapM, d_vmapP, d_elemtobndy) where {dim, N, nmoist,
                                                          ntrace}
  nelem = length(elems)
  @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
        knl_facegrad!(Val(dim), Val(N), Val(nmoist), Val(ntrace), d_grad, d_Q,
                      d_vgeo, d_sgeo, gravity, nelem, d_vmapM, d_vmapP,
                      d_elemtobndy))
end

function volumerhs!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                    d_rhs::CuArray, d_Q, d_grad, d_vgeo, gravity, viscosity,
                    d_D, elems) where {dim, N, nmoist, ntrace}
  ngrad = _nstategrad + 3nmoist

  Qshape    = (ntuple(j->N+1, dim)..., size(d_Q, 2), size(d_Q, 3))
  gradshape = (ntuple(j->N+1, dim)..., ngrad, size(d_Q, 3))
  vgeoshape = (ntuple(j->N+1, dim)..., size(d_vgeo,2), size(d_Q, 3))

  d_rhsC = reshape(d_rhs, Qshape...)
  d_QC = reshape(d_Q, Qshape)
  d_gradC = reshape(d_grad, gradshape)
  d_vgeoC = reshape(d_vgeo, vgeoshape)

  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_volumerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), d_rhsC, d_QC,
                       d_gradC, d_vgeoC, gravity, viscosity, d_D, nelem))
end

function facerhs!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                  d_rhs::CuArray, d_Q, d_grad, d_vgeo, d_sgeo, gravity,
                  viscosity, elems, d_vmapM, d_vmapP,
                  d_elemtobndy) where {dim, N, nmoist, ntrace}
  nelem = length(elems)
  @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
        knl_facerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), d_rhs, d_Q,
                     d_grad, d_vgeo, d_sgeo, gravity, viscosity, nelem,
                     d_vmapM, d_vmapP, d_elemtobndy))
end

# }}}
