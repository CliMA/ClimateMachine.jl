# {{{ RHS functions
rhs!(::Type{Val{:vanilla}}, rhs, s, p, c) = vanillarhs!(rhs, s, p, c)

function vanillarhs!(rhs, state, parameters, configuration)
  mesh = configuration.mesh
  mpicomm = configuration.mpicomm
  sendreq = configuration.sendreq
  recvreq = configuration.recvreq
  host_recvQ = configuration.host_recvQ
  host_sendQ = configuration.host_sendQ
  device_recvQ = configuration.device_recvQ
  device_sendQ = configuration.device_sendQ
  sendelems = configuration.sendelems

  vgeo = configuration.vgeo
  sgeo = configuration.sgeo
  Dmat = configuration.D
  vmapM = configuration.vmapM
  vmapP = configuration.vmapP
  elemtobndy = configuration.elemtobndy

  Q = state.Q

  N   = parameters.N
  dim = parameters.dim

  nnabr = length(mesh.nabrtorank)
  nrealelem = length(mesh.realelems)

  # post MPI receives
  for n = 1:nnabr
    recvreq[n] = MPI.Irecv!((@view host_recvQ[:, :, mesh.nabrtorecv[n]]),
                            mesh.nabrtorank[n], 777, mpicomm)
  end

  # wait on (prior) MPI sends
  MPI.Waitall!(sendreq)

  # pack data in send buffer
  vanilla_fillsendQ!(Val(dim), Val(N), host_sendQ, device_sendQ, Q, sendelems)

  # post MPI sends
  for n = 1:nnabr
    sendreq[n] = MPI.Isend((@view host_sendQ[:, :, mesh.nabrtosend[n]]),
                           mesh.nabrtorank[n], 777, mpicomm)
  end

  # volume RHS computation
  vanilla_volumerhs!(Val(dim), Val(N), rhs, Q, vgeo, Dmat, mesh.realelems)

  # wait on MPI receives
  MPI.Waitall!(recvreq)

  # copy data to state vectors
  vanilla_transferrecvQ!(Val(dim), Val(N), device_recvQ, host_recvQ, Q,
                         nrealelem)

  # face RHS computation
  vanilla_facerhs!(Val(dim), Val(N), rhs, Q, vgeo, sgeo,
           mesh.realelems, vmapM, vmapP, elemtobndy)
end
# }}}

# {{{ MPI Buffer handling
function vanilla_fillsendQ!(::Val{dim}, ::Val{N}, host_sendQ,
                            device_sendQ::Array, Q, sendelems) where {dim, N}
  host_sendQ[:, :, :] .= Q[:, :, sendelems]
end

function vanilla_transferrecvQ!(::Val{dim}, ::Val{N}, device_recvQ::Array,
                                host_recvQ, Q, nrealelem) where {dim, N}
  Q[:, :, nrealelem+1:end] .= host_recvQ[:, :, :]
end
# }}}

# {{{ Volume RHS for 2-D
function vanilla_volumerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, D,
                            elems) where N
  DFloat = eltype(Q)

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, _nstate, nelem)
  rhs = reshape(rhs, Nq, Nq, _nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

  s_F = Array{DFloat}(undef, Nq, Nq, _nstate)
  s_G = Array{DFloat}(undef, Nq, Nq, _nstate)

  @inbounds for e in elems
    for j = 1:Nq, i = 1:Nq
      MJ = vgeo[i, j, _MJ, e]
      ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
      ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
      y = vgeo[i,j,_y,e]

      U, V = Q[i, j, _U, e], Q[i, j, _V, e]
      ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]

      P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*grav*y)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U + P
      fluxV_x = ρinv * U * V
      fluxE_x = ρinv * U * (E + P)

      fluxρ_y = V
      fluxU_y = ρinv * V * U
      fluxV_y = ρinv * V * V + P
      fluxE_y = ρinv * V * (E + P)

      s_F[i, j, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y)
      s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
      s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)
      s_F[i, j, _W] = 0
      s_F[i, j, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y)

      s_G[i, j, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y)
      s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
      s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
      s_G[i, j, _W] = 0
      s_G[i, j, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y)

      # buoyancy term
      rhs[i, j, _V, e] -= MJ * ρ * grav
    end

    # loop of ξ-grid lines
    for s = 1:_nstate, j = 1:Nq, i = 1:Nq, n = 1:Nq
      rhs[i, j, s, e] += D[n, i] * s_F[n, j, s]
    end
    # loop of η-grid lines
    for s = 1:_nstate, j = 1:Nq, i = 1:Nq, n = 1:Nq
      rhs[i, j, s, e] += D[n, j] * s_G[i, n, s]
    end
  end
end
#}}}

# {{{ Volume RHS for 3-D
function vanilla_volumerhs!(::Val{3}, ::Val{N}, rhs::Array, Q, vgeo, D,
                            elems) where N
  DFloat = eltype(Q)

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, Nq, _nstate, nelem)
  rhs = reshape(rhs, Nq, Nq, Nq, _nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

  s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
  s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
  s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)

  @inbounds for e in elems
    for k = 1:Nq, j = 1:Nq, i = 1:Nq
      MJ = vgeo[i, j, k, _MJ, e]
      ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
      ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
      ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
      z = vgeo[i,j,k,_z,e]

      U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
      ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

      P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*grav*z)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U  + P
      fluxV_x = ρinv * U * V
      fluxW_x = ρinv * U * W
      fluxE_x = ρinv * U * (E + P)

      fluxρ_y = V
      fluxU_y = ρinv * V * U
      fluxV_y = ρinv * V * V + P
      fluxW_y = ρinv * V * W
      fluxE_y = ρinv * V * (E + P)

      fluxρ_z = W
      fluxU_z = ρinv * W * U
      fluxV_z = ρinv * W * V
      fluxW_z = ρinv * W * W + P
      fluxE_z = ρinv * W * (E + P)

      s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
      s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
      s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
      s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
      s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

      s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
      s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
      s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
      s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
      s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

      s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
      s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
      s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
      s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
      s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

      # buoyancy term
      rhs[i, j, k, _W, e] -= MJ * ρ * grav
    end

    # loop of ξ-grid lines
    for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
      rhs[i, j, k, s, e] += D[n, i] * s_F[n, j, k, s]
    end
    # loop of η-grid lines
    for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
      rhs[i, j, k, s, e] += D[n, j] * s_G[i, n, k, s]
    end
    # loop of ζ-grid lines
    for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
      rhs[i, j, k, s, e] += D[n, k] * s_H[i, j, n, s]
    end
  end
end
#}}}

# {{{ Face RHS for 2-D
function vanilla_facerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, sgeo, elems,
                          vmapM, vmapP, elemtobndy) where N
  DFloat = eltype(Q)

  Np = (N+1)^2
  Nfp = N+1
  nface = 4

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        EM = Q[vidM, _E, eM]
        yM = vgeo[vidM, _y, eM]

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2)/(2*ρM) - ρM*grav*yM)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          EP = Q[vidP, _E, eP]
          yP = vgeo[vidP, _y, eP]
          PP = gdm1*(EP - (UP^2 + VP^2)/(2*ρP) - ρP*grav*yP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          ρP = ρM
          EP = EM
          PP = PM
        else
          error("Invalid boundary conditions $bc on face $f of element $e")
        end

        ρMinv = 1 / ρM
        fluxρM_x = UM
        fluxUM_x = ρMinv * UM * UM + PM
        fluxVM_x = ρMinv * UM * VM
        fluxEM_x = ρMinv * UM * (EM + PM)

        fluxρM_y = VM
        fluxUM_y = ρMinv * VM * UM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxEM_y = ρMinv * VM * (EM + PM)

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * UP * VP
        fluxEP_x = ρPinv * UP * (EP + PP)

        fluxρP_y = VP
        fluxUP_y = ρPinv * VP * UP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxEP_y = ρPinv * VP * (EP + PP)

        λM = ρMinv * abs(nxM * UM + nyM * VM) + sqrt(ρMinv * gamma_d * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP) + sqrt(ρPinv * gamma_d * PP)
        λ  =  max(λM, λP)

        #Compute Numerical Flux and Update
        fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                  - λ * (ρP - ρM)) / 2
        fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                  - λ * (UP - UM)) / 2
        fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                  - λ * (VP - VM)) / 2
        fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                  - λ * (EP - EM)) / 2


        #Update RHS
        rhs[vidM, _ρ, eM] -= sMJ * fluxρS
        rhs[vidM, _U, eM] -= sMJ * fluxUS
        rhs[vidM, _V, eM] -= sMJ * fluxVS
        rhs[vidM, _E, eM] -= sMJ * fluxES
      end
    end
  end
end
# }}}

# {{{ Face RHS for 3-D
function vanilla_facerhs!(::Val{3}, ::Val{N}, rhs::Array, Q, vgeo, sgeo, elems,
                          vmapM, vmapP, elemtobndy) where N
  DFloat = eltype(Q)

  Np = (N+1)^3
  Nfp = (N+1)^2
  nface = 6

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        WM = Q[vidM, _W, eM]
        EM = Q[vidM, _E, eM]
        zM = vgeo[vidM, _z, eM]

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*grav*zM)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          zP = vgeo[vidP, _z, eP]
          PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*grav*zP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM + nzM * WM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          WP = WM - 2 * UnM * nzM
          ρP = ρM
          EP = EM
          PP = PM
        else
          error("Invalid boundary conditions $bc on face $f of element $e")
        end

        ρMinv = 1 / ρM
        fluxρM_x = UM
        fluxUM_x = ρMinv * UM * UM + PM
        fluxVM_x = ρMinv * UM * VM
        fluxWM_x = ρMinv * UM * WM
        fluxEM_x = ρMinv * UM * (EM + PM)

        fluxρM_y = VM
        fluxUM_y = ρMinv * VM * UM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxWM_y = ρMinv * VM * WM
        fluxEM_y = ρMinv * VM * (EM + PM)

        fluxρM_z = WM
        fluxUM_z = ρMinv * WM * UM
        fluxVM_z = ρMinv * WM * VM
        fluxWM_z = ρMinv * WM * WM + PM
        fluxEM_z = ρMinv * WM * (EM + PM)

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * UP * VP
        fluxWP_x = ρPinv * UP * WP
        fluxEP_x = ρPinv * UP * (EP + PP)

        fluxρP_y = VP
        fluxUP_y = ρPinv * VP * UP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxWP_y = ρPinv * VP * WP
        fluxEP_y = ρPinv * VP * (EP + PP)

        fluxρP_z = WP
        fluxUP_z = ρPinv * WP * UP
        fluxVP_z = ρPinv * WP * VP
        fluxWP_z = ρPinv * WP * WP + PP
        fluxEP_z = ρPinv * WP * (EP + PP)

        λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + sqrt(ρMinv * gamma_d * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + sqrt(ρPinv * gamma_d * PP)
        λ  =  max(λM, λP)

        #Compute Numerical Flux and Update
        fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                  nzM * (fluxρM_z + fluxρP_z) - λ * (ρP - ρM)) / 2
        fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                  nzM * (fluxUM_z + fluxUP_z) - λ * (UP - UM)) / 2
        fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                  nzM * (fluxVM_z + fluxVP_z) - λ * (VP - VM)) / 2
        fluxWS = (nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) +
                  nzM * (fluxWM_z + fluxWP_z) - λ * (WP - WM)) / 2
        fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                  nzM * (fluxEM_z + fluxEP_z) - λ * (EP - EM)) / 2


        #Update RHS
        rhs[vidM, _ρ, eM] -= sMJ * fluxρS
        rhs[vidM, _U, eM] -= sMJ * fluxUS
        rhs[vidM, _V, eM] -= sMJ * fluxVS
        rhs[vidM, _W, eM] -= sMJ * fluxWS
        rhs[vidM, _E, eM] -= sMJ * fluxES
      end
    end
  end
end
# }}}
