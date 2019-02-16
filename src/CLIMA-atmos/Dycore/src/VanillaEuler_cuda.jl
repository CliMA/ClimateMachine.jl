# FIXME: Add link to https://github.com/paranumal/libparanumal here and in
# advection (also update the license)

# {{{ Volume Gradient for 2-D
function knl_volumegrad!(::Val{2}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                         grad, Q, vgeo, gravity, D, nelem) where {N, nmoist,
                                                                  ntrace}
  DFloat = eltype(D)

  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
  s_q = s_ρ = @cuStaticSharedMem(eltype(Q), (Nq, Nq))
  s_u = @cuStaticSharedMem(eltype(Q), (Nq, Nq))
  s_v = @cuStaticSharedMem(eltype(Q), (Nq, Nq))
  s_T = @cuStaticSharedMem(eltype(Q), (Nq, Nq))

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    # Load derivative into shared memory
    if k == 1
      s_D[i, j] = D[i, j]
    end

    U, V = Q[i, j, _U, e], Q[i, j, _V, e]
    ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
    y = vgeo[i, j, _y, e]
    P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

    s_ρ[i, j] = ρ
    s_u[i, j] = U/ρ
    s_v[i, j] = V/ρ
    s_T[i, j] = P/(R_d*ρ)
  end

  sync_threads()

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    ρξ = ρη = zero(DFloat)
    uξ = uη = zero(DFloat)
    vξ = vη = zero(DFloat)
    Tξ = Tη = zero(DFloat)

    for n = 1:Nq
      ρξ += D[i, n] * s_ρ[n, j]
      ρη += D[j, n] * s_ρ[i, n]

      uξ += D[i, n] * s_u[n, j]
      uη += D[j, n] * s_u[i, n]

      vξ += D[i, n] * s_v[n, j]
      vη += D[j, n] * s_v[i, n]

      Tξ += D[i, n] * s_T[n, j]
      Tη += D[j, n] * s_T[i, n]
    end

    ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
    ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]

    ρx = ξx*ρξ + ηx*ρη
    ρy = ξy*ρξ + ηy*ρη

    ux = ξx*uξ + ηx*uη
    uy = ξy*uξ + ηy*uη

    vx = ξx*vξ + ηx*vη
    vy = ξy*vξ + ηy*vη

    Tx = ξx*Tξ + ηx*Tη
    Ty = ξy*Tξ + ηy*Tη

    grad[i, j, _ρx, e] = ρx
    grad[i, j, _ρy, e] = ρy
    grad[i, j, _ρz, e] = zero(DFloat)

    grad[i, j, _ux, e] = ux
    grad[i, j, _uy, e] = uy
    grad[i, j, _uz, e] = zero(DFloat)

    grad[i, j, _vx, e] = vx
    grad[i, j, _vy, e] = vy
    grad[i, j, _vz, e] = zero(DFloat)

    grad[i, j, _wx, e] = zero(DFloat)
    grad[i, j, _wy, e] = zero(DFloat)
    grad[i, j, _wz, e] = zero(DFloat)

    grad[i, j, _Tx, e] = Tx
    grad[i, j, _Ty, e] = Ty
    grad[i, j, _Tz, e] = zero(DFloat)
  end

  # loop over moist variables
  # FIXME: Currently just passive advection
  # TODO: This should probably be unrolled by some factor
  for m = 1:nmoist
    s = _nstate + m
    sout = _nstategrad + 3*(m-1)

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      s_q[i, j] = Q[i, j, s, e]
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      qξ = qη = zero(DFloat)

      for n = 1:Nq
        qξ += D[i, n] * s_q[n, j]
        qη += D[j, n] * s_q[i, n]
      end

      ξx, ξy = vgeo[i, j, _ξx, e], vgeo[i, j, _ξy, e]
      ηx, ηy = vgeo[i, j, _ηx, e], vgeo[i, j, _ηy, e]

      qx = ξx*qξ + ηx*qη
      qy = ξy*qξ + ηy*qη

      grad[i, j, sout + 1, e] = qx
      grad[i, j, sout + 2, e] = qy
      grad[i, j, sout + 3, e] = zero(DFloat)
    end
  end

  nothing
end
# }}}

# {{{ Volume Gradient for 3-D
function knl_volumegrad!(::Val{3}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                         grad, Q, vgeo, gravity, D, nelem) where {N, nmoist,
                                                                       ntrace}
  DFloat = eltype(D)

  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
  s_q = s_ρ = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq))
  s_u = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq))
  s_v = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq))
  s_w = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq))
  s_T = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq))

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    # Load derivative into shared memory
    if k == 1
      s_D[i, j] = D[i, j]
    end

    # Load values will need into registers
    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]
    z = vgeo[i,j,k,_z,e]
    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

    s_ρ[i, j, k] = ρ
    s_u[i, j, k] = U/ρ
    s_v[i, j, k] = V/ρ
    s_w[i, j, k] = W/ρ
    s_T[i, j, k] = P/(R_d*ρ)
  end

  sync_threads()

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    ρξ = ρη = ρζ = zero(DFloat)
    uξ = uη = uζ = zero(DFloat)
    vξ = vη = vζ = zero(DFloat)
    wξ = wη = wζ = zero(DFloat)
    Tξ = Tη = Tζ = zero(DFloat)

    for n = 1:Nq
      ρξ += D[i, n] * s_ρ[n, j, k]
      ρη += D[j, n] * s_ρ[i, n, k]
      ρζ += D[k, n] * s_ρ[i, j, n]

      uξ += D[i, n] * s_u[n, j, k]
      uη += D[j, n] * s_u[i, n, k]
      uζ += D[k, n] * s_u[i, j, n]

      vξ += D[i, n] * s_v[n, j, k]
      vη += D[j, n] * s_v[i, n, k]
      vζ += D[k, n] * s_v[i, j, n]

      wξ += D[i, n] * s_w[n, j, k]
      wη += D[j, n] * s_w[i, n, k]
      wζ += D[k, n] * s_w[i, j, n]

      Tξ += D[i, n] * s_T[n, j, k]
      Tη += D[j, n] * s_T[i, n, k]
      Tζ += D[k, n] * s_T[i, j, n]
    end

    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

    ρx = ξx*ρξ + ηx*ρη + ζx*ρζ
    ρy = ξy*ρξ + ηy*ρη + ζy*ρζ
    ρz = ξz*ρξ + ηz*ρη + ζz*ρζ

    ux = ξx*uξ + ηx*uη + ζx*uζ
    uy = ξy*uξ + ηy*uη + ζy*uζ
    uz = ξz*uξ + ηz*uη + ζz*uζ

    vx = ξx*vξ + ηx*vη + ζx*vζ
    vy = ξy*vξ + ηy*vη + ζy*vζ
    vz = ξz*vξ + ηz*vη + ζz*vζ

    wx = ξx*wξ + ηx*wη + ζx*wζ
    wy = ξy*wξ + ηy*wη + ζy*wζ
    wz = ξz*wξ + ηz*wη + ζz*wζ

    Tx = ξx*Tξ + ηx*Tη + ζx*Tζ
    Ty = ξy*Tξ + ηy*Tη + ζy*Tζ
    Tz = ξz*Tξ + ηz*Tη + ζz*Tζ

    grad[i, j, k, _ρx, e] = ρx
    grad[i, j, k, _ρy, e] = ρy
    grad[i, j, k, _ρz, e] = ρz

    grad[i, j, k, _ux, e] = ux
    grad[i, j, k, _uy, e] = uy
    grad[i, j, k, _uz, e] = uz

    grad[i, j, k, _vx, e] = vx
    grad[i, j, k, _vy, e] = vy
    grad[i, j, k, _vz, e] = vz

    grad[i, j, k, _wx, e] = wx
    grad[i, j, k, _wy, e] = wy
    grad[i, j, k, _wz, e] = wz

    grad[i, j, k, _Tx, e] = Tx
    grad[i, j, k, _Ty, e] = Ty
    grad[i, j, k, _Tz, e] = Tz
  end

  # loop over moist variables
  # FIXME: Currently just passive advection
  # TODO: This should probably be unrolled by some factor
  for m = 1:nmoist
    s = _nstate + m
    sout = _nstategrad + 3*(m-1)

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      s_q[i, j, k] = Q[i, j, k, s, e]
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      qξ = qη = qζ = zero(DFloat)

      for n = 1:Nq
        qξ += D[i, n] * s_q[n, j, k]
        qη += D[j, n] * s_q[i, n, k]
        qζ += D[k, n] * s_q[i, j, n]
      end

      ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
      ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
      ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

      qx = ξx*qξ + ηx*qη + ζx*qζ
      qy = ξy*qξ + ηy*qη + ζy*qζ
      qz = ξz*qξ + ηz*qη + ζz*qζ

      grad[i, j, sout + 1, e] = qx
      grad[i, j, sout + 2, e] = qy
      grad[i, j, sout + 3, e] = qz
    end
  end

  nothing
end
# }}}

# {{{ Face Gradient (all dimensions)
function knl_facegrad!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                       grad, Q, vgeo, sgeo, gravity, nelem, vmapM, vmapP,
                       elemtobndy) where {dim, N, nmoist, ntrace}
  DFloat = eltype(Q)

  if dim == 1
    Np = (N+1)
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    nface = 6
  end

  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    n = i + (j-1) * Nq
    for lf = 1:2:nface
      for f = lf:lf+1
        (nxM, nyM) = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e])
        (nzM, sMJ) = (sgeo[_nz, n, f, e], sgeo[_sMJ, n, f, e])
        vMJI = sgeo[_vMJI, n, f, e]

        (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

        (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
        (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        WM = Q[vidM, _W, eM]
        EM = Q[vidM, _E, eM]
        yorzM = (dim == 2) ? vgeo[vidM, _y, eM] : vgeo[vidM, _z, eM]

        PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*yorzM)
        uM=UM/ρM
        vM=VM/ρM
        wM=WM/ρM
        TM=PM/(R_d*ρM)

        bc = elemtobndy[f, e]
        ρP = UP = VP = WP = EP = PP = zero(eltype(Q))
        uP = vP = wP = TP = zero(eltype(Q))
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          yorzP = (dim == 2) ? vgeo[vidP, _y, eP] : vgeo[vidP, _z, eP]
          PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*yorzP)
          uP=UP/ρP
          vP=VP/ρP
          wP=WP/ρP
          TP=PP/(R_d*ρP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM + nzM * WM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          WP = WM - 2 * UnM * nzM
          ρP = ρM
          EP = EM
          PP = PM
          uP = UP/ρP
          vP = VP/ρP
          wP = WP/ρP
          TP = TM
        end

        fluxρS = (ρP - ρM)/2
        fluxuS = (uP - uM)/2
        fluxvS = (vP - vM)/2
        fluxwS = (wP - wM)/2
        fluxTS = (TP - TM)/2

        grad[vidM, _ρx, eM] += vMJI*sMJ*nxM*fluxρS
        grad[vidM, _ρy, eM] += vMJI*sMJ*nyM*fluxρS
        grad[vidM, _ρz, eM] += vMJI*sMJ*nzM*fluxρS
        grad[vidM, _ux, eM] += vMJI*sMJ*nxM*fluxuS
        grad[vidM, _uy, eM] += vMJI*sMJ*nyM*fluxuS
        grad[vidM, _uz, eM] += vMJI*sMJ*nzM*fluxuS
        grad[vidM, _vx, eM] += vMJI*sMJ*nxM*fluxvS
        grad[vidM, _vy, eM] += vMJI*sMJ*nyM*fluxvS
        grad[vidM, _vz, eM] += vMJI*sMJ*nzM*fluxvS
        grad[vidM, _wx, eM] += vMJI*sMJ*nxM*fluxwS
        grad[vidM, _wy, eM] += vMJI*sMJ*nyM*fluxwS
        grad[vidM, _wz, eM] += vMJI*sMJ*nzM*fluxwS
        grad[vidM, _Tx, eM] += vMJI*sMJ*nxM*fluxTS
        grad[vidM, _Ty, eM] += vMJI*sMJ*nyM*fluxTS
        grad[vidM, _Tz, eM] += vMJI*sMJ*nzM*fluxTS

        # loop over moist variables
        for m = 1:nmoist
          s = _nstate + m
          ss = _nstategrad + 3*(m-1)
          qM, qP = Q[vidM, s, eM], Q[vidP, s, eP]

          fluxqS = (qP - qM)/2

          grad[vidM, ss+1, eM] += vMJI*sMJ*nxM*fluxqS
          grad[vidM, ss+2, eM] += vMJI*sMJ*nyM*fluxqS
          grad[vidM, ss+3, eM] += vMJI*sMJ*nzM*fluxqS
        end
      end
      sync_threads()
    end
  end
  nothing
end
# }}}

# {{{ Volume RHS for 2-D
function knl_volumerhs!(::Val{2}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace}, rhs,
                        Q, grad, vgeo, gravity, viscosity, D,
                        nelem) where {N, nmoist, ntrace}
  DFloat = eltype(D)

  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
  s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate))
  s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate))

  MJI = rhsU = rhsV = rhsρ = rhsE = zero(eltype(rhs))
  MJ = ξx = ξy = ηx = ηy = zero(eltype(rhs))
  u = v = zero(eltype(rhs))
  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    # Load derivative into shared memory
    if k == 1
      s_D[i, j] = D[i, j]
    end

    # Load values will need into registers
    MJ, MJI = vgeo[i, j, _MJ, e], vgeo[i, j, _MJI, e]
    ξx, ξy = vgeo[i, j, _ξx, e], vgeo[i, j, _ξy, e]
    ηx, ηy = vgeo[i, j, _ηx, e], vgeo[i, j, _ηy, e]
    y = vgeo[i, j, _y, e]
    U, V = Q[i, j, _U, e], Q[i, j, _V, e]
    ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
    rhsU, rhsV = rhs[i, j, _U, e], rhs[i, j, _V, e]
    rhsρ, rhsE = rhs[i, j, _ρ, e], rhs[i, j, _E, e]

    P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

    ρx, ρy = grad[i,j,_ρx,e], grad[i,j,_ρy,e]
    ux, uy = grad[i,j,_ux,e], grad[i,j,_uy,e]
    vx, vy = grad[i,j,_vx,e], grad[i,j,_vy,e]
    Tx, Ty = grad[i,j,_Tx,e], grad[i,j,_Ty,e]

    ρinv = 1 / ρ

    ldivu = stokes*(ux + vy)
    u = ρinv*U
    v = ρinv*V

    vfluxρ_x = 0*ρx
    vfluxρ_y = 0*ρy

    vfluxU_x = 2*ux + ldivu
    vfluxU_y = uy + vx

    vfluxV_x = vx + uy
    vfluxV_y = 2*vy + ldivu

    vfluxE_x = u*(2*ux + ldivu) + v*(uy + vx) + k_μ*Tx
    vfluxE_y = u*(vx + uy) + v*(2*vy + ldivu) + k_μ*Ty

    fluxρ_x = U                  - viscosity*vfluxρ_x
    fluxU_x = ρinv * U * U + P   - viscosity*vfluxU_x
    fluxV_x = ρinv * U * V       - viscosity*vfluxV_x
    fluxE_x = ρinv * U * (E + P) - viscosity*vfluxE_x

    fluxρ_y = V                  - viscosity*vfluxρ_y
    fluxU_y = ρinv * V * U       - viscosity*vfluxU_y
    fluxV_y = ρinv * V * V + P   - viscosity*vfluxV_y
    fluxE_y = ρinv * V * (E + P) - viscosity*vfluxE_y

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
    rhsV -= ρ * gravity
  end

  sync_threads()

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    # loop of ξ-grid lines
    for n = 1:Nq
      MJI_Dni = MJI * s_D[n, i]
      MJI_Dnj = MJI * s_D[n, j]

      rhsρ += MJI_Dni * s_F[n, j, _ρ]
      rhsρ += MJI_Dnj * s_G[i, n, _ρ]

      rhsU += MJI_Dni * s_F[n, j, _U]
      rhsU += MJI_Dnj * s_G[i, n, _U]

      rhsV += MJI_Dni * s_F[n, j, _V]
      rhsV += MJI_Dnj * s_G[i, n, _V]

      rhsE += MJI_Dni * s_F[n, j, _E]
      rhsE += MJI_Dnj * s_G[i, n, _E]
    end

    rhs[i, j, _U, e] = rhsU
    rhs[i, j, _V, e] = rhsV
    rhs[i, j, _ρ, e] = rhsρ
    rhs[i, j, _E, e] = rhsE
  end

  # loop over moist variables
  # FIXME: Currently just passive advection
  # TODO: This should probably be unrolled by some factor
  rhsmoist = zero(eltype(rhs))
  for m = 1:nmoist
    s = _nstate + m
    ss = _nstategrad + 3*(m-1)

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      q = Q[i, j, s, e]
      qx = grad[i, j, ss+1, e]
      qy = grad[i, j, ss+2, e]

      fx = u * q - viscosity*0*qx
      fy = v * q - viscosity*0*qy

      rhsmoist = rhs[i, j, s, e]

      s_F[i, j, 1] = MJ * (ξx * fx + ξy * fy)
      s_G[i, j, 1] = MJ * (ηx * fx + ηy * fy)
    end

    sync_threads()
    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      for n = 1:Nq
        MJI_Dni = MJI * s_D[n, i]
        MJI_Dnj = MJI * s_D[n, j]

        rhsmoist += MJI_Dni * s_F[n, j, 1]
        rhsmoist += MJI_Dnj * s_G[i, n, 1]
      end
      rhs[i, j, s, e] = rhsmoist
    end
  end

  # loop over tracer variables
  # TODO: This should probably be unrolled by some factor
  rhstrace = zero(eltype(rhs))
  for t = 1:ntrace
    s = _nstate + nmoist + t

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      Qtrace   = Q[i, j, s, e]
      rhstrace = rhs[i, j, s, e]

      s_F[i, j, 1] = MJ * (ξx * u * Qtrace + ξy * v * Qtrace)
      s_G[i, j, 1] = MJ * (ηx * u * Qtrace + ηy * v * Qtrace)
    end

    sync_threads()
    for n = 1:Nq
      MJI_Dni = MJI * s_D[n, i]
      MJI_Dnj = MJI * s_D[n, j]

      rhstrace += MJI_Dni * s_F[n, j, 1]
      rhstrace += MJI_Dnj * s_G[i, n, 1]
    end
    rhs[i, j, s, e] = rhstrace
  end

  nothing
end
# }}}

# {{{ Volume RHS for 3-D
function knl_volumerhs!(::Val{3}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace}, rhs,
                        Q, grad, vgeo, gravity, viscosity, D,
                        nelem) where {N, nmoist, ntrace}
  DFloat = eltype(D)

  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
  s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
  s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
  s_H = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))

  MJI = rhsU = rhsV = rhsW = rhsρ = rhsE = zero(eltype(rhs))
  MJ = ξx = ξy = ξz = ηx = ηy = ηz = ζx = ζy = ζz = zero(eltype(rhs))
  u = v = w = zero(eltype(rhs))
  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    # Load derivative into shared memory
    if k == 1
      s_D[i, j] = D[i, j]
    end

    # Load values will need into registers
    MJ, MJI= vgeo[i, j, k, _MJ, e], vgeo[i, j, k, _MJI, e]
    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
    z = vgeo[i,j,k,_z,e]

    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

    ρx, ρy, ρz = grad[i,j,k,_ρx,e], grad[i,j,k,_ρy,e], grad[i,j,k,_ρz,e]
    ux, uy, uz = grad[i,j,k,_ux,e], grad[i,j,k,_uy,e], grad[i,j,k,_uz,e]
    vx, vy, vz = grad[i,j,k,_vx,e], grad[i,j,k,_vy,e], grad[i,j,k,_vz,e]
    wx, wy, wz = grad[i,j,k,_wx,e], grad[i,j,k,_wy,e], grad[i,j,k,_wz,e]
    Tx, Ty, Tz = grad[i,j,k,_Tx,e], grad[i,j,k,_Ty,e], grad[i,j,k,_Tz,e]

    ρinv = 1 / ρ

    ldivu = stokes*(ux + vy + wz)
    u = ρinv*U
    v = ρinv*V
    w = ρinv*W

    vfluxρ_x = 0*ρx
    vfluxρ_y = 0*ρy
    vfluxρ_z = 0*ρz

    vfluxU_x = 2*ux + ldivu
    vfluxU_y = uy + vx
    vfluxU_z = uz + wx

    vfluxV_x = vx + uy
    vfluxV_y = 2*vy + ldivu
    vfluxV_z = vz + wy

    vfluxW_x = wx + uz
    vfluxW_y = wy + vz
    vfluxW_z = 2*wz + ldivu

    vfluxE_x = u*(2*ux + ldivu) + v*(uy + vx) + w*(uz + wx) + k_μ*Tx
    vfluxE_y = u*(vx + uy) + v*(2*vy + ldivu) + w*(vz + wy) + k_μ*Ty
    vfluxE_z = u*(wx + uz) + v*(wy + vz) + w*(2*wz + ldivu) + k_μ*Tz

    fluxρ_x = U                   - viscosity*vfluxρ_x
    fluxU_x = ρinv * U * U + P    - viscosity*vfluxU_x
    fluxV_x = ρinv * U * V        - viscosity*vfluxV_x
    fluxW_x = ρinv * U * W        - viscosity*vfluxW_x
    fluxE_x = ρinv * U * (E + P)  - viscosity*vfluxE_x

    fluxρ_y = V                   - viscosity*vfluxρ_y
    fluxU_y = ρinv * V * U        - viscosity*vfluxU_y
    fluxV_y = ρinv * V * V + P    - viscosity*vfluxV_y
    fluxW_y = ρinv * V * W        - viscosity*vfluxW_y
    fluxE_y = ρinv * V * (E + P)  - viscosity*vfluxE_y

    fluxρ_z = W                   - viscosity*vfluxρ_z
    fluxU_z = ρinv * W * U        - viscosity*vfluxU_z
    fluxV_z = ρinv * W * V        - viscosity*vfluxV_z
    fluxW_z = ρinv * W * W + P    - viscosity*vfluxW_z
    fluxE_z = ρinv * W * (E + P)  - viscosity*vfluxE_z

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

    rhsU, rhsV, rhsW = (rhs[i, j, k, _U, e],
                        rhs[i, j, k, _V, e],
                        rhs[i, j, k, _W, e])
    rhsρ, rhsE = rhs[i, j, k, _ρ, e], rhs[i, j, k, _E, e]

    # buoyancy term
    rhsW -= ρ * gravity
  end

  sync_threads()

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    # loop of ξ-grid lines
    for n = 1:Nq
      MJI_Dni = MJI * s_D[n, i]
      MJI_Dnj = MJI * s_D[n, j]
      MJI_Dnk = MJI * s_D[n, k]

      rhsρ += MJI_Dni * s_F[n, j, k, _ρ]
      rhsρ += MJI_Dnj * s_G[i, n, k, _ρ]
      rhsρ += MJI_Dnk * s_H[i, j, n, _ρ]

      rhsU += MJI_Dni * s_F[n, j, k, _U]
      rhsU += MJI_Dnj * s_G[i, n, k, _U]
      rhsU += MJI_Dnk * s_H[i, j, n, _U]

      rhsV += MJI_Dni * s_F[n, j, k, _V]
      rhsV += MJI_Dnj * s_G[i, n, k, _V]
      rhsV += MJI_Dnk * s_H[i, j, n, _V]

      rhsW += MJI_Dni * s_F[n, j, k, _W]
      rhsW += MJI_Dnj * s_G[i, n, k, _W]
      rhsW += MJI_Dnk * s_H[i, j, n, _W]

      rhsE += MJI_Dni * s_F[n, j, k, _E]
      rhsE += MJI_Dnj * s_G[i, n, k, _E]
      rhsE += MJI_Dnk * s_H[i, j, n, _E]
    end

    rhs[i, j, k, _U, e] = rhsU
    rhs[i, j, k, _V, e] = rhsV
    rhs[i, j, k, _W, e] = rhsW
    rhs[i, j, k, _ρ, e] = rhsρ
    rhs[i, j, k, _E, e] = rhsE
  end

  # loop over moist variables
  # FIXME: Currently just passive advection
  # TODO: This should probably be unrolled by some factor
  rhsmoist = zero(eltype(rhs))
  for m = 1:nmoist
    s = _nstate + m
    ss = _nstategrad + 3*(m-1)

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      q = Q[i, j, k, s, e]
      qx = grad[i, j, k, ss+1, e]
      qy = grad[i, j, k, ss+2, e]
      qz = grad[i, j, k, ss+3, e]

      rhsmoist = rhs[i, j, k, s, e]

      fx = u*q - viscosity*0*qx
      fy = v*q - viscosity*0*qy
      fz = w*q - viscosity*0*qz

      s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
      s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
      s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      for n = 1:Nq
        MJI_Dni = MJI * s_D[n, i]
        MJI_Dnj = MJI * s_D[n, j]
        MJI_Dnk = MJI * s_D[n, k]

        rhsmoist += MJI_Dni * s_F[n, j, k, 1]
        rhsmoist += MJI_Dnj * s_G[i, n, k, 1]
        rhsmoist += MJI_Dnk * s_H[i, j, n, 1]
      end
      rhs[i, j, k, s, e] = rhsmoist
    end
  end

  # loop over tracer variables
  # TODO: This should probably be unrolled by some factor
  rhstrace = zero(eltype(rhs))
  for t = 1:ntrace
    s = _nstate + nmoist + t

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      Qtrace = Q[i, j, k, s, e]
      rhstrace = rhs[i, j, k, s, e]

      fx = u * Qtrace
      fy = v * Qtrace
      fz = w * Qtrace

      s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
      s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
      s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      for n = 1:Nq
        MJI_Dni = MJI * s_D[n, i]
        MJI_Dnj = MJI * s_D[n, j]
        MJI_Dnk = MJI * s_D[n, k]

        rhstrace += MJI_Dni * s_F[n, j, k, 1]
        rhstrace += MJI_Dnj * s_G[i, n, k, 1]
        rhstrace += MJI_Dnk * s_H[i, j, n, 1]
      end
      rhs[i, j, k, s, e] = rhstrace
    end
  end

  nothing
end
# }}}

# {{{ Face RHS (all dimensions)
function knl_facerhs!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace}, rhs,
                      Q, grad, vgeo, sgeo, gravity, viscosity, nelem, vmapM,
                      vmapP, elemtobndy) where {dim, N, nmoist, ntrace}
  DFloat = eltype(Q)

  if dim == 1
    Np = (N+1)
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    nface = 6
  end

  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    n = i + (j-1) * Nq
    for lf = 1:2:nface
      for f = lf:lf+1
        (nxM, nyM) = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e])
        (nzM, sMJ) = (sgeo[_nz, n, f, e], sgeo[_sMJ, n, f, e])
        vMJI = sgeo[_vMJI, n, f, e]

        (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

        (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
        (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

        ρxM = grad[vidM, _ρx, eM]
        ρyM = grad[vidM, _ρy, eM]
        ρzM = grad[vidM, _ρz, eM]
        uxM = grad[vidM, _ux, eM]
        uyM = grad[vidM, _uy, eM]
        uzM = grad[vidM, _uz, eM]
        vxM = grad[vidM, _vx, eM]
        vyM = grad[vidM, _vy, eM]
        vzM = grad[vidM, _vz, eM]
        wxM = grad[vidM, _wx, eM]
        wyM = grad[vidM, _wy, eM]
        wzM = grad[vidM, _wz, eM]
        TxM = grad[vidM, _Tx, eM]
        TyM = grad[vidM, _Ty, eM]
        TzM = grad[vidM, _Tz, eM]

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        WM = Q[vidM, _W, eM]
        EM = Q[vidM, _E, eM]

        yorzM = (dim == 2) ? vgeo[vidM, _y, eM] : vgeo[vidM, _z, eM]

        ρMinv = 1 / ρM
        uM, vM, wM = ρMinv * UM, ρMinv * VM, ρMinv * WM

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*yorzM)
        ρP = UP = VP = WP = EP = PP = zero(eltype(Q))
        ρxP = ρyP = ρzP = uxP = uyP = uzP = vxP = vyP = vzP = zero(eltype(grad))
        wxP = wyP = wzP = TxP = TyP = TzP = zero(eltype(grad))
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          yorzP = (dim == 2) ? vgeo[vidP, _y, eP] : vgeo[vidP, _z, eP]
          PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*yorzP)

          ρxP = grad[vidP, _ρx, eP]
          ρyP = grad[vidP, _ρy, eP]
          ρzP = grad[vidP, _ρz, eP]
          uxP = grad[vidP, _ux, eP]
          uyP = grad[vidP, _uy, eP]
          uzP = grad[vidP, _uz, eP]
          vxP = grad[vidP, _vx, eP]
          vyP = grad[vidP, _vy, eP]
          vzP = grad[vidP, _vz, eP]
          wxP = grad[vidP, _wx, eP]
          wyP = grad[vidP, _wy, eP]
          wzP = grad[vidP, _wz, eP]
          TxP = grad[vidP, _Tx, eP]
          TyP = grad[vidP, _Ty, eP]
          TzP = grad[vidP, _Tz, eP]
        elseif bc == 1
          UnM = nxM * UM + nyM * VM + nzM * WM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          WP = WM - 2 * UnM * nzM
          ρP = ρM
          EP = EM
          PP = PM

          ρnM = nxM * ρxM + nyM * ρyM + nzM * ρzM
          ρxP = ρxM - 2 * ρnM * nxM
          ρyP = ρyM - 2 * ρnM * nyM
          ρzP = ρzM - 2 * ρnM * nzM

          unM = nxM * uxM + nyM * uyM + nzM * uzM
          uxP = uxM - 2 * unM * nxM
          uyP = uyM - 2 * unM * nyM
          uzP = uzM - 2 * unM * nzM

          vnM = nxM * vxM + nyM * vyM + nzM * vzM
          vxP = vxM - 2 * vnM * nxM
          vyP = vyM - 2 * vnM * nyM
          vzP = vzM - 2 * vnM * nzM

          wnM = nxM * wxM + nyM * wyM + nzM * wzM
          wxP = wxM - 2 * wnM * nxM
          wyP = wyM - 2 * wnM * nyM
          wzP = wzM - 2 * wnM * nzM

          TxP, TyP, TzP = TxM, TyM, TzM
        end

        ρPinv = 1 / ρP
        uP, vP, wP = ρPinv * UP, ρPinv * VP, ρPinv * WP

        fluxρM_x = UM
        fluxUM_x = uM * UM + PM
        fluxVM_x = uM * VM
        fluxWM_x = uM * WM
        fluxEM_x = uM * (EM + PM)

        fluxρM_y = VM
        fluxUM_y = vM * UM
        fluxVM_y = vM * VM + PM
        fluxWM_y = vM * WM
        fluxEM_y = vM * (EM + PM)

        fluxρM_z = WM
        fluxUM_z = wM * UM
        fluxVM_z = wM * VM
        fluxWM_z = wM * WM + PM
        fluxEM_z = wM * (EM + PM)

        fluxρP_x = UP
        fluxUP_x = uP * UP + PP
        fluxVP_x = uP * VP
        fluxWP_x = uP * WP
        fluxEP_x = uP * (EP + PP)

        fluxρP_y = VP
        fluxUP_y = vP * UP
        fluxVP_y = vP * VP + PP
        fluxWP_y = vP * WP
        fluxEP_y = vP * (EP + PP)

        fluxρP_z = WP
        fluxUP_z = wP * UP
        fluxVP_z = wP * VP
        fluxWP_z = wP * WP + PP
        fluxEP_z = wP * (EP + PP)

        λM = abs(nxM * uM + nyM * vM + nzM * wM) + CUDAnative.sqrt(ρMinv * gamma_d * PM)
        λP = abs(nxM * uP + nyM * vP + nzM * wP) + CUDAnative.sqrt(ρPinv * gamma_d * PP)
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

        #Compute Viscous Numerical Flux
        #Left Fluxes
        ldivuM = stokes*(uxM + vyM + wzM)
        vfρM_x = 0*ρxM
        vfρM_y = 0*ρyM
        vfρM_z = 0*ρzM
        vfUM_x = 2*uxM + ldivuM
        vfUM_y = uyM + vxM
        vfUM_z = uzM + wxM
        vfVM_x = vxM + uyM
        vfVM_y = 2*vyM + ldivuM
        vfVM_z = vzM + wyM
        vfWM_x = wxM + uzM
        vfWM_y = wyM + vzM
        vfWM_z = 2*wzM + ldivuM
        vfEM_x = uM*(2*uxM + ldivuM) + vM*(uyM + vxM) + wM*(uzM + wxM) + k_μ*TxM
        vfEM_y = uM*(vxM + uyM) + vM*(2*vyM + ldivuM) + wM*(vzM + wyM) + k_μ*TyM
        vfEM_z = uM*(wxM + uzM) + vM*(wyM + vzM) + wM*(2*wzM + ldivuM) + k_μ*TzM

        #Right Fluxes
        ldivuP = stokes*(uxP + vyP + wzP)
        vfρP_x = 0*ρxP
        vfρP_y = 0*ρyP
        vfρP_z = 0*ρzP
        vfUP_x = 2*uxP + ldivuP
        vfUP_y = uyP + vxP
        vfUP_z = uzP + wxP
        vfVP_x = vxP + uyP
        vfVP_y = 2*vyP + ldivuP
        vfVP_z = vzP + wyP
        vfWP_x = wxP + uzP
        vfWP_y = wyP + vzP
        vfWP_z = 2*wzP + ldivuP
        vfEP_x = uP*(2*uxP + ldivuP) + vP*(uyP + vxP) + wP*(uzP + wxP) + k_μ*TxP
        vfEP_y = uP*(vxP + uyP) + vP*(2*vyP + ldivuP) + wP*(vzP + wyP) + k_μ*TyP
        vfEP_z = uP*(wxP + uzP) + vP*(wyP + vzP) + wP*(2*wzP + ldivuP) + k_μ*TzP

        #Compute Numerical Flux
        vfluxρS = (nxM*(vfρM_x + vfρP_x) + nyM*(vfρM_y + vfρP_y) + nzM*(vfρM_z + vfρP_z))/2
        vfluxUS = (nxM*(vfUM_x + vfUP_x) + nyM*(vfUM_y + vfUP_y) + nzM*(vfUM_z + vfUP_z))/2
        vfluxVS = (nxM*(vfVM_x + vfVP_x) + nyM*(vfVM_y + vfVP_y) + nzM*(vfVM_z + vfVP_z))/2
        vfluxWS = (nxM*(vfWM_x + vfWP_x) + nyM*(vfWM_y + vfWP_y) + nzM*(vfWM_z + vfWP_z))/2
        vfluxES = (nxM*(vfEM_x + vfEP_x) + nyM*(vfEM_y + vfEP_y) + nzM*(vfEM_z + vfEP_z))/2

        #Update RHS
        rhs[vidM, _ρ, eM] -= vMJI * sMJ * (fluxρS - viscosity*vfluxρS)
        rhs[vidM, _U, eM] -= vMJI * sMJ * (fluxUS - viscosity*vfluxUS)
        rhs[vidM, _V, eM] -= vMJI * sMJ * (fluxVS - viscosity*vfluxVS)
        rhs[vidM, _W, eM] -= vMJI * sMJ * (fluxWS - viscosity*vfluxWS)
        rhs[vidM, _E, eM] -= vMJI * sMJ * (fluxES - viscosity*vfluxES)


        # loop over moist variables
        # FIXME: Currently just passive advection
        # TODO: This should probably be unrolled by some factor
        for m = 1:nmoist
          s = _nstate + m
          ss = _nstategrad + 3(m-1)

          QmoistM = Q[vidM, s, eM]

          qxM = grad[vidM, ss + 1, eM]
          qyM = grad[vidM, ss + 2, eM]
          qzM = grad[vidM, ss + 3, eM]

          QmoistP = qxP = qyP = qzP = zero(eltype(grad))
          if bc == 0
            QmoistP = Q[vidP, s, eP]
            qxP = grad[vidP, ss + 1, eP]
            qyP = grad[vidP, ss + 2, eP]
            qzP = grad[vidP, ss + 3, eP]
          elseif bc == 1
            QmoistP = QmoistM
            qnM = nxM * qxM + nyM * qyM + nzM * qzM
            qxP = qxM - 2 * qnM * nxM
            qyP = qyM - 2 * qnM * nyM
            qzP = qzM - 2 * qnM * nzM
          end

          fluxM_x, fluxP_x = uM * QmoistM, uP * QmoistP
          fluxM_y, fluxP_y = vM * QmoistM, vP * QmoistP
          fluxM_z, fluxP_z = wM * QmoistM, wP * QmoistP

          fluxS = (nxM * (fluxM_x + fluxP_x) + nyM * (fluxM_y + fluxP_y) +
                   nzM * (fluxM_z + fluxP_z) - λ * (QmoistP - QmoistM)) / 2

          vfqM_x, vfqP_x = 0*qxM, 0*qxP
          vfqM_y, vfqP_y = 0*qyM, 0*qyP
          vfqM_z, vfqP_z = 0*qzM, 0*qzP
          vfluxqS = (nxM*(vfqM_x + vfqP_x) + nyM*(vfqM_y + vfqP_y) +
                     nzM*(vfqM_z + vfqP_z))/2

          rhs[vidM, s, eM] -= vMJI * sMJ * (fluxS - viscosity*vfluxqS)
        end

        # FIXME: Will need to be updated for other bcs...
        vidP = bc == 0 ? vidP : vidM

        # loop over tracer variables
        # TODO: This should probably be unrolled by some factor
        for t = 1:ntrace
          s = _nstate + nmoist + t
          QtraceM, QtraceP = Q[vidM, s, eM], Q[vidP, s, eP]

          fluxM_x, fluxP_x = uM * QtraceM, uP * QtraceP
          fluxM_y, fluxP_y = vM * QtraceM, vP * QtraceP
          fluxM_z, fluxP_z = wM * QtraceM, wP * QtraceP

          fluxS = (nxM * (fluxM_x + fluxP_x) + nyM * (fluxM_y + fluxP_y) +
                   nzM * (fluxM_z + fluxP_z) - λ * (QtraceP - QtraceM)) / 2

          rhs[vidM, s, eM] -= vMJI * sMJ * fluxS
        end
      end
      sync_threads()
    end
  end
  nothing
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
