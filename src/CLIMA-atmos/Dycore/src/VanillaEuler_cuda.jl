# FIXME: Add link to https://github.com/paranumal/libparanumal here and in
# advection (also update the license)

# {{{ Volume RHS for 2-D
function knl_volumerhs!(::Val{2}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace}, rhs,
                        Q, vgeo, gravity, D, nelem) where {N, nmoist, ntrace}
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
  if i <= Nq && j <= Nq && k == 1 && e <= nelem
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
    rhsV -= ρ * gravity

    u, v = U * ρinv, V * ρinv
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

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      Qmoist   = Q[i, j, s, e]
      rhsmoist = rhs[i, j, s, e]

      s_F[i, j, 1] = MJ * (ξx * u * Qmoist + ξy * v * Qmoist)
      s_G[i, j, 1] = MJ * (ηx * u * Qmoist + ηy * v * Qmoist)
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
                        Q, vgeo, gravity, D, nelem) where {N, nmoist, ntrace}
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

    ρinv = 1 / ρ
    fluxρ_x = U
    fluxU_x = ρinv * U * U + P
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

    rhsU, rhsV, rhsW = (rhs[i, j, k, _U, e],
                        rhs[i, j, k, _V, e],
                        rhs[i, j, k, _W, e])
    rhsρ, rhsE = rhs[i, j, k, _ρ, e], rhs[i, j, k, _E, e]

    # buoyancy term
    rhsW -= ρ * gravity

    u, v, w = U * ρinv, V * ρinv, W * ρinv
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

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
      Qmoist = Q[i, j, k, s, e]
      rhsmoist = rhs[i, j, k, s, e]

      fx = u * Qmoist
      fy = v * Qmoist
      fz = w * Qmoist

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
                      Q, vgeo, sgeo, gravity, nelem, vmapM, vmapP,
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
  half = convert(eltype(Q), 0.5)

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

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*yorzM)
        ρP = UP = VP = WP = EP = PP = zero(eltype(Q))
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          yorzP = (dim == 2) ? vgeo[vidP, _y, eP] : vgeo[vidP, _z, eP]
          PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*yorzP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM + nzM * WM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          WP = WM - 2 * UnM * nzM
          ρP = ρM
          EP = EM
          PP = PM
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

        λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + CUDAnative.sqrt(ρMinv * gamma_d * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + CUDAnative.sqrt(ρPinv * gamma_d * PP)
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
        rhs[vidM, _ρ, eM] -= vMJI * sMJ * fluxρS
        rhs[vidM, _U, eM] -= vMJI * sMJ * fluxUS
        rhs[vidM, _V, eM] -= vMJI * sMJ * fluxVS
        rhs[vidM, _W, eM] -= vMJI * sMJ * fluxWS
        rhs[vidM, _E, eM] -= vMJI * sMJ * fluxES

        # Calculate the velocity
        uM, vM, wM = ρMinv * UM, ρMinv * VM, ρMinv * WM
        uP, vP, wP = ρPinv * UP, ρPinv * VP, ρPinv * WP

        # FIXME: Will need to be updated for other bcs...
        vidP = bc == 0 ? vidP : vidM

        # loop over moist variables
        # FIXME: Currently just passive advection
        # TODO: This should probably be unrolled by some factor
        for m = 1:nmoist
          s = _nstate + m
          QmoistM, QmoistP = Q[vidM, s, eM], Q[vidP, s, eP]

          fluxM_x, fluxP_x = uM * QmoistM, uP * QmoistP
          fluxM_y, fluxP_y = vM * QmoistM, vP * QmoistP
          fluxM_z, fluxP_z = wM * QmoistM, wP * QmoistP

          fluxS = (nxM * (fluxM_x + fluxP_x) + nyM * (fluxM_y + fluxP_y) +
                   nzM * (fluxM_z + fluxP_z) - λ * (QmoistP - QmoistM)) / 2

          rhs[vidM, s, eM] -= vMJI * sMJ * fluxS
        end

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
function fillsendbuf!(sendbuf, d_sendbuf::CuArray, d_buf, d_sendelems)
  nsendelem = length(d_sendelems)
  Np = size(d_buf, 1)
  nvar = size(d_buf, 2)
  if nsendelem > 0
    @cuda(threads=Np, blocks=nsendelem,
          knl_fillsendbuf!(Val(Np), Val(nvar), d_sendbuf, d_buf, d_sendelems))
    sendbuf .= d_sendbuf
  end
end

function transferrecvbuf!(d_recvbuf::CuArray, recvbuf, d_buf, nrealelem)
  nrecvelem = size(recvbuf)[end]
  Np = size(d_buf, 1)
  nvar = size(d_buf, 2)
  if nrecvelem > 0
    d_recvbuf .= recvbuf
    @cuda(threads=Np, blocks=nrecvelem,
          knl_transferrecvbuf!(Val(Np), Val(nvar), d_buf, d_recvbuf,
                               nrecvelem, nrealelem))
  end
end
# }}}

# {{{ Kernel wrappers
function volumerhs!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                    d_rhs::CuArray, d_Q, d_vgeo, gravity, d_D,
                    elems) where {dim, N, nmoist, ntrace}
  Qshape    = (ntuple(j->N+1, dim)..., size(d_Q, 2), size(d_Q, 3))
  vgeoshape = (ntuple(j->N+1, dim)..., _nvgeo, size(d_Q, 3))

  d_rhsC = reshape(d_rhs, Qshape...)
  d_QC = reshape(d_Q, Qshape)
  d_vgeoC = reshape(d_vgeo, vgeoshape)

  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_volumerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), d_rhsC, d_QC,
                       d_vgeoC, gravity, d_D, nelem))
end

function facerhs!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                  d_rhs::CuArray, d_Q, d_vgeo, d_sgeo, gravity, elems, d_vmapM,
                  d_vmapP, d_elemtobndy) where {dim, N, nmoist, ntrace}
  nelem = length(elems)
  @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
        knl_facerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), d_rhs, d_Q,
                     d_vgeo, d_sgeo, gravity, nelem, d_vmapM, d_vmapP,
                     d_elemtobndy))
end

# }}}
