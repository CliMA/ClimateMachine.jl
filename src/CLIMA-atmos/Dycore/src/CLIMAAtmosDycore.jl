module CLIMAAtmosDycore

using Printf: @sprintf
using Canary, MPI, Requires

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" include("gpu.jl")

include("vtk.jl")
export atmosdycore

# To think about:
# - How to handle parameters for different case? Dictionaries?

# FIXME: Be consistent with tuple assignments (either with or without parens)
#
# FIXME: Add logging

# FIXME: Add link to https://github.com/paranumal/libparanumal here and in
# advection (also update the license)

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _U, _V, _W, _ρ, _E = 1:_nstate
const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                 x = _x,   y = _y,   z = _z)

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, nz = _nz, sMJ = _sMJ, vMJI = _vMJI)

const _γ = 14  // 10
const _p0 = 100000
const _R_gas = 28717 // 100
const _c_p = 100467 // 100
const _c_v = 7175 // 10
const _gravity = 10

# }}}

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
      E = Q[n, _E, e]
      P = p0 * (R_gas * E / p0)^(c_p / c_v)
      ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                       vgeo[n, _ηx, e], vgeo[n, _ηy, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy) + ρ * sqrt(γ * P / ρ),
                        abs(U * ηx + V * ηy) + ρ * sqrt(γ * P / ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V, W = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
      E = Q[n, _E, e]
      P = p0 * (R_gas * E / p0)^(c_p / c_v)

      ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
      ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
      ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * sqrt(γ * P / ρ),
                        abs(U * ηx + V * ηy + W * ηz) + ρ * sqrt(γ * P / ρ),
                        abs(U * ζx + V * ζy + W * ζz) + ρ * sqrt(γ * P / ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
end
# }}}

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
  # Compute metric terms
  Nq = size(D, 1)
  DFloat = eltype(D)

  (nface, nelem) = size(mesh.elemtoelem)

  crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

  vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
  sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)

  (ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, MJ, MJI, x, y, z) =
      ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x)
  (nx, ny, nz, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
  creategrid!(X..., mesh.elemtocoord, ξ)

  @inbounds for j = 1:length(x)
    (x[j], y[j], z[j]) = meshwarp(x[j], y[j], z[j])
  end

  # Compute the metric terms
  if dim == 1
    computemetric!(x, J, ξx, sJ, nx, D)
  elseif dim == 2
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  elseif dim == 3
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)
  end

  M = kron(1, ntuple(j->ω, dim)...)
  MJ .= M .* J
  MJI .= 1 ./ MJ
  vMJI .= MJI[vmapM]

  sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
  sMJ .= sM .* sJ

  (vgeo, sgeo)
end
# }}}

# {{{ CPU Kernels
sync(::Type{Array}) = nothing

# {{{ 2-D
# Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where N
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

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

      U, V = Q[i, j, _U, e], Q[i, j, _V, e]
      ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]

      P = p0 * (R_gas * E / p0)^(c_p / c_v)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U + P
      fluxV_x = ρinv * V * U
      fluxE_x = ρinv * U * E

      fluxρ_y = V
      fluxU_y = ρinv * U * V
      fluxV_y = ρinv * V * V + P
      fluxE_y = ρinv * V * E

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
      rhs[i, j, _V, e] -= MJ * ρ * gravity
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

# Face RHS for 2-D
function facerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, sgeo, elems, vmapM,
                  vmapP, elemtobndy) where N
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

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

        bc = elemtobndy[f, e]
        PM = p0 * (R_gas * EM / p0)^(c_p / c_v)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          EP = Q[vidP, _E, eP]
          PP = p0 * (R_gas * EP / p0)^(c_p / c_v)
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
        fluxVM_x = ρMinv * VM * UM
        fluxEM_x = ρMinv * UM * EM

        fluxρM_y = VM
        fluxUM_y = ρMinv * UM * VM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxEM_y = ρMinv * VM * EM

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * VP * UP
        fluxEP_x = ρPinv * UP * EP

        fluxρP_y = VP
        fluxUP_y = ρPinv * UP * VP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxEP_y = ρPinv * VP * EP

        λM = ρMinv * abs(nxM * UM + nyM * VM) + sqrt(ρMinv * γ * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP) + sqrt(ρPinv * γ * PP)
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

# {{{ 3-D
# Volume RHS for 3-D
function volumerhs!(::Val{3}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where N
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

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

      U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
      ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

      P = p0 * (R_gas * E / p0)^(c_p / c_v)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U + P
      fluxV_x = ρinv * V * U
      fluxW_x = ρinv * W * U
      fluxE_x = E * ρinv * U

      fluxρ_y = V
      fluxU_y = ρinv * U * V
      fluxV_y = ρinv * V * V + P
      fluxW_y = ρinv * W * V
      fluxE_y = E * ρinv * V

      fluxρ_z = W
      fluxU_z = ρinv * U * W
      fluxV_z = ρinv * V * W
      fluxW_z = ρinv * W * W + P
      fluxE_z = E * ρinv * W

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
      rhs[i, j, k, _W, e] -= MJ * ρ * gravity
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

# Face RHS for 3-D
function facerhs!(::Val{3}, ::Val{N}, rhs::Array, Q, sgeo, elems, vmapM,
                  vmapP, elemtobndy) where N
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

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

        bc = elemtobndy[f, e]
        PM = p0 * (R_gas * EM / p0)^(c_p / c_v)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          PP = p0 * (R_gas * EP / p0)^(c_p / c_v)
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
        fluxVM_x = ρMinv * VM * UM
        fluxWM_x = ρMinv * WM * UM
        fluxEM_x = ρMinv * UM * EM

        fluxρM_y = VM
        fluxUM_y = ρMinv * UM * VM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxWM_y = ρMinv * WM * VM
        fluxEM_y = ρMinv * VM * EM

        fluxρM_z = WM
        fluxUM_z = ρMinv * UM * WM
        fluxVM_z = ρMinv * VM * WM
        fluxWM_z = ρMinv * WM * WM + PM
        fluxEM_z = ρMinv * WM * EM

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * VP * UP
        fluxWP_x = ρPinv * WP * UP
        fluxEP_x = ρPinv * UP * EP

        fluxρP_y = VP
        fluxUP_y = ρPinv * UP * VP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxWP_y = ρPinv * WP * VP
        fluxEP_y = ρPinv * VP * EP

        fluxρP_z = WP
        fluxUP_z = ρPinv * UP * WP
        fluxVP_z = ρPinv * VP * WP
        fluxWP_z = ρPinv * WP * WP + PP
        fluxEP_z = ρPinv * WP * EP

        λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + sqrt(ρMinv * γ * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + sqrt(ρPinv * γ * PP)
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

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, elems, rka,
                         rkb, dt) where {dim, N}
  @inbounds for e = elems, s = 1:_nstate, i = 1:(N+1)^dim
    Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
    rhs[i, s, e] *= rka
  end
end

# }}}
# }}}

# {{{ MPI Buffer handling
function fillsendQ!(::Val{dim}, ::Val{N}, sendQ, d_sendQ::Array, Q,
                    sendelems) where {dim, N}
  sendQ[:, :, :] .= Q[:, :, sendelems]
end

function transferrecvQ!(::Val{dim}, ::Val{N}, d_recvQ::Array, recvQ, Q,
                        nrealelem) where {dim, N}
  Q[:, :, nrealelem+1:end] .= recvQ[:, :, :]
end
# }}}

# {{{ L2 Energy (for all dimensions)
function L2energysquared(::Val{dim}, ::Val{N}, Q, vgeo, elems) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += vgeo[i, _MJ, e] * Q[i, q, e]^2
  end

  energy
end
# }}}

# {{{ RK loop
function lowstorageRK(::Val{dim}, ::Val{N}, mesh, vgeo, sgeo, Q, rhs, D,
                      dt, nsteps, tout, vmapM, vmapP, mpicomm;
                      ArrType=ArrType, plotstep=0) where {dim, N}
  DFloat = eltype(Q)
  mpirank = MPI.Comm_rank(mpicomm)

  # Fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy
  # (1994) ((5,4) 2N-Storage RK scheme.
  #
  # Ref:
  # @TECHREPORT{CarpenterKennedy1994,
  #   author = {M.~H. Carpenter and C.~A. Kennedy},
  #   title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},
  #   institution = {National Aeronautics and Space Administration},
  #   year = {1994},
  #   number = {NASA TM-109112},
  #   address = {Langley Research Center, Hampton, VA},
  # }
  RKA = (DFloat(0),
         DFloat(-567301805773)  / DFloat(1357537059087),
         DFloat(-2404267990393) / DFloat(2016746695238),
         DFloat(-3550918686646) / DFloat(2091501179385),
         DFloat(-1275806237668) / DFloat(842570457699 ))

  RKB = (DFloat(1432997174477) / DFloat(9575080441755 ),
         DFloat(5161836677717) / DFloat(13612068292357),
         DFloat(1720146321549) / DFloat(2090206949498 ),
         DFloat(3134564353537) / DFloat(4481467310338 ),
         DFloat(2277821191437) / DFloat(14882151754819))

  RKC = (DFloat(0),
         DFloat(1432997174477) / DFloat(9575080441755),
         DFloat(2526269341429) / DFloat(6820363962896),
         DFloat(2006345519317) / DFloat(3224310063776),
         DFloat(2802321613138) / DFloat(2924317926251))

  # Create send and recv request array
  nnabr = length(mesh.nabrtorank)
  sendreq = fill(MPI.REQUEST_NULL, nnabr)
  recvreq = fill(MPI.REQUEST_NULL, nnabr)

  # Create send and recv buffer
  sendQ = zeros(DFloat, (N+1)^dim, size(Q,2), length(mesh.sendelems))
  recvQ = zeros(DFloat, (N+1)^dim, size(Q,2), length(mesh.ghostelems))

  nrealelem = length(mesh.realelems)
  nsendelem = length(mesh.sendelems)
  nrecvelem = length(mesh.ghostelems)
  nelem = length(mesh.elems)

  d_QL, d_rhsL = ArrType(Q), ArrType(rhs)
  d_vgeoL, d_sgeo = ArrType(vgeo), ArrType(sgeo)
  d_vmapM, d_vmapP = ArrType(vmapM), ArrType(vmapP)
  d_sendelems, d_elemtobndy = ArrType(mesh.sendelems), ArrType(mesh.elemtobndy)
  d_sendQ, d_recvQ = ArrType(sendQ), ArrType(recvQ)
  d_D = ArrType(D)

  Qshape    = (fill(N+1, dim)..., size(Q, 2), size(Q, 3))
  vgeoshape = (fill(N+1, dim)..., _nvgeo, size(Q, 3))

  d_QC = reshape(d_QL, Qshape)
  d_rhsC = reshape(d_rhsL, Qshape...)
  d_vgeoC = reshape(d_vgeoL, vgeoshape)

  start_time = t1 = time_ns()
  for step = 1:nsteps
    for s = 1:length(RKA)
      # post MPI receives
      for n = 1:nnabr
        recvreq[n] = MPI.Irecv!((@view recvQ[:, :, mesh.nabrtorecv[n]]),
                                mesh.nabrtorank[n], 777, mpicomm)
      end

      # wait on (prior) MPI sends
      MPI.Waitall!(sendreq)

      # pack data in send buffer
      fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)

      # post MPI sends
      for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
      end

      # volume RHS computation
      volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems)

      # wait on MPI receives
      MPI.Waitall!(recvreq)

      # copy data to state vectors
      transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)

      # face RHS computation
      facerhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, mesh.realelems, d_vmapM,
               d_vmapP, d_elemtobndy)

      # update solution and scale RHS
      updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, mesh.realelems,
                      RKA[s%length(RKA)+1], RKB[s], dt)
    end
    if step == 1
      sync(ArrType)
      start_time = time_ns()
    end
    if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
      sync(ArrType)
      t1 = time_ns()
      avg_stage_time = (time_ns() - start_time) * 1e-9 / ((step-1) * length(RKA))
      @show (step, nsteps, avg_stage_time)
    end
    # TODO: Fix VTK for 1-D
    if dim > 1 && plotstep > 0 && step % plotstep == 0
      Q .= d_QL
      X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                            nelem), dim)
      ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
      U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
      V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
      W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
      E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
      writemesh(@sprintf("viz/euler%dD_%s_rank_%04d_step_%05d",
                         dim, ArrType, mpirank, step), X...;
                fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
                realelems=mesh.realelems)
    end
  end
  if mpirank == 0
    avg_stage_time = (time_ns() - start_time) * 1e-9 / ((nsteps-1) * length(RKA))
    @show (nsteps, avg_stage_time)
  end
  Q .= d_QL
  rhs .= d_rhsL
end
# }}}

# {{{ atmosdycore driver
function atmosdycore(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend;
                     meshwarp=(x...)->identity(x), tout = 1, ArrType=Array,
                     plotstep=0) where {dim, N}
  DFloat = typeof(tend)

  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # Partion the mesh using a Hilbert curve based partitioning
  mpirank == 0 && println("[CPU] partiting mesh...")
  mesh = partition(mpicomm, mesh...)

  # Connect the mesh in parallel
  mpirank == 0 && println("[CPU] connecting mesh...")
  mesh = connectmesh(mpicomm, mesh...)

  # Get the vmaps
  mpirank == 0 && println("[CPU] computing mappings...")
  (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                            mesh.elemtoordr)

  # Create 1-D operators
  (ξ, ω) = lglpoints(DFloat, N)
  D = spectralderivative(ξ)

  # Compute the geometry
  mpirank == 0 && println("[CPU] computing metrics...")
  (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
  (nface, nelem) = size(mesh.elemtoelem)

  # Storage for the solution, rhs, and error
  mpirank == 0 && println("[CPU] creating fields (CPU)...")
  Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)
  rhs = zeros(DFloat, (N+1)^dim, _nstate, nelem)

  # setup the initial condition
  mpirank == 0 && println("[CPU] computing initial conditions (CPU)...")
  @inbounds for e = 1:nelem, i = 1:(N+1)^dim
    x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
    ρ, U, V, W, E = ic(x, y, z)
    Q[i, _ρ, e] = ρ
    Q[i, _U, e] = U
    Q[i, _V, e] = V
    Q[i, _W, e] = W
    Q[i, _E, e] = E
  end

  # Compute time step
  mpirank == 0 && println("[CPU] computing dt (CPU)...")
  base_dt = cfl(Val(dim), Val(N), vgeo, Q, mpicomm) / N^√2
  mpirank == 0 && @show base_dt

  nsteps = ceil(Int64, tend / base_dt)
  dt = tend / nsteps
  mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

  # Do time stepping
  stats = zeros(DFloat, 2)
  mpirank == 0 && println("[CPU] computing initial energy...")
  stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

  # plot the initial condition
  mkpath("viz")
  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    writemesh(@sprintf("viz/euler%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
              realelems=mesh.realelems)
  end

  mpirank == 0 && println("[DEV] starting time stepper...")
  lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
               vmapM, vmapP, mpicomm; ArrType=ArrType, plotstep=plotstep)

  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    writemesh(@sprintf("viz/euler%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
              realelems=mesh.realelems)
  end

  mpirank == 0 && println("[CPU] computing final energy...")
  stats[2] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

  stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

  if  mpirank == 0
    @show eng0 = stats[1]
    @show engf = stats[2]
    @show Δeng = engf - eng0
  end
end
# }}}

end # module
