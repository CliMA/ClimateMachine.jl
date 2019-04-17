module VanillaAtmosDiscretizations
using MPI

using ..AtmosDycore
AD = AtmosDycore
using ...Grids
using ...MPIStateArrays

export VanillaAtmosDiscretization

using ...ParametersType
using ...PlanetParameters: cp_d, cv_d, R_d, grav, MSLP
using CLIMA.MoistThermodynamics

# ASR Correction to Prandtl number from 7.1 to 0.71
@parameter prandtl 71//100 "Prandtl number: ratio of momentum diffusivity to thermal diffusivity"
@parameter λ_stokes  -2//3  "scaling for viscous effect associated with volume change"
@parameter k_μ cp_d/prandtl "thermal conductivity / dynamic viscosity"
@parameter Cs      12//100  "Smagorinsky constant"

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const _nstategrad = 18
const (_ρx, _ρy, _ρz, _ux, _uy, _uz, _vx, _vy, _vz, _wx, _wy, _wz,
       _Tx, _Ty, _Tz, _θx, _θy, _θz) = 1:_nstategrad

"""
    VanillaAtmosDiscretization{nmoist, ntrace}(grid; gravity = true,
    viscosity = 0)

Given a 'grid <: AbstractGrid' this construct all the data necessary to run a
vanilla discontinuous Galerkin discretization of the the compressible Euler
equations with `nmoist` moisture variables and `ntrace` tracer variables. If the
boolean keyword argument `gravity` is `true` then gravity is used otherwise it
is not. Isotropic viscosity can be used if `viscosity` is set to a positive
constant.
"""
struct VanillaAtmosDiscretization{T, dim, polynomialorder, numberofDOFs,
                                  DeviceArray, nmoist, ntrace, DASAT3,
                                  GT } <: AD.AbstractAtmosDiscretization
  "grid"
  grid::GT

  "gravitational acceleration (m/s^2)"
  gravity::T

  "viscosity constant"
  viscosity::T

  "storage for the grad"
  grad::DASAT3

  VanillaAtmosDiscretization(grid;
                             # How many tracer variables
                             ntrace=0,
                             # How many moisture variables
                             nmoist=0,
                             kw...) =
  VanillaAtmosDiscretization{nmoist, ntrace}(grid; kw...)

  function VanillaAtmosDiscretization{nmoist, ntrace
                                     }(grid::AbstractGrid{T, dim, N, Np, DA};
                                       # Use gravity?
                                       gravity = true,
                                       # viscosity constant
                                       viscosity = 0
                                      ) where {T, dim, N, Np, DA,
                                               nmoist, ntrace}
    topology = grid.topology
    
    ngrad = _nstategrad + 3*nmoist
    # FIXME: Remove after updating CUDA
    h_vgeo = Array(grid.vgeo)
    grad = MPIStateArray{Tuple{Np, ngrad}, T, DA}(topology.mpicomm,
                                                  length(topology.elems),
                                                  realelems=topology.realelems,
                                                  ghostelems=topology.ghostelems,
                                                  sendelems=topology.sendelems,
                                                  nabrtorank=topology.nabrtorank,
                                                  nabrtorecv=topology.nabrtorecv,
                                                  nabrtosend=topology.nabrtosend,
                                                  weights=view(h_vgeo, :, grid.Mid, :))

    GT = typeof(grid)
    DASAT3 = typeof(grad)

    new{T, dim, N, Np, DA, nmoist, ntrace, DASAT3, GT}(grid,
                                                       gravity ? grav : 0,
                                                       viscosity, grad)
  end
end

function Base.getproperty(X::VanillaAtmosDiscretization{T, dim, polynomialorder,
                                                        numberofDOFs,
                                                        DeviceArray, nmoist,
                                                        ntrace},
                          s::Symbol) where {T, dim, polynomialorder,
                                            numberofDOFs, DeviceArray, nmoist,
                                            ntrace}
  if s ∈ keys(stateid)
    stateid[s]
  elseif s == :nstate
    _nstate
  elseif s == :moistrange
    _nstate .+ (1:nmoist)
  elseif s == :tracerange
    _nstate+nmoist .+ (1:ntrace)
  else
    getfield(X, s)
  end
end

function Base.propertynames(X::VanillaAtmosDiscretization)
  (fieldnames(VanillaAtmosDiscretization)..., keys(stateid)...)
end

"""
    MPIStateArray(disc::VanillaAtmosDiscretization)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state
"""
function MPIStateArrays.MPIStateArray(disc::VanillaAtmosDiscretization{
                                            T, dim, N, Np, DA, nmoist, ntrace}
                                      ) where {T, dim, N, Np, DA, nmoist,
                                               ntrace}
  topology = disc.grid.topology
  nvar = _nstate + nmoist + ntrace
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(disc.grid.vgeo)
  MPIStateArray{Tuple{Np, nvar}, T, DA}(topology.mpicomm,
                                        length(topology.elems),
                                        realelems=topology.realelems,
                                        ghostelems=topology.ghostelems,
                                        sendelems=topology.sendelems,
                                        nabrtorank=topology.nabrtorank,
                                        nabrtorecv=topology.nabrtorecv,
                                        nabrtosend=topology.nabrtosend,
                                        weights=view(h_vgeo, :, disc.grid.Mid, :))
end

function MPIStateArrays.MPIStateArray(disc::VanillaAtmosDiscretization{
                                               T, dim, N, Np, DA, nmoist,
                                               ntrace}, ic::Function
                                       ) where {T, dim, N, Np, DA, nmoist,
                                                ntrace}
  Q = MPIStateArray(disc)

  nvar = _nstate + nmoist + ntrace
  G    = disc.grid
  vgeo = G.vgeo

  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q) : (Array(vgeo), Array(Q))
  @inbounds for e = 1:size(Q, 3), i = 1:Np
    x, y, z = h_vgeo[i, G.xid, e], h_vgeo[i, G.yid, e], h_vgeo[i, G.zid, e]
    q0 = ic(x, y, z)

    # Assume that this will be compile time?

    @assert ((nmoist >  0 && nmoist == length(q0.Qmoist)) ||
             (nmoist == 0 && :Qmoist ∉ fieldnames(typeof(q0))))
    @assert ((ntrace >  0 && ntrace == length(q0.Qtrace)) ||
             (ntrace == 0 && :Qtrace ∉ fieldnames(typeof(q0))))

    h_Q[i, [_ρ, _U, _V, _W, _E], e] .= (q0.ρ, q0.U, q0.V, q0.W, q0.E)
    (nmoist > 0) && (h_Q[i, _nstate .+           (1:nmoist), e] .= q0.Qmoist)
    (ntrace > 0) && (h_Q[i, _nstate .+ nmoist .+ (1:ntrace), e] .= q0.Qtrace)
  end
  if !host_array
    Q .= h_Q
  end

  Q
end

MPIStateArrays.MPIStateArray(f::Function,
                             d::VanillaAtmosDiscretization
                            ) = MPIStateArray(d, f)

"""
    estimatedt(disc::VanillaAtmosDiscretization, Q::MPIStateArray)

Given a discretization `disc` and a state `Q` compute an estimate for the time
step

!!! todo
    This estimate is currently very conservative, needs to be revisited
"""
function estimatedt(disc::VanillaAtmosDiscretization{T, dim, N, Np, DA, nmoist},
                    Q::MPIStateArray) where {T, dim, N, Np, DA, nmoist}
  
  @assert T == eltype(Q)

  G = disc.grid
  vgeo = G.vgeo
  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q) : (Array(vgeo), Array(Q))
  estimatedt(Val(dim), Val(N), Val(nmoist), G, disc.gravity, h_Q, h_vgeo, G.topology.mpicomm)
end

# FIXME: This needs cleaning up
function estimatedt(::Val{dim}, ::Val{N}, ::Val{nmoist}, G, gravity, Q, vgeo,
                    mpicomm) where {dim, N, nmoist}

  DFloat = eltype(Q)

  Np = (N+1)^dim
  (_, _, nelem) = size(Q)

  dt = [floatmax(DFloat)]
  
  # Allocate 3 spaces for moist tracers qm, with a zero default value
  q_m = zeros(DFloat, max(3, nmoist))

  #Allocate at least three spaces for qm, with a zero default value
  #q_m = zeros(DFloat, 3)
    
  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
      E = Q[n, _E, e]
      y = vgeo[n, G.yid, e]
      
      #compute temperature and internal energy
      #get moist variables from state vector
      for m = 1:nmoist
          s = _nstate + m
          q_m[m] = Q[n, s, e]/ρ
      end
      
      E_int = E - (U^2 + V^2)/(2*ρ) - ρ * gravity * y
      # get adjusted temperature and liquid and ice specific humidities
      T            = saturation_adjustment(E_int/ρ, ρ, q_m[1])
     
      ξx, ξy, ηx, ηy = vgeo[n, G.ξxid, e], vgeo[n, G.ξyid, e],
                       vgeo[n, G.ηxid, e], vgeo[n, G.ηyid, e]
      
       # Calculate local dt
      loc_dt = 2ρ / max(abs(U * ξx + V * ξy) + ρ * soundspeed_air(T),
                        abs(U * ηx + V * ηy) + ρ * soundspeed_air(T))
      dt[1] = min(dt[1], loc_dt)
  
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V, W = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
      E = Q[n, _E, e]
      z = vgeo[n, G.zid, e]
      
      #Compute (Temperature) and (E_int per unit mass)
      E_int = E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * z 
      #Loop over moist variables and extract q_m where q_m[1] = q_t, q_m[2] = q_liq, q_m[3] = q_ice
      for m = 1:nmoist
          s = _nstate + m 
          q_m[m] = Q[n, s, e] / ρ 
      end

      # get adjusted temperature and liquid and ice specific humidities
      T            = saturation_adjustment(E_int/ρ, ρ, q_m[1])
     
      ξx, ξy, ξz = vgeo[n, G.ξxid, e], vgeo[n, G.ξyid, e], vgeo[n, G.ξzid, e]
      ηx, ηy, ηz = vgeo[n, G.ηxid, e], vgeo[n, G.ηyid, e], vgeo[n, G.ηzid, e]
      ζx, ζy, ζz = vgeo[n, G.ζxid, e], vgeo[n, G.ζyid, e], vgeo[n, G.ζzid, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * soundspeed_air(T),
                        abs(U * ηx + V * ηy + W * ηz) + ρ * soundspeed_air(T),
                        abs(U * ζx + V * ζy + W * ζz) + ρ * soundspeed_air(T))
      dt[1] = min(dt[1], loc_dt)
   
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm) / N^√2
end

#{{{ rhs!

function AD.getrhsfunction(disc::VanillaAtmosDiscretization)
  (x...) -> rhs!(x..., disc)
end

rhs!(dQ, Q, p::Nothing, t, sd::VanillaAtmosDiscretization) = rhs!(dQ, Q, t, sd)

function rhs!(dQ::MPIStateArray{S, T}, Q::MPIStateArray{S, T}, t::T,
              disc::VanillaAtmosDiscretization{T, dim, N, Np, DA, nmoist,
                                               ntrace}
             ) where {S, T, dim, N, Np, DA, nmoist, ntrace}
  grid = disc.grid
  topology = grid.topology

  gravity = disc.gravity

  grad = disc.grad

  vgeo = grid.vgeo
  sgeo = grid.sgeo
  Dmat = grid.D
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy

  DFloat = eltype(Q)
  @assert DFloat == eltype(Q)
  @assert DFloat == eltype(vgeo)
  @assert DFloat == eltype(sgeo)
  @assert DFloat == eltype(Dmat)
  @assert DFloat == eltype(grad)
  @assert DFloat == eltype(dQ)

  ########################
  # Gradient Computation #
  ########################
  MPIStateArrays.start_ghost_exchange!(Q)

  volumegrad!(Val(dim), Val(N), Val(nmoist), Val(ntrace), grad.Q, Q.Q, vgeo,
              gravity, Dmat, topology.realelems)

  MPIStateArrays.finish_ghost_exchange!(Q)

  facegrad!(Val(dim), Val(N), Val(nmoist), Val(ntrace), grad.Q, Q.Q, vgeo,
            sgeo, gravity, topology.realelems, vmapM, vmapP, elemtobndy)


  ###################
  # RHS Computation #
  ###################

  viscosity::DFloat = disc.viscosity
 
  MPIStateArrays.start_ghost_exchange!(grad)

  volumerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), dQ.Q, Q.Q, grad.Q,
             vgeo, gravity, viscosity, Dmat, topology.realelems)

  MPIStateArrays.finish_ghost_exchange!(grad)

  facerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), dQ.Q, Q.Q, grad.Q,
           vgeo, sgeo, gravity, viscosity, topology.realelems, vmapM, vmapP,
           elemtobndy)

end
# }}}

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
# }}}

using Requires

include("VanillaAtmosDiscretizations_kernels.jl")

include("../../../Mesh/vtk.jl")
function writevtk(prefix, Q::MPIStateArray, disc::VanillaAtmosDiscretization)
  vgeo = disc.grid.vgeo
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q.Q) : (Array(vgeo), Array(Q))
  writevtk(prefix, h_vgeo, h_Q, disc.grid)
end

function writevtk(prefix, vgeo::Array, Q::Array,
                  G::Grids.AbstractGrid{T, dim, N}) where {T, dim, N}

  Nq  = N+1
  nelem = size(Q)[end]
  Xid = (G.xid, G.yid, G.zid)
  X = ntuple(j->reshape((@view vgeo[:, Xid[j], :]),
                        ntuple(j->Nq, dim)...,
                        nelem), dim)
 
  # Tuple splat output for final plots
  #ntuple(j->(reshape(@view Q[:, j, :]),
  #             ntuple(j->Nq,dim)...,nelem),dim)

  ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->Nq, dim)..., nelem)
  U = reshape((@view Q[:, _U, :]), ntuple(j->Nq, dim)..., nelem)
  V = reshape((@view Q[:, _V, :]), ntuple(j->Nq, dim)..., nelem)
  W = reshape((@view Q[:, _W, :]), ntuple(j->Nq, dim)..., nelem)
  E = reshape((@view Q[:, _E, :]), ntuple(j->Nq, dim)..., nelem)
  Qt= reshape((@view Q[:, _nstate+1, :]), ntuple(j->Nq, dim)..., nelem)
  Ql= reshape((@view Q[:, _nstate+2, :]), ntuple(j->Nq, dim)..., nelem)

  #[_, nstates, _] = size(Q)    
  #nmoist = nstates - (ndim + 2)
        
 # var_string = ["DENSI" , "U" , "U" , "W", "E"]
 #    error("NSTATES ", size(Q))
 # fields = ntuple(i->(var_string[i], reshape((@view Q[:,i,:]), ntuple(j->Nq,dim)...,nelem)), size(Q,2))
 # 
 # writemesh(prefix, X...; fields=fields, realelems=G.topology.realelems)

  fields = ntuple(i->("Q$i", reshape((@view Q[:,i,:]), ntuple(j->Nq,dim)...,nelem)), size(Q,2)) 
  
  writemesh(prefix, X...;
            fields=fields,
            realelems=G.topology.realelems)
  
end
end


