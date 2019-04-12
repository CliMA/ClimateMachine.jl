module DGBalanceLawDiscretizations
using MPI
using ...Grids
using ...MPIStateArrays
using Documenter
using StaticArrays
using ...SpaceMethods

export DGBalanceLaw

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
# }}}

include("DGBalanceLawDiscretizations_kernels.jl")
include("NumericalFluxes.jl")

"""
    DGBalanceLaw(;grid,
                 length_state_vector,
                 flux!,
                 numericalflux!,
                 gradstates=(),
                 length_constant_auxiliary=0,
                 length_dynamic_auxiliary=0,
                 dynamic_auxiliary_update! = (auxd, Q, auxc, t) -> error(),
                 constant_auxiliary_init! = nothing)

Given a balance law for `length_state_vector` fields of the form

   ``q_{,t} + Σ_{i=1,...d} F_{i,i} = s``

The flux function `F_{i}` can depend on the state `q`, gradient `q` for `j =
1,...,d`, time `t`, and a set of user defined "constant" state `ϕ`.

The flux functions `flux!` has syntax
```
    flux!(F, Q, G, ϕ_c, ϕ_d, t)
```
where:
- `F` is an `MArray` of size `(d, length_state_vector)` to be filled
- `Q` is the state to evaluate
- `G` is an array of size `(d, ngradstate)` for `Q` for `j = 1,...,d` for the
  subset of variables sepcified by `gradstates`
- `ϕ_c` is the user-defined constant state
- `ϕ_d` is the user-defined dynamic state
- `t` is the time

!!! todo

    Add docs for other arguments...

!!! todo

    Stil need to add
    - `bcfun!`
    - `source!`
    - `initϕ_c!`
    - `updateϕ_d!`
    - `gradnumericalflux!`?

"""
struct DGBalanceLaw <: AbstractDGMethod
  grid::DiscontinuousSpectralElementGrid

  "number of state"
  nstate::Int

  "Tuple of states to take the gradient of"
  gradstates::Tuple

  "dynamic auxiliary state"
  nauxdstate::Int

  "physical flux function"
  flux!::Function

  "numerical flux function"
  numericalflux!::Function

  "storage for the grad"
  Qgrad_auxd::MPIStateArray

  "constant auxiliary state"
  auxc::MPIStateArray

  "update function for auxd state"
  auxdfun!::Function
end

function DGBalanceLaw(;grid, length_state_vector, flux!, numericalflux!,
                      gradstates=(), length_constant_auxiliary=0,
                      length_dynamic_auxiliary=0, dynamic_auxiliary_update! =
                      (auxd, Q, auxc, t) -> error(), constant_auxiliary_init! =
                      nothing)
  ngradstate = length(gradstates)
  topology = grid.topology
  Np = dofs_per_element(grid)
  h_vgeo = Array(grid.vgeo)
  DFloat = eltype(h_vgeo)
  DA = arraytype(grid)
  # TODO: Clean up this MPIStateArray interface...
  Qgrad_auxd = MPIStateArray{Tuple{Np, ngradstate + length_dynamic_auxiliary},
                             DFloat, DA
                            }(topology.mpicomm,
                              length(topology.elems),
                              realelems=topology.realelems,
                              ghostelems=topology.ghostelems,
                              sendelems=topology.sendelems,
                              nabrtorank=topology.nabrtorank,
                              nabrtorecv=topology.nabrtorecv,
                              nabrtosend=topology.nabrtosend,
                              weights=view(h_vgeo, :, grid.Mid, :),
                              commtag=111)

  auxc = MPIStateArray{Tuple{Np, length_constant_auxiliary}, DFloat, DA
                      }(topology.mpicomm,
                        length(topology.elems),
                        realelems=topology.realelems,
                        ghostelems=topology.ghostelems,
                        sendelems=topology.sendelems,
                        nabrtorank=topology.nabrtorank,
                        nabrtorecv=topology.nabrtorecv,
                        nabrtosend=topology.nabrtosend,
                        weights=view(h_vgeo, :, grid.Mid, :),
                        commtag=222)

  if constant_auxiliary_init! !== nothing
    @assert length_constant_auxiliary > 0
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    vgeo = grid.vgeo
    initauxc!(Val(dim), Val(N), Val(length_constant_auxiliary),
              constant_auxiliary_init!, auxc, vgeo, topology.realelems)
    MPIStateArrays.start_ghost_exchange!(auxc)
    MPIStateArrays.finish_ghost_exchange!(auxc)
  end

  DGBalanceLaw(grid, length_state_vector, gradstates, length_dynamic_auxiliary,
               flux!, numericalflux!, Qgrad_auxd, auxc,
               dynamic_auxiliary_update!)
end

"""
    MPIStateArray(disc::DGBalanceLaw)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state
"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw; commtag=888)
  grid = disc.grid
  topology = disc.grid.topology
  nvar = disc.nstate
  # FIXME: Remove after updating CUDA
  h_vgeo = Array(disc.grid.vgeo)
  DFloat = eltype(h_vgeo)
  Np = dofs_per_element(grid)
  DA = arraytype(grid)
  MPIStateArray{Tuple{Np, nvar}, DFloat, DA}(topology.mpicomm,
                                             length(topology.elems),
                                             realelems=topology.realelems,
                                             ghostelems=topology.ghostelems,
                                             sendelems=topology.sendelems,
                                             nabrtorank=topology.nabrtorank,
                                             nabrtorecv=topology.nabrtorecv,
                                             nabrtosend=topology.nabrtosend,
                                             weights=view(h_vgeo, :,
                                                          disc.grid.Mid, :),
                                             commtag=commtag)
end

function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw,
                                      ic!::Function; commtag=888)
  Q = MPIStateArray(disc; commtag=commtag)

  nvar = disc.nstate
  grid = disc.grid
  vgeo = grid.vgeo
  Np = dofs_per_element(grid)
  auxc = disc.auxc
  nauxcstate = size(auxc, 2)

  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q, h_auxc) = host_array ? (vgeo, Q, auxc) :
                                       (Array(vgeo), Array(Q), Array(auxc))
  Qdof = MArray{Tuple{nvar}, eltype(h_Q)}(undef)
  ϕcdof = MArray{Tuple{nauxcstate}, eltype(h_Q)}(undef)
  @inbounds for e = 1:size(Q, 3), i = 1:Np
    (x, y, z) = (h_vgeo[i, grid.xid, e], h_vgeo[i, grid.yid, e],
                 h_vgeo[i, grid.zid, e])
    if nauxcstate > 0
      for s = 1:nauxcstate
        ϕcdof[s] = h_auxc[i, s, e]
      end
      ic!(Qdof, x, y, z, ϕcdof)
    else
      ic!(Qdof, x, y, z)
    end
    for n = 1:nvar
      h_Q[i, n, e] = Qdof[n]
    end
  end
  if !host_array
    Q .= h_Q
  end

  Q
end

MPIStateArrays.MPIStateArray(f::Function,
                             d::DGBalanceLaw; commtag=888
                            ) = MPIStateArray(d, f; commtag=commtag)

#TODO: Need to think about where this should really live. Grid? MPIStateArrays?
include("../Mesh/vtk.jl")
function writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw,
                  fieldnames=nothing)
  vgeo = disc.grid.vgeo
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q.Q) : (Array(vgeo), Array(Q))
  writevtk(prefix, h_vgeo, h_Q, disc.grid, fieldnames)
end

function writevtk(prefix, vgeo::Array, Q::Array,
                  grid, fieldnames)

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq  = N+1

  nelem = size(Q)[end]
  Xid = (grid.xid, grid.yid, grid.zid)
  X = ntuple(j->reshape((@view vgeo[:, Xid[j], :]),
                        ntuple(j->Nq, dim)...,
                        nelem), dim)
  if fieldnames == nothing
    fields = ntuple(i->("Q$i", reshape((@view Q[:, i, :]),
                                       ntuple(j->Nq, dim)..., nelem)),
                    size(Q, 2))
  else
    fields = ntuple(i->(fieldnames[i], reshape((@view Q[:, i, :]),
                                               ntuple(j->Nq, dim)..., nelem)),
                    size(Q, 2))
  end
  writemesh(prefix, X...; fields=fields, realelems=grid.topology.realelems)
end

function SpaceMethods.odefun!(disc::DGBalanceLaw, dQ::MPIStateArray,
                              Q::MPIStateArray, t)
  grid = disc.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)

  Qgrad_auxd = disc.Qgrad_auxd
  auxc = disc.auxc

  nstate = disc.nstate
  ngradstate = length(disc.gradstates)
  nauxcstate = size(auxc, 2)
  nauxdstate = disc.nauxdstate

  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy

  ########################
  # Gradient Computation #
  ########################
  MPIStateArrays.start_ghost_exchange!(Q)

  if ngradstate > 0
    error("Grad not implemented yet")

    # TODO: volumegrad!

    MPIStateArrays.finish_ghost_exchange!(Q)

    # TODO: facegrad!

    MPIStateArrays.start_ghost_exchange!(Qgrad_auxd)
  elseif nauxdstate > 0
    # TODO: compute dynamic aux state
    updateauxd!(Val(dim), Val(N), Val(nstate), Val(ngradstate), Val(nauxcstate),
               Val(nauxdstate), disc.auxdfun!, Q.Q, Qgrad_auxd.Q,
               auxc.Q, t, topology.realelems)

    MPIStateArrays.start_ghost_exchange!(Qgrad_auxd)
  end

  ###################
  # RHS Computation #
  ###################

  volumerhs!(Val(dim), Val(N), Val(nstate), Val(ngradstate), Val(nauxcstate),
             Val(nauxdstate), disc.flux!, dQ.Q, Q.Q, Qgrad_auxd.Q,
             auxc.Q, vgeo, t, Dmat, topology.realelems)

  MPIStateArrays.finish_ghost_exchange!(ngradstate > 0 ? Qgrad_auxd : Q)

  (ngradstate > 0 || nauxdstate > 0) && MPIStateArrays.finish_ghost_exchange!(Qgrad_auxd)
  ngradstate == 0 && MPIStateArrays.finish_ghost_exchange!(Q)

  facerhs!(Val(dim), Val(N), Val(nstate), Val(ngradstate), Val(nauxcstate),
           Val(nauxdstate), disc.numericalflux!, dQ.Q, Q.Q, Qgrad_auxd.Q,
           auxc.Q, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
           topology.realelems)
end

end
