module DGBalanceLawDiscretizations
using MPI
using ...Grids
using ...MPIStateArrays
using Documenter
using StaticArrays

export DGBalanceLaw

"""
    DGBalanceLaw(;grid::DiscontinuousSpectralElementGrid,
                 nstate::Int,
                 flux!::Function,
                 numericalflux!::Function,
                 gradstates::NTuple{X, Int} = (),
                 nauxcstate=0,
                 nauxdstate=0
                 )

Given a balance law for `nstate` fields of the form

   ``q_{,t} + Σ_{i=1,...d} F_{i,i} = s``

The flux function `F_{i}` can depend on the state `q`, gradient `q` for `j =
1,...,d`, time `t`, and a set of user defined "constant" state `ϕ`.

The flux functions `flux!` has syntax `flux!(F, Q, G, ϕ_c, ϕ_d, t)` where:
- `F` is an `MArray` of size `(d, nstate)` to be filled
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
struct DGBalanceLaw
  grid::DiscontinuousSpectralElementGrid

  "number of state"
  nstate::Int

  "Tuple of states to take the gradient of"
  gradstates::Tuple

  "physical flux function"
  flux!::Function

  "numerical flux function"
  numericalflux!::Function

  "storage for the grad"
  Qgrad::Union{Nothing, MPIStateArray}

end

function DGBalanceLaw(;grid, nstate, flux!, numericalflux!, gradstates=())
  ngradstate = length(gradstates)
  if ngradstate > 0
    topology = grid.topology
    Np = dofs_per_element(grid)
    h_vgeo = Array(grid.vgeo)
    DFloat = eltype(h_vgeo)
    DA = arraytype(grid)
    # TODO: Clean up this MPIStateArray interface...
    Qgrad = MPIStateArray{Tuple{Np, ngradstate}, DFloat, DA
                         }(topology.mpicomm,
                           length(topology.elems),
                           realelems=topology.realelems,
                           ghostelems=topology.ghostelems,
                           sendelems=topology.sendelems,
                           nabrtorank=topology.nabrtorank,
                           nabrtorecv=topology.nabrtorecv,
                           nabrtosend=topology.nabrtosend,
                           weights=view(h_vgeo, :, grid.Mid, :))
  else
    Qgrad = nothing
  end

  DGBalanceLaw(grid, nstate, gradstates, flux!, numericalflux!, Qgrad)
end

"""
    MPIStateArray(disc::DGBalanceLaw)

Given a discretization `disc` constructs an `MPIStateArrays` for holding a
solution state
"""
function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw)
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
                                                          disc.grid.Mid, :))
end

function MPIStateArrays.MPIStateArray(disc::DGBalanceLaw,
                                      ic!::Function)
  Q = MPIStateArray(disc)

  nvar = disc.nstate
  grid = disc.grid
  vgeo = grid.vgeo
  Np = dofs_per_element(grid)

  # FIXME: GPUify me
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q) : (Array(vgeo), Array(Q))
  Qdof = MArray{Tuple{nvar}, eltype(h_Q)}(undef)
  @inbounds for e = 1:size(Q, 3), i = 1:Np
    (x, y, z) = (h_vgeo[i, grid.xid, e], h_vgeo[i, grid.yid, e],
                 h_vgeo[i, grid.zid, e])
    ic!(Qdof, x, y, z)
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
                             d::DGBalanceLaw
                            ) = MPIStateArray(d, f)

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

end
