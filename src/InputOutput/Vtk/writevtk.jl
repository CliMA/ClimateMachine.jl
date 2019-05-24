using ..Grids
using ..MPIStateArrays
using ..DGBalanceLawDiscretizations
using ..AtmosDycore.VanillaAtmosDiscretizations

"""
    writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw [, fieldnames])

Write a vtk file for all the fields in the state array `Q` using geometry and
connectivity information from `disc.grid`. The filename will start with `prefix`
which may also contain a directory path. The names used for each of the fields
in the vtk file can be specified through the collection of strings `fieldnames`;
if not specified the fields names will be `"Q1"` through `"Qk"` where `k` is the
number of states in `Q`, i.e., `k = size(Q,2)`.

"""
function writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw,
                  fieldnames=nothing)
  vgeo = disc.grid.vgeo
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q) = host_array ? (vgeo, Q.Q) : (Array(vgeo), Array(Q))
  writevtk_helper(prefix, h_vgeo, h_Q, disc.grid, fieldnames)
end

"""
    writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw, fieldnames,
             auxstate::MPIStateArray, auxfieldnames)

Write a vtk file for all the fields in the state array `Q` and auxiliary state
`auxstate` using geometry and connectivity information from `disc.grid`. The
filename will start with `prefix` which may also contain a directory path. The
names used for each of the fields in the vtk file can be specified through the
collection of strings `fieldnames` and `auxfieldnames`.

If `fieldnames === nothing` then the fields names will be `"Q1"` through `"Qk"`
where `k` is the number of states in `Q`, i.e., `k = size(Q,2)`.

If `auxfieldnames === nothing` then the fields names will be `"aux1"` through
`"auxk"` where `k` is the number of states in `auxstate`, i.e., `k =
size(auxstate,2)`.

"""
function writevtk(prefix, Q::MPIStateArray, disc::DGBalanceLaw,
                  fieldnames, auxstate, auxfieldnames)
  vgeo = disc.grid.vgeo
  host_array = Array ∈ typeof(Q).parameters
  (h_vgeo, h_Q, h_aux) = host_array ? (vgeo, Q.Q, auxstate.Q) :
                                      (Array(vgeo), Array(Q), Array(auxstate))
  writevtk_helper(prefix, h_vgeo, h_Q, disc.grid, fieldnames, h_aux,
                  auxfieldnames)
end


"""
    writevtk_helper(prefix, vgeo::Array, Q::Array, grid, fieldnames)

Internal helper function for `writevtk`
"""
function writevtk_helper(prefix, vgeo::Array, Q::Array, grid,
                         fieldnames, auxstate=nothing, auxfieldnames=nothing)

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
  if auxstate !== nothing
    if auxfieldnames === nothing
      auxfields = ntuple(i->("aux$i", reshape((@view auxstate[:, i, :]),
                                              ntuple(j->Nq, dim)..., nelem)),
                         size(auxstate, 2))
    else
      auxfields = ntuple(i->(auxfieldnames[i], reshape((@view auxstate[:, i, :]),
                                                       ntuple(j->Nq, dim)...,
                                                       nelem)),
                         size(auxstate, 2))
    end
    fields = (fields..., auxfields...)
  end
  writemesh(prefix, X...; fields=fields, realelems=grid.topology.realelems)
end

function writevtk(prefix, Q::MPIStateArray, disc::VanillaAtmosDiscretization)
  vgeo = disc.grid.vgeo
  host_array = Array ∈ typeof(Q).parameters
  (vgeo, Q) = host_array ? (vgeo, Q.Q) : (Array(vgeo), Array(Q))
  writevtk_VanillaAtmosDiscretization(prefix, vgeo, Q, disc.grid)
end

function writevtk_VanillaAtmosDiscretization(prefix, vgeo::Array, Q::Array,
                                             G::Grids.AbstractGrid{T, dim, N}
                                            ) where {T, dim, N}

  Nq  = N+1

  nelem = size(Q)[end]
  Xid = (G.xid, G.yid, G.zid)
  X = ntuple(j->reshape((@view vgeo[:, Xid[j], :]),
                        ntuple(j->Nq, dim)...,
                        nelem), dim)

  _ρ = VanillaAtmosDiscretizations._ρ
  _U = VanillaAtmosDiscretizations._U
  _V = VanillaAtmosDiscretizations._V
  _W = VanillaAtmosDiscretizations._W
  _E = VanillaAtmosDiscretizations._E

  ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->Nq, dim)..., nelem)
  U = reshape((@view Q[:, _U, :]), ntuple(j->Nq, dim)..., nelem)
  V = reshape((@view Q[:, _V, :]), ntuple(j->Nq, dim)..., nelem)
  W = reshape((@view Q[:, _W, :]), ntuple(j->Nq, dim)..., nelem)
  E = reshape((@view Q[:, _E, :]), ntuple(j->Nq, dim)..., nelem)
  writemesh(prefix, X...;
            fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
            realelems=G.topology.realelems)
end

