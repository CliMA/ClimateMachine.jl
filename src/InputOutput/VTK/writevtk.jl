using ..Mesh.Grids
using ..MPIStateArrays
using ..DGmethods
using ..TicToc

"""
    writevtk(prefix, Q::MPIStateArray, dg::DGModel [, fieldnames])

Write a vtk file for all the fields in the state array `Q` using geometry and
connectivity information from `dg.grid`. The filename will start with `prefix`
which may also contain a directory path. The names used for each of the fields
in the vtk file can be specified through the collection of strings `fieldnames`;
if not specified the fields names will be `"Q1"` through `"Qk"` where `k` is the
number of states in `Q`, i.e., `k = size(Q,2)`.

"""
function writevtk(prefix, Q::MPIStateArray, dg::DGModel, fieldnames = nothing)
    vgeo = dg.grid.vgeo
    host_array = Array ∈ typeof(Q).parameters
    (h_vgeo, h_Q) = host_array ? (vgeo, Q.data) : (Array(vgeo), Array(Q))
    writevtk_helper(prefix, h_vgeo, h_Q, dg.grid, fieldnames)
    return nothing
end

"""
    writevtk(prefix, Q::MPIStateArray, dg::DGModel, fieldnames,
             state_auxiliary::MPIStateArray, auxfieldnames)

Write a vtk file for all the fields in the state array `Q` and auxiliary state
`state_auxiliary` using geometry and connectivity information from `dg.grid`. The
filename will start with `prefix` which may also contain a directory path. The
names used for each of the fields in the vtk file can be specified through the
collection of strings `fieldnames` and `auxfieldnames`.

If `fieldnames === nothing` then the fields names will be `"Q1"` through `"Qk"`
where `k` is the number of states in `Q`, i.e., `k = size(Q,2)`.

If `auxfieldnames === nothing` then the fields names will be `"aux1"` through
`"auxk"` where `k` is the number of states in `state_auxiliary`, i.e., `k =
size(state_auxiliary,2)`.

"""
function writevtk(
    prefix,
    Q::MPIStateArray,
    dg::DGModel,
    fieldnames,
    state_auxiliary,
    auxfieldnames,
)
    vgeo = dg.grid.vgeo
    host_array = Array ∈ typeof(Q).parameters
    (h_vgeo, h_Q, h_aux) = host_array ? (vgeo, Q.data, state_auxiliary.data) :
        (Array(vgeo), Array(Q), Array(state_auxiliary))
    writevtk_helper(
        prefix,
        h_vgeo,
        h_Q,
        dg.grid,
        fieldnames,
        h_aux,
        auxfieldnames,
    )
    return nothing
end


"""
    writevtk_helper(prefix, vgeo::Array, Q::Array, grid, fieldnames)

Internal helper function for `writevtk`
"""
function writevtk_helper(
    prefix,
    vgeo::Array,
    Q::Array,
    grid,
    fieldnames,
    state_auxiliary = nothing,
    auxfieldnames = nothing,
)

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1

    nelem = size(Q)[end]
    Xid = (grid.x1id, grid.x2id, grid.x3id)
    X = ntuple(
        j -> reshape((@view vgeo[:, Xid[j], :]), ntuple(j -> Nq, dim)..., nelem),
        dim,
    )
    if fieldnames == nothing
        fields = ntuple(
            i -> ("Q$i", reshape((@view Q[:, i, :]), ntuple(j -> Nq, dim)..., nelem)),
            size(Q, 2),
        )
    else
        fields = ntuple(
            i -> (
                fieldnames[i],
                reshape((@view Q[:, i, :]), ntuple(j -> Nq, dim)..., nelem),
            ),
            size(Q, 2),
        )
    end
    if state_auxiliary !== nothing
        if auxfieldnames === nothing
            auxfields = ntuple(
                i -> (
                    "aux$i",
                    reshape(
                        (@view state_auxiliary[:, i, :]),
                        ntuple(j -> Nq, dim)...,
                        nelem,
                    ),
                ),
                size(state_auxiliary, 2),
            )
        else
            auxfields = ntuple(
                i -> (
                    auxfieldnames[i],
                    reshape(
                        (@view state_auxiliary[:, i, :]),
                        ntuple(j -> Nq, dim)...,
                        nelem,
                    ),
                ),
                size(state_auxiliary, 2),
            )
        end
        fields = (fields..., auxfields...)
    end
    writemesh(
        prefix,
        X...;
        fields = fields,
        realelems = grid.topology.realelems,
    )
end

"""
    writegrid(prefix, grid::DiscontinuousSpectralElementGrid)

Write a vtk file for the grid.  The filename will start with `prefix` which
may also contain a directory path.
"""
function writevtk(prefix, grid::DiscontinuousSpectralElementGrid)
    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1

    vgeo = grid.vgeo isa Array ? grid.vgeo : Array(grid.vgeo)

    nelem = size(vgeo)[end]
    Xid = (grid.x1id, grid.x2id, grid.x3id)
    X = ntuple(
        j -> reshape((@view vgeo[:, Xid[j], :]), ntuple(j -> Nq, dim)..., nelem),
        dim,
    )
    writemesh(prefix, X...; realelems = grid.topology.realelems)
end
