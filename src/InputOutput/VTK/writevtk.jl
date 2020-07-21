import KernelAbstractions: CPU

using ..Mesh.Grids
using ..Mesh.Elements: interpolationmatrix
using ..MPIStateArrays
using ..DGMethods
using ..TicToc

"""
    writevtk(prefix, Q::MPIStateArray, dg::DGModel [, fieldnames]; number_sample_points)

Write a vtk file for all the fields in the state array `Q` using geometry and
connectivity information from `dg.grid`. The filename will start with `prefix`
which may also contain a directory path. The names used for each of the fields
in the vtk file can be specified through the collection of strings `fieldnames`;
if not specified the fields names will be `"Q1"` through `"Qk"` where `k` is the
number of states in `Q`, i.e., `k = size(Q,2)`.

If specified, fields are sampled on an equally spaced, tensor-product grid of
points with 'number_sample_points' in each direction.
"""
function writevtk(
    prefix,
    Q::MPIStateArray,
    dg::DGModel,
    fieldnames = nothing;
    number_sample_points = 0,
)
    vgeo = dg.grid.vgeo
    device = array_device(Q)
    (h_vgeo, h_Q) = device isa CPU ? (vgeo, Q.data) : (Array(vgeo), Array(Q))
    writevtk_helper(
        prefix,
        h_vgeo,
        h_Q,
        dg.grid,
        fieldnames;
        number_sample_points = 0,
    )
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

If specified, fields are sampled on an equally spaced, tensor-product grid of
points with 'number_sample_points' in each direction.
"""
function writevtk(
    prefix,
    Q::MPIStateArray,
    dg::DGModel,
    fieldnames,
    state_auxiliary,
    auxfieldnames;
    number_sample_points = 0,
)
    vgeo = dg.grid.vgeo
    device = array_device(Q)
    (h_vgeo, h_Q, h_aux) =
        device isa CPU ? (vgeo, Q.data, state_auxiliary.data) :
        (Array(vgeo), Array(Q), Array(state_auxiliary))
    writevtk_helper(
        prefix,
        h_vgeo,
        h_Q,
        dg.grid,
        fieldnames,
        h_aux,
        auxfieldnames;
        number_sample_points = number_sample_points,
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
    auxfieldnames = nothing;
    number_sample_points,
)
    @assert number_sample_points >= 0

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1

    nelem = size(Q)[end]

    Xid = (grid.x1id, grid.x2id, grid.x3id)
    X = ntuple(j -> (@view vgeo[:, Xid[j], :]), dim)
    fields = ntuple(j -> (@view Q[:, j, :]), size(Q, 2))
    auxfields = isnothing(state_auxiliary) ? () :
        (
        auxfields = ntuple(
            j -> (@view state_auxiliary[:, j, :]),
            size(state_auxiliary, 2),
        )
    )

    # Interpolate to an equally spaced grid if necessary
    if number_sample_points > 0
        FT = eltype(Q)
        ξsrc = referencepoints(grid)
        ξdst = range(FT(-1); length = number_sample_points, stop = 1)
        I1d = interpolationmatrix(ξsrc, ξdst)
        I = kron(ntuple(i -> I1d, dim)...)
        fields = ntuple(i -> I * fields[i], length(fields))
        auxfields = ntuple(i -> I * auxfields[i], length(auxfields))
        X = ntuple(i -> I * X[i], length(X))
        Nq = number_sample_points
    end

    X = ntuple(i -> reshape(X[i], ntuple(j -> Nq, dim)..., nelem), length(X))
    fields = ntuple(
        i -> reshape(fields[i], ntuple(j -> Nq, dim)..., nelem),
        length(fields),
    )
    auxfields = ntuple(
        i -> reshape(auxfields[i], ntuple(j -> Nq, dim)..., nelem),
        length(auxfields),
    )

    if fieldnames === nothing
        fields = ntuple(i -> ("Q$i", fields[i]), length(fields))
    else
        fields = ntuple(i -> (fieldnames[i], fields[i]), length(fields))
    end

    if auxfieldnames === nothing
        auxfields = ntuple(i -> ("aux$i", auxfields[i]), length(auxfields))
    else
        auxfields =
            ntuple(i -> (auxfieldnames[i], auxfields[i]), length(auxfields))
    end

    fields = (fields..., auxfields...)
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
