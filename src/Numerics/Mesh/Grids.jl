module Grids
using ..Topologies
import ..Metrics, ..Elements
import ..BrickMesh

using MPI
using LinearAlgebra
using KernelAbstractions

export DiscontinuousSpectralElementGrid, AbstractGrid
export dofs_per_element, arraytype, dimensionality, polynomialorders
export referencepoints, min_node_distance, get_z
export EveryDirection, HorizontalDirection, VerticalDirection, Direction

abstract type Direction end
struct EveryDirection <: Direction end
struct HorizontalDirection <: Direction end
struct VerticalDirection <: Direction end
Base.in(::T, ::S) where {T <: Direction, S <: Direction} = T == S

abstract type AbstractGrid{
    FloatType,
    dim,
    polynomialorder,
    numberofDOFs,
    DeviceArray,
} end

dofs_per_element(::AbstractGrid{T, D, N, Np}) where {T, D, N, Np} = Np

polynomialorders(::AbstractGrid{T, dim, N}) where {T, dim, N} = N

dimensionality(::AbstractGrid{T, dim}) where {T, dim} = dim

Base.eltype(::AbstractGrid{T}) where {T} = T

arraytype(::AbstractGrid{T, D, N, Np, DA}) where {T, D, N, Np, DA} = DA

"""
    referencepoints(::AbstractGrid)

Returns the points on the reference element.
"""
referencepoints(::AbstractGrid) = error("needs to be implemented")

"""
    min_node_distance(::AbstractGrid, direction::Direction=EveryDirection() )

Returns an approximation of the minimum node distance in physical space.
"""
function min_node_distance(
    ::AbstractGrid,
    direction::Direction = EveryDirection(),
)
    error("needs to be implemented")
end

# {{{
const _nvgeo = 16
const _ξ1x1,
_ξ2x1,
_ξ3x1,
_ξ1x2,
_ξ2x2,
_ξ3x2,
_ξ1x3,
_ξ2x3,
_ξ3x3,
_M,
_MI,
_MH,
_x1,
_x2,
_x3,
_JcV = 1:_nvgeo
const vgeoid = (
    ξ1x1id = _ξ1x1,
    ξ2x1id = _ξ2x1,
    ξ3x1id = _ξ3x1,
    ξ1x2id = _ξ1x2,
    ξ2x2id = _ξ2x2,
    ξ3x2id = _ξ3x2,
    ξ1x3id = _ξ1x3,
    ξ2x3id = _ξ2x3,
    ξ3x3id = _ξ3x3,
    Mid = _M,
    MIid = _MI,
    MHid = _MH,
    x1id = _x1,
    x2id = _x2,
    x3id = _x3,
    JcVid = _JcV,
)
# JcV is the vertical line integral Jacobian
# The MH terms are for integrating over a plane.
const _nsgeo = 5
const _n1, _n2, _n3, _sM, _vMI = 1:_nsgeo
const sgeoid = (n1id = _n1, n2id = _n2, n3id = _n3, sMid = _sM, vMIid = _vMI)
# }}}

"""
    DiscontinuousSpectralElementGrid(topology; FloatType, DeviceArray,
                                     polynomialorder,
                                     meshwarp = (x...)->identity(x))

Generate a discontinuous spectral element (tensor product,
Legendre-Gauss-Lobatto) grid/mesh from a `topology`, where the order of the
elements is given by `polynomialorder`. `DeviceArray` gives the array type used
to store the data (`CuArray` or `Array`), and the coordinate points will be of
`FloatType`.

The polynomial order can be different in each direction (specified as a
`NTuple`). If only a single integer is specified, then each dimension will use
the same order. If the topology dimension is 3 and the `polynomialorder` has
dimension 2, then the first value will be used for horizontal and the second for
the vertical.

The optional `meshwarp` function allows the coordinate points to be warped after
the mesh is created; the mesh degrees of freedom are orginally assigned using a
trilinear blend of the element corner locations.
"""
struct DiscontinuousSpectralElementGrid{
    T,
    dim,
    N,
    Np,
    DA,
    DAT1,
    DAT2,
    DAT3,
    DAT4,
    DAI1,
    DAI2,
    DAI3,
    TOP,
} <: AbstractGrid{T, dim, N, Np, DA}
    "mesh topology"
    topology::TOP

    "volume metric terms"
    vgeo::DAT3

    "surface metric terms"
    sgeo::DAT4

    "element to boundary condition map"
    elemtobndy::DAI2

    "volume DOF to element minus side map"
    vmap⁻::DAI3

    "volume DOF to element plus side map"
    vmap⁺::DAI3

    "list of DOFs that need to be received (in neighbors order)"
    vmaprecv::DAI1

    "list of DOFs that need to be sent (in neighbors order)"
    vmapsend::DAI1

    "An array of ranges in `vmaprecv` to receive from each neighbor"
    nabrtovmaprecv::Any

    "An array of ranges in `vmapsend` to send to each neighbor"
    nabrtovmapsend::Any

    "Array of real elements that do not have a ghost element as a neighbor"
    interiorelems::Any

    "Array of real elements that have at least one ghost element as a neighbor"
    exteriorelems::Any

    "Array indicating if a degree of freedom (real or ghost) is active"
    activedofs::Any

    "1-D lgl weights on the device (one for each dimension)"
    ω::DAT1

    "1-D derivative operator on the device (one for each dimension)"
    D::DAT2

    "1-D indefinite integral operator on the device (one for each dimension)"
    Imat::DAT2

    # Constructor for a tuple of polynomial orders
    function DiscontinuousSpectralElementGrid(
        topology::AbstractTopology{dim};
        polynomialorder,
        FloatType,
        DeviceArray,
        meshwarp::Function = (x...) -> identity(x),
    ) where {dim}

        if polynomialorder isa Integer
            polynomialorder = ntuple(j -> polynomialorder, dim)
        elseif polynomialorder isa NTuple{2} && dim == 3
            polynomialorder =
                (polynomialorder[1], polynomialorder[1], polynomialorder[2])
        end

        @assert dim == length(polynomialorder)

        N = polynomialorder

        (vmap⁻, vmap⁺) = mappings(
            N,
            topology.elemtoelem,
            topology.elemtoface,
            topology.elemtoordr,
        )

        (vmaprecv, nabrtovmaprecv) = commmapping(
            N,
            topology.ghostelems,
            topology.ghostfaces,
            topology.nabrtorecv,
        )
        (vmapsend, nabrtovmapsend) = commmapping(
            N,
            topology.sendelems,
            topology.sendfaces,
            topology.nabrtosend,
        )

        Np = prod(N .+ 1)

        # Create element operators for each polynomial order
        ξω = ntuple(j -> Elements.lglpoints(FloatType, N[j]), dim)
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)

        Imat = ntuple(
            j -> indefinite_integral_interpolation_matrix(ξ[j], ω[j]),
            dim,
        )
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)

        (vgeo, sgeo) = computegeometry(topology.elemtocoord, D, ξ, ω, meshwarp)

        @assert Np == size(vgeo, 1)

        activedofs = zeros(Bool, Np * length(topology.elems))
        activedofs[1:(Np * length(topology.realelems))] .= true
        activedofs[vmaprecv] .= true

        # Create arrays on the device
        vgeo = DeviceArray(vgeo)
        sgeo = DeviceArray(sgeo)
        elemtobndy = DeviceArray(topology.elemtobndy)
        vmap⁻ = DeviceArray(vmap⁻)
        vmap⁺ = DeviceArray(vmap⁺)
        vmapsend = DeviceArray(vmapsend)
        vmaprecv = DeviceArray(vmaprecv)
        activedofs = DeviceArray(activedofs)
        ω = DeviceArray.(ω)
        D = DeviceArray.(D)
        Imat = DeviceArray.(Imat)

        # FIXME: There has got to be a better way!
        DAT1 = typeof(ω)
        DAT2 = typeof(D)
        DAT3 = typeof(vgeo)
        DAT4 = typeof(sgeo)
        DAI1 = typeof(vmapsend)
        DAI2 = typeof(elemtobndy)
        DAI3 = typeof(vmap⁻)
        TOP = typeof(topology)

        new{
            FloatType,
            dim,
            N,
            Np,
            DeviceArray,
            DAT1,
            DAT2,
            DAT3,
            DAT4,
            DAI1,
            DAI2,
            DAI3,
            TOP,
        }(
            topology,
            vgeo,
            sgeo,
            elemtobndy,
            vmap⁻,
            vmap⁺,
            vmaprecv,
            vmapsend,
            nabrtovmaprecv,
            nabrtovmapsend,
            DeviceArray(topology.interiorelems),
            DeviceArray(topology.exteriorelems),
            activedofs,
            ω,
            D,
            Imat,
        )
    end
end

"""
    referencepoints(::DiscontinuousSpectralElementGrid)

Returns the 1D interpolation points used for the reference element.
"""
function referencepoints(
    ::DiscontinuousSpectralElementGrid{FT, dim, N},
) where {FT, dim, N}
    ξω = ntuple(j -> Elements.lglpoints(FT, N[j]), dim)
    ξ, _ = ntuple(j -> map(x -> x[j], ξω), 2)
    return ξ
end

"""
    min_node_distance(::DiscontinuousSpectralElementGrid,
                      direction::Direction=EveryDirection()))

Returns an approximation of the minimum node distance in physical space along
the reference coordinate directions.  The direction controls which reference
directions are considered.
"""
function min_node_distance(
    grid::DiscontinuousSpectralElementGrid{T, dim, Ns},
    direction::Direction = EveryDirection(),
) where {T, dim, Ns}
    topology = grid.topology
    nrealelem = length(topology.realelems)

    if nrealelem > 0
        # XXX: Needs updating for multiple polynomial orders
        # Currently only support single polynomial order
        @assert all(Ns[1] .== Ns)
        N = Ns[1]
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
        device = grid.vgeo isa Array ? CPU() : CUDADevice()
        min_neighbor_distance = similar(grid.vgeo, Nq^dim, nrealelem)
        event = Event(device)
        event = kernel_min_neighbor_distance!(device, min(Nq * Nq * Nqk, 1024))(
            Val(N),
            Val(dim),
            direction,
            min_neighbor_distance,
            grid.vgeo,
            topology.realelems;
            ndrange = (Nq * Nq * Nqk * nrealelem),
            dependencies = (event,),
        )
        wait(device, event)
        locmin = minimum(min_neighbor_distance)
    else
        locmin = typemax(T)
    end

    MPI.Allreduce(locmin, min, topology.mpicomm)
end

"""
    get_z(grid; z_scale = 1, rm_dupes = false)

Get the Gauss-Lobatto points along the Z-coordinate.

 - `grid`: DG grid
 - `z_scale`: multiplies `z-coordinate`
 - `rm_dupes`: removes duplicate Gauss-Lobatto points
"""
function get_z(
    grid::DiscontinuousSpectralElementGrid{T, dim, Ns};
    z_scale = 1,
    rm_dupes = false,
) where {T, dim, Ns}
    # XXX: Needs updating for multiple polynomial orders
    # Currently only support single polynomial order
    @assert all(Ns[1] .== Ns)
    N = Ns[1]
    if rm_dupes
        ijk_range = (1:((N + 1)^2):(((N + 1)^3) - (N + 1)^2))
        vgeo = Array(grid.vgeo)
        z = reshape(vgeo[ijk_range, _x3, :], :)
        z = [z..., vgeo[(N + 1)^3, _x3, end]]
        return z * z_scale
    else
        ijk_range = (1:((N + 1)^2):((N + 1)^3))
        z = Array(reshape(grid.vgeo[ijk_range, _x3, :], :))
        return z * z_scale
    end
    return reshape(grid.vgeo[(1:((N + 1)^2):((N + 1)^3)), _x3, :], :) * z_scale
end

function Base.getproperty(G::DiscontinuousSpectralElementGrid, s::Symbol)
    if s ∈ keys(vgeoid)
        vgeoid[s]
    elseif s ∈ keys(sgeoid)
        sgeoid[s]
    else
        getfield(G, s)
    end
end

function Base.propertynames(G::DiscontinuousSpectralElementGrid)
    (
        fieldnames(DiscontinuousSpectralElementGrid)...,
        keys(vgeoid)...,
        keys(sgeoid)...,
    )
end

# {{{ mappings
"""
    mappings(N, elemtoelem, elemtoface, elemtoordr)

This function takes in a tuple of polynomial orders `N` and parts of a topology
(as returned from `connectmesh`) and returns index mappings for the element
surface flux computation. The returned `Tuple` contains:

 - `vmap⁻` an array of linear indices into the volume degrees of freedom where
   `vmap⁻[:,f,e]` are the degrees of freedom indices for face `f` of element
    `e`.

 - `vmap⁺` an array of linear indices into the volume degrees of freedom where
   `vmap⁺[:,f,e]` are the degrees of freedom indices for the face neighboring
   face `f` of element `e`.
"""
function mappings(N, elemtoelem, elemtoface, elemtoordr)
    nfaces, nelem = size(elemtoelem)

    d = div(nfaces, 2)
    Nq = N .+ 1
    # number of points in the element
    Np = prod(Nq)

    # Compute the maximum number of points on a face
    Nfp = div.(Np, Nq)

    # linear index for each direction, e.g., (i, j, k) -> n
    p = reshape(1:Np, ntuple(j -> Nq[j], d))

    # fmask[f] -> returns an array of all degrees of freedom on face f
    fmask = if d == 1
        (
            p[1:1],    # Face 1
            p[Nq[1]:Nq[1]], # Face 2
        )
    elseif d == 2
        (
            p[1, :][:],     # Face 1
            p[Nq[1], :][:], # Face 2
            p[:, 1][:],     # Face 3
            p[:, Nq[2]][:], # Face 4
        )
    elseif d == 3
        (
            p[1, :, :][:],     # Face 1
            p[Nq[1], :, :][:], # Face 2
            p[:, 1, :][:],     # Face 3
            p[:, Nq[2], :][:], # Face 4
            p[:, :, 1][:],     # Face 5
            p[:, :, Nq[3]][:], # Face 6
        )
    else
        error("unknown dimensionality")
    end

    # Create a map from Cartesian face dof number to linear face dof numbering
    # inds[face][i, j] -> n
    inds = ntuple(
        f -> dropdims(
            LinearIndices(ntuple(j -> j == cld(f, 2) ? 1 : Nq[j], d));
            dims = cld(f, 2),
        ),
        nfaces,
    )

    # Use the largest possible storage
    vmap⁻ = fill!(similar(elemtoelem, maximum(Nfp), nfaces, nelem), 0)
    vmap⁺ = fill!(similar(elemtoelem, maximum(Nfp), nfaces, nelem), 0)

    for e1 in 1:nelem, f1 in 1:nfaces
        e2 = elemtoelem[f1, e1]
        f2 = elemtoface[f1, e1]
        o2 = elemtoordr[f1, e1]
        d1, d2 = cld(f1, 2), cld(f2, 2)

        # Check to make sure the dof grid is conforming
        @assert Nfp[d1] == Nfp[d2]

        # Always pull out minus side without any flips / rotations
        vmap⁻[1:Nfp[d1], f1, e1] .= Np * (e1 - 1) .+ fmask[f1][1:Nfp[d1]][:]

        # Orientation codes defined in BrickMesh.jl (arbitrary numbers in 3D)
        if o2 == 1 # Neighbor oriented same as minus
            vmap⁺[1:Nfp[d1], f1, e1] .= Np * (e2 - 1) .+ fmask[f2][1:Nfp[d1]][:]
        elseif d == 3 && o2 == 3 # Neighbor fliped in first index
            vmap⁺[1:Nfp[d1], f1, e1] =
                Np * (e2 - 1) .+ fmask[f2][inds[f2][end:-1:1, :]][:]
        else
            error("Orientation '$o2' with dim '$d' not supported yet")
        end
    end

    (vmap⁻, vmap⁺)
end
# }}}

"""
   commmapping(N, commelems, commfaces, nabrtocomm)

This function takes in a tuple of polynomial orders `N` and parts of a mesh (as
returned from `connectmesh` such as `sendelems`, `sendfaces`, and `nabrtosend`)
and returns index mappings for the element surface flux parallel communcation.
The returned `Tuple` contains:

 - `vmapC` an array of linear indices into the volume degrees of freedom to be
   communicated.

 - `nabrtovmapC` a range in `vmapC` to communicate with each neighbor.
"""
function commmapping(N, commelems, commfaces, nabrtocomm)
    nface, nelem = size(commfaces)

    @assert nelem == length(commelems)

    d = div(nface, 2)
    Nq = N .+ 1
    Np = prod(Nq)

    vmapC = similar(commelems, nelem * Np)
    nabrtovmapC = similar(nabrtocomm)

    i = 1
    e = 1
    for neighbor in 1:length(nabrtocomm)
        rbegin = i
        for ne in nabrtocomm[neighbor]
            ce = commelems[ne]

            # Whole element sending
            # for n = 1:Np
            #   vmapC[i] = (ce-1)*Np + n
            #   i += 1
            # end

            CI = CartesianIndices(ntuple(j -> 1:Nq[j], d))
            for (ci, li) in zip(CI, LinearIndices(CI))
                addpoint = false
                for j in 1:d
                    addpoint |=
                        (commfaces[2 * (j - 1) + 1, e] && ci[j] == 1) ||
                        (commfaces[2 * (j - 1) + 2, e] && ci[j] == Nq[j])
                end

                if addpoint
                    vmapC[i] = (ce - 1) * Np + li
                    i += 1
                end
            end

            e += 1
        end
        rend = i - 1

        nabrtovmapC[neighbor] = rbegin:rend
    end

    resize!(vmapC, i - 1)

    (vmapC, nabrtovmapC)
end

# {{{ compute geometry
function computegeometry(elemtocoord, D, ξ, ω, meshwarp)
    dim = length(D)
    nface = 2dim
    nelem = size(elemtocoord, 3)

    # Compute metric terms
    Nq = ntuple(j -> size(D[j], 1), dim)
    Np = prod(Nq)
    Nfp = div.(Np, Nq)

    FT = eltype(D[1])

    vgeo = zeros(FT, Np, _nvgeo, nelem)
    sgeo = zeros(FT, _nsgeo, maximum(Nfp), nface, nelem)

    (
        #! format: off
        ξ1x1, ξ2x1, ξ3x1, ξ1x2, ξ2x2, ξ3x2, ξ1x3, ξ2x3, ξ3x3,
        MJ, MJI, MHJH,
        x1, x2, x3,
        JcV,
       #! format: on
    ) = ntuple(j -> (@view vgeo[:, j, :]), _nvgeo)
    J = similar(x1)
    (n1, n2, n3, sMJ, vMJI) = ntuple(j -> (@view sgeo[j, :, :, :]), _nsgeo)
    sJ = similar(sMJ)

    X = ntuple(j -> (@view vgeo[:, _x1 + j - 1, :]), dim)
    Metrics.creategrid!(X..., elemtocoord, ξ...)

    @inbounds for j in 1:length(x1)
        (x1[j], x2[j], x3[j]) = meshwarp(x1[j], x2[j], x3[j])
    end

    # Compute the metric terms
    p = reshape(1:Np, Nq)
    if dim == 1
        Metrics.computemetric!(x1, J, ξ1x1, sJ, n1, D...)
        fmask = (p[1:1], p[Nq[1]:Nq[1]])
    elseif dim == 2
        Metrics.computemetric!(
            #! format: off
            x1, x2,
            J,
            ξ1x1, ξ2x1, ξ1x2, ξ2x2,
            sJ,
            n1, n2,
            D...,
            #! format: on
        )
        fmask = (p[1, :][:], p[Nq[1], :][:], p[:, 1][:], p[:, Nq[2]][:])
    elseif dim == 3
        Metrics.computemetric!(
            #! format: off
            x1, x2, x3,
            J,
            ξ1x1, ξ2x1, ξ3x1, ξ1x2, ξ2x2, ξ3x2, ξ1x3, ξ2x3, ξ3x3,
            sJ,
            n1, n2, n3,
            D...,
            #! format: on
        )
        fmask = (
            p[1, :, :][:],
            p[Nq[1], :, :][:],
            p[:, 1, :][:],
            p[:, Nq[2], :][:],
            p[:, :, 1][:],
            p[:, :, Nq[3]][:],
        )
    end

    # since `ξ1` is the fastest dimension and `ξdim` the slowest the tensor
    # product order is reversed
    M = kron(1, reverse(ω)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    for d in 1:dim
        for f in (2d - 1):(2d)
            vMJI[1:Nfp[d], f, :] .= MJI[fmask[f], :]
        end
    end

    MH =
        dim == 1 ? ones(FT, 1) :
        kron(ones(FT, Nq[dim]), reverse(ω[1:(dim - 1)])...)

    sM = fill!(similar(sJ, maximum(Nfp), nface), NaN)
    for d in 1:dim
        for f in (2d - 1):(2d)
            ωf = ntuple(j -> ω[mod1(d + j, dim)], dim - 1)
            # Because of the `mod1` this face is already flipped
            if !(dim == 3 && d == 2)
                ωf = reverse(ωf)
            end
            sM[1:Nfp[d], f] .= dim > 1 ? kron(1, ωf...) : one(FT)
        end
    end
    sMJ .= sM .* sJ

    # Compute |r'(ξ3)| for vertical line integrals
    if dim == 1
        MHJH .= 1
        JcV .= J
    elseif dim == 2
        map!(JcV, J, ξ1x1, ξ1x2) do J, ξ1x1, ξ1x2
            x1ξ1 = J * ξ1x2
            x2ξ2 = J * ξ1x1
            hypot(x1ξ1, x2ξ2)
        end
        map!(MHJH, J, ξ2x1, ξ2x2) do J, ξ2x1, ξ2x2
            hypot(J * ξ2x1, J * ξ2x2)
        end
        MHJH .= MH .* MHJH

    elseif dim == 3
        map!(
            #! format: off
            JcV, J,
            ξ1x1, ξ1x2, ξ1x3, ξ2x1, ξ2x2, ξ2x3,
            #! format: on
        ) do J, ξ1x1, ξ1x2, ξ1x3, ξ2x1, ξ2x2, ξ2x3
            x1ξ3 = J * (ξ1x2 * ξ2x3 - ξ2x2 * ξ1x3)
            x2ξ3 = J * (ξ1x3 * ξ2x1 - ξ2x3 * ξ1x1)
            x3ξ3 = J * (ξ1x1 * ξ2x2 - ξ2x1 * ξ1x2)
            hypot(x1ξ3, x2ξ3, x3ξ3)
        end
        map!(MHJH, J, ξ3x1, ξ3x2, ξ3x3) do J, ξ3x1, ξ3x2, ξ3x3
            hypot(J * ξ3x1, J * ξ3x2, J * ξ3x3)
        end
        MHJH .= MH .* MHJH
    else
        error("dim $dim not implemented")
    end
    (vgeo, sgeo)
end
# }}}

# {{{ indefinite integral matrix
"""
    indefinite_integral_interpolation_matrix(r, ω)

Given a set of integration points `r` and integration weights `ω` this computes
a matrix that will compute the indefinite integral of the (interpolant) of a
function and evaluate the indefinite integral at the points `r`.

Namely, let
```math
    q(ξ) = ∫_{ξ_{0}}^{ξ} f(ξ') dξ'
```
then we have that
```
I∫ * f.(r) = q.(r)
```
where `I∫` is the integration and interpolation matrix defined by this function.

!!! note

    The integration is done using the provided quadrature weight, so if these
    cannot integrate `f(ξ)` exactly, `f` is first interpolated and then
    integrated using quadrature. Namely, we have that:
    ```math
        q(ξ) = ∫_{ξ_{0}}^{ξ} I(f(ξ')) dξ'
    ```
    where `I` is the interpolation operator.

"""
function indefinite_integral_interpolation_matrix(r, ω)
    Nq = length(r)

    I∫ = similar(r, Nq, Nq)
    # first value is zero
    I∫[1, :] .= 0

    # barycentric weights for interpolation
    wbary = Elements.baryweights(r)

    # Compute the interpolant of the indefinite integral
    for n in 2:Nq
        # grid from first dof to current point
        rdst = (1 .- r) / 2 * r[1] + (1 .+ r) / 2 * r[n]
        # interpolation matrix
        In = Elements.interpolationmatrix(r, rdst, wbary)
        # scaling from LGL to current of the interval
        Δ = (r[n] - r[1]) / 2
        # row of the matrix we have computed
        I∫[n, :] .= (Δ * ω' * In)[:]
    end
    I∫
end
# }}}

using KernelAbstractions.Extras: @unroll

using StaticArrays

const _x1 = Grids._x1
const _x2 = Grids._x2
const _x3 = Grids._x3

@doc """
    kernel_min_neighbor_distance!(::Val{N}, ::Val{dim}, direction,
                             min_neighbor_distance, vgeo, topology.realelems)

Computational kernel: Computes the minimum physical distance between node
neighbors within an element.

The `direction` in the reference element controls which nodes are considered
neighbors.
""" kernel_min_neighbor_distance!
@kernel function kernel_min_neighbor_distance!(
    ::Val{N},
    ::Val{dim},
    direction,
    min_neighbor_distance,
    vgeo,
    elems,
) where {N, dim}

    @uniform begin
        FT = eltype(min_neighbor_distance)
        # XXX: Needs updating for multiple polynomial orders
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
        Np = Nq * Nq * Nqk

        if direction isa EveryDirection
            mininξ = (true, true, true)
        elseif direction isa HorizontalDirection
            mininξ = (true, dim == 2 ? false : true, false)
        elseif direction isa VerticalDirection
            mininξ = (false, dim == 2 ? true : false, dim == 2 ? false : true)
        end
    end

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    ijk = (I - 1) % Np + 1

    i = (ijk - 1) % Nq + 1
    j = (ijk - 1) ÷ Nq % Nq + 1
    k = (ijk - 1) ÷ Nq^2 % Nqk + 1

    md = typemax(FT)

    x = SVector(vgeo[ijk, _x1, e], vgeo[ijk, _x2, e], vgeo[ijk, _x3, e])

    if mininξ[1]
        @unroll for î in (i - 1, i + 1)
            if 1 ≤ î ≤ Nq
                îjk = î + Nq * (j - 1) + Nq * Nq * (k - 1)
                x̂ = SVector(
                    vgeo[îjk, _x1, e],
                    vgeo[îjk, _x2, e],
                    vgeo[îjk, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    if mininξ[2]
        @unroll for ĵ in (j - 1, j + 1)
            if 1 ≤ ĵ ≤ Nq
                iĵk = i + Nq * (ĵ - 1) + Nq * Nq * (k - 1)
                x̂ = SVector(
                    vgeo[iĵk, _x1, e],
                    vgeo[iĵk, _x2, e],
                    vgeo[iĵk, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    if mininξ[3]
        @unroll for k̂ in (k - 1, k + 1)
            if 1 ≤ k̂ ≤ Nqk
                ijk̂ = i + Nq * (j - 1) + Nq * Nq * (k̂ - 1)
                x̂ = SVector(
                    vgeo[ijk̂, _x1, e],
                    vgeo[ijk̂, _x2, e],
                    vgeo[ijk̂, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    min_neighbor_distance[ijk, e] = md
end

end # module
