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
    TVTK,
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

    "1-D lgl nodes on the device (one for each dimension) for over-integration"
    ξ_m2::DAT1

    "1-D lgl weights on the device (one for each dimension) for over-integration"
    ω_m2::DAT1

    "1-D basis function matrix, on mesh 2 (for over-integration), on the device (one for each dimension)"
    B_m2::DAT2

    "transpose of 1-D basis function matrix, on mesh 2 (for over-integration), on the device (one for each dimension)"
    B_m2ᵀ::DAT2

    "1-D derivative operator on the device (one for each dimension)"
    D::DAT2

    "transpose of 1-D derivative operator on the device (one for each dimension)"
    Dᵀ::DAT2

    "1-D derivative operator, on mesh 2 (for over-integration), on the device (one for each dimension)"
    D_m2::DAT2

    "transpose of 1-D derivative operator, on mesh 2 (for over-integration), on the device (one for each dimension)"
    D_m2ᵀ::DAT2

    "1-D indefinite integral operator on the device (one for each dimension)"
    Imat::DAT2

    """
    tuple of (x1, x2, x3) to use for vtk output (Needed for the `N = 0` case) in
    other cases these match `vgeo` values
    """
    x_vtk::TVTK

    "Temporary Storage for FTP"
    ftp_storage::DAT3

    "Temporary Storage for FTP on m1"
    m1_storage::DAT3

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
        N_m2 = Int.(ceil.(N .* 1.5)) # polynomial order for overintegration

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
        ξω = ntuple(
            j ->
                N[j] == 0 ? Elements.glpoints(FloatType, N[j]) :
                Elements.lglpoints(FloatType, N[j]),
            dim,
        )
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
        # for over-integration grid
        ξω_m2 = ntuple(
            j ->
                N_m2[j] == 0 ? Elements.glpoints(FloatType, N_m2[j]) :
                Elements.lglpoints(FloatType, N_m2[j]),
            dim,
        )
        ξ_m2, ω_m2 = ntuple(j -> map(x -> x[j], ξω_m2), 2)

        Imat = ntuple(
            j -> indefinite_integral_interpolation_matrix(ξ[j], ω[j]),
            dim,
        )
        wb = Elements.baryweights.(ξ)
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)
        Dᵀ = ntuple(j -> Array(transpose(D[j])), dim)

        B_m2 =
            ntuple(j -> Elements.interpolationmatrix(ξ[j], ξ_m2[j], wb[j]), dim)
        B_m2ᵀ = ntuple(j -> Array(transpose(B_m2[j])), dim)
        D_m2 = ntuple(j -> B_m2[j] * D[j], dim)
        D_m2ᵀ = ntuple(j -> Array(transpose(D_m2[j])), dim)

        (vgeo, sgeo, x_vtk) =
            computegeometry(topology.elemtocoord, D, ξ, ω, meshwarp)

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
        ξ_m2 = DeviceArray.(ξ_m2)
        ω_m2 = DeviceArray.(ω_m2)
        B_m2 = DeviceArray.(B_m2)
        B_m2ᵀ = DeviceArray.(B_m2ᵀ)
        D = DeviceArray.(D)
        Dᵀ = DeviceArray.(Dᵀ)
        D_m2 = DeviceArray.(D_m2)
        D_m2ᵀ = DeviceArray.(D_m2ᵀ)
        Imat = DeviceArray.(Imat)

        ftp_storage = DeviceArray(Array{FloatType}(
            undef,
            Np,
            2,
            length(topology.realelems),
        ))
        m1_storage = DeviceArray(Array{FloatType}(
            undef,
            Np,
            3,
            length(topology.realelems),
        ))
        # FIXME: There has got to be a better way!
        DAT1 = typeof(ω)
        DAT2 = typeof(D)
        DAT3 = typeof(vgeo)
        DAT4 = typeof(sgeo)
        DAI1 = typeof(vmapsend)
        DAI2 = typeof(elemtobndy)
        DAI3 = typeof(vmap⁻)
        TOP = typeof(topology)
        TVTK = typeof(x_vtk)

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
            TVTK,
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
            ξ_m2,
            ω_m2,
            B_m2,
            B_m2ᵀ,
            D,
            Dᵀ,
            D_m2,
            D_m2ᵀ,
            Imat,
            x_vtk,
            ftp_storage,
            m1_storage,
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
    ξω = ntuple(
        j ->
            N[j] == 0 ? Elements.glpoints(FT, N[j]) :
            Elements.lglpoints(FT, N[j]),
        dim,
    )
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
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    direction::Direction = EveryDirection(),
) where {T, dim, N}
    topology = grid.topology
    nrealelem = length(topology.realelems)

    if nrealelem > 0
        Nq = N .+ 1
        Np = prod(Nq)
        device = grid.vgeo isa Array ? CPU() : CUDADevice()
        min_neighbor_distance = similar(grid.vgeo, Np, nrealelem)
        event = Event(device)
        event = kernel_min_neighbor_distance!(device, min(Np, 1024))(
            Val(N),
            Val(dim),
            direction,
            min_neighbor_distance,
            grid.vgeo,
            topology.realelems;
            ndrange = (Np * nrealelem),
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
    grid::DiscontinuousSpectralElementGrid{T, dim, N};
    z_scale = 1,
    rm_dupes = false,
) where {T, dim, N}
    # Assumes same polynomial orders in all horizontal directions
    @assert dim < 3 || N[1] == N[2]
    Nhoriz = N[1]
    Nvert = N[end]
    Nph = (Nhoriz + 1)^2
    Np = Nph * (Nvert + 1)
    if rm_dupes
        ijk_range = (1:Nph:(Np - Nph))
        vgeo = Array(grid.vgeo)
        z = reshape(vgeo[ijk_range, _x3, :], :)
        z = [z..., vgeo[Np, _x3, end]]
        return z * z_scale
    else
        ijk_range = (1:Nph:Np)
        z = Array(reshape(grid.vgeo[ijk_range, _x3, :], :))
        return z * z_scale
    end
    return reshape(grid.vgeo[(1:Nph:Np), _x3, :], :) * z_scale
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

# Compute geometry
function computegeometry_fvm(elemtocoord, D, ξ, ω, meshwarp)
    FT = eltype(D[1])
    dim = length(D)
    nface = 2dim
    nelem = size(elemtocoord, 3)

    Nq = ntuple(j -> size(D[j], 1), dim)
    Np = prod(Nq)
    Nfp = div.(Np, Nq)

    Nq_N1 = max.(Nq, 2)
    Np_N1 = prod(Nq_N1)
    Nfp_N1 = div.(Np_N1, Nq_N1)

    # First we compute the geometry with all the N = 0 dimension set to N = 1
    # so that we can later compute the geometry for the case N = 0, as the
    # average of two N = 1 cases
    ξ1, ω1 = Elements.lglpoints(FT, 1)
    D1 = Elements.spectralderivative(ξ1)
    D_N1 = ntuple(j -> Nq[j] == 1 ? D1 : D[j], dim)
    ξ_N1 = ntuple(j -> Nq[j] == 1 ? ξ1 : ξ[j], dim)
    ω_N1 = ntuple(j -> Nq[j] == 1 ? ω1 : ω[j], dim)
    (vgeo_N1, sgeo_N1, x_vtk) =
        computegeometry(elemtocoord, D_N1, ξ_N1, ω_N1, meshwarp)

    # Sort out the vgeo terms
    @views begin
        vgeo_N1_flds =
            ntuple(fld -> reshape(vgeo_N1[:, fld, :], Nq_N1..., nelem), _nvgeo)
    end

    # Allocate the storage for N = 0 volume metrics
    vgeo = zeros(FT, Np, _nvgeo, nelem)

    # Counter to make sure we got all the vgeo terms
    num_vgeo_handled = 0

    X = ntuple(j -> (@view vgeo[:, _x1 + j - 1, :]), dim)
    Metrics.creategrid!(X..., elemtocoord, ξ...)
    x1 = @view vgeo[:, _x1, :]
    x2 = @view vgeo[:, _x2, :]
    x3 = @view vgeo[:, _x3, :]
    @inbounds for j in 1:length(x1)
        (x1[j], x2[j], x3[j]) = meshwarp(x1[j], x2[j], x3[j])
    end

    num_vgeo_handled += 3

    @views begin
        # _M should be a sum
        vgeo[:, _M, :][:] .= sum(vgeo_N1_flds[_M], dims = findall(Nq .== 1))[:]
        num_vgeo_handled += 1

        # need to recompute _MI
        vgeo[:, _MI, :] = 1 ./ vgeo[:, _M, :]
        num_vgeo_handled += 1

        # coordinates should just be averages
        avg_den = 2 .^ sum(Nq .== 1)
        for fld in (_JcV,)
            vgeo[:, fld, :] =
                sum(vgeo_N1_flds[fld], dims = findall(Nq .== 1))[:] / avg_den
            num_vgeo_handled += 1
        end

        # For the metrics it is J * ξixk we approximate so multiply and divide the
        # mass matrix (which has the Jacobian determinant and the proper averaging
        # due to the quadrature weights)
        M_N1 = vgeo_N1_flds[_M]
        MI = vgeo[:, _MI, :]
        for fld in
            (_ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3)
            vgeo[:, fld, :] =
                sum(M_N1 .* vgeo_N1_flds[fld], dims = findall(Nq .== 1))[:] .*
                MI[:]
            num_vgeo_handled += 1
        end

        # compute MH and JvC
        horizontal_metrics(vgeo, Nq, ω)
        num_vgeo_handled += 1

        # Make sure we handled all the vgeo terms
        @assert _nvgeo == num_vgeo_handled
    end

    # Sort out the sgeo terms
    @views begin
        sgeo = zeros(FT, _nsgeo, maximum(Nfp), nface, nelem)

        # for the volume inverse mass matrix
        MI = vgeo[:, _MI, :]
        p = reshape(1:Np, Nq)
        if dim == 1
            fmask = (p[1:1], p[Nq[1]:Nq[1]])
        elseif dim == 2
            fmask = (p[1, :][:], p[Nq[1], :][:], p[:, 1][:], p[:, Nq[2]][:])
        elseif dim == 3
            fmask = (
                p[1, :, :][:],
                p[Nq[1], :, :][:],
                p[:, 1, :][:],
                p[:, Nq[2], :][:],
                p[:, :, 1][:],
                p[:, :, Nq[3]][:],
            )
        end
        for d in 1:dim
            for f in (2d - 1):(2d)
                # number of points matches means that we keep all the data
                # (N = 0 is not on the face)
                if Nfp[d] == Nfp_N1[d]
                    sgeo[:, 1:Nfp[d], f, :] .= sgeo_N1[:, 1:Nfp[d], f, :]

                    # Volume inverse mass will be wrong so reset it
                    sgeo[_vMI, 1:Nfp[d], f, :] .= MI[fmask[f], :]
                else
                    # Counter to make sure we got all the sgeo terms
                    num_sgeo_handled = 0

                    # sum to get sM
                    Nq_f = (Nq[1:(d - 1)]..., Nq[(d + 1):dim]...)
                    Nq_f_N1 = (Nq_N1[1:(d - 1)]..., Nq_N1[(d + 1):dim]...)
                    sM_N1 = reshape(
                        sgeo_N1[_sM, 1:Nfp_N1[d], f, :],
                        Nq_f_N1...,
                        nelem,
                    )
                    sgeo[_sM, 1:Nfp[d], f, :][:] .=
                        sum(sM_N1, dims = findall(Nq_f .== 1))[:]
                    num_sgeo_handled += 1

                    # Normals (like metrics in the volume) need to be computed
                    # scaled by surface Jacobian which we can do with the
                    # surface mass matrices
                    sM = sgeo[_sM, 1:Nfp[d], f, :]
                    for fld in (_n1, _n2, _n3)
                        fld_N1 = reshape(
                            sgeo_N1[fld, 1:Nfp_N1[d], f, :],
                            Nq_f_N1...,
                            nelem,
                        )
                        sgeo[fld, 1:Nfp[d], f, :][:] .=
                            sum(sM_N1 .* fld_N1, dims = findall(Nq_f .== 1))[:] ./
                            sM[:]
                        num_sgeo_handled += 1
                    end

                    # set the volume inverse mass matrix
                    sgeo[_vMI, 1:Nfp[d], f, :] .= MI[fmask[f], :]
                    num_sgeo_handled += 1

                    # Make sure we handled all the vgeo terms
                    @assert _nsgeo == num_sgeo_handled
                end
            end
        end
    end

    (vgeo, sgeo, x_vtk)
end

function computegeometry(elemtocoord, D, ξ, ω, meshwarp)
    dim = length(D)
    nface = 2dim
    nelem = size(elemtocoord, 3)

    Nq = ntuple(j -> size(D[j], 1), dim)

    # Compute metric terms for FVM
    if any(Nq .== 1)
        return computegeometry_fvm(elemtocoord, D, ξ, ω, meshwarp)
    end

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
        Metrics.computemetric!(x1, J, JcV, ξ1x1, sJ, n1, D...)
        fmask = (p[1:1], p[Nq[1]:Nq[1]])
    elseif dim == 2
        Metrics.computemetric!(
            #! format: off
            x1, x2,
            J, JcV,
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
            J, JcV,
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

    # compute MH and JvC
    horizontal_metrics(vgeo, Nq, ω)

    # This is mainly done to support FVM plotting when N=0 (since we need cell
    # edge values)
    x_vtk = (vgeo[:, _x1, :], vgeo[:, _x2, :], vgeo[:, _x3, :])

    (vgeo, sgeo, x_vtk)
end

function horizontal_metrics(vgeo, Nq, ω)
    dim = length(Nq)

    MH = dim == 1 ? 1 : kron(ones(1, Nq[dim]), reverse(ω[1:(dim - 1)])...)[:]
    M = kron(1, reverse(ω)...)[:]

    (
        #! format: off
        ξ1x1, ξ2x1, ξ3x1, ξ1x2, ξ2x2, ξ3x2, ξ1x3, ξ2x3, ξ3x3,
        MJ, MJI, MHJH,
        x1, x2, x3,
        JcV,
       #! format: on
    ) = ntuple(j -> (@view vgeo[:, j, :]), _nvgeo)
    J = MJ ./ M[:]

    # Compute |r'(ξ3)| for vertical line integrals
    if dim == 1
        MHJH .= 1
    elseif dim == 2
        map!(MHJH, J, ξ2x1, ξ2x2) do J, ξ2x1, ξ2x2
            hypot(J * ξ2x1, J * ξ2x2)
        end
        MHJH .= MH .* MHJH

    elseif dim == 3
        map!(MHJH, J, ξ3x1, ξ3x2, ξ3x3) do J, ξ3x1, ξ3x2, ξ3x3
            hypot(J * ξ3x1, J * ξ3x2, J * ξ3x3)
        end
        MHJH .= MH .* MHJH
    else
        error("dim $dim not implemented")
    end
end

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
    I∫[1, :] .= Nq == 1 ? ω[1] : 0

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
const _JcV = Grids._JcV

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
        Nq = N .+ 1
        Np = prod(Nq)

        if direction isa EveryDirection
            mininξ = (true, true, true)
        elseif direction isa HorizontalDirection
            mininξ = (true, dim == 2 ? false : true, false)
        elseif direction isa VerticalDirection
            mininξ = (false, dim == 2 ? true : false, dim == 2 ? false : true)
        end

        @inbounds begin
            # 2D Nq = (nh, nv)
            # 3D Nq = (nh, nh, nv)
            Nq1 = Nq[1]
            Nq2 = Nq[2]
            Nqk = dim == 2 ? 1 : Nq[end]
            mininξ1 = mininξ[1]
            mininξ2 = mininξ[2]
            mininξ3 = mininξ[3]
        end
    end


    I = @index(Global, Linear)
    # local element id
    e = (I - 1) ÷ Np + 1
    # local quadrature id
    ijk = (I - 1) % Np + 1

    # local i, j, k quadrature id
    i = (ijk - 1) % Nq1 + 1
    j = (ijk - 1) ÷ Nq1 % Nq2 + 1
    k = (ijk - 1) ÷ (Nq1 * Nq2) % Nqk + 1

    md = typemax(FT)

    x = SVector(vgeo[ijk, _x1, e], vgeo[ijk, _x2, e], vgeo[ijk, _x3, e])

    # first horizontal distance
    if mininξ1
        @unroll for î in (i - 1, i + 1)
            if 1 ≤ î ≤ Nq1
                îjk = î + Nq1 * (j - 1) + Nq1 * Nq2 * (k - 1)
                x̂ = SVector(
                    vgeo[îjk, _x1, e],
                    vgeo[îjk, _x2, e],
                    vgeo[îjk, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    # second horizontal distance or vertical distance (dim=2)
    if mininξ2
        # FV Vercial direction, use 2vgeo[ijk, _JcV, e]
        if dim == 2 && Nq2 == 1
            md = min(md, 2vgeo[ijk, _JcV, e])
        else
            @unroll for ĵ in (j - 1, j + 1)
                if 1 ≤ ĵ ≤ Nq2
                    iĵk = i + Nq1 * (ĵ - 1) + Nq1 * Nq2 * (k - 1)
                    x̂ = SVector(
                        vgeo[iĵk, _x1, e],
                        vgeo[iĵk, _x2, e],
                        vgeo[iĵk, _x3, e],
                    )
                    md = min(md, norm(x - x̂))
                end
            end
        end
    end

    # vertical distance (dim=3)
    if mininξ3
        # FV Vercial direction, use 2vgeo[ijk, _JcV, e]
        if dim == 3 && Nqk == 1
            md = min(md, 2vgeo[ijk, _JcV, e])
        else
            @unroll for k̂ in (k - 1, k + 1)
                if 1 ≤ k̂ ≤ Nqk
                    ijk̂ = i + Nq1 * (j - 1) + Nq1 * Nq2 * (k̂ - 1)
                    x̂ = SVector(
                        vgeo[ijk̂, _x1, e],
                        vgeo[ijk̂, _x2, e],
                        vgeo[ijk̂, _x3, e],
                    )
                    md = min(md, norm(x - x̂))
                end
            end
        end
    end

    min_neighbor_distance[ijk, e] = md
end

end # module
