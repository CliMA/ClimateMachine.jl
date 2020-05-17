module Grids
using ..Topologies
import ..Metrics, ..Elements
import ..BrickMesh

using MPI
using LinearAlgebra
using KernelAbstractions

export DiscontinuousSpectralElementGrid, AbstractGrid
export dofs_per_element, arraytype, dimensionality, polynomialorder
export referencepoints, min_node_distance, get_z
export EveryDirection, HorizontalDirection, VerticalDirection

abstract type Direction end
struct EveryDirection <: Direction end
struct HorizontalDirection <: Direction end
struct VerticalDirection <: Direction end

abstract type AbstractGrid{
    FloatType,
    dim,
    polynomialorder,
    numberofDOFs,
    DeviceArray,
} end

dofs_per_element(::AbstractGrid{T, D, N, Np}) where {T, D, N, Np} = Np

polynomialorder(::AbstractGrid{T, dim, N}) where {T, dim, N} = N

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
    nabrtovmaprecv

    "An array of ranges in `vmapsend` to send to each neighbor"
    nabrtovmapsend

    "Array of real elements that do not have a ghost element as a neighbor"
    interiorelems

    "Array of real elements that have at least one ghost element as a neighbor"
    exteriorelems

    "Array indicating if a degree of freedom (real or ghost) is active"
    activedofs

    "1-D lvl weights on the device"
    ω::DAT1

    "1-D derivative operator on the device"
    D::DAT2

    "1-D indefinite integral operator on the device"
    Imat::DAT2

    function DiscontinuousSpectralElementGrid(
        topology::AbstractTopology{dim};
        FloatType,
        DeviceArray,
        polynomialorder,
        meshwarp::Function = (x...) -> identity(x),
    ) where {dim}

        N = polynomialorder
        (ξ, ω) = Elements.lglpoints(FloatType, N)
        Imat = indefinite_integral_interpolation_matrix(ξ, ω)
        D = Elements.spectralderivative(ξ)

        (vmap⁻, vmap⁺) = mappings(
            N,
            topology.elemtoelem,
            topology.elemtoface,
            topology.elemtoordr,
        )

        (vmaprecv, nabrtovmaprecv) = BrickMesh.commmapping(
            N,
            topology.ghostelems,
            topology.ghostfaces,
            topology.nabrtorecv,
        )
        (vmapsend, nabrtovmapsend) = BrickMesh.commmapping(
            N,
            topology.sendelems,
            topology.sendfaces,
            topology.nabrtosend,
        )

        (vgeo, sgeo) = computegeometry(topology, D, ξ, ω, meshwarp, vmap⁻)
        Np = (N + 1)^dim
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
        ω = DeviceArray(ω)
        D = DeviceArray(D)
        Imat = DeviceArray(Imat)

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
    ::DiscontinuousSpectralElementGrid{T, dim, N},
) where {T, dim, N}
    ξ, _ = Elements.lglpoints(T, N)
    ξ
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
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
        device = grid.vgeo isa Array ? CPU() : CUDA()
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
    get_z(grid, z_scale = 1)

Get the Gauss-Lobatto points along the Z-coordinate.

 - `grid`: DG grid
 - `z_scale`: multiplies `z-coordinate`
"""
function get_z(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    z_scale = 1,
) where {T, dim, N}
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

This function takes in a polynomial order `N` and parts of a topology (as
returned from `connectmesh`) and returns index mappings for the element surface
flux computation.  The returned `Tuple` contains:

 - `vmap⁻` an array of linear indices into the volume degrees of freedom where
   `vmap⁻[:,f,e]` are the degrees of freedom indices for face `f` of element
    `e`.

 - `vmap⁺` an array of linear indices into the volume degrees of freedom where
   `vmap⁺[:,f,e]` are the degrees of freedom indices for the face neighboring
   face `f` of element `e`.
"""
function mappings(N, elemtoelem, elemtoface, elemtoordr)
    nface, nelem = size(elemtoelem)

    d = div(nface, 2)
    Np, Nfp = (N + 1)^d, (N + 1)^(d - 1)

    p = reshape(1:Np, ntuple(j -> N + 1, d))
    fd(f) = div(f - 1, 2) + 1
    fe(f) = N * mod(f - 1, 2) + 1
    fmask = hcat((
        p[ntuple(j -> (j == fd(f)) ? (fe(f):fe(f)) : (:), d)...][:] for
        f in 1:nface
    )...)
    inds = LinearIndices(ntuple(j -> N + 1, d - 1))

    vmap⁻ = similar(elemtoelem, Nfp, nface, nelem)
    vmap⁺ = similar(elemtoelem, Nfp, nface, nelem)

    for e1 in 1:nelem, f1 in 1:nface
        e2 = elemtoelem[f1, e1]
        f2 = elemtoface[f1, e1]
        o2 = elemtoordr[f1, e1]

        vmap⁻[:, f1, e1] .= Np * (e1 - 1) .+ fmask[:, f1]

        if o2 == 1
            vmap⁺[:, f1, e1] .= Np * (e2 - 1) .+ fmask[:, f2]
        elseif d == 3 && o2 == 3
            n = 1
            @inbounds for j in 1:(N + 1), i in (N + 1):-1:1
                vmap⁺[n, f1, e1] = Np * (e2 - 1) + fmask[inds[i, j], f2]
                n += 1
            end
        else
            error("Orientation '$o2' with dim '$d' not supported yet")
        end
    end

    (vmap⁻, vmap⁺)
end
# }}}

# {{{ compute geometry
function computegeometry(
    topology::AbstractTopology{dim},
    D,
    ξ,
    ω,
    meshwarp,
    vmap⁻,
) where {dim}
    # Compute metric terms
    Nq = size(D, 1)
    FT = eltype(D)

    (nface, nelem) = size(topology.elemtoelem)

    # crd = creategrid(Val(dim), elemtocoord(topology), ξ)

    vgeo = zeros(FT, Nq^dim, _nvgeo, nelem)
    sgeo = zeros(FT, _nsgeo, Nq^(dim - 1), nface, nelem)

    (
        ξ1x1,
        ξ2x1,
        ξ3x1,
        ξ1x2,
        ξ2x2,
        ξ3x2,
        ξ1x3,
        ξ2x3,
        ξ3x3,
        MJ,
        MJI,
        MHJH,
        x1,
        x2,
        x3,
        JcV,
    ) = ntuple(j -> (@view vgeo[:, j, :]), _nvgeo)
    J = similar(x1)
    (n1, n2, n3, sMJ, vMJI) = ntuple(j -> (@view sgeo[j, :, :, :]), _nsgeo)
    sJ = similar(sMJ)

    X = ntuple(j -> (@view vgeo[:, _x1 + j - 1, :]), dim)
    Metrics.creategrid!(X..., topology.elemtocoord, ξ)

    @inbounds for j in 1:length(x1)
        (x1[j], x2[j], x3[j]) = meshwarp(x1[j], x2[j], x3[j])
    end

    # Compute the metric terms
    if dim == 1
        Metrics.computemetric!(x1, J, ξ1x1, sJ, n1, D)
    elseif dim == 2
        Metrics.computemetric!(x1, x2, J, ξ1x1, ξ2x1, ξ1x2, ξ2x2, sJ, n1, n2, D)
    elseif dim == 3
        Metrics.computemetric!(
            x1,
            x2,
            x3,
            J,
            ξ1x1,
            ξ2x1,
            ξ3x1,
            ξ1x2,
            ξ2x2,
            ξ3x2,
            ξ1x3,
            ξ2x3,
            ξ3x3,
            sJ,
            n1,
            n2,
            n3,
            D,
        )
    end

    M = kron(1, ntuple(j -> ω, dim)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    vMJI .= MJI[vmap⁻]

    MH = kron(ones(FT, Nq), ntuple(j -> ω, dim - 1)...)

    sM = dim > 1 ? kron(1, ntuple(j -> ω, dim - 1)...) : one(FT)
    sMJ .= sM .* sJ

    # Compute |r'(ξ3)| for vertical line integrals
    if dim == 2
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
            JcV,
            J,
            ξ1x1,
            ξ1x2,
            ξ1x3,
            ξ2x1,
            ξ2x2,
            ξ2x3,
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
