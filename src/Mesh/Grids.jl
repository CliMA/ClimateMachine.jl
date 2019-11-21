module Grids
using ..Topologies
import ..Metrics, ..Elements
import ..BrickMesh

using LinearAlgebra

export DiscontinuousSpectralElementGrid, AbstractGrid
export dofs_per_element, arraytype, dimensionality, polynomialorder
export referencepoints

abstract type AbstractGrid{FloatType, dim, polynomialorder, numberofDOFs,
                         DeviceArray} end

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

# {{{
const _nvgeo = 15
const _ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3, _M, _MI,
       _x1, _x2, _x3, _JcV = 1:_nvgeo
const vgeoid = (ξ1x1id = _ξ1x1, ξ2x1id = _ξ2x1, ξ3x1id = _ξ3x1,
                ξ1x2id = _ξ1x2, ξ2x2id = _ξ2x2, ξ3x2id = _ξ3x2,
                ξ1x3id = _ξ1x3, ξ2x3id = _ξ2x3, ξ3x3id = _ξ3x3,
                Mid  = _M , MIid = _MI,
                x1id  = _x1 , x2id  = _x2 , x3id  = _x3,
                JcVid = _JcV)
# JcV is the vertical line integral Jacobian

const _nsgeo = 5
const _n1, _n2, _n3, _sM, _vMI = 1:_nsgeo
const sgeoid = (n1id = _n1, n2id = _n2, n3id = _n3, sMid = _sM,
                vMIid = _vMI)
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
struct DiscontinuousSpectralElementGrid{T, dim, N, Np, DA,
                                        DAT1, DAT2, DAT3, DAT4,
                                        DAI1, DAI2, DAI3,
                                        TOP
                                       } <: AbstractGrid{T, dim, N, Np, DA }
  "mesh topology"
  topology::TOP

  "volume metric terms"
  vgeo::DAT3

  "surface metric terms"
  sgeo::DAT4

  "element to boundary condition map"
  elemtobndy::DAI2

  "volume DOF to element minus side map"
  vmapM::DAI3

  "volume DOF to element plus side map"
  vmapP::DAI3

  "list of DOFs that need to be received (in neighbors order)"
  vmaprecv::DAI1

  "list of DOFs that need to be sent (in neighbors order)"
  vmapsend::DAI1

  "An array of ranges in `vmaprecv` to receive from each neighbor"
  nabrtovmaprecv

  "An array of ranges in `vmapsend` to send to each neighbor"
  nabrtovmapsend

  "1-D lvl weights on the device"
  ω::DAT1

  "1-D derivative operator on the device"
  D::DAT2

  "1-D indefinite integral operator on the device"
  Imat::DAT2

  function DiscontinuousSpectralElementGrid(topology::AbstractTopology{dim};
                                            FloatType = nothing,
                                            DeviceArray = nothing,
                                            polynomialorder = nothing,
                                            meshwarp::Function =
                                            (x...)->identity(x)) where dim
    @assert FloatType != nothing
    @assert DeviceArray != nothing
    @assert polynomialorder != nothing

    N = polynomialorder
    (ξ, ω) = Elements.lglpoints(FloatType, N)
    Imat = indefinite_integral_interpolation_matrix(ξ, ω)
    D = Elements.spectralderivative(ξ)

    (vmapM, vmapP) = mappings(N, topology.elemtoelem, topology.elemtoface,
                              topology.elemtoordr)

    (vmaprecv, nabrtovmaprecv) =
      BrickMesh.commmapping(N, topology.ghostelems, topology.ghostfaces, topology.nabrtorecv)
    (vmapsend, nabrtovmapsend) =
      BrickMesh.commmapping(N, topology.sendelems, topology.sendfaces, topology.nabrtosend)

    (vgeo, sgeo) = computegeometry(topology, D, ξ, ω, meshwarp, vmapM)
    Np = (N+1)^dim
    @assert Np == size(vgeo, 1)

     # Create arrays on the device
     vgeo = DeviceArray(vgeo)
     sgeo = DeviceArray(sgeo)
     elemtobndy = DeviceArray(topology.elemtobndy)
     vmapM = DeviceArray(vmapM)
     vmapP = DeviceArray(vmapP)
     vmapsend = DeviceArray(vmapsend)
     vmaprecv = DeviceArray(vmaprecv)
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
     DAI3 = typeof(vmapM)
     TOP = typeof(topology)

    new{FloatType, dim, N, Np, DeviceArray, DAT1, DAT2, DAT3, DAT4, DAI1, DAI2,
        DAI3, TOP}(topology, vgeo, sgeo, elemtobndy, vmapM, vmapP, vmaprecv, vmapsend,
                   nabrtovmaprecv, nabrtovmapsend, ω, D, Imat)
  end
end

"""
    referencepoints(::DiscontinuousSpectralElementGrid)

Returns the 1D interpolation points used for the reference element.
"""
function referencepoints(::DiscontinuousSpectralElementGrid{T, dim, N}
                        ) where {T, dim, N}
  ξ, _ = Elements.lglpoints(T, N)
  ξ
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
  (fieldnames(DiscontinuousSpectralElementGrid)...,
   keys(vgeoid)..., keys(sgeoid)...)
end

# {{{ mappings
"""
    mappings(N, elemtoelem, elemtoface, elemtoordr)

This function takes in a polynomial order `N` and parts of a topology (as
returned from `connectmesh`) and returns index mappings for the element surface
flux computation.  The returned `Tuple` contains:

 - `vmapM` an array of linear indices into the volume degrees of freedom where
   `vmapM[:,f,e]` are the degrees of freedom indices for face `f` of element
    `e`.

 - `vmapP` an array of linear indices into the volume degrees of freedom where
   `vmapP[:,f,e]` are the degrees of freedom indices for the face neighboring
   face `f` of element `e`.
"""
function mappings(N, elemtoelem, elemtoface, elemtoordr)
  nface, nelem = size(elemtoelem)

  d = div(nface, 2)
  Np, Nfp = (N+1)^d, (N+1)^(d-1)

  p = reshape(1:Np, ntuple(j->N+1, d))
  fd(f) =   div(f-1,2)+1
  fe(f) = N*mod(f-1,2)+1
  fmask = hcat((p[ntuple(j->(j==fd(f)) ? (fe(f):fe(f)) : (:), d)...][:]
                for f=1:nface)...)
  inds = LinearIndices(ntuple(j->N+1, d-1))

  vmapM = similar(elemtoelem, Nfp, nface, nelem)
  vmapP = similar(elemtoelem, Nfp, nface, nelem)

  for e1 = 1:nelem, f1 = 1:nface
    e2 = elemtoelem[f1,e1]
    f2 = elemtoface[f1,e1]
    o2 = elemtoordr[f1,e1]

    vmapM[:,f1,e1] .= Np*(e1-1) .+ fmask[:,f1]

    if o2 == 1
      vmapP[:,f1,e1] .= Np*(e2-1) .+ fmask[:,f2]
    elseif d == 3 && o2 == 3
      n = 1
      @inbounds for j = 1:N+1, i = N+1:-1:1
        vmapP[n,f1,e1] = Np*(e2-1) + fmask[inds[i,j],f2]
        n+=1
      end
    else
      error("Orientation '$o2' with dim '$d' not supported yet")
    end
  end

  (vmapM, vmapP)
end
# }}}

# {{{ compute geometry
function computegeometry(topology::AbstractTopology{dim}, D, ξ, ω, meshwarp,
                         vmapM) where {dim}
  # Compute metric terms
  Nq = size(D, 1)
  FT = eltype(D)

  (nface, nelem) = size(topology.elemtoelem)

  # crd = creategrid(Val(dim), elemtocoord(topology), ξ)

  vgeo = zeros(FT, Nq^dim, _nvgeo, nelem)
  sgeo = zeros(FT, _nsgeo, Nq^(dim-1), nface, nelem)

  (ξ1x1, ξ2x1, ξ3x1, ξ1x2, ξ2x2, ξ3x2, ξ1x3, ξ2x3, ξ3x3, MJ, MJI, x1, x2, x3,
   JcV) = ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x1)
  (n1, n2, n3, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x1+j-1, :]), dim)
  Metrics.creategrid!(X..., topology.elemtocoord, ξ)

  @inbounds for j = 1:length(x1)
    (x1[j], x2[j], x3[j]) = meshwarp(x1[j], x2[j], x3[j])
  end

  # Compute the metric terms
  if dim == 1
    Metrics.computemetric!(x1, J, ξ1x1, sJ, n1, D)
  elseif dim == 2
    Metrics.computemetric!(x1, x2, J, ξ1x1, ξ2x1, ξ1x2, ξ2x2, sJ, n1, n2, D)
  elseif dim == 3
    Metrics.computemetric!(x1, x2, x3, J, ξ1x1, ξ2x1, ξ3x1, ξ1x2, ξ2x2, ξ3x2,
                           ξ1x3, ξ2x3, ξ3x3, sJ, n1, n2, n3, D)
  end

  M = kron(1, ntuple(j->ω, dim)...)
  MJ .= M .* J
  MJI .= 1 ./ MJ
  vMJI .= MJI[vmapM]

  sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(FT)
  sMJ .= sM .* sJ

  # Compute |r'(ξ3)| for vertical line integrals
  if dim == 2
    map!(JcV, J, ξ1x1, ξ1x2) do J, ξ1x1, ξ1x2
      x1ξ1 = J * ξ1x2
      x2ξ2 = J * ξ1x1
      hypot(x1ξ1, x2ξ2)
    end
  elseif dim == 3
    map!(JcV, J, ξ1x1, ξ1x2, ξ1x3, ξ2x1, ξ2x2, ξ2x3
        ) do J, ξ1x1, ξ1x2, ξ1x3, ξ2x1, ξ2x2, ξ2x3
      x1ξ3 = J * (ξ1x2 * ξ2x3 - ξ2x2 * ξ1x3)
      x2ξ3 = J * (ξ1x3 * ξ2x1 - ξ2x3 * ξ1x1)
      x3ξ3 = J * (ξ1x1 * ξ2x2 - ξ2x1 * ξ1x2)
      hypot(x1ξ3, x2ξ3, x3ξ3)
    end
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
  for n = 2:Nq
    # grid from first dof to current point
    rdst = (1 .- r)/2 * r[1] + (1 .+ r)/2 * r[n]
    # interpolation matrix
    In = Elements.interpolationmatrix(r, rdst, wbary)
    # scaling from LGL to current of the interval
    Δ = (r[n] -  r[1]) / 2
    # row of the matrix we have computed
    I∫[n, :] .= (Δ * ω' * In)[:]
  end
  I∫
end
# }}}

end # module
