module GeomData

export VolumeGeometry, SurfaceGeometry

struct VolumeGeometry{Nq, A<:AbstractArray}
    ξ1x1::A
    ξ2x1::A
    ξ3x1::A
    ξ1x2::A
    ξ2x2::A
    ξ3x2::A
    ξ1x3::A
    ξ2x3::A
    ξ3x3::A
    x1ξ1::A
    x2ξ1::A
    x3ξ1::A
    x1ξ2::A
    x2ξ2::A
    x3ξ2::A
    x1ξ3::A
    x2ξ3::A
    x3ξ3::A
    ωJ::A
    ωJI::A
    ωJH::A
    x1::A
    x2::A
    x3::A
    JcV::A
    function VolumeGeometry{Nq}(args::A...) where {Nq, A<:AbstractArray}
        new{Nq,A}(args...)
    end
end

"""
    VolumeGeometry(FT, Nq::Tuple, nelems::Integer)

Construct an empty `VolumeGeometry` object, in `FT` precision.
- `Nq` is a tuple containing the number of quadrature points in each direction.
- `nelem` is the number of elements.
"""
function VolumeGeometry(FT, Nq::NTuple{N,Int}, nelem::Int) where {N}
    array = zeros(FT, prod(Nq), fieldcount(VolumeGeometry), nelem)
    VolumeGeometry{Nq}(ntuple(j -> @view(array[:, j, :]), fieldcount(VolumeGeometry))...)
end

struct SurfaceGeometry{Nq, A<:AbstractArray}
    n1::A
    n2::A
    n3::A
    sωJ::A
    vωJI::A
    function SurfaceGeometry{Nq}(args::A...) where {Nq, A<:AbstractArray}
        new{Nq,A}(args...)
    end
end

"""
SurfaceGeometry(FT, Nq::Tuple, nface::Integer, nelem::Integer)

Construct an empty `SurfaceGeometry` object, in `FT` precision.
- `Nq` is a tuple containing the number of quadrature points in each direction.
- `nface` is the number of faces.
- `nelem` is the number of elements.
"""
function SurfaceGeometry(FT, Nq::NTuple{N,Int}, nface::Int, nelem::Int) where {N}
    Np = prod(Nq)
    Nfp = div.(Np, Nq)
    array = zeros(FT, maximum(Nfp), fieldcount(SurfaceGeometry), nface, nelem)
    SurfaceGeometry{Nfp}(ntuple(j -> @view(array[:, j, :, :]), fieldcount(SurfaceGeometry))...)
end

end # module
