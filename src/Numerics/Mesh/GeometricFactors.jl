module GeometricFactors

export VolumeGeometry, SurfaceGeometry

"""
    VolumeGeometry{Nq, AA <: AbstractArray, A <: AbstractArray}

A struct that collects `VolumeGeometry` fields:
- array: Array contatining the data stored in a VolumeGeometry struct (the following fields are views into this array)
- ∂ξk/∂xi: Derivative of the Cartesian reference element coordinate `ξ_k` with respect to the Cartesian physical coordinate `x_i`
- ωJ: Mass matrix. This is the physical mass matrix, and thus contains the Jacobian determinant, J .* (ωᵢ ⊗ ωⱼ ⊗ ωₖ), where ωᵢ are the quadrature weights and J is the Jacobian determinant, det(∂x/∂ξ)
- ωJI: Inverse mass matrix: 1 ./ ωJ
- ωJH: Horizontal mass matrix (used in diagnostics),  J .* norm(∂ξ3/∂x) * (ωᵢ ⊗ ωⱼ); for integrating over a plane (in 2-D ξ2 is used, not ξ3)
- xi: Nodal degrees of freedom locations in Cartesian physical space
- JcV: Metric terms for vertical line integrals norm(∂x/∂ξ3) (in 2-D ξ2 is used, not ξ3)
- ∂xk/∂ξi: Inverse of matrix `∂ξk/∂xi` that represents the derivative of Cartesian physical coordinate `x_i` with respect to Cartesian reference element coordinate `ξ_k`
"""
struct VolumeGeometry{Nq, AA <: AbstractArray, A <: AbstractArray}
    array::AA
    ξ1x1::A
    ξ2x1::A
    ξ3x1::A
    ξ1x2::A
    ξ2x2::A
    ξ3x2::A
    ξ1x3::A
    ξ2x3::A
    ξ3x3::A
    ωJ::A
    ωJI::A
    ωJH::A
    x1::A
    x2::A
    x3::A
    JcV::A
    x1ξ1::A
    x2ξ1::A
    x3ξ1::A
    x1ξ2::A
    x2ξ2::A
    x3ξ2::A
    x1ξ3::A
    x2ξ3::A
    x3ξ3::A
    function VolumeGeometry{Nq}(
        array::AA,
        args::A...,
    ) where {Nq, AA, A <: AbstractArray}
        new{Nq, AA, A}(array, args...)
    end
end

"""
    VolumeGeometry(FT, Nq::NTuple{N, Int}, nelems::Int)

Construct an empty `VolumeGeometry` object, in `FT` precision.
- `Nq` is a tuple containing the number of quadrature points in each direction.
- `nelem` is the number of elements.
"""
function VolumeGeometry(FT, Nq::NTuple{N, Int}, nelem::Int) where {N}
    # - 1 after fieldcount is to remove the `array` field from the array allocation
    array = zeros(FT, prod(Nq), fieldcount(VolumeGeometry) - 1, nelem)
    VolumeGeometry{Nq}(
        array,
        ntuple(j -> @view(array[:, j, :]), fieldcount(VolumeGeometry) - 1)...,
    )
end

"""
    SurfaceGeometry{Nq, AA, A <: AbstractArray}

A struct that collects `VolumeGeometry` fields:
- array: Array contatining the data stored in a SurfaceGeometry struct (the following fields are views into this array)
- ni: Outward pointing unit normal in physical space
- sωJ: Surface mass matrix. This is the physical mass matrix, and thus contains the surface Jacobian determinant, sJ .* (ωⱼ ⊗ ωₖ), where ωᵢ are the quadrature weights and sJ is the surface Jacobian determinant
- vωJI: Volume mass matrix at the surface nodes (needed in the lift operation, i.e., the projection of a face field back to the volume). Since DGSEM is used only collocated, volume mass matrices are required.
"""
struct SurfaceGeometry{Nq, AA, A <: AbstractArray}
    array::AA
    n1::A
    n2::A
    n3::A
    sωJ::A
    vωJI::A
    function SurfaceGeometry{Nq}(
        array::AA,
        args::A...,
    ) where {Nq, AA, A <: AbstractArray}
        new{Nq, AA, A}(array, args...)
    end
end

"""
    SurfaceGeometry(FT, Nq::NTuple{N, Int}, nface::Int, nelem::Int)

Construct an empty `SurfaceGeometry` object, in `FT` precision.
- `Nq` is a tuple containing the number of quadrature points in each direction.
- `nface` is the number of faces.
- `nelem` is the number of elements.
"""
function SurfaceGeometry(
    FT,
    Nq::NTuple{N, Int},
    nface::Int,
    nelem::Int,
) where {N}
    Np = prod(Nq)
    Nfp = div.(Np, Nq)
    # - 1 after fieldcount is to remove the `array` field from the array allocation
    array =
        zeros(FT, fieldcount(SurfaceGeometry) - 1, maximum(Nfp), nface, nelem)
    SurfaceGeometry{Nfp}(
        array,
        ntuple(
            j -> @view(array[j, :, :, :]),
            fieldcount(SurfaceGeometry) - 1,
        )...,
    )
end

end # module
