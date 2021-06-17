⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::AbstractArray, b::AbstractArray) = a * b'