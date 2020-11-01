##### ContinuousData

export ContinuousData

using Interpolations

"""
    ContinuousData(coordinate_data::A, data::A) where {A<:AbstractArray}

Creates a continuous representation
of discrete data. Example:

```julia
FT = Float64

data_discrete = FT[1, 2, 3, 4]
coordinate_discrete = FT[0, 10, 20, 30]

data_continuous = ContinuousData(coordinate_discrete, data_discrete)

data_at_40 = data_continuous(40)
```
"""
struct ContinuousData{FT <: AbstractFloat, I, E}
    itp::I
    ext::E
    bounds::Tuple{FT, FT}
    function ContinuousData(
        coordinate_data::A,
        data::A,
    ) where {A <: AbstractArray}
        FT = eltype(A)
        itp = interpolate((coordinate_data,), data, Gridded(Linear()))
        ext = extrapolate(itp, Flat())
        bounds = (first(coordinate_data), last(coordinate_data))
        I = typeof(itp)
        E = typeof(ext)
        return new{FT, I, E}(itp, ext, bounds)
    end
end

function (cont_data::ContinuousData)(x::A) where {A <: AbstractArray}
    return [
        cont_data.bounds[1] < x_i < cont_data.bounds[2] ? cont_data.itp(x_i) :
        cont_data.ext(x_i) for x_i in x
    ]
end

function (cont_data::ContinuousData)(x_i::FT) where {FT <: AbstractFloat}
    if cont_data.bounds[1] < x_i < cont_data.bounds[2]
        return cont_data.itp(x_i)
    else
        return cont_data.ext(x_i)
    end
end
