#####
##### RectangularElement
#####

"""
    RectangularElement(data, x, y, z)

Returns...
"""
struct RectangularElement{D, X, Y, Z}
    data::D
    x::X
    y::Y
    z::Z
    # M::Mⁱʲ
end

Base.eltype(::RectangularElement) = eltype(elem.data)

Base.size(element::RectangularElement) = size(element.data)

Base.collect(element::RectangularElement) =
    RectangularElement(collect(data), collect(x), collect(y), collect(z))

Base.@propagate_inbounds Base.getindex(elem::RectangularElement, i, j, k) =
    elem.data[i, j, k]

Base.maximum(f, element::RectangularElement) = maximum(f, element.data)
Base.minimum(f, element::RectangularElement) = minimum(f, element.data)

Base.maximum(element::RectangularElement) = maximum(element.data)
Base.minimum(element::RectangularElement) = minimum(element.data)

function Base.show(io::IO, elem::RectangularElement{D}) where {D}
    intro = "RectangularElement{$(D.name.wrapper)} with "
    data = @sprintf("data ∈ [%.2e, %.2e]\n", minimum(elem.data), maximum(elem.data))
    x = @sprintf("    x ∈ [%.2e, %.2e]\n", minimum(elem.x), maximum(elem.x))
    y = @sprintf("    y ∈ [%.2e, %.2e]", minimum(elem.y), maximum(elem.y))
    z = @sprintf("    z ∈ [%.2e, %.2e]", minimum(elem.z), maximum(elem.z))

    return print(io, intro, data, x, y, z)
end    
    

#####
##### ⟨⟨ Assemble! ⟩⟩
#####

""" Assemble an array along the first dimension. """
function assemble(::Val{1}, west::AbstractArray, east::AbstractArray)
    contact = @. (west[end:end, :, :] + east[1:1, :, :]) / 2

    east = east[2:end, :, :]
    west = west[1:(end - 1), :, :]

    assembled = cat(west, contact, east, dims = 1)

    return assembled
end

""" Assemble an array along the second dimension. """
function assemble(::Val{2}, south::AbstractArray, north::AbstractArray)
    contact = @. (south[:, end:end, :] + north[:, 1:1, :]) / 2

    north = north[:, 2:end, :]
    south = south[:, 1:(end - 1), :]

    assembled = cat(south, contact, north, dims = 2)

    return assembled
end

""" Assemble an array along the third dimension. """
function assemble(::Val{3}, bottom::AbstractArray, top::AbstractArray)
    contact = @. (bottom[:, :, end:end] + top[:, :, 1:1]) / 2

    top = top[:, :, 2:end]
    bottom = bottom[:, :, 1:(end - 1)]

    assembled = cat(bottom, contact, top, dims = 3)

    return assembled
end

""" Assemble elements along `dim`ension. """
function assemble(dim::Val, left::RectangularElement, right::RectangularElement)
    data = assemble(dim, left.data, right.data)
    x = assemble(dim, left.x, right.x)
    y = assemble(dim, left.y, right.y)
    z = assemble(dim, left.z, right.z)

    return RectangularElement(data, x, y, z)
end

assemble(dim::Val, e1, e2, e3...) = assemble(dim, e1, assemble(dim, e2, e3...))
assemble(elem) = elem

data(elem::RectangularElement) = elem.data

"""
    assemble(elements::Array{<:RectangularElement, 3})

Assemble the three-dimensional data in `elements` into a single `Array`,
averaging data on shared nodes.
"""
function assemble(elements::Array{T, 3}) where T <: Union{RectangularElement, AbstractArray}

    Nx, Ny, Nz = size(elements)

    pencils = [assemble(Val(1), elements[:, j, k]...) for j in 1:Ny, k in 1:Nz]

    slabs = [assemble(Val(2), pencils[:, k]...) for k in 1:Nz]

    volume = assemble(Val(3), slabs...)

    return volume
end
