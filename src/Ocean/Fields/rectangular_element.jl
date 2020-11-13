#####
##### RectangularElement
#####

struct RectangularElement{D, X, Y, Z}
    data::D
    x::X
    y::Y
    z::Z
end

Base.size(element::RectangularElement) = size(element.data)
Base.collect(element::RectangularElement) =
    RectangularElement(collect(data), collect(x), collect(y), collect(z))

Base.maximum(f, element::RectangularElement) = maximum(f, element.data)
Base.minimum(f, element::RectangularElement) = minimum(f, element.data)

Base.maximum(element::RectangularElement) = maximum(element.data)
Base.minimum(element::RectangularElement) = minimum(element.data)

Base.show(io::IO, elem::RectangularElement{D}) where {D} = print(
    io,
    "RectangularElement{$(D.name.wrapper)} with ",
    @sprintf("data ∈ [%.2e, %.2e]", minimum(elem.data), maximum(elem.data)),
    '\n',
    @sprintf("    x ∈ [%.2e, %.2e]", minimum(elem.x), maximum(elem.x)),
    '\n',
    @sprintf("    y ∈ [%.2e, %.2e]", minimum(elem.y), maximum(elem.y)),
    '\n',
    @sprintf("    z ∈ [%.2e, %.2e]", minimum(elem.z), maximum(elem.z)),
    '\n',
)

Base.@propagate_inbounds Base.getindex(elem::RectangularElement, i, j, k) =
    elem.data[i, j, k]

eltype(::RectangularElement{<:AbstractArray{FT}}) where {FT} = FT

#####
##### ⟨⟨ Assemble! ⟩⟩
#####


function x_assemble(west::RectangularElement, east::RectangularElement)
    west.x[end] ≈ east.x[1] ||
    error("Element end-points $((west.x[end], east.x[1])) are not x-adjacent!")

    all(west.y .≈ east.y) || error("Elements do not share y nodes!")
    all(west.z .≈ east.z) || error("Elements do not share z nodes!")

    x = vcat(west.x, east.x[2:end])
    y = west.y
    z = west.z

    contact = @. 1 / 2 * (west.data[end:end, :, :] + east.data[1:1, :, :])

    data = cat(
        west.data[1:(end - 1), :, :],
        contact,
        east.data[2:end, :, :],
        dims = 1,
    )

    return RectangularElement(data, x, y, z)
end

function y_assemble(south::RectangularElement, north::RectangularElement)

    FT = eltype(south)

    all(south.x .≈ north.x) || error("Elements do not share x nodes!")

    isapprox(south.y[end], north.y[1], atol = sqrt(eps(FT))) ||
    error("Elements are not y-adjacent!")

    all(south.z .≈ north.z) || error("Elements do not share z nodes!")

    x = south.x
    y = vcat(south.y, north.y[2:end])
    z = south.z

    contact = @. 1 / 2 * (south.data[:, end:end, :] + north.data[:, 1:1, :])

    data = cat(
        south.data[:, 1:(end - 1), :],
        contact,
        north.data[:, 2:end, :],
        dims = 2,
    )

    return RectangularElement(data, x, y, z)
end

function z_assemble(bottom::RectangularElement, top::RectangularElement)

    FT = eltype(bottom)

    all(bottom.x .≈ top.x) || error("Elements do not share x nodes!")
    all(bottom.y .≈ top.y) || error("Elements do not share y nodes!")

    isapprox(bottom.z[end], top.z[1], atol = sqrt(eps(FT))) ||
    error("Elements are not z-adjacent!")

    x = bottom.x
    y = bottom.y
    z = vcat(bottom.z, top.z[2:end])

    contact = @. 1 / 2 * (bottom.data[:, :, end:end] + top.data[:, :, 1:1])
    data = cat(
        bottom.data[:, :, 1:(end - 1)],
        contact,
        top.data[:, :, 2:end],
        dims = 3,
    )

    return RectangularElement(data, x, y, z)
end

x_assemble(e1, e2, e3...) = x_assemble(e1, x_assemble(e2, e3...))
y_assemble(e1, e2, e3...) = y_assemble(e1, y_assemble(e2, e3...))
z_assemble(e1, e2, e3...) = z_assemble(e1, z_assemble(e2, e3...))

x_assemble(e1) = e1
y_assemble(e1) = e1
z_assemble(e1) = e1

"""
    assemble(elements::Array{<:RectangularElement, 3})

Assemble the three-dimensional data in `elements` into a single `RectangularElement`,
averaging data on shared nodes.
"""
function assemble(elements::Array{<:RectangularElement, 3})

    Ne = size(elements)

    pencils = [x_assemble(elements[:, j, k]...) for j in 1:Ne[2], k in 1:Ne[3]]
    slabs = [y_assemble(pencils[:, k]...) for k in 1:Ne[3]]
    volume = z_assemble(slabs...)

    return volume
end
