#####
##### CartesianElement
#####

struct CartesianElement{D, X, Y, Z}
    data :: D
       x :: X
       y :: Y
       z :: Z
end

Base.size(element::CartesianElement) = size(element.data)
Base.collect(element::CartesianElement) = CartesianElement(collect(data), collect(x), collect(y), collect(z))

Base.show(io::IO, elem::CartesianElement{D}) where D =
    print(io, "CartesianElement{$(D.name.wrapper)} with ",
          @sprintf("data ∈ [%.2e, %.2e]", minimum(elem.data), maximum(elem.data)), '\n',
          @sprintf("    x ∈ [%.2e, %.2e]", minimum(elem.x), maximum(elem.x)), '\n',
          @sprintf("    y ∈ [%.2e, %.2e]", minimum(elem.y), maximum(elem.y)), '\n',
          @sprintf("    z ∈ [%.2e, %.2e]", minimum(elem.z), maximum(elem.z)), '\n')

Base.@propagate_inbounds Base.getindex(elem::CartesianElement, i, j, k) = elem.data[i, j, k]

function x_join(eL, eR)
    eL.x[end] ≈ eR.x[1] || error("Element end-points $((eL.x[end], eR.x[1])) are not x-adjacent!")
    all(eL.y .≈ eR.y)   || error("Elements do not share y nodes!")
    all(eL.z .≈ eR.z)   || error("Elements do not share z nodes!")

    x = vcat(eL.x, eR.x[2:end])
    y = eL.y
    z = eL.z

    contact = @. 1/2 * (eL.data[end:end, :, :] + eR.data[1:1, :, :])

    data = cat(eL.data[1:end-1, :, :], contact, eR.data[2:end, :, :], dims=1)

    return CartesianElement(data, x, y, z)
end

function y_join(eL, eR)
    all(eL.x .≈ eR.x)   || error("Elements do not share x nodes!")
    eL.y[end] ≈ eR.y[1] || error("Elements are not y-adjacent!")
    all(eL.z .≈ eR.z)   || error("Elements do not share z nodes!")

    x = eL.x
    y = vcat(eL.y, eR.y[2:end])
    z = eL.z

    contact = @. 1/2 * (eL.data[:, end:end, :] + eR.data[:, 1:1, :])
    data = cat(eL.data[:, 1:end-1, :], contact, eR.data[:, 2:end, :], dims=2)

    return CartesianElement(data, x, y, z)
end

function z_join(eL, eR)
    all(eL.x .≈ eR.x)   || error("Elements do not share x nodes!")
    all(eL.y .≈ eR.y)   || error("Elements do not share y nodes!")
    eL.z[end] ≈ eR.z[1] || error("Elements are not z-adjacent!")

    x = eL.x
    y = eL.y
    z = vcat(eL.z, eR.z[2:end])

    contact = @. 1/2 * (eL.data[:, :, end:end] + eR.data[:, :, 1:1])
    data = cat(eL.data[:, :, 1:end-1], contact, eR.data[:, :, 2:end], dims=3)

    return CartesianElement(data, x, y, z)
end

x_join(e1, e2, e3...) = x_join(e1, x_join(e2, e3...))
y_join(e1, e2, e3...) = y_join(e1, y_join(e2, e3...))
z_join(e1, e2, e3...) = z_join(e1, z_join(e2, e3...))

x_join(e1) = e1
y_join(e1) = e1
z_join(e1) = e1

function join(elements)

    Ne = size(elements)

    pencils = [x_join(elements[:, j, k]...) for j = 1:Ne[2], k = 1:Ne[3]]
    slabs = [y_join(pencils[:, k]...) for k = 1:Ne[3]]
    volume = z_join(slabs...)

    return volume
end


