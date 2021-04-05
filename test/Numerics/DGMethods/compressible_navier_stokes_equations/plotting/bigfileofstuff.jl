using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Elements
import ClimateMachine.Mesh.Elements: baryweights
import ClimateMachine.Mesh.Grids: polynomialorders
using GaussQuadrature
using Base.Threads


# Depending on CliMa version 
# old, should return a tuple of polynomial orders
# polynomialorders(::DiscontinuousSpectralElementGrid{T, dim, N}) where {T, dim, N} = Tuple([N for i in 1:dim])
# new, should return a tuple of polynomial orders
# polynomialorders(::DiscontinuousSpectralElementGrid{T, dim, N}) where {T, dim, N} = N

# utils.jl
"""
function cellaverage(Q; M = nothing)

# Description
Compute the cell-average of Q given the mass matrix M.
Assumes that Q and M are the same size

# Arguments
`Q`: MPIStateArrays (array)

# Keyword Arguments
`M`: Mass Matrix (array)

# Return
The cell-average of Q
"""
function cellaverage(Q; M = nothing)
    if M == nothing
        return nothing
    end
    return (sum(M .* Q, dims = 1) ./ sum(M, dims = 1))[:]
end

"""
function coordinates(grid::DiscontinuousSpectralElementGrid)

# Description
Gets the (x,y,z) coordinates corresponding to the grid

# Arguments
- `grid`: DiscontinuousSpectralElementGrid

# Return
- `x, y, z`: views of x, y, z coordinates
"""
function coordinates(grid::DiscontinuousSpectralElementGrid)
    x = view(grid.vgeo, :, grid.x1id, :)   # x-direction	
    y = view(grid.vgeo, :, grid.x2id, :)   # y-direction	
    z = view(grid.vgeo, :, grid.x3id, :)   # z-direction
    return x, y, z
end

"""
function cellcenters(grid; M = nothing)

# Description
Get the cell-centers of every element in the grid

# Arguments
- `grid`: DiscontinuousSpectralElementGrid

# Return
- Tuple of cell-centers
"""
function cellcenters(grid::DiscontinuousSpectralElementGrid)
    x, y, z = coordinates(grid)
    M = massmatrix(grid)  # mass matrix
    xC = cellaverage(x, M = M)
    yC = cellaverage(y, M = M)
    zC = cellaverage(z, M = M)
    return xC[:], yC[:], zC[:]
end


"""
function massmatrix(grid; M = nothing)

# Description
Get the mass matrix of the grid

# Arguments
- `grid`: DiscontinuousSpectralElementGrid

# Return
- Tuple of cell-centers
"""
function massmatrix(grid)
    return view(grid.vgeo, :, grid.Mid, :)
end

# find_element.jl
# 3D version
function findelement(xC, yC, zC, location, p, lin)
    ex, ey, ez = size(lin)
    # i 
    currentmin = ones(1)
    minind = ones(Int64, 1)
    currentmin[1] = abs.(xC[p[lin[1, 1, 1]]] .- location[1])
    for i in 2:ex
        current = abs.(xC[p[lin[i, 1, 1]]] .- location[1])
        if current < currentmin[1]
            currentmin[1] = current
            minind[1] = i
        end
    end
    i = minind[1]
    # j 
    currentmin[1] = abs.(yC[p[lin[1, 1, 1]]] .- location[2])
    minind[1] = 1
    for i in 2:ey
        current = abs.(yC[p[lin[1, i, 1]]] .- location[2])
        if current < currentmin[1]
            currentmin[1] = current
            minind[1] = i
        end
    end
    j = minind[1]
    # k 
    currentmin[1] = abs.(zC[p[lin[1, 1, 1]]] .- location[3])
    minind[1] = 1
    for i in 2:ez
        current = abs.(zC[p[lin[1, 1, i]]] .- location[3])
        if current < currentmin[1]
            currentmin[1] = current
            minind[1] = i
        end
    end
    k = minind[1]
    return p[lin[i, j, k]]
end

# 2D version
function findelement(xC, yC, location, p, lin)
    ex, ey = size(lin)
    # i 
    currentmin = ones(1)
    minind = ones(Int64, 1)
    currentmin[1] = abs.(xC[p[lin[1, 1]]] .- location[1])
    for i in 2:ex
        current = abs.(xC[p[lin[i, 1]]] .- location[1])
        if current < currentmin[1]
            currentmin[1] = current
            minind[1] = i
        end
    end
    i = minind[1]
    # j 
    currentmin[1] = abs.(yC[p[lin[1, 1]]] .- location[2])
    minind[1] = 1
    for i in 2:ey
        current = abs.(yC[p[lin[1, i]]] .- location[2])
        if current < currentmin[1]
            currentmin[1] = current
            minind[1] = i
        end
    end
    j = minind[1]
    return p[lin[i, j]]
end

# gridhelper.jl

struct InterpolationHelper{S, T}
    points::S
    quadrature::S
    interpolation::S
    cartesianindex::T
end

function InterpolationHelper(g::DiscontinuousSpectralElementGrid)
    porders = polynomialorders(g)
    if length(porders) == 3
        npx, npy, npz = porders
        rx, wx = GaussQuadrature.legendre(npx + 1, both)
        ωx = baryweights(rx)
        ry, wy = GaussQuadrature.legendre(npy + 1, both)
        ωy = baryweights(ry)
        rz, wz = GaussQuadrature.legendre(npz + 1, both)
        ωz = baryweights(rz)
        linlocal = reshape(
            collect(1:((npx + 1) * (npy + 1) * (npz + 1))),
            (npx + 1, npy + 1, npz + 1),
        )
        return InterpolationHelper(
            (rx, ry, rz),
            (wx, wy, wz),
            (ωx, ωy, ωz),
            linlocal,
        )
    elseif length(porders) == 2
        npx, npy = porders
        rx, wx = GaussQuadrature.legendre(npx + 1, both)
        ωx = baryweights(rx)
        ry, wy = GaussQuadrature.legendre(npy + 1, both)
        ωy = baryweights(ry)
        linlocal =
            reshape(collect(1:((npx + 1) * (npy + 1))), (npx + 1, npy + 1))
        return InterpolationHelper((rx, ry), (wx, wy), (ωx, ωy), linlocal)
    else
        println("Not supported")
        return nothing
    end
    return nothing
end

struct ElementHelper{S, T, U, Q, V, W}
    cellcenters::S
    coordinates::T
    cartesiansizes::U
    polynomialorders::Q
    permutation::V
    cartesianindex::W
end

addup(xC, tol) = sum(abs.(xC[1] .- xC) .≤ tol)

# only valid for cartesian domains
function ElementHelper(g::DiscontinuousSpectralElementGrid)
    porders = polynomialorders(g)
    x, y, z = coordinates(g)
    xC, yC, zC = cellcenters(g)
    ne = size(x)[2]
    ex = round(Int64, ne / addup(xC, 10^4 * eps(maximum(abs.(x)))))
    ey = round(Int64, ne / addup(yC, 10^4 * eps(maximum(abs.(y)))))
    ez = round(Int64, ne / addup(zC, 10^4 * eps(maximum(abs.(z)))))

    check = ne == ex * ey * ez
    check ? true : error("improper counting")
    p = getperm(xC, yC, zC, ex, ey, ez)
    # should use dispatch ...
    if length(porders) == 3
        npx, npy, npz = porders
        lin = reshape(collect(1:length(xC)), (ex, ey, ez))
        return ElementHelper(
            (xC, yC, zC),
            (x, y, z),
            (ex, ey, ez),
            porders,
            p,
            lin,
        )
    elseif length(porders) == 2
        npx, npy = porders
        lin = reshape(collect(1:length(xC)), (ex, ey))
        check = ne == ex * ey
        check ? true : error("improper counting")
        return ElementHelper((xC, yC), (x, y), (ex, ey), porders, p, lin)
    else
        println("no constructor for polynomial order = ", porders)
        return nothing
    end
    return nothing
end

struct GridHelper{S, T, V}
    interpolation::S
    element::T
    grid::V
end

function GridHelper(g::DiscontinuousSpectralElementGrid)
    return GridHelper(InterpolationHelper(g), ElementHelper(g), g)
end

function getvalue(f, location, gridhelper::GridHelper)
    ih = gridhelper.interpolation
    eh = gridhelper.element
    porders = gridhelper.element.polynomialorders
    if length(porders) == 3
        npx, npy, npz = gridhelper.element.polynomialorders
        fl = reshape(f, (npx + 1, npy + 1, npz + 1, prod(eh.cartesiansizes)))
        ip = getvalue(
            fl,
            eh.cellcenters...,
            location,
            eh.permutation,
            eh.cartesianindex,
            ih.cartesianindex,
            eh.coordinates...,
            ih.points...,
            ih.interpolation...,
        )
        return ip
    elseif length(porders) == 2
        npx, npy = gridhelper.element.polynomialorders
        fl = reshape(f, (npx + 1, npy + 1, prod(eh.cartesiansizes)))
        ip = getvalue(
            fl,
            eh.cellcenters...,
            location,
            eh.permutation,
            eh.cartesianindex,
            ih.cartesianindex,
            eh.coordinates...,
            ih.points...,
            ih.interpolation...,
        )
        return ip
    end
    return nothing
end

# lagrange_interpolation.jl
function checkgl(x, rx)
    for i in eachindex(rx)
        if abs(x - rx[i]) ≤ eps(rx[i])
            return i
        end
    end
    return 0
end

function lagrange_eval(f, newx, newy, newz, rx, ry, rz, ωx, ωy, ωz)
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    kcheck = checkgl(newz, rz)
    numerator = zeros(1)
    denominator = zeros(1)
    for k in eachindex(rz)
        if kcheck == 0
            Δz = (newz .- rz[k])
            polez = ωz[k] ./ Δz
            kk = k
        else
            polez = 1.0
            k = eachindex(rz)[end]
            kk = kcheck
        end
        for j in eachindex(ry)
            if jcheck == 0
                Δy = (newy .- ry[j])
                poley = ωy[j] ./ Δy
                jj = j
            else
                poley = 1.0
                j = eachindex(ry)[end]
                jj = jcheck
            end
            for i in eachindex(rx)
                if icheck == 0
                    Δx = (newx .- rx[i])
                    polex = ωx[i] ./ Δx
                    ii = i
                else
                    polex = 1.0
                    i = eachindex(rx)[end]
                    ii = icheck
                end
                numerator[1] += f[ii, jj, kk] * polex * poley * polez
                denominator[1] += polex * poley * polez
            end
        end
    end
    return numerator[1] / denominator[1]
end

function lagrange_eval(f, newx, newy, rx, ry, ωx, ωy)
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    numerator = zeros(1)
    denominator = zeros(1)
    for j in eachindex(ry)
        if jcheck == 0
            Δy = (newy .- ry[j])
            poley = ωy[j] ./ Δy
            jj = j
        else
            poley = 1.0
            j = eachindex(ry)[end]
            jj = jcheck
        end
        for i in eachindex(rx)
            if icheck == 0
                Δx = (newx .- rx[i])
                polex = ωx[i] ./ Δx
                ii = i
            else
                polex = 1.0
                i = eachindex(rx)[end]
                ii = icheck
            end
            numerator[1] += f[ii, jj] * polex * poley
            denominator[1] += polex * poley
        end
    end
    return numerator[1] / denominator[1]
end


function lagrange_eval_nocheck(f, newx, newy, newz, rx, ry, rz, ωx, ωy, ωz)
    numerator = zeros(1)
    denominator = zeros(1)
    for k in eachindex(rz)
        Δz = (newz .- rz[k])
        polez = ωz[k] ./ Δz
        for j in eachindex(ry)
            Δy = (newy .- ry[j])
            poley = ωy[j] ./ Δy
            for i in eachindex(rx)
                Δx = (newx .- rx[i])
                polex = ωx[i] ./ Δx
                numerator[1] += f[i, j, k] * polex * poley * polez
                denominator[1] += polex * poley * polez
            end
        end
    end
    return numerator[1] / denominator[1]
end

function lagrange_eval_nocheck(f, newx, newy, rx, ry, ωx, ωy)
    numerator = zeros(1)
    denominator = zeros(1)
    for j in eachindex(ry)
        Δy = (newy .- ry[j])
        poley = ωy[j] ./ Δy
        for i in eachindex(rx)
            Δx = (newx .- rx[i])
            polex = ωx[i] ./ Δx
            numerator[1] += f[i, j] * polex * poley
            denominator[1] += polex * poley
        end
    end
    return numerator[1] / denominator[1]
end

function lagrange_eval_nocheck(f, newx, rx, ωx)
    numerator = zeros(1)
    denominator = zeros(1)
    for i in eachindex(rx)
        Δx = (newx .- rx[i])
        polex = ωx[i] ./ Δx
        numerator[1] += f[i] * polex * poley
        denominator[1] += polex * poley
    end
    return numerator[1] / denominator[1]
end

# 3D, only valid for rectangles
function getvalue(
    fl,
    xC,
    yC,
    zC,
    location,
    p,
    lin,
    linlocal,
    x,
    y,
    z,
    rx,
    ry,
    rz,
    ωx,
    ωy,
    ωz,
)
    e = findelement(xC, yC, zC, location, p, lin)
    # need bounds to rescale, only value for cartesian

    xmax = x[linlocal[length(rx), 1, 1], e]
    xmin = x[linlocal[1, 1, 1], e]
    ymax = y[linlocal[1, length(ry), 1], e]
    ymin = y[linlocal[1, 1, 1], e]
    zmax = z[linlocal[1, 1, length(rz)], e]
    zmin = z[linlocal[1, 1, 1], e]

    # rescale new point to [-1,1]³
    newx = 2 * (location[1] - xmin) / (xmax - xmin) - 1
    newy = 2 * (location[2] - ymin) / (ymax - ymin) - 1
    newz = 2 * (location[3] - zmin) / (zmax - zmin) - 1

    return lagrange_eval(
        view(fl, :, :, :, e),
        newx,
        newy,
        newz,
        rx,
        ry,
        rz,
        ωx,
        ωy,
        ωz,
    )
end

# 2D
function getvalue(fl, xC, yC, location, p, lin, linlocal, x, y, rx, ry, ωx, ωy)
    e = findelement(xC, yC, location, p, lin)
    # need bounds to rescale
    xmax = x[linlocal[length(rx), 1, 1], e]
    xmin = x[linlocal[1, 1, 1], e]
    ymax = y[linlocal[1, length(ry), 1], e]
    ymin = y[linlocal[1, 1, 1], e]

    # rescale new point to [-1,1]²
    newx = 2 * (location[1] - xmin) / (xmax - xmin) - 1
    newy = 2 * (location[2] - ymin) / (ymax - ymin) - 1

    return lagrange_eval(view(fl, :, :, e), newx, newy, rx, ry, ωx, ωy)
end

# permutations.jl
function getperm(xC, yC, zC, ex, ey, ez)
    pz = sortperm(zC)
    tmpY = reshape(yC[pz], (ex * ey, ez))
    tmp_py = [sortperm(tmpY[:, i]) for i in 1:ez]
    py = zeros(Int64, length(pz))
    for i in eachindex(tmp_py)
        n = length(tmp_py[i])
        ii = (i - 1) * n + 1
        py[ii:(ii + n - 1)] .= tmp_py[i] .+ ii .- 1
    end
    tmpX = reshape(xC[pz][py], (ex, ey * ez))
    tmp_px = [sortperm(tmpX[:, i]) for i in 1:(ey * ez)]
    px = zeros(Int64, length(pz))
    for i in eachindex(tmp_px)
        n = length(tmp_px[i])
        ii = (i - 1) * n + 1
        px[ii:(ii + n - 1)] .= tmp_px[i] .+ ii .- 1
    end
    p = [pz[py[px[i]]] for i in eachindex(px)]
    return p
end
