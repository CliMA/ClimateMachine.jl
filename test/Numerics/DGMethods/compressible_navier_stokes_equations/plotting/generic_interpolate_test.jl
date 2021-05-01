using ClimateMachine
using Optim
ClimateMachine.init()
##
include("../boilerplate.jl")
include("../plotting/vizinanigans.jl")

## Functions 
"""
lagrange_eval(f, ξ⃗, ihelper::InterpolationHelper)

# Description
evaluate the function f at ξ⃗

# arguments
f: an array within an element, usually an mpi-state array Q[:,s,e]
where s and e are fixed indices
ξ⃗: an array of numbers
"""
function lagrange_eval(f, ξ⃗, ihelper::InterpolationHelper)
    fnew = lagrange_eval(f, ξ⃗[1], ξ⃗[2], ξ⃗[3],  
                  ihelper.points[1], ihelper.points[2], ihelper.points[3], 
                  ihelper.quadrature[1], ihelper.quadrature[2], ihelper.quadrature[3])
    return fnew
end

"""
closure_position(x,y,z,e,grid, ihelper)

returns polynomial approximation to mapping x⃗(ξ⃗) at element e, where
ξ⃗ ∈ [-1,1]^d (should be true but not necessarily)
position(ξ¹, ξ², ξ³)


"""
function closure_position(x,y,z,e,grid, ihelper)
    function position(ξ¹, ξ², ξ³)
        ξ⃗ = (ξ¹, ξ², ξ³)
        x¹ = lagrange_eval(reshape(x[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
        x² = lagrange_eval(reshape(y[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
        x³ = lagrange_eval(reshape(z[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
        x⃗ = [x¹, x², x³]
        return x⃗
    end
end

"""
closure_cost(x1,y1,z1, position_function)

# Description 
returns the loss function |x⃗(ξ⃗) - y⃗ |

"""
function closure_cost(x1,y1,z1, position_function)
    function cost(x)
        norm(position_function(x...) - [x1, y1, z1])
    end
end

"""
findelement(cellcenter, newpoint)

# Description
Given the cell centers and the point we want to interpolate to, find the closest element
"""
function findelement(cellcenter, newpoint)
    minind = argmin((cellcenter[1] .- newpoint[1]) .^2 + (cellcenter[2] .- newpoint[2]) .^2 + (cellcenter[3] .- newpoint[3]) .^2)
    return minind
end

"""
findelement(cellcenter, newpoint)

# Description
Given the cell centers and a grid we want to interpolate to, find the closest elements for each point on the new grid
"""
function findelements(cellcenters, newgrid)
    elementlist = zeros(Int, length(newgrid))
    Threads.@threads for i in eachindex(elementlist)
        elementlist[i] = findelement(cellcenters, newgrid[i]) 
    end
    return elementlist
end

"""
findξlist(elementlist, x,y,z , newgrid, grid, ihelper)

# Description
Given the cell centers and a grid we want to interpolate to, find the closest elements for each point on the new grid
"""
function findξlist(elementlist, newgrid, grid, ihelper; f_calls_limit = 50)
    x,y,z = coordinates(grid)

    ξlist = zeros(length(elementlist), 3)
    losslist = zeros(length(elementlist))

    Threads.@threads for i in eachindex(elementlist)
        e = elementlist[i]
        position = closure_position(x,y,z, e, grid, ihelper)
        xnew = newgrid[i][1]
        ynew = newgrid[i][2]
        znew = newgrid[i][3]
        loss = closure_cost(xnew, ynew, znew, position)
        # get initial guess via closest nodal point
        rx = reshape(x[:,e], polynomialorders(grid) .+ 1)
        ry = reshape(y[:,e], polynomialorders(grid) .+ 1)
        rz = reshape(z[:,e], polynomialorders(grid) .+ 1)
        minind = argmin((rx .- xnew) .^2 + (ry .- ynew) .^2 + (rz .- znew) .^2)
        ξᴳ = [ihelper.points[i][minind[i]] for i in 1:3]
        # invert the mapping x⃗(ξ⃗) = y⃗ 
        result = optimize(loss, ξᴳ, BFGS(), Optim.Options(f_calls_limit = f_calls_limit))
        ξlist[i,:] .= result.minimizer
        losslist[i] = loss(ξlist[i,:])
    end
    return ξlist
end

"""
interpolatefield(elementlist, ξlist, newgrid, grid, ihelper, oldϕ)

# Description 
interpolates to new field

# return
newϕ

"""
function interpolatefield(elementlist, ξlist, newgrid, grid, ihelper, oldϕ)
    newϕ = zeros(size(newgrid))
    Threads.@threads for i in eachindex(elementlist)
        e = elementlist[i]
        ξ⃗ = ξlist[i,:]
        newϕ[i] = lagrange_eval(reshape(oldϕ[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
    end
    return newϕ
end

## Test
domain = AtmosDomain(radius = 1, height = 0.1)

ddomain = DiscretizedDomain(
    domain;
    elements = (vertical = 5, horizontal = 8),
    polynomial_order = (vertical = 3, horizontal = 6),
    overintegration_order = 0,
)

grid = ddomain.numerical
xC, yC, zC = cellcenters(grid)
x, y, z = coordinates(grid)
ihelper = InterpolationHelper(grid)

## latitude is λ, longitude if ϕ
rC = sqrt.(xC .^2 .+ yC .^2  .+ zC .^2)
rlist = collect(range(minimum(rC), maximum(rC), length = 4))
r = [maximum(rC) + minimum(rC)] * 0.5 # collect(range(domain.radius,domain.radius + domain.height,length = 10))
λ = collect(range(0,2π, length = 90))
ϕ = collect(range(0 + π/9 , π - π/9, length = 43))
newgrid = [ [r[i] * cos(λ[j]) * sin(ϕ[k]), r[i] * sin(λ[j]) * sin(ϕ[k]) , r[i] * cos(ϕ[k])] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]

elementlist = findelements(cellcenters(grid), newgrid)
ξlist = findξlist(elementlist, newgrid, grid, ihelper)

##
rad(x,y,z) = sqrt(x^2 + y^2 + z^2)
lat(x,y,z) = asin(z/rad(x,y,z)) # ϕ ∈ [-π/2, π/2] 
lon(x,y,z) = atan(y,x) # λ ∈ [-π, π) 

oldr = sin.(lon.(x,y,z)) # sqrt.(x .^2  + y .^ 2 + z .^2)
oldfield = sin.(lon.(x,y,z))
newr = zeros(size(newgrid))

tic = Base.time()
newfield = interpolatefield(elementlist, ξlist, newgrid, grid, ihelper, oldfield)
toc = Base.time()
println(toc-tic)
println("Time for finding oneself ", toc - tic, " seconds")

newr .= newfield

##
fig = Figure(resolution = (1086, 828))

clims = [-1,1] # extrema(newr)

xnew = reshape([newgrid[i][1] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
ynew = reshape([newgrid[i][2] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
znew = reshape([newgrid[i][3] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))

n = 1
ax = fig[1:3,1:3] = LScene(fig) # make plot area wider
surface!(ax, xnew[1,:,:], ynew[1,:,:], znew[1,:,:], color=newr[1,:,:], colormap= :balance, colorrange=clims,  shading = false, show_axis=false)
rotate_cam!(ax.scene, (π/2, π/6, 0))
zoom!(ax.scene, (0, 0, 0), 5, false)
fig[4,2] = Label(fig, "Sphere Plot", textsize = 50) # put names in center

ax2 = fig[2, 5:7] = LScene(fig)
heatmap!(ax2, λ, ϕ, newr[1,:,:], colormap = :balance, interpolate = true)
fig[4, 6]  = Label(fig, "Lat-Lon Plot", textsize = 50)

zonal_mean_newr = mean(newr[1,:,:], dims = 1)

scatter(zonal_mean_newr[:])

display(fig)
#=
iterations = 1:360
record(fig, "makieprelim.mp4", iterations, framerate=30) do i
    rotate_cam!(fig.scene.children[1], (2π/360, 0, 0))
end
=#