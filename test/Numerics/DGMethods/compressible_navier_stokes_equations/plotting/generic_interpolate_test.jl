using ClimateMachine
using Optim
ClimateMachine.init()
##
include("../boilerplate.jl")
include("../plotting/vizinanigans.jl")

##
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

##
function lagrange_eval(f, newr, ihelper::InterpolationHelper)
    fnew = lagrange_eval(f, newr[1], newr[2], newr[3],  
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

function closure_cost(x1,y1,z1, position_function)
    function cost(x)
        norm(position_function(x...) - [x1, y1, z1])
    end
end

function findelement(cellcenter, newpoint)
    minind = argmin((cellcenter[1] .- newpoint[1]) .^2 + (cellcenter[2] .- newpoint[2]) .^2 + (cellcenter[3] .- newpoint[3]) .^2)
    return minind
end
##
fig = Figure()
lscene = fig[1:4, 2:4] = LScene(fig.scene)
e = 1
scatter!(lscene, x[:, e], y[:, e], z[:, e], color = :black, markersize = 100.0, strokewidth = 0)
scatter!(lscene, xC[e:e], yC[e:e], zC[e:e], color = :red, markersize = 100.0, strokewidth = 0)

## latitude is λ, longitude if ϕ
r = [domain.radius + domain.height/2]# collect(range(domain.radius,domain.radius + domain.height,length = 10))
λ = collect(range(0,2π, length = 180))
ϕ = collect(range(0 + π/18 , π - π/18, length = 43))

newgrid = [ [r[i] * cos(λ[j]) * sin(ϕ[k]), r[i] * sin(λ[j]) * sin(ϕ[k]) , r[i] * cos(ϕ[k])] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]

xnew = reshape([newgrid[i][1] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
ynew = reshape([newgrid[i][2] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
znew = reshape([newgrid[i][3] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))

#=
fig = Figure()
lscene = fig[1:4, 2:4] = LScene(fig.scene)
e = 1
scatter!(lscene, x[:], y[:], z[:], color = :black, markersize = 100.0, strokewidth = 0)

scatter!(lscene, xnew[end,:,5], ynew[end,:,5], znew[end,:,5], color = :red, markersize = 100)
display(fig)
=#
##

elementlist = [findelement([xC,yC,zC], newgrid[i]) for i in eachindex(newgrid)]

ξlist = zeros(length(elementlist), 3)
losslist = zeros(length(elementlist))
# Threads.@threads 
Threads.@threads for i in eachindex(elementlist)
    e = elementlist[i]
    println("iteration ", i)
    position = closure_position(x,y,z, e, grid, ihelper)
    xnew = newgrid[i][1]
    ynew = newgrid[i][2]
    znew = newgrid[i][3]
    loss = closure_cost(xnew, ynew, znew, position)
    tolin = reshape(1:length(x[:,1]), polynomialorders(grid) .+1)
    rx = reshape(x[:,e], polynomialorders(grid) .+ 1)
    ry = reshape(y[:,e], polynomialorders(grid) .+ 1)
    rz = reshape(z[:,e], polynomialorders(grid) .+ 1)
    minind = argmin((rx .- xnew) .^2 + (ry .- ynew) .^2 + (rz .- znew) .^2)
    ξᴳ = [ihelper.points[i][minind[i]] for i in 1:3]
    # println("before loss = ", loss(ξᴳ))
    result = optimize(loss, ξᴳ, BFGS(), Optim.Options(f_calls_limit = 50))
    ξlist[i,:] .= result.minimizer
    # abs(ξlist[i, 1]) > 1 + eps(10.0) ? println("error!") : nothing 
    # abs(ξlist[i, 2]) > 1 + eps(10.0) ? println("error!") : nothing
    # abs(ξlist[i, 3]) > 1 + eps(10.0) ? println("error!") : nothing
    # println("after loss = ", loss(ξlist[i,:]))
    losslist[i] = loss(ξlist[i,:])
end

##
rad(x,y,z) = sqrt(x^2 + y^2 + z^2)
lat(x,y,z) = asin(z/rad(x,y,z)) # ϕ ∈ [-π/2, π/2] 
lon(x,y,z) = atan(y,x) # λ ∈ [-π, π) 

oldr = sin.(lon.(x,y,z)) # sqrt.(x .^2  + y .^ 2 + z .^2)
newr = zeros(size(newgrid))

Threads.@threads for i in eachindex(elementlist)
    e = elementlist[i]
    ξ⃗ = ξlist[i,:]
    newr[i] = lagrange_eval(reshape(oldr[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
end

##
heatmap(λ, ϕ, newr[1,:,:], colormap = :balance)
##
fig = Figure(resolution = (3156, 1074))

clims = [-1,1] # extrema(newr)

xnew = reshape([newgrid[i][1] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
ynew = reshape([newgrid[i][2] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
znew = reshape([newgrid[i][3] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))


n = 1
ax = fig[3:7, 3n-2:3n] = LScene(fig) # make plot area wider
surface!(ax, xnew[1,:,:], ynew[1,:,:], znew[1,:,:], color=newr[1,:,:], colormap=:balance, colorrange=clims,  shading = false)
rotate_cam!(ax.scene, (π/2, π/6, 0))
zoom!(ax.scene, (0, 0, 0), 5, false)
fig[2, 2 + 3*(n-1)] = Label(fig, "Sphere Plot", textsize = 50) # put names in center

display(fig)