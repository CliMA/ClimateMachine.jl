using ClimateMachine
using Optim
ClimateMachine.init()
##
# Restart File 
restart_file = jldopen("held_suarez_restart.jld2")
resolution = restart_file["grid"]["resolution"]
close(restart_file)
##
include("generic_interpolate_test.jl")
##

domain = AtmosDomain(
    radius = 6.371e6, 
    height = 30e3,
)

ddomain = DiscretizedDomain(
    domain;
    elements = (vertical = 5, horizontal = 9),
    polynomial_order = (vertical = 3, horizontal = 6),
    overintegration_order = (vertical = 0, horizontal = 0),
)

grid = ddomain.numerical
xC, yC, zC = cellcenters(grid)
x, y, z = coordinates(grid)
ihelper = InterpolationHelper(grid)
resolution = ddomain.resolution
## latitude is λ, longitude if ϕ
rr = sqrt.( x .^2 .+ y .^2 .+ z .^2)
rC = sqrt.(xC .^2 .+ yC .^2  .+ zC .^2) # extrema of rC are not contained in rr
rlist = collect(range(minimum(rr), maximum(rr), length = 8))
r = rlist[[1, 4, 6]] .+ 1e1 # [6.38465836e6] # collect(range(domain.radius,domain.radius + domain.height,length = 10))
λ = collect(range(0,2π, length = 180 * 2))
ϕ = collect(range(0 + π/9 , π - π/9, length = 43 * 2))
newgrid = [ [r[i] * cos(λ[j]) * sin(ϕ[k]), r[i] * sin(λ[j]) * sin(ϕ[k]) , -r[i] * cos(ϕ[k])] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]
rλϕ = [ϕ[k] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]

# only do once
# elementlist = findelements(cellcenters(grid), newgrid)

# findelements for sphere resolution.polynomial.vertical
po = (resolution.polynomial_order.horizontal, resolution.polynomial_order.horizontal, resolution.polynomial_order.vertical) .+ 1
es = (resolution.elements.vertical, resolution.elements.horizontal^2 * 6)
rr = sqrt.( x .^2 + y .^2 + z .^2)
elementlist = zeros(Int, length(newgrid));
evolverr = reshape(rr,(po..., es...));
rstack = evolverr[1,1,:,:,1];
for i in eachindex(elementlist)
    pv = argmin(abs.(rstack .- norm(newgrid[i])))[1]
    ev = argmin(abs.(rstack .- norm(newgrid[i])))[2]
    reshx = reshape(x,(po..., es...))
    reshy = reshape(y,(po..., es...))
    reshz = reshape(z,(po..., es...))
    greed = 4:4 # check over few indices, 4,4 is in the center for poly order 6
    ff = (reshx[greed,greed,pv,ev,:] .- newgrid[i][1]) .^ 2 + (reshy[greed,greed,pv,ev,:] .- newgrid[i][2]) .^2 + (reshz[greed,greed,pv,ev,:] .- newgrid[i][3]) .^2;
    eh = argmin(ff)[3]
    elementlist[i] = ev + 5 * (eh - 1)
end

# norm(checkele - elementlist)


##
ξlist =  greedyfindξlist(elementlist, newgrid, grid, ihelper, f_calls_limit = 50, outer_iterations = 2)

# ξlist = findξlist(elementlist, newgrid, grid, ihelper, f_calls_limit = 50, outer_iterations = 2)
##
jlfile = jldopen("newheldsuarez.jld2")
jlkeys = keys(jlfile["state"])
Q = jlfile["state"][jlkeys[1]]

# uᴿ = grabvelocities(Q, ddomain.numerical, Spherical())

# sphereviz(Q, ddomain)

##
ρ   = Q[:,1,:]
ρu¹ = Q[:,2,:]
ρu² = Q[:,3,:]
ρu³ = Q[:,4,:]
ρe = Q[:,5,:]
geo = rr * 9.8
γ = 1.4

p = (γ - 1) * (ρe - (ρu¹ .^2 + ρu² .^2 + ρu³ .^2 ) ./ (2 .* ρ) - ρ .* geo)

oldfield = Q[:,2,:]
tic = Base.time()
newfield = interpolatefield(elementlist, ξlist, newgrid, grid, ihelper, oldfield)
toc = Base.time()
println(toc-tic)
println("Time for finding oneself ", toc - tic, " seconds")
##

fig = Figure(resolution = (1086, 828))

level = 3

clims = quantile.(Ref(newfield[level,:,:][:]), [0.01,0.99])


xnew = reshape([newgrid[i][1] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
ynew = reshape([newgrid[i][2] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
znew = reshape([newgrid[i][3] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))

fieldslice = newfield[level,:,:] .- mean(newfield[level,:,:], dims = 2)
clims = quantile.(Ref(fieldslice[:]), [0.01,0.99])
ax = fig[1:3,1:3] = LScene(fig) # make plot area wider
surface!(ax, xnew[1,:,:], ynew[1,:,:], znew[1,:,:], color=fieldslice, colormap= :balance, colorrange=clims,  shading = false, show_axis=false)
rotate_cam!(ax.scene, (π/2, π/6, 0))
zoom!(ax.scene, (0, 0, 0), 5, false)
fig[4,2] = Label(fig, "Sphere Plot", textsize = 50) # put names in center

ax2 = fig[2, 5:7] = LScene(fig)
heatmap!(ax2, λ, ϕ, fieldslice, colormap = :balance, interpolate = true, colorrange = clims)
fig[4, 6]  = Label(fig, "Lat-Lon Plot", textsize = 50)

display(fig)

##
iterations = collect(eachindex(jlkeys))
record(fig, "xvelocity.mp4", iterations, framerate=10) do i
    Q = jlfile["state"][jlkeys[i]]

    ρ   = Q[:,1,:]
    ρu¹ = Q[:,2,:]
    ρu² = Q[:,3,:]
    ρu³ = Q[:,4,:]
    ρe = Q[:,5,:]
    geo = rr * 9.8
    γ = 1.4
    p = (γ - 1) * (ρe - (ρu¹ .^2 + ρu² .^2 + ρu³ .^2 ) ./ (2 .* ρ) - ρ .* geo)

    oldfield = Q[:,2,:]
    newfield = interpolatefield(elementlist, ξlist, newgrid, grid, ihelper, oldfield)
    # clims = extrema(newfield)
    fieldslice = newfield[level,:,:] .- mean(newfield[level,:,:], dims = 2)
    surface!(ax, xnew[1,:,:], ynew[1,:,:], znew[1,:,:], color= fieldslice, colormap= :balance, colorrange=clims,  shading = false, show_axis=false)
    heatmap!(ax2, λ, ϕ, fieldslice, colormap = :balance, interpolate = true, colorrange = clims)
end