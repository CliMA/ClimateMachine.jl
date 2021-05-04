using ClimateMachine
using Optim
ClimateMachine.init()
##
include("generic_interpolate_test.jl")
##

domain = AtmosDomain(
    radius = 6.371e6, 
    height = 30e3,
)

ddomain = DiscretizedDomain(
    domain;
    elements = (vertical = 5, horizontal = 7),
    polynomial_order = (vertical = 3, horizontal = 6),
    overintegration_order = (vertical = 0, horizontal = 0),
)

grid = ddomain.numerical
xC, yC, zC = cellcenters(grid)
x, y, z = coordinates(grid)
ihelper = InterpolationHelper(grid)

## latitude is λ, longitude if ϕ
rr = sqrt.( x .^2 .+ y .^2 .+ z .^2)
rC = sqrt.(xC .^2 .+ yC .^2  .+ zC .^2)
rlist = collect(range(minimum(rC), maximum(rC), length = 8))
r = rlist # [6.38465836e6] # collect(range(domain.radius,domain.radius + domain.height,length = 10))
λ = collect(range(0,2π, length = 90))
ϕ = collect(range(0 + π/9 , π - π/9, length = 43))
newgrid = [ [r[i] * cos(λ[j]) * sin(ϕ[k]), r[i] * sin(λ[j]) * sin(ϕ[k]) , -r[i] * cos(ϕ[k])] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]
rλϕ = [ϕ[k] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]

# only do once
# elementlist = findelements(cellcenters(grid), newgrid)

# findelements for sphere 
po = convention(ddomain.resolution.polynomial_order, Val(3))  .+1
es = (ddomain.resolution.elements.vertical, ddomain.resolution.elements.horizontal^2 * 6)
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
    greed = 4:4 # check over few indices
    ff = (reshx[greed,greed,pv,ev,:] .- newgrid[i][1]) .^ 2 + (reshy[greed,greed,pv,ev,:] .- newgrid[i][2]) .^2 + (reshz[greed,greed,pv,ev,:] .- newgrid[i][3]) .^2;
    pi = argmin(ff)[1]
    pj = argmin(ff)[2]
    eh = argmin(ff)[3]
    elementlist[i] = ev + 5 * (eh - 1)
end

norm(checkele - elementlist)


ξlist = findξlist(elementlist, newgrid, grid, ihelper, f_calls_limit = 50, outer_iterations = 2)
##
jlfile = jldopen("downloadhs.jld2")
jlkeys = keys(jlfile["state"])
Q = jlfile["state"][jlkeys[end]]
# sphereviz(Q, ddomain)
fig = sphereviz(Q, ddomain)
seconds = 30
fps = 30
frames = round(Int, fps * seconds )
record(fig.scene, pwd() * "/held_suarez.mp4"; framerate = fps) do io
    for i = 1:frames
        sleep(1/fps)
        recordframe!(io)
    end
end
##

oldfield = Q[:,4,:]

# oldfield = sqrt.(x .^2 .+ y .^2 .+ z .^2) / 6.371e6 

tic = Base.time()
newfield = interpolatefield(elementlist, ξlist, newgrid, grid, ihelper, oldfield)
toc = Base.time()
println(toc-tic)
println("Time for finding oneself ", toc - tic, " seconds")
##
# newfield = rλϕ
fig = Figure(resolution = (1086, 828))

clims = quantile.(Ref(newfield[:]), [0.01,0.99])
# clims = extrema(newfield)

xnew = reshape([newgrid[i][1] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
ynew = reshape([newgrid[i][2] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))
znew = reshape([newgrid[i][3] for i in eachindex(newgrid)], (length(r),length(λ),length(ϕ)))

fieldslice = newfield[end,:,:]
ax = fig[1:3,1:3] = LScene(fig) # make plot area wider
surface!(ax, xnew[1,:,:], ynew[1,:,:], znew[1,:,:], color=fieldslice, colormap= :balance, colorrange=clims,  shading = false, show_axis=false)
rotate_cam!(ax.scene, (π/2, π/6, 0))
zoom!(ax.scene, (0, 0, 0), 5, false)
fig[4,2] = Label(fig, "Sphere Plot", textsize = 50) # put names in center

ax2 = fig[2, 5:7] = LScene(fig)
heatmap!(ax2, λ, ϕ, fieldslice, colormap = :balance, interpolate = true, colorrange = clims)
fig[4, 6]  = Label(fig, "Lat-Lon Plot", textsize = 50)

display(fig)
