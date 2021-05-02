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
    elements = (vertical = 5, horizontal = 10),
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
rlist = collect(range(minimum(rC), maximum(rC), length = 4))
r = [6.38465836e6] # collect(range(domain.radius,domain.radius + domain.height,length = 10))
λ = collect(range(0,2π, length = 360))
ϕ = collect(range(0 + π/9 , π - π/9, length = 43))
newgrid = [ [r[i] * cos(λ[j]) * sin(ϕ[k]), r[i] * sin(λ[j]) * sin(ϕ[k]) , -r[i] * cos(ϕ[k])] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]
rλϕ = [ϕ[k] for i in eachindex(r), j in eachindex(λ), k in eachindex(ϕ)]

# only do once
elementlist = findelements(cellcenters(grid), newgrid)
ξlist = findξlist(elementlist, newgrid, grid, ihelper, f_calls_limit = 100, outer_iterations = 2)
##
jlfile = jldopen("baroclinic_wave.jld2")
jlkeys = keys(jlfile["state"])

# sphereviz(Q, ddomain)
# closure_position(x,y,z, e, grid, ihelper)
# e = elementlist[1]
# lagrange_eval(f, ξ⃗, ihelper::InterpolationHelper)
# rf = reshape(oldfield[:,e], polynomialorders(grid).+1)
# lagrange_eval(rf, [-1,-1,0], ihelper)
# ρ = Q[:,1,:]
##
#=
for i in eachindex(elementlist)
    ff = (x .- newgrid[i][1]) .^ 2 + (y .- newgrid[i][2]) .^2 + (z .- newgrid[i][3]) .^2
    elementlist[i] = argmin(ff)[2]
end
=#
##
i = 7484

po = (7,7,4)
es = (5, 10^2 * 6)
rr = sqrt.( x .^2 + y .^2 + z .^2)
evolverr = reshape(rr,(po..., es...))
rstack = evolverr[1,1,:,:,1]
for i in eachindex(elementlist)
    pv = argmin(abs.(rstack .- norm(newgrid[i])))[1]
    ev = argmin(abs.(rstack .- norm(newgrid[i])))[2]
    reshx = reshape(x,(po..., es...))
    reshy = reshape(y,(po..., es...))
    reshz = reshape(z,(po..., es...))
    ff = (reshx[4,4,pv,ev,:] .- newgrid[i][1]) .^ 2 + (reshy[4,4,pv,ev,:] .- newgrid[i][2]) .^2 + (reshz[4,4,pv,ev,:] .- newgrid[i][3]) .^2
    #pi = argmin(ff)[1]
    #pj = argmin(ff)[2]
    eh = argmin(ff)[1]
    elementlist[i] = ev + 5 * (eh - 1)
end
##
i = 2351
e = elementlist[i]
newx, newy, newz = ξlist[i,:]
rx, ry, rz = ihelper.points[1], ihelper.points[2], ihelper.points[3]
ωx, ωy, ωz = ihelper.interpolation[1], ihelper.interpolation[2], ihelper.interpolation[3]
rf = reshape(oldfield[:,e], polynomialorders(grid).+1);

tmpx = reshape(x[:,e], polynomialorders(grid).+1);
tmpy = reshape(y[:,e], polynomialorders(grid).+1);
tmpz = reshape(z[:,e], polynomialorders(grid).+1);


lagrange_eval(tmpx, newx, newy, -0.4, rx, ry, rz,  ωx, ωy, ωz)
lagrange_eval(tmpy, newx, newy, -0.4, rx, ry, rz,  ωx, ωy, ωz)
lagrange_eval(tmpz, newx, newy, -0.4, rx, ry, rz,  ωx, ωy, ωz)

newgrid[i]

exact = 6.38465836e6 / 6.371e6

lagrange_eval(rf, newx, newy, -0.4, rx, ry, rz,  ωx, ωy, ωz)

lagrange_eval(rf, newx, newy, newz, rx, ry, rz,  ωx, ωy, ωz) - exact

##
i = 1
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
        lower = [-1,-1,-1] * 1.2 # allow a little extrapolation 
        upper = [1,1,1] * 1.2 # allow a little extrapolation 
        result = optimize(loss, lower, upper, ξᴳ, Fminbox(ConjugateGradient()), Optim.Options(f_calls_limit = 50, outer_iterations = 2))
        # result = optimize(loss, ξᴳ, ConjugateGradient(), Optim.Options(f_calls_limit = 100))
        # ξlist[i,:] .= result.minimizer
        # losslist[i] = loss(ξlist[i,:])

##

##
Q = jlfile["state"][jlkeys[end]]
oldfield = Q[:,1,:]

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

ax = fig[1:3,1:3] = LScene(fig) # make plot area wider
surface!(ax, xnew[1,:,:], ynew[1,:,:], znew[1,:,:], color=newfield[1,:,:], colormap= :balance, colorrange=clims,  shading = false, show_axis=false)
rotate_cam!(ax.scene, (π/2, π/6, 0))
zoom!(ax.scene, (0, 0, 0), 5, false)
fig[4,2] = Label(fig, "Sphere Plot", textsize = 50) # put names in center

ax2 = fig[2, 5:7] = LScene(fig)
heatmap!(ax2, λ, ϕ, newfield[1,:,:], colormap = :balance, interpolate = true, colorrange = clims)
fig[4, 6]  = Label(fig, "Lat-Lon Plot", textsize = 50)

display(fig)
##