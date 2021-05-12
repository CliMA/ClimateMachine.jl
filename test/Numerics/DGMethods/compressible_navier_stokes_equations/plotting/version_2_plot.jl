fig = Figure(resolution = (1086, 828))

ax  = fig[1:3,1:3] = LScene(fig) # make plot area wider
ax2 = fig[2, 5:7]  = LScene(fig)

rad(x,y,z) = sqrt(x^2 + y^2 + z^2)
lat(x,y,z) = asin(z/rad(x,y,z)) # ϕ ∈ [-π/2, π/2] 
lon(x,y,z) = atan(y,x) # λ ∈ [-π, π) 


function displayfig!(fig, ax, ax2, ddomain, Q)
    x,y,z = coordinates(ddomain)
    glpoints = polynomialorders(ddomain) .+ 1
    # for the sphere
    ne = (ddomain.resolution.elements.vertical, 6 * ddomain.resolution.elements.horizontal^2)
    # expand to true higher dimensional form
    rx, ry, rz = reshape.( (x,y,z), Ref((glpoints..., ne...)))
    # Sliders
    k = 3
    ev = 1

    ox = rx[:,:,k,ev, :]
    oy = ry[:,:,k,ev, :]
    oz = rz[:,:,k,ev, :]

    or = rad.(ox,oy,oz) 
    oϕ = lat.(ox,oy,oz)
    oλ = lon.(ox,oy,oz) 

    statenode = 1

    u = reshape(Q[:,statenode,:], (glpoints..., ne...))
    field = u[:,:,k,ev, :]
    clims = quantile.(Ref(field[:]), [0.01,0.99])


    for eh in 1:1:ne[2]
        surface!(ax, ox[:,:,eh], ry[:,:,k,ev, eh], rz[:,:,k,ev, eh], color= field[:,:,eh], colormap=:balance, colorrange  = clims, shading = false, show_axis = false)
        # surface!(ax, oλ[:,:,eh], oϕ[:,:,eh], or[:,:,eh] .* 0.0, color= field[:,:,eh], colormap=:balance, colorrange  = clims)
        a,b = extrema(oλ[:,:,eh])
        if (sign(a) == sign(b)) | (abs(b) < π/2) # take care of elements that go from -π to π
            surface!(ax2, oλ[:,:,eh], oϕ[:,:,eh], color= field[:,:,eh], colormap=:balance, colorrange  = clims, shading = false, interpolated = true, interpolate = true)
        end
    end


    fig[4,2] = Label(fig, "Sphere Plot", textsize = 50) # put names in center
    fig[4, 6]  = Label(fig, "Lat-Lon Plot", textsize = 50)
    display(fig)
end

Q = jlfile["state"][jlkeys[end]]
displayfig!(fig, ax, ax2, ddomain, Q)

##
iterations = collect(eachindex(jlkeys))
record(fig, "densityplot.mp4", iterations, framerate=10) do i
    Q = jlfile["state"][jlkeys[i]]
    displayfig!(fig, ax, ax2, ddomain, Q)
end

