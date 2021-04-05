    # Clim sliders
x,y,z = coordinates(grid)
ijksize, esize = size(x)


fig = Figure()
sc = fig[1:3,1:3]
scene = scatter(sc, x[:], y[:], z[:], markersize = 4, show_axis=false)


ijk_slider =
    Slider(fig, range = Int.(range(1, ijksize, length = ijksize)), startvalue = 1)
ijk = ijk_slider.value
e_slider =
    Slider(fig, range = Int.(range(1, esize, length = esize)), startvalue = 1)
e = e_slider.value

arrowsize = 0.5
pts =    @lift(Point3f0(getposition(grid, $ijk, $e)[:]))
ptdir1 = @lift(Point3f0(getjacobian(grid, $ijk, $e)[1,:]/norm(getjacobian(grid, $ijk, $e)[1,:])) )
ptdir2 = @lift(Point3f0(getjacobian(grid, $ijk, $e)[2,:]/norm(getjacobian(grid, $ijk, $e)[2,:])) )
ptdir3 = @lift(Point3f0(getjacobian(grid, $ijk, $e)[3,:]/norm(getjacobian(grid, $ijk, $e)[3,:])) )

tpts = @lift( [$pts, $pts, $pts] )
tptdir = @lift(arrowsize .* [$ptdir1, $ptdir2, $ptdir3] )

ptdir4 = @lift(Point3f0(inv(getjacobian(grid, $ijk, $e))[:,1]/norm(inv(getjacobian(grid, $ijk, $e))[:,1])))
ptdir5 = @lift(Point3f0(inv(getjacobian(grid, $ijk, $e))[:,2]/norm(inv(getjacobian(grid, $ijk, $e))[:,2])))
ptdir6 = @lift(Point3f0(inv(getjacobian(grid, $ijk, $e))[:,3]/norm(inv(getjacobian(grid, $ijk, $e))[:,3])))

tptdir2 = @lift(arrowsize .*[$ptdir4, $ptdir5, $ptdir6])

# scatter!(scene.figure[1,1], x[edges], y[edges], z[edges], color = :red, markersize = 30)
GLMakie.arrows!(sc, tpts, tptdir, arrowsize = 0.05, linecolor = :blue, arrowcolor = :darkblue, linewidth = 10)
GLMakie.arrows!(sc, tpts, tptdir2, arrowsize = 0.05, linecolor = :red, arrowcolor = :darkred, linewidth = 10)

n = elements.horizontal

r = range(-1,1,length=n+1)

xa = zeros(n+1,n+1)
xb = zeros(n+1,n+1)
xc = zeros(n+1,n+1)

xa .= r
xb .= r'
xc .= 1

a = [cubedshellwarp(a,b,c)[1] for (a,b,c) in zip(xa,xb,xc)]
b = [cubedshellwarp(a,b,c)[2] for (a,b,c) in zip(xa,xb,xc)]
c = [cubedshellwarp(a,b,c)[3] for (a,b,c) in zip(xa,xb,xc)]


lw = 10
wireframe!(sc,
           a,b,c,
           show_axis=false,
           linewidth=lw)
wireframe!(sc,
           a,b,-c,
           show_axis=false,
           linewidth=lw)
wireframe!(sc,
           c,a,b,
           show_axis=false,
           linewidth=lw)
wireframe!(sc,
           -c,a,b,
           show_axis=false,
           linewidth=lw)
wireframe!(sc,
           b,c,a,
           show_axis=false,
           linewidth=lw)
wireframe!(sc,
           b,-c,a,
           show_axis=false,
           linewidth=lw)

fig[1, 4] = vgrid!(
    Label(fig, "GL Point", width = nothing),
    ijk_slider,
    Label(fig, "Element", width = nothing),
    e_slider,
)

display(fig)