ρ   = Q[:,1,:]
ρuˣ = Q[:,2,:]
ρuʸ = Q[:,3,:]
ρuᶻ = Q[:,4,:]
ρe = Q[:,5,:]
geo = rr * 9.8
γ = 1.4

p = (γ - 1) * (ρe - (ρuˣ .^2 + ρuʸ .^2 + ρuᶻ .^2 ) ./ (2 .* ρ) - ρ .* geo)

cr = sqrt.(x .^2 + y .^2) .+ eps(1e6)
uᶻᵒⁿᵃˡ = (-y .* ρuˣ .+ x .* ρuʸ ) ./ cr ./ ρ
uʳ = (x .* ρuˣ .+ y .* ρuʸ .+ z .* ρuᶻ) ./ rr ./ ρ
# [x*z, y*z, -(x^2 + y^2)] ./ ( norm([x, y, z]) * norm([x, y, 0]))
uᵐᵉʳ = (x .* z .* ρuˣ .+ y .* z .* ρuʸ - (x .^2 .+ y .^2) .* ρuᶻ ) ./ ( rr .* (cr) )
oldfield = uᶻᵒⁿᵃˡ

states = [uᶻᵒⁿᵃˡ, p, uᵐᵉʳ]
statenames = ["zonal velocity", "pressure", "meridional velocity"]
stateindex = collect(1:length(states))


fig = Figure(resolution = (1603, 986))
n = 1
ax = fig[2:4, 2:4] = LScene(fig)

x,y,z = coordinates(grid)
glpoints = polynomialorders(grid) .+ 1
# for the sphere
ne = (ddomain.resolution.elements.vertical, 6 * ddomain.resolution.elements.horizontal^2)
# expand to true higher dimensional form
rx, ry, rz = reshape.( (x,y,z), Ref((glpoints..., ne...)))

# pick a vertical level contour 
# k  = Node(1)  # GL point
# ev = Node(1) # vertical element

# Sliders
ijk_slider =
    Slider(fig, range = Int.(range(1, glpoints[3], length = glpoints[3])), startvalue = 1)
k = ijk_slider.value
e_slider =
    Slider(fig, range = Int.(range(1, ne[1], length = ne[1])), startvalue = 1)
ev = e_slider.value

upperclim_slider =
    Slider(fig, range = range(0, 1, length = 101), startvalue = 0.99)
upperclim_node = upperclim_slider.value
lowerclim_slider =
    Slider(fig, range = range(0, 1, length = 101), startvalue = 0.01)
lowerclim_node = lowerclim_slider.value

ox = @lift(rx[:,:,$k,$ev, :])
oy = @lift(ry[:,:,$k,$ev, :])
oz = @lift(rz[:,:,$k,$ev, :])
statenode = 1
statenode = Node(stateindex[1])
name = @lift(statenames[$statenode])

u = @lift(reshape(states[$statenode], (glpoints..., ne...)))
field = @lift($u[:,:,$k,$ev, :] )
# clims = @lift(extrema($u[broadcast(!, isnan.($u))]))

clims = @lift((
    quantile($field[broadcast(!, isnan.($field))], $lowerclim_node),
    quantile($field[broadcast(!, isnan.($field))], $upperclim_node),
))


for eh in 1:1:ne[2]
    surface!(ax, @lift(rx[:,:,$k,$ev, $eh]), @lift(ry[:,:,$k,$ev, $eh]), @lift(rz[:,:,$k,$ev, $eh]), color=@lift($u[:,:,$k,$ev, $eh]), colormap=:balance, colorrange  = clims, show_axis = false)
end

r_0 = sqrt(rx[1,1,1,1, 1]^2 + ry[1,1,1,1, 1]^2 + rz[1,1,1,1, 1]^2)
height = @lift(sqrt(rx[1,1,$k,$ev, 1]^2 + ry[1,1,$k,$ev, 1]^2 + rz[1,1,$k,$ev, 1]^2) .- r_0)
level_string = @lift( "height = " * @sprintf("%0.2f", $height) )
titlestring = @lift( "State = " * $name)
fig[1, 2:4] = Label(fig, titlestring , textsize = 50)


MHid = ddomain.numerical.MHid
MH = ddomain.numerical.vgeo[:, MHid, :]
rMH = reshape(MH, (glpoints..., ne...))

Mid = ddomain.numerical.Mid
M = ddomain.numerical.vgeo[:, Mid, :]
rM = reshape(M, (glpoints..., ne...))

v_average = @lift(sum($u .* rM ) / sum(rM) )
v_2moment= @lift(sum($u .^2 .* rM ) / sum(rM) )
v_variance = @lift($v_2moment  - $v_average^2)
s_average = @lift(sum($u[:,:,$k,$ev, :] .* rMH[:,:,$k,$ev, :] ) / sum(rMH[:,:,$k,$ev, :]) )
s_2moment = @lift(sum($u[:,:,$k,$ev, :] .^2 .* rMH[:,:,$k,$ev, :] ) / sum(rMH[:,:,$k,$ev, :]) )
s_variance = @lift($s_2moment - $s_average^2)

volumestring = @lift(
    "volume average = " *
    @sprintf("%0.2f", $v_variance)
)
volumevariancestring = @lift(
    "volume variance = " *
    @sprintf("%0.2f", $v_variance)
)

surfacestring = @lift(
    "surface average = " *
    @sprintf("%0.2f", $s_average)
)

surfacevariancestring = @lift(
    "surface variance = " *
    @sprintf("%0.2f", $s_variance)
)

fig[1,5] = Label(fig, "Vertical Level", textsize = 40)
fig[2, 5] = vgrid!(
    Label(fig, level_string, textsize = 30),
    Label(fig, "GL Point", width = nothing, textsize = 20),
    ijk_slider,
    Label(fig, "Element", width = nothing, textsize = 20),
    e_slider,
)

fig[3, 5] = vgrid!(
    Label(fig, "Statistics", textsize = 40),
    Label(fig, volumestring, width = nothing, textsize = 20),
    Label(fig, volumevariancestring, width = nothing, textsize = 20),
    Label(fig, surfacestring, width = nothing, textsize = 20),
    Label(fig, surfacevariancestring, width = nothing, textsize = 20),
)


statemenu = Menu(fig, options = zip(statenames, stateindex))
    on(statemenu.selection) do s
        statenode[] = s
    end


lowerclim_string = @lift(
    "quantile = " *
    @sprintf("%0.2f", $lowerclim_node) *
    ", value = " *
    @sprintf("%0.1e", $clims[1])
)
upperclim_string = @lift(
    "quantile = " *
    @sprintf("%0.2f", $upperclim_node) *
    ", value = " *
    @sprintf("%0.1e", $clims[2])
)

fig[1, 1] = Label(fig, "State Menu", textsize = 50, width = 400)
fig[2, 1] = vgrid!(
    statemenu,
    Label(fig, lowerclim_string, width = nothing),
    lowerclim_slider,
    Label(fig, upperclim_string, width = nothing),
    upperclim_slider,
)

fig[3, 1] = Label(fig, "Statistics", textsize = 50)
llscene = fig[4, 1] = Axis(
    fig,
    xlabel = @lift(statenames[$statenode]),
    xlabelcolor = :black,
    ylabel = "pdf",
    ylabelcolor = :black,
    xlabelsize = 40,
    ylabelsize = 40,
    xticklabelsize = 25,
    yticklabelsize = 25,
    xtickcolor = :black,
    ytickcolor = :black,
    xticklabelcolor = :black,
    yticklabelcolor = :black,
)
histogram_node = @lift(histogram($field, bins = 100))
xs = @lift($histogram_node[1])
ys = @lift($histogram_node[2])
pdf = GLMakie.AbstractPlotting.barplot!(
    llscene,
    xs,
    ys,
    color = :red,
    strokecolor = :red,
    strokewidth = 1,
)

@lift(GLMakie.AbstractPlotting.xlims!(llscene, extrema($xs)))
@lift(GLMakie.AbstractPlotting.ylims!(
    llscene,
    extrema($ys),
))
vlines!(
    llscene,
    @lift($clims[1]),
    color = :black,
)
vlines!(
    llscene,
    @lift($clims[2]),
    color = :black,
)
display(fig)