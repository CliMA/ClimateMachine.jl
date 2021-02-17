using GLMakie, Statistics, Printf

"""
visualize(states::AbstractArray; statenames = string.(1:length(states)), quantiles = (0.1, 0.99), aspect = (1,1,1), resolution = (1920, 1080), statistics = false, title = "Field = ")
# Description 
Visualize 3D states 
# Arguments
- `states`: Array{Array{Float64,3},1}. An array of arrays containing different fields
# Keyword Arguments
- `statenames`: Array{String,1}. An array of stringnames
- `aspect`: Tuple{Int64,Int64,Float64}. Determines aspect ratio of box for volumes
- `resolution`: Resolution of preliminary makie window
- `statistics`: boolean. toggle for displaying statistics 
# Return
- `scene`: Scene. A preliminary scene object for manipulation
"""
function visualize(states::AbstractArray; statenames = string.(1:length(states)), units = ["" for i in eachindex(states)], aspect = (1,1,1), resolution = (1920, 1080), statistics = false, title = "Field = ", bins = 300)
    # Create scene
    scene, layout = layoutscene(resolution = resolution)
    lscene = layout[2:4, 2:4] = LScene(scene) 
    width = round(Int, resolution[1] / 4) # make menu 1/4 of preliminary resolution

    # Create choices and nodes
    stateindex = collect(1:length(states))
    statenode = Node(stateindex[1])

    colorchoices = [:balance, :thermal, :dense, :deep, :curl, :thermometer]
    colornode = Node(colorchoices[1])

    if statistics
        llscene = layout[4,1] = Axis(scene, xlabel = @lift(statenames[$statenode] * units[$statenode]), 
                         xlabelcolor = :black, ylabel = "pdf", 
                         ylabelcolor = :black, xlabelsize = 40, ylabelsize = 40,
                         xticklabelsize = 25, yticklabelsize = 25,
                         xtickcolor = :black, ytickcolor = :black,
                         xticklabelcolor  = :black, yticklabelcolor = :black)
        layout[3, 1] = Label(scene, "Statistics", width = width, textsize = 50)
    end

    # x,y,z are for determining the aspect ratio of the box
    if (typeof(aspect) <: Tuple) & (length(aspect) == 3)
        x, y, z = aspect
    else
        x, y, z = size(states[1])
    end

    # Clim sliders
    upperclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.99)
    upperclim_node = upperclim_slider.value
    lowerclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.01)
    lowerclim_node = lowerclim_slider.value

    # Lift Nodes
    state = @lift(states[$statenode])
    statename = @lift(statenames[$statenode])
    clims = @lift((quantile($state[:], $lowerclim_node) , quantile($state[:], $upperclim_node)))
    cmap_rgb = @lift(to_colormap($colornode))
    titlename = @lift(title * $statename) # use padding and appropriate centering

    # Statistics
    if statistics
        histogram_node = @lift(histogram($state, bins = bins))
        xs = @lift($histogram_node[1])
        ys = @lift($histogram_node[2])
        pdf = GLMakie.AbstractPlotting.barplot!(llscene, xs, ys, color = :red, 
                        strokecolor = :red, 
                        strokewidth = 1)
        @lift(GLMakie.AbstractPlotting.xlims!(llscene, extrema($state)))
        @lift(GLMakie.AbstractPlotting.ylims!(llscene, extrema($histogram_node[2])))
        vlines!(llscene, @lift($clims[1]), color = :black, linewidth = width / 100)
        vlines!(llscene, @lift($clims[2]), color = :black, linewidth = width / 100)
    end

    # Volume Plot 
    volume!(lscene, 0..x, 0..y, 0..z, state, 
            camera = cam3d!, 
            colormap = cmap_rgb, 
            colorrange = clims)
    # Camera
    cam = cameracontrols(scene.children[1])
    eyeposition = Float32[2, 2, 1.3]
    lookat = Float32[0.82, 0.82, 0.1]
    # Title
    supertitle = layout[1, 2:4] = Label(scene, titlename , textsize = 50, color = :black)
    

    # Menus
    statemenu = Menu(scene, options = zip(statenames, stateindex))
    on(statemenu.selection) do s
        statenode[] = s
    end

    colormenu = Menu(scene, options = zip(colorchoices, colorchoices))
    on(colormenu.selection) do s
        colornode[] = s
    end
    lowerclim_string = @lift("lower clim quantile = " *  @sprintf("%0.2f", $lowerclim_node) * ", value = " * @sprintf("%0.1e", $clims[1]))
    upperclim_string = @lift("upper clim quantile = " *  @sprintf("%0.2f", $upperclim_node) * ", value = " * @sprintf("%0.1e", $clims[2]))
    # depends on makie version, vbox for old, vgrid for new
    layout[2, 1] = vgrid!(
        Label(scene, "State", width = nothing),
        statemenu,
        Label(scene, "Color", width = nothing),
        colormenu,
        Label(scene, lowerclim_string, width = nothing),
        lowerclim_slider,
        Label(scene, upperclim_string, width = nothing),
        upperclim_slider,
    )
    layout[1,1] = Label(scene, "Menu", width = width, textsize = 50)

    # Modify Axis
    axis = scene.children[1][OldAxis] 
    # axis[:names][:axisnames] = ("↓ Zonal [m] ", "Meriodonal [m]↓ ", "Depth [m]↓ ")
    axis[:names][:axisnames] = ("↓", "↓ ", "↓ ")
    axis[:names][:align] = ((:left, :center), (:right, :center), (:right, :center))
    # need to adjust size of ticks first and then size of axis names
    axis[:names][:textsize] = (50.0, 50.0, 50.0)
    axis[:ticks][:textsize] = (00.0, 00.0, 00.0)
    # axis[:ticks][:ranges_labels].val # current axis labels
    xticks = collect(range(-0, aspect[1], length = 2))
    yticks = collect(range(-0, aspect[2], length = 6))
    zticks = collect(range(-0, aspect[3], length = 2))
    ticks = (xticks, yticks, zticks)
    axis[:ticks][:ranges] = ticks
    xtickslabels = [@sprintf("%0.1f", (xtick)) for xtick in xticks]
    xtickslabels[end] = "1e6"
    ytickslabels = ["", "south","", "", "north", ""]
    ztickslabels = [@sprintf("%0.1f", (xtick)) for xtick in xticks]
    labels = (xtickslabels, ytickslabels, ztickslabels)
    axis[:ticks][:labels] = labels

    display(scene)
    # Change the default camera position after the fact
    # note that these change dynamically as the plot is manipulated
    return scene
end


"""
histogram(array; bins = 100)
# Description
return arrays for plotting histogram
"""
function histogram(array; bins = minimum([100, length(array)]), normalize = true)
    tmp = zeros(bins)
    down, up = extrema(array)
    down, up = down == up ? (down-1, up+1) : (down, up) # edge case
    bucket = collect(range(down, up, length = bins+1))
    normalization = normalize ? length(array) : 1
    for i in eachindex(array)
        # normalize then multiply by bins
        val = (array[i] - down) / (up - down) * bins
        ind = ceil(Int, val)
        # handle edge cases
        ind = maximum([ind, 1])
        ind = minimum([ind, bins])
        tmp[ind] += 1/normalization
    end
    return (bucket[2:end] + bucket[1:end-1]) .* 0.5, tmp
end

# 2D visualization
function visualize(states::Array{Array{S, 2},1}; statenames = string.(1:length(states)), units = ["" for i in eachindex(states)], aspect = (1,1,1), resolution = (2412, 1158), title = "Zonal and Temporal Average of ", xlims = (0,1), ylims = (0,1), bins = 300) where {S}
    # Create scene
    scene, layout = layoutscene(resolution = resolution)
    lscene = layout[2:4, 2:4] = Axis(scene, xlabel = "South to North [m]", 
                         xlabelcolor = :black, ylabel = "Depth [m]", 
                         ylabelcolor = :black, xlabelsize = 40, ylabelsize = 40,
                         xticklabelsize = 25, yticklabelsize = 25,
                         xtickcolor = :black, ytickcolor = :black,
                         xticklabelcolor  = :black, yticklabelcolor = :black,
                         titlesize = 50) 
    width = round(Int, resolution[1] / 4) # make menu 1/4 of preliminary resolution

    # Create choices and nodes
    stateindex = collect(1:length(states))
    statenode = Node(stateindex[1])

    colorchoices = [:balance, :thermal, :dense, :deep, :curl, :thermometer]
    colornode = Node(colorchoices[1])

    interpolationlabels = ["contour", "heatmap"]
    interpolationchoices = [true, false]
    interpolationnode = Node(interpolationchoices[1])

    # Statistics
    llscene = layout[4,1] = Axis(scene, xlabel = @lift(statenames[$statenode] * " " * units[$statenode]), 
                    xlabelcolor = :black, ylabel = "pdf", 
                    ylabelcolor = :black, xlabelsize = 40, ylabelsize = 40,
                    xticklabelsize = 25, yticklabelsize = 25,
                    xtickcolor = :black, ytickcolor = :black,
                    xticklabelcolor  = :black, yticklabelcolor = :black)
    layout[3, 1] = Label(scene, "Statistics", width = width, textsize = 50)

    # Clim sliders
    upperclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.99)
    upperclim_node = upperclim_slider.value
    lowerclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.01)
    lowerclim_node = lowerclim_slider.value
   
    #ylims = @lift(range($lowerval, $upperval, length = $))
    # Lift Nodes
    state = @lift(states[$statenode])
    statename = @lift(statenames[$statenode])
    unit = @lift(units[$statenode])
    oclims = @lift((quantile($state[:], $lowerclim_node) , quantile($state[:], $upperclim_node)))
    cmap_rgb = @lift($oclims[1] < $oclims[2] ? to_colormap($colornode) : reverse(to_colormap($colornode)))
    clims = @lift($oclims[1] != $oclims[2] ? (minimum($oclims), maximum($oclims)) : (minimum($oclims)-1, maximum($oclims)+1))
    xlims = Array(range(xlims[1], xlims[2], length = 4)) #collect(range(xlims[1], xlims[2], length = size(states[1])[1]))
    ylims = Array(range(ylims[1], ylims[2], length = 4)) #@lift(collect(range($lowerval], $upperval, length = size($state)[2])))
    # newrange = @lift(range($lowerval, $upperval, length = 4))
    # lscene.yticks = @lift(Array($newrange))
    titlename = @lift(title * $statename) # use padding and appropriate centering
    layout[1, 2:4] = Label(scene, titlename, textsize = 50)
    # heatmap 
    heatmap1 = heatmap!(lscene, xlims, 
            ylims,
            state, interpolate = interpolationnode,
            colormap = cmap_rgb, colorrange = clims)


    # statistics
    histogram_node = @lift(histogram($state, bins = bins))
    xs = @lift($histogram_node[1])
    ys = @lift($histogram_node[2])
    pdf = GLMakie.AbstractPlotting.barplot!(llscene, xs, ys, color = :red, 
                    strokecolor = :red, 
                    strokewidth = 1)
    @lift(GLMakie.AbstractPlotting.xlims!(llscene, extrema($state)))
    @lift(GLMakie.AbstractPlotting.ylims!(llscene, extrema($histogram_node[2])))
    vlines!(llscene, @lift($clims[1]), color = :black, linewidth = width / 100)
    vlines!(llscene, @lift($clims[2]), color = :black, linewidth = width / 100)

    # Menus
    statemenu = Menu(scene, options = zip(statenames, stateindex))
    on(statemenu.selection) do s
        statenode[] = s
    end

    colormenu = Menu(scene, options = zip(colorchoices, colorchoices))
    on(colormenu.selection) do s
        colornode[] = s
    end

    interpolationmenu = Menu(scene, options = zip(interpolationlabels, interpolationchoices))
    on(interpolationmenu.selection) do s
        interpolationnode[] = s
    heatmap1 = heatmap!(lscene, xlims, 
            ylims,
            state, interpolate = s,
            colormap = cmap_rgb, colorrange = clims)
    end

    newlabel = @lift($statename * " " * $unit)
    cbar = Colorbar(scene, heatmap1, label = newlabel)
    cbar.width = Relative(1/3)
    cbar.height = Relative(5/6)
    cbar.halign = :center
    cbar.flipaxisposition = true
    # cbar.labelpadding = -350
    cbar.labelsize = 50

    lowerclim_string = @lift("clim quantile = " *  @sprintf("%0.2f", $lowerclim_node) * ", value = " * @sprintf("%0.1e", $clims[1]))
    upperclim_string = @lift("clim quantile = " *  @sprintf("%0.2f", $upperclim_node) * ", value = " * @sprintf("%0.1e", $clims[2]))

    # depends on makie version, vbox for old, vgrid for new
    layout[2, 1] = vgrid!(
        Label(scene, "State", width = nothing),
        statemenu,
        Label(scene, "plotting options", width = width, textsize = 30, padding = (0,0, 10, 0)),
        interpolationmenu,
        Label(scene, "Color", width = nothing),
        colormenu,
        Label(scene, lowerclim_string, width = nothing),
        lowerclim_slider,
        Label(scene, upperclim_string, width = nothing),
        upperclim_slider,
    )

    layout[2:4, 5] = vgrid!(
        Label(scene, "Color Bar", width = width/2, textsize = 50, padding = (25, 0, 0, 00)),
        cbar,
    )
    layout[1,1] = Label(scene, "Menu", width = width, textsize = 50)
    display(scene)
    return scene
end

function volumeslice(states::AbstractArray; statenames = string.(1:length(states)), units = ["" for i in eachindex(states)], aspect = (1, 1, 32/192), resolution = (2678, 1030), statistics = false, title = "Volume plot of ", bins = 300, statlabelsize = (20,20))
scene, layout = layoutscene(resolution = resolution)
volumescene = layout[2:4, 2:4] = LScene(scene)
menuwidth = round(Int, 350)
layout[1,1] = Label(scene, "Menu", width = menuwidth, textsize = 50)

slice_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.0)
slice_node = slice_slider.value

directionindex = [1, 2, 3]
directionnames = ["x-slice", "y-slice", "z-slice"]
directionnode = Node(directionindex[1])

stateindex = collect(1:length(states))
statenode = Node(stateindex[1])

layout[1, 2:4] = Label(scene, @lift(title * statenames[$statenode]), textsize = 50)

colorchoices = [:balance, :thermal, :dense, :deep, :curl, :thermometer]
colornode = Node(colorchoices[1])

state = @lift(states[$statenode])
statename = @lift(statenames[$statenode])
unit = @lift(units[$statenode])
nx = @lift(size($state)[1])
ny = @lift(size($state)[2])
nz = @lift(size($state)[3])
nr = @lift([$nx, $ny, $nz])

nslider = 100
xrange = range(0.00, aspect[1], length = nslider)
yrange = range(0.00, aspect[2], length = nslider)
zrange = range(0.00, aspect[3], length = nslider)
constx = collect(reshape(xrange, (nslider,1,1)))
consty = collect(reshape(yrange, (1,nslider,1)))
constz = collect(reshape(zrange, (1,1,nslider)))
matx = zeros(nslider,nslider,nslider)
maty = zeros(nslider,nslider,nslider)
matz = zeros(nslider,nslider,nslider)
matx .= constx
maty .= consty
matz .= constz
sliceconst = [matx, maty, matz]
planeslice = @lift(sliceconst[$directionnode])

upperclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.99)
upperclim_node = upperclim_slider.value
lowerclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.01)
lowerclim_node = lowerclim_slider.value

clims = @lift((quantile($state[:], $lowerclim_node) , quantile($state[:], $upperclim_node)))

volume!(volumescene, 0..aspect[1], 0..aspect[2], 0..aspect[3], state, overdraw = false, colorrange = clims, colormap = @lift(to_colormap($colornode)))


alpha_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.5)
alphanode = alpha_slider.value

slicecolormap = @lift(cgrad(:viridis, alpha = $alphanode))
v = volume!(volumescene, 0..aspect[1], 0..aspect[2], 0..aspect[3],
        planeslice, algorithm = :iso, isorange = 0.005, 
        isovalue = @lift($slice_node * aspect[$directionnode]),
        transparency = true, overdraw = false, visible = true, 
        colormap = slicecolormap, colorrange = [-1,0] )

# Volume histogram

layout[3, 1] = Label(scene, "Statistics", textsize = 50)
hscene = layout[4, 1] = Axis(scene, xlabel = @lift(statenames[$statenode] * " " * units[$statenode]), 
                    xlabelcolor = :black, ylabel = "pdf", 
                    ylabelcolor = :black, xlabelsize = 40, ylabelsize = 40,
                    xticklabelsize = statlabelsize[1], yticklabelsize = statlabelsize[2],
                    xtickcolor = :black, ytickcolor = :black,
                    xticklabelcolor  = :black, yticklabelcolor = :black)

histogram_node = @lift(histogram($state, bins = bins))
vxs = @lift($histogram_node[1])
vys = @lift($histogram_node[2])
pdf = GLMakie.AbstractPlotting.barplot!(hscene, vxs, vys, color = :red, 
                strokecolor = :red, 
                strokewidth = 1)

@lift(GLMakie.AbstractPlotting.xlims!(hscene, extrema($vxs))) 
@lift(GLMakie.AbstractPlotting.ylims!(hscene, extrema($vys)))
vlines!(hscene, @lift($clims[1]), color = :black, linewidth = menuwidth / 100)
vlines!(hscene, @lift($clims[2]), color = :black, linewidth = menuwidth / 100)


# Slice
sliceupperclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.99)
sliceupperclim_node = sliceupperclim_slider.value
slicelowerclim_slider = Slider(scene, range = range(0, 1, length = 101), startvalue = 0.01)
slicelowerclim_node = slicelowerclim_slider.value


slicexaxislabel = @lift(["y", "x", "x"][$directionnode])
sliceyaxislabel = @lift(["z", "z", "y"][$directionnode])

slicexaxis = @lift([[1,$ny], [1, $nx], [1,$nx]][$directionnode])
sliceyaxis = @lift([[1,$nz], [1, $nz], [1,$ny]][$directionnode])

slicescene = layout[2:4, 5:6] = Axis(scene, xlabel = slicexaxislabel, ylabel = sliceyaxislabel)

sliced_state1 = @lift( $state[round(Int, 1 + $slice_node * (size($state)[1]-1)), 1:size($state)[2], 1:size($state)[3]])
sliced_state2 = @lift( $state[1:size($state)[1], round(Int, 1 + $slice_node * (size($state)[2]-1)), 1:size($state)[3]])
sliced_state3 = @lift( $state[1:size($state)[1], 1:size($state)[2], round(Int, 1 + $slice_node * (size($state)[3]-1))]) 
sliced_states = @lift([$sliced_state1, $sliced_state2, $sliced_state3])
sliced_state = @lift($sliced_states[$directionnode]) 

oclims = @lift((quantile($sliced_state[:], $slicelowerclim_node) , quantile($sliced_state[:], $sliceupperclim_node)))
slicecolormapnode = @lift($oclims[1] < $oclims[2] ? to_colormap($colornode) : reverse(to_colormap($colornode)))
sliceclims = @lift($oclims[1] != $oclims[2] ? (minimum($oclims), maximum($oclims)) : (minimum($oclims)-1, maximum($oclims)+1))

heatmap1 = heatmap!(slicescene, slicexaxis, sliceyaxis, sliced_state, interpolate = true, colormap = slicecolormapnode, colorrange = sliceclims)

# Colorbar
newlabel = @lift($statename * " " * $unit)
cbar = Colorbar(scene, heatmap1, label = newlabel)
cbar.width = Relative(1/3)
# cbar.height = Relative(5/6)
cbar.halign = :left
# cbar.flipaxisposition = true
# cbar.labelpadding = -250
cbar.labelsize = 50

@lift(GLMakie.AbstractPlotting.xlims!(slicescene, extrema($slicexaxis))) 
@lift(GLMakie.AbstractPlotting.ylims!(slicescene, extrema($sliceyaxis)))

sliceindex = @lift([round(Int, 1 + $slice_node * ($nx-1)), round(Int, 1 + $slice_node * ($ny-1)), round(Int, 1 + $slice_node * ($nz-1))][$directionnode])
slicestring = @lift(directionnames[$directionnode] * " of " * statenames[$statenode] ) 
layout[1, 5:6] = Label(scene, slicestring, textsize = 50)


axis = scene.children[1][OldAxis] 
axis[:names][:axisnames] = ("↓", "↓ ", "↓ ")
axis[:names][:align] = ((:left, :center), (:right, :center), (:right, :center))
axis[:names][:textsize] = (50.0, 50.0, 50.0)
axis[:ticks][:textsize] = (00.0, 00.0, 00.0)


# Menus
statemenu = Menu(scene, options = zip(statenames, stateindex))
on(statemenu.selection) do s
    statenode[] = s
end

colormenu = Menu(scene, options = zip(colorchoices, colorchoices))
on(colormenu.selection) do s
    colornode[] = s
end


# Slice Statistics
layout[1, 7] = Label(scene, "Slice Menu", width = menuwidth, textsize = 50)
layout[3, 7] = Label(scene, "Slice Statistics", textsize = 50)
hslicescene = layout[4, 7] = Axis(scene, xlabel = @lift(statenames[$statenode] * " " * units[$statenode]), 
                    xlabelcolor = :black, ylabel = "pdf", 
                    ylabelcolor = :black, xlabelsize = 40, ylabelsize = 40,
                    xticklabelsize = statlabelsize[1], yticklabelsize = statlabelsize[2],
                    xtickcolor = :black, ytickcolor = :black,
                    xticklabelcolor  = :black, yticklabelcolor = :black)

slicehistogram_node = @lift(histogram($sliced_state, bins = bins))
xs = @lift($slicehistogram_node[1])
ys = @lift($slicehistogram_node[2])
pdf = GLMakie.AbstractPlotting.barplot!(hslicescene, xs, ys, color = :blue, 
                strokecolor = :blue, 
                strokewidth = 1)

@lift(GLMakie.AbstractPlotting.xlims!(hslicescene, extrema($xs))) 
@lift(GLMakie.AbstractPlotting.ylims!(hslicescene, extrema($ys)))
vlines!(hslicescene, @lift($sliceclims[1]), color = :black, linewidth = menuwidth / 100)
vlines!(hslicescene, @lift($sliceclims[2]), color = :black, linewidth = menuwidth / 100)

interpolationnames = ["contour", "heatmap"]
interpolationchoices = [true, false]
interpolationnode = Node(interpolationchoices[1])
interpolationmenu = Menu(scene, options = zip(interpolationnames, interpolationchoices))

on(interpolationmenu.selection) do s
    interpolationnode[] = s
    # hack
    heatmap!(slicescene, slicexaxis, sliceyaxis, sliced_state, interpolate = s, colormap = slicecolormapnode, colorrange = sliceclims)
end

directionmenu = Menu(scene, options = zip(directionnames, directionindex))

on(directionmenu.selection) do s
    directionnode[] = s
end

slicemenustring = @lift(directionnames[$directionnode] * " at index "  * string(round(Int, 1 + $slice_node * ($nr[$directionnode]-1)))) 
lowerclim_string = @lift("quantile = " *  @sprintf("%0.2f", $lowerclim_node) * ", value = " * @sprintf("%0.1e", $clims[1]))
upperclim_string = @lift("quantile = " *  @sprintf("%0.2f", $upperclim_node) * ", value = " * @sprintf("%0.1e", $clims[2]))
alphastring = @lift("Slice alpha = " * @sprintf("%0.2f", $alphanode))
layout[2, 1] = vgrid!(
    Label(scene, "State", width = nothing),
    statemenu,
    Label(scene, "Color", width = nothing),
    colormenu,
    Label(scene, "Slice Direction", width = nothing),
    directionmenu,
    Label(scene, alphastring, width = nothing),
    alpha_slider,
    Label(scene, slicemenustring, width = nothing),
    slice_slider,
    Label(scene, lowerclim_string, width = nothing),
    lowerclim_slider,
    Label(scene, upperclim_string, width = nothing),
    upperclim_slider,
)

slicelowerclim_string = @lift("quantile = " *  @sprintf("%0.2f", $slicelowerclim_node) * ", value = " * @sprintf("%0.1e", $sliceclims[1]))
sliceupperclim_string = @lift("quantile = " *  @sprintf("%0.2f", $sliceupperclim_node) * ", value = " * @sprintf("%0.1e", $sliceclims[2]))

layout[2,7] = vgrid!(
    Label(scene, "Contour Plot Type", width = nothing), 
    interpolationmenu,
    Label(scene, slicelowerclim_string, width = nothing),
    slicelowerclim_slider,
    Label(scene, sliceupperclim_string, width = nothing), 
    sliceupperclim_slider,
    cbar,
)

display(scene)
return scene
end