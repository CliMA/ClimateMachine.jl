"""
   function create_interpolation_grid(xbnd, xres, thegrid)
Given boundaries of a domain, and a resolution in each direction, create 
an interpolation grid. This is where the interpolation functions for 
a set of variables will be evaluated.
"""
function create_interpolation_grid(xbnd, xres, thegrid)
    x1g = collect(range(xbnd[1, 1], xbnd[2, 1], step = xres[1]))
    x2g = collect(range(xbnd[1, 2], xbnd[2, 2], step = xres[2]))
    x3g = collect(range(xbnd[1, 3], xbnd[2, 3], step = xres[3]))
    intrp_brck = ClimateMachine.InterpolationBrick(thegrid, xbnd, x1g, x2g, x3g)
    return intrp_brck
end

"""
   function interpolate_variables(objects, brick)
Create an interpolation function from data and evaluate that
function on the interpolation brick passed.
"""
function interpolate_variables(objects, brick)
    i_objects = []
    for object in objects
        nvars = size(object.data, 2)
        i_object = Array{FT}(undef, brick.Npl, nvars)
        ClimateMachine.interpolate_local!(brick, object.data, i_object)
        push!(i_objects, i_object)
    end

    return i_objects
end
