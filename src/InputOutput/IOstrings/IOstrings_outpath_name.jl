#
# Define output path string:
#
# called by a driver right before run()
#
# ex. 
#    #Create unique output path directory:
#    OUTPATH = IOstrings_outpath_name(problem_name, grid_resolution)
#
#
function IOstrings_outpath_name(problem_name, grid_resolution)
    #
    # Arguments:
    # problem_name = "NAME"
    # grid_resolution[1]   = Δx
    # grid_resolution[2]   = Δy
    # grid_resolution[end] = Δz
    #
    ndim = length(grid_resolution)

    outpath_string = string(grid_resolution[1], "mx")
    for i = 2:ndim
        ds = grid_resolution[i]
        outpath_string = string(outpath_string, ds, "mx")
    end
    current_time = string(Dates.format(convert(Dates.DateTime, Dates.now()), Dates.dateformat"yyyymmdd_HHMMSS"))
    OUTPATH = string("./output/",problem_name, "/", outpath_string,"_", current_time)

    mkpath(OUTPATH)
    
    return OUTPATH
end

