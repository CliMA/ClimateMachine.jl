#
# Define output path string:
#
function IOstrings_outpath_name(problem_name, grid_resolution)

    ndim = length(grid_resolution)

    outpath_string = string(grid_resolution[1], "mx")
    for i = 2:ndim
        ds = grid_resolution[i]
        outpath_string = string(outpath_string, ds, "mx")
    end        
    OUTPATH = string("./output/",problem_name, "/", outpath_string,"_", randstring(6))

    #                                                                                                                                                                                                                                          
    # Create output directories to store diagnostics and vtks:
    #
    if (isdir(OUTPATH) == false)
        mkpath(OUTPATH)
    else
        #Move an existing directory to dir_previous if it already exists
        auxi = OUTPATH[1:end]
        previous_run = string(OUTPATH, "_previous");
        mv(OUTPATH, previous_run, force=true)
    end

    return OUTPATH
end

