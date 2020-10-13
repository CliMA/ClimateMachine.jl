"""
loop through all subdirectories of cwd
    loop through all .jl files to get their prefixes
    open corresponding .mem files and read everything
        record in dict the file and line number, get allocation
    compare the resulting dicts for forward diff/AD
"""
allocDictFD = Dict()
allocDictAD = Dict()
function loop(dict, suffix)
    for (root, dirs, files) in walkdir(joinpath(pwd(),"src"))
        for f in joinpath.(root, files) # files is a Vector{String}, can be empty
            if endswith(f, suffix)
                open(f) do file
                    for (i,line) in enumerate(eachline(file))
                        l = lstrip(line)
                        if l[1] != '-' # || l[1] != '0'
                            dict[string(f, ':', i)] = parse(Int64, split(l)[1])
                        end
                    end
                end
            end
        end
    end
end
loop(allocDictAD, ".34352.mem")
#loop(allocDictFD, ".?????.mem")

# FD: 16.189625175, 1668404937, 0.477505845, Base.GC_Diff(1668404937, 3218, 34, 30167244, 5119, 0, 477505845, 13, 0)
# AD: 21.785127897, 2653736782, 0.678276927, Base.GC_Diff(2653736782, 3171, 70, 50396953, 8086, 0, 678276927, 24, 0)

# FD: 73.610377481, 3554536293, 0.844688455, Base.GC_Diff(3554536293, 43036, 34, 38930882, 4522, 0, 844688455, 27, 0)
# AD: 92.269193670, 4322492547, 1.351398259, Base.GC_Diff(4322492547, 37409, 70, 58513905, 8202, 0, 1351398259, 42, 1)

"""
@withprogress name="test" begin
           count = 1
           lastfrac = 0.0
           while count < last
               @info "hello"
               count += 1
               sleep(.1)
               frac = count / last
               if frac - lastfrac > .02
                   @logprogress frac
                   lastfrac = frac
               end
           end
       end
"""