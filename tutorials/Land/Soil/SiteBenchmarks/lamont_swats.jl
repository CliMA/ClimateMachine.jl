using NCDatasets
using Dates
using Statistics
using DelimitedFiles
start = DateTime(2016,04,01)
endtime = DateTime(2016,07,01)
n = 1
output_ln = []
output_dir = @__DIR__;
mydir = joinpath(output_dir, "data/lamont/arms_swat")
for (root, dirs, files) in walkdir(mydir)
    filepaths = joinpath.(root, files)
    for filepath in filepaths
        println(filepath)
        datestring = filepath[125:132]
        
        println(datestring)
        date = DateTime(datestring,"yyyymmdd")
        if (date <= endtime ) & (date >= start)
            ds = Dataset(filepath)
            swc = ds["watcont_w"][:]
            if size(swc) == (5, 24)
                depths = reshape(repeat(ds["depth"][:], 24), (5,24))
                mask = ds["qc_watcont_w"][:].== 0
                
                d = depths[mask]
                swc = swc[mask]
                sc =[]
                dlist = unique(depths)
                for depth in dlist
                    newmask = d .== depth
                    if sum(newmask) == 0
                        append!(sc, [-9999])
                    else
                        append!(sc,[mean(swc[newmask])])
                    end
                end
                output = reshape(append!([datestring], string.(sc)),(1, 6))
                if n == 1
                    global output_ln = output
                    global n = 2
                else
                    global output_ln = vcat(output_ln, output)
                end
                
            else
                println(string("not enough data on ",string(date)))
                
            end
            close(ds)
        end

    end
end


open("./data/lamont/swat_swc_depth.txt", "w") do io
    writedlm(io, output_ln)
end
 
