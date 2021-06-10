using NCDatasets
using Dates
using Statistics
using DelimitedFiles
start = DateTime(2016,03,01)
endtime = DateTime(2016,07,01)
n = 1
output_ln = []
output_dir = @__DIR__;
mydir = joinpath(output_dir, "data/lamont/arms_stamps")
for (root, dirs, files) in walkdir(mydir)
    filepaths = joinpath.(root, files)
    for filepath in filepaths
        println(filepath)
        datestring = filepath[127:134]# fix this
        
        println(datestring)
        date = DateTime(datestring,"yyyymmdd")
        if (date <= endtime ) & (date >= start)
            ds = Dataset(filepath)
            swc = ds["soil_specific_water_content_south"][:]
            if size(swc) == (6, 48)
                depths = reshape(repeat(ds["depth"][:], 48), (6,48))
                qc = ds["qc_soil_specific_water_content_south"][:]

                mask = qc.== 0
                
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
                output = reshape(append!([datestring], string.(sc)),(1, 7))
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


open("data/lamont/stamps_swc_depth_south.txt", "w") do io
    writedlm(io, output_ln)
end
 



#
south = readdlm("data/lamont/stamps_swc_depth_south.txt",'\t', String)
south_data = tryparse.(Float64, south[:, 2:end])
MS =south_data .== -9999
south_data[MS] .=0

west = readdlm("data/lamont/stamps_swc_depth_west.txt",'\t', String)
west_data = tryparse.(Float64, west[:, 2:end])
MW =west_data .== -9999
west_data[MW] .=0

east = readdlm("data/lamont/stamps_swc_depth_east.txt",'\t', String)
east_data = tryparse.(Float64, east[:, 2:end])
ME =east_data .== -9999
east_data[ME] .=0

num = east_data .+ south_data .+ west_data
denom  = 3.0 .- (Array(MW) .+Array(ME) .+ Array(MS))

swc = num ./ denom
open("data/lamont/stamps_swc_depth.txt", "w") do io
    writedlm(io, hcat(south[:,1], string.(swc)))
end

output_dir = @__DIR__

plot(31:1:122,swc[31:end,1], label = "Aggregated", title = "SWC (%), 5cm")
plot!(31:1:122,east_data[31:end,1], label = "East")
plot!(31:1:122,south_data[31:end,1], label = "South")
plot!(31:1:122,west_data[31:end,1], label = "West")
plot!(xticks = ([31, 61, 92, 122], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
savefig(joinpath(output_dir, "data/lamont/arms_stamps/layer1.png"))

plot(31:1:122,swc[31:end,2], label = "Aggregated", title = "SWC (%), 10cm")
plot!(31:1:122,east_data[31:end,2], label = "East")
plot!(31:1:122,south_data[31:end,2], label = "South")
plot!(31:1:122,west_data[31:end,2], label = "West")
plot!(xticks = ([31, 61, 92, 122], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
savefig(joinpath(output_dir, "data/lamont/arms_stamps/layer2.png"))


plot(31:1:122,swc[31:end,3], label = "Aggregated", title = "SWC (%), 20cm")
plot!(31:1:122,east_data[31:end,3], label = "East")
plot!(31:1:122,south_data[31:end,3], label = "South")
plot!(31:1:122,west_data[31:end,3], label = "West")
plot!(xticks = ([31, 61, 92, 122], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
savefig(joinpath(output_dir, "data/lamont/arms_stamps/layer3.png"))

plot(31:1:122,swc[31:end,4], label = "Aggregated", title = "SWC (%), 50cm")
plot!(31:1:122,east_data[31:end,4], label = "East")
plot!(31:1:122,south_data[31:end,4], label = "South")
plot!(31:1:122,west_data[31:end,4], label = "West")
plot!(xticks = ([31, 61, 92, 122], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
savefig(joinpath(output_dir, "data/lamont/arms_stamps/layer4.png"))

plot(31:1:122,swc[31:end,5], label = "Aggregated", title = "SWC (%), 75cm")
plot!(31:1:122,east_data[31:end,5], label = "East")
plot!(31:1:122,south_data[31:end,5], label = "South")
plot!(31:1:122,west_data[31:end,5], label = "West")
plot!(xticks = ([31, 61, 92, 122], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
savefig(joinpath(output_dir, "data/lamont/arms_stamps/layer5.png"))

plot(31:1:122,swc[31:end,6], label = "Aggregated", title = "SWC (%), 100cm")
plot!(31:1:122,east_data[31:end,6], label = "East")
plot!(31:1:122,south_data[31:end,6], label = "South")
plot!(31:1:122,west_data[31:end,6], label = "West")
plot!(xticks = ([31, 61, 92, 122], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
savefig(joinpath(output_dir, "data/lamont/arms_stamps/layer6.png"))
