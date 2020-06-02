using FileIO
using JLD2

struct JLD2Writer <: AbstractWriter end

function write_data(jld::JLD2Writer, filename, dims, varvals, simtime)
    jldopen(full_name(jld, filename), "a+") do ds
        dimnames = collect(keys(dims))
        for di in 1:length(dimnames)
            ds["dim_$di"] = dimnames[di]
        end
        ds["dim_time"] = "time"
        for (dn, (dv, _)) in dims
            ds[dn] = dv
        end
        ds["time"] = [1]
        ds["simtime"] = [simtime]
        for (vn, (_, vv, _)) in varvals
            ds[vn] = vv
        end
    end
    return nothing
end
