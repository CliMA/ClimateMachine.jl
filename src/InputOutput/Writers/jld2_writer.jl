using FileIO
using JLD2

struct JLD2Writer <: AbstractWriter end

function write_data(jld::JLD2Writer, filename, dims, varvals, simtime)
    jldopen(full_name(jld, filename), "a+") do ds
        dimnames = collect(keys(dims))
        for di in 1:length(dimnames)
            ds["dim_$di"] = dimnames[di]
        end
        for (dn, dv) in dims
            ds[dn] = dv
        end
        ds["t"] = [1]
        for (vn, vv) in varvals
            ds[vn] = vv[2]
        end
        ds["simtime"] = [simtime]
    end
    return nothing
end
