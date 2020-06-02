using FileIO
using JLD2

struct JLD2Writer <: AbstractWriter end

function write_data(jld::JLD2Writer, filename, dims, varvals, simtime)
    jldopen(full_name(jld, filename), "a+") do ds
        ds["dim_1"] = "time"
        dimnames = collect(keys(dims))
        for di in 1:length(dimnames)
            ds["dim_$(di + 1)"] = dimnames[di]
        end
        for (dn, (dv, _)) in dims
            ds[dn] = dv
        end
        ds["time"] = [simtime]
        for (vn, (_, vv, _)) in varvals
            ds[vn] = vv
        end
    end
    return nothing
end
