using NCDatasets

struct NetCDFWriter <: AbstractWriter end

function write_data(nc::NetCDFWriter, filename, dims, varvals, simtime)
    Dataset(full_name(nc, filename), "c") do ds
        for (dn, dv) in dims
            ds.dim[dn] = length(dv)
        end
        ds.dim["time"] = 1
        for (dn, dv) in dims
            defVar(ds, dn, dv, (dn,))
        end
        defVar(ds, "time", 1, ("time",))
        for (vn, vv) in varvals
            defVar(ds, vn, vv, collect(keys(dims)))
        end
        defVar(ds, "simtime", [simtime], ("time",))
    end
    return nothing
end
