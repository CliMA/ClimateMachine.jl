using NCDatasets

struct NetCDFWriter <: AbstractWriter end

function write_data(nc::NetCDFWriter, filename, dims, varvals, simtime)
    Dataset(full_name(nc, filename), "c") do ds
        for (dn, dv) in dims
            ds.dim[dn] = length(dv)
        end
        ds.dim["t"] = 1
        for (dn, dv) in dims
            defVar(ds, dn, dv, (dn,))
        end
        defVar(ds, "t", 1, ("t",))
        for (vn, vv) in varvals
            defVar(ds, vn, vv[2], vv[1])
        end
        defVar(ds, "simtime", [simtime], ("t",))
    end
    return nothing
end
