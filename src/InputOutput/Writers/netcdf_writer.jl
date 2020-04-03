using NCDatasets

struct NetCDFWriter <: AbstractWriter end

function write_data(nc::NetCDFWriter, filename, dims, varvals)
    Dataset(full_name(nc, filename), "c") do ds
        for (dn, dv) in dims
            ds.dim[dn] = length(dv)
        end
        for (dn, dv) in dims
            defVar(ds, dn, dv, (dn,))
        end
        for (vn, vv) in varvals
            defVar(ds, vn, vv, collect(keys(dims)))
        end
    end
    return nothing
end
