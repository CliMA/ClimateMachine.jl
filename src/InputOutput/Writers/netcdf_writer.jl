using NCDatasets
using OrderedCollections

struct NetCDFWriter <: AbstractWriter end

function write_data(nc::NetCDFWriter, filename, dims, varvals, simtime)
    Dataset(full_name(nc, filename), "c") do ds
        ds.dim["time"] = 1
        for (dn, (dv, da)) in dims
            ds.dim[dn] = length(dv)
        end
        defVar(
            ds,
            "time",
            [simtime],
            ("time",),
            attrib = OrderedDict(
                "units" => "seconds",
                "long_name" => "simulation time",
            ),
        )
        for (dn, (dv, da)) in dims
            defVar(ds, dn, dv, (dn,), attrib = da)
        end
        for (vn, (vd, vv, va)) in varvals
            defVar(ds, vn, vv, vd, attrib = va)
        end
    end
    return nothing
end
