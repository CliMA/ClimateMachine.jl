# we order the time dimension last because Julia is column-major (see:
# https://github.com/Alexander-Barth/NCDatasets.jl/issues/87#issuecomment-636098859)

using NCDatasets
using OrderedCollections

mutable struct NetCDFWriter <: AbstractWriter
    filename::Union{Nothing, String}

    NetCDFWriter() = new(nothing)
end

function full_name(writer::NetCDFWriter, filename = nothing)
    if !isnothing(filename)
        return filename * ".nc"
    else
        return writer.filename * ".nc"
    end
end

function init_data(nc::NetCDFWriter, filename, dims, vars)
    Dataset(full_name(nc, filename), "c") do ds
        # define spatial and time dimensions
        for (dn, (dv, da)) in dims
            defDim(ds, dn, length(dv))
        end
        defDim(ds, "time", Inf) # Inf sets UNLIMITED dimension

        # include dimensions as variables
        for (dn, (dv, da)) in dims
            defVar(ds, dn, dv, (dn,), attrib = da)
        end
        defVar(
            ds,
            "time",
            Float64,
            ("time",),
            attrib = OrderedDict(
                "units" => "seconds since 1900-01-01 00:00:00",
                "long_name" => "time",
            ),
        )

        # define variables
        for (vn, (vd, vt, va)) in vars
            defVar(ds, vn, vt, (vd..., "time"), attrib = va)
        end
    end
    nc.filename = filename
    return nothing
end

function append_data(nc::NetCDFWriter, varvals, simtime)
    Dataset(full_name(nc), "a") do ds
        timevar = ds["time"]
        t = length(timevar) + 1
        timevar[t] = simtime
        for (vn, vv) in varvals
            dsvar = ds[vn]
            dsvar[ntuple(_ -> Colon(), ndims(vv))..., t] = vv
        end
    end
    return nothing
end
