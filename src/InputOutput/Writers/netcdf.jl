using NCDatasets

"""
    write_data(filename::String,
               dimnames::Tuple,
               dims::Tuple,
               axes::Tuple,
               varnames::Tuple,
               varvals::Array)

Writes dimensions, axes and variables to a NetCDF file. Supports any number
of dimensions.
"""
function write_data(filename, dimnames, dims, axes, varnames, varvals)
    nd = length(dimnames)
    nv = length(varnames)
    clns = [Colon() for _ in 1:nd]
    Dataset(filename, "c") do ds
        for n in 1:nd
            ds.dim[dimnames[n]] = dims[n]
        end
        for n in 1:nd
            defVar(ds, dimnames[n], axes[n], (dimnames[n],))
        end
        for v in 1:nv
            defVar(ds, varnames[v], varvals[clns...,v], dimnames)
        end
    end
    return nothing
end
