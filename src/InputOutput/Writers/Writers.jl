"""
    Writers

Abstracts writing dimensioned data so that output can be to a NetCDF
file or another file format.
"""

module Writers

export AbstractWriter, NetCDFWriter, full_name, init_data, append_data

abstract type AbstractWriter end

"""
    full_name(
        writer,
        filename,
    )

Appends the appropriate (based on `writer`) extension to the specified
filename.
"""
function full_name end

"""
    init_data(
        writer,
        filename,
        dims,
        vars,
    )

Creates the specified file, initializing it with the specified dimension
information. An unlimited `time` dimension is implicitly created. The
specified variables are also defined. This function must be called before
`append_data()`. Specialized by every `Writer` subtype.

# Arguments:
# - `writer`: instance of a subtype of `AbstractWriter`.
# - `filename`: into which to write data (without extension).
# - `dims`: Dict of dimension name to 2-tuple of dimension values and Dict
#   of attributes.
# - `vars`: Dict of variable name to 3-tuple of a k-tuple of dimension
#   names, variable type, and Dict of attributes.
"""
function init_data end

"""
    append_data(
        writer,
        filename,
        varvals,
        simtime,
    )

Appends the specified variables to the specified file. The file must have
been previously created with `init_data()`. `simtime` is appended to the
`time` dimension variable. Specialized by every `Writer` subtype.

# Arguments:
# - `writer`: instance of a subtype of `AbstractWriter`.
# - `filename`: into which to write data (without extension).
# - `varvals`: Dict of variable name to k-dimensional array of values.
# - `simtime`: Current simulation time.
"""
function append_data end

include("netcdf_writer.jl")

end
