"""
    Writers

Abstracts writing dimensioned data so that output can be to a NetCDF
file or to a JLD2 file.

Currently, a single file per time of writing is used. Thus, a `time`
dimension is implicitly defined of length 1, with a value of the
current simulation time.

TODO: use an unlimited dimension for time and append?
"""

module Writers

export AbstractWriter, NetCDFWriter, JLD2Writer, write_data, full_name

abstract type AbstractWriter end

"""
    write_data(
        writer,
        filename,
        dims,
        varvals,
        simtime,
    )

Writes the specified dimension names, dimensions, axes, variable names
and variable values to a file. Specialized by every `Writer` subtype.

# Arguments:
# - `writer`: instance of a subtype of `AbstractWriter`.
# - `filename`: into which to write data (without extension).
# - `dims`: Dict of dimension name to 2-tuple of dimension values and Dict
#   of attributes.
# - `varvals`: Dict of variable name to 3-tuple of a k-tuple of dimension
#   names, k-dimensional array of values, and Dict of attributes.
#   variable name to k-dimensional array of values.
# - `simtime`: Current simulation time.
"""
function write_data end

"""
    full_name(
        writer,
        filename
    )

Appends the appropriate (based on `writer`) extension to the specified
filename.
"""
function full_name(writer::AbstractWriter, filename::String)
    if writer isa NetCDFWriter
        return string(filename, ".nc")
    elseif writer isa JLD2Writer
        return string(filename, ".jld2")
    else
        error("Unrecognized Writer $(writer)")
    end
end

include("netcdf_writer.jl")
include("jld2_writer.jl")

end
