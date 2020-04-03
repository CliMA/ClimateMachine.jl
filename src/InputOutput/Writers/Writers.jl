module Writers

export AbstractWriter, NetCDFWriter, JLD2Writer, write_data, full_name

abstract type AbstractWriter end

"""
    write_data(
        writer,
        filename,
        dims,
        varvals,
    )

Writes the specified dimension names, dimensions, axes, variable names
and variable values to a file. Specialized by every `Writer` subtype.

# Arguments:
# - `writer`: instance of a subtype of `AbstractWriter`.
# - `filename`: into which to write data (without extension).
# - `dims`: Dict of dimension name to axis.
# - `varvals`: Dict of variable name to array of values.
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
