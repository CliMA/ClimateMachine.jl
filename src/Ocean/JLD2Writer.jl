module JLD2Writers

using JLD2

using ClimateMachine.Ocean.Domains: DiscontinuousSpectralElementGrid
using ClimateMachine.Ocean: current_step, current_time
using ..Fields: SpectralElementField

struct JLD2Writer{A, F, M, O}
    filepath :: F
    model :: M
    outputs :: O
    array_type :: A
end

function Base.show(io::IO, writer::JLD2Writer{A}) where A

    header =   "JLD2Writer{$A}"
    filepath = "    ├── filepath: $(writer.filepath)"
    outputs =  "    └── $(length(writer.outputs)) outputs: $(keys(ow.outputs))"

    print(io, header, '\n', filepath, '\n', outputs)

    return nothing
end

"""
    JLD2Writer(model, outputs=model.fields; filepath, array_type=Array, overwrite_existing=true)

Returns a utility for writing field output to JLD2 files, `overwrite_existing` file at `filepath`.

`write!(jld2_writer::JLD2Writer)` writes `outputs` to `filepath`, where `outputs` is either a `NamedTuple` or
`Dict`ionary of `fields` or functions of the form `output(model)`.

Field data is converted to `array_type` before outputting.
"""
function JLD2Writer(model, outputs=model.fields; filepath, array_type=Array, overwrite_existing=true)

    # Convert grid to CPU
    cpu_grid = DiscontinuousSpectralElementGrid(model.domain, array_type=array_type)

    # Initialize output
    overwrite_existing && isfile(filepath) && rm(filepath; force=true)

    file = jldopen(filepath, "a+")

    file["domain"] = model.domain
    file["grid"] = cpu_grid

    close(file)

    writer = JLD2Writer(filepath, model, outputs, array_type)

    write!(writer, first=true)
    
    return writer
end

initialize_output!(file, args...) = nothing

initialize_output!(file, field::SpectralElementField, name) =
    file["$name/meta/realelems"] = field.realelems

function write!(writer::JLD2Writer; first=false)

    model = writer.model
    filepath = writer.filepath
    outputs = writer.outputs

    # Add new data to file
    file = jldopen(filepath, "a+")

    N_output = first ? 0 : length(keys(file["times"]))

    step = current_step(model)
    time = current_time(model)

    file["times/$N_output"] = time
    file["steps/$N_output"] = step

    for (name, output) in zip(keys(outputs), values(outputs))
        first && initialize_output!(file, output, name)
        write_single_output!(file, output, name, N_output, writer)
    end

    close(file)

    return nothing
end

function write_field!(file, field, name, N_output, writer)
    data = convert(writer.array_type, field.data)
    file["$name/$N_output"] = data
    return nothing
end

write_single_output!(file, field::SpectralElementField, args...) =
    write_field!(file, field, args...)

function write_single_output!(file, output, name, N_output, writer)
    data = output(writer.model)
    file["$name/$N_output"] = data
    return nothing
end

struct OutputTimeSeries{F, N, D, G, T, S}
    filepath :: F
    name :: N
    domain :: D
    grid :: G
    times :: T
    steps :: S
end

function OutputTimeSeries(name, filepath)
    file = jldopen(filepath)

    domain, grid, times, steps = [nothing for i=1:4]

    try
        domain = file["domain"]
        grid = file["grid"]

        output_indices = keys(file["times"])

        times = [file["times/$i"] for i in output_indices]
        steps = [file["steps/$i"] for i in output_indices]

    catch err
        @warn "Could not build time series of $name from $filepath because $(sprint(showerror, err))"

    finally
        close(file)

    end

    return OutputTimeSeries(filepath, name, domain, grid, times, steps)
end

function Base.length(timeseries::OutputTimeSeries)
    file = jldopen(timeseries.filepath)

    timeseries_length = length(keys(file["times"]))

    close(file)

    return timeseries_length
end


function Base.getindex(timeseries::OutputTimeSeries, i)

    name = timeseries.name
    domain = timeseries.domain
    grid = timeseries.grid

    file = jldopen(timeseries.filepath)

    data = file["$name/$(i-1)"]
    realelems = file["$name/meta/realelems"]

    close(file)

    realdata = view(data, :, realelems)

    return SpectralElementField(domain, grid, realdata, data, realelems)
end

end # module
