function data_to_file(path::String, value::String)
    open(path, "w") do file
        write(file, value)
    end
end