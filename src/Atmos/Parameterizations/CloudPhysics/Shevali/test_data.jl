"""
Text file format:
material, molar mass, osmotic coefficient, density, mass fraction, mass mixing ratio (?), dissociation
"""

function get_data(material::String, path::String)
    line = []
    open(path, "r") do f
        while !eof(f)
            r = readline(f)
            value = ""
            for index in 1:length(r)
                if r[index] != '"' && r[index] != ','
                    value *= r[index] 
                end
                if (r[index] == ',' || index == length(r))
                    if (length(line) == 0)
                        if (value != material)
                            continue
                        end
                    end
                    push!(line, value)
                    value = "" 
                end
            end       
        end
        println(line)
    end
    for values in 2:length(line)
        line[values] = parse(Float64, line[values])
    end
    println(line[2:length(line)])
    return line[2:length(line)]
    
end

get_data("seasalt", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Shevali/data.txt")