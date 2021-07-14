"""
Text file format:
material, molar mass, osmotic coefficient, density, mass fraction, mass mixing ratio (?), dissociation
"""

function get_data(material::String, path::String)
    line = []
    open(path, "r") do f
        while !eof(f)
            r = eachline(f)
            println("THIS IS R")
            println(r)
            value = ""
            for index in 1:length(r)
                if r[index] != '"' && r[index] != ','
                    value *= r[index] 
                end
                if (r[index] == ',' || index == length(r))
                    if (length(line) == 0)
                        if (value != material)
                            continue
                        else
                            push!(line, value)
                            value = "" 
                        end
                    end
                    
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

function get_data_2(material::String, path::String)
    line = []
    for r in eachline(path)
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
        if (length(line) != 0)
            break
        end
    end
    for values in 2:length(line)
        line[values] = parse(Float64, line[values])
    end
    println(line[2:length(line)])
    return line[2:length(line)]
end

get_data_2("seasalt", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Shevali/data.txt")
get_data_2("dust", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Shevali/data.txt")