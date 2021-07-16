function get_data(material::String, path::String)
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

get_data("seasalt", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/particle_data.txt")
get_data("dust", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/particle_data.txt")