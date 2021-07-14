"""
Text file format:
material, molar mass, osmotic coefficient, density, mass fraction, mass mixing ratio (?), dissociation
"""

function get_data(material::String, path::String)
    open(path, "r") do f
    println(path)
        line = []
        while !eof(f)
            r = readline(f)
            value = ""
            if r[1] == material
                for index in 2:length(r)
                    if r[index] != '"' && r[index] != ','
                        value *= r[index] 
                    end
                    if (r[index] == ',')
                        println(value)
                        push!(line, parse(Float64, value))
                        println(line)
                        # index += 1
                        value = "" 
                    end
                end
            end       
        end
        println(line)
    end
end

get_data("seasalt", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Shevali/data.txt")