"""
Text file format:
material, molar mass, osmotic coefficient, density, mass fraction, mass mixing ratio (?), dissociation
"""

open("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Shevali/data.txt", "r") do f
    line = []
    while !eof(f)
        r = readline(f)
        value = " "
        for index in 1:(length(r) - 1)
            value *= r[index]
            println(value)
            if (r[index + 1] == ',')
                append!(line, string(value))
                index += 1
            end
        end       
    end
    println(line)
end