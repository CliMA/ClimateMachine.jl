"""
Text file format:
material, molar mass, osmotic coefficient, density, mass fraction, mass mixing ratio (?), dissociation
"""

open("/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/Shevali/data.txt", "r") do f
    line = 0
    while !eof(f)
    r = readline(f)
        s = read(r, Char)
        if (s == ',')
            println("ahe")
        else
            line += 1
            println("$line . $s")
        end
    end
end