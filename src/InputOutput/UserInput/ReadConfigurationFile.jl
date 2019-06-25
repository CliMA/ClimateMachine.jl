module ReadConfigurationFile

#import Base
export read_configuration_file

using MPI
using CLIMA
using Dierckx
using DelimitedFiles
using Logging, Printf, Dates

if haspkg("CuArrays")
    using CUDAdrv
    using CUDAnative
    using CuArrays
    CuArrays.allowscalar(false)
    const ArrayType = CuArray
else
    const ArrayType = Array
end
macro datatype(str); :($(Symbol(str))); end


function read_configuration_file()

    local DInt
    local DFloat

    split_chars = ["=", ",", ";", "->", "_"]
    filein      = open(joinpath(@__DIR__, "./configuration.ini"))
    
    #Detect precision first of all:
    for line in eachline(filein)
        
        r_split = Regex( join(split_chars, "|") ) 
        words  = split( line, r_split )
        
        if uppercase(strip(words[1])) == "PRECISION"
            precision = uppercase(strip(words[2]))
            
            if precision == "DOUBLE"

                DFloat = @datatype(Float64)
                DInt   = @datatype(Int64)
                @info @sprintf """ Precision: %s, %s, %s """ precision DFloat DInt
                break 
            elseif precision == "SINGLE"            
                DFloat = @datatype(Float32)
                DInt   = @datatype(Int32)
                @info @sprintf """ Precision: %s, %s, %s """ precision DFloat DInt
                break 
                
            elseif precision == "HALF" 
                DFloat = @datatype(Float16)
                DInt   = @datatype(Int16)              
                @info @sprintf """ Precision: %s, %s, %s """ precision DFloat DInt
                break 
            end
            
        end
    end
    
    #Read in everything else:
    for line in eachline(filein)

        r_split = Regex( join(split_chars, "|") ) 
        words  = split( line, r_split )
        
        if uppercase(strip(words[1])) == "PROBLEM"
            problem = words[2] 
            @info @sprintf """ Problem:               %s""" problem            
        end
        if uppercase(strip(words[1])) == "GEOMETRY"
            geometry = words[2]
            @info @sprintf """ Geometry:              %s""" geometry           
        end       
        if uppercase(strip(words[1])) == "NSD"
            nsd = parse(DInt, words[2])
            @info @sprintf """ Space dimensions:       %d""" nsd
        end
        if uppercase(strip(words[1])) == "SGS"
            sgs = words[2]
            cs  = parse(DFloat, words[3])
            @info @sprintf """ Stabilization model:   %s""" sgs
            @info @sprintf """ Coefficient:            %.6f""" cs
        end
        if uppercase(strip(words[1])) == "ΔX"
            Δx = parse(DFloat, words[2])            
            @info @sprintf """ Δx:                     %.6f""" Δx
        end        
        if uppercase(strip(words[1])) == "ΔY"
            Δy = parse(DFloat, words[2])            
            @info @sprintf """ Δy:                     %.6f""" Δy
        end
        if uppercase(strip(words[1])) == "ΔZ"
            Δz = parse(DFloat, words[2])            
            @info @sprintf """ Δz:                     %.6f""" Δz
        end

        
        if uppercase(strip(words[1])) == "TFINAL"
            tfinal = parse(DFloat, words[2])
            @info @sprintf """ tfinal:                 %.2f""" tfinal
        end
        if uppercase(strip(words[1])) == "DT"
            dt = parse(DFloat, words[2])
            @info @sprintf """ Explicit time step:     %.6f""" dt
        end
        if uppercase(strip(words[1])) == "TINTEGRATOR"
            tintegrator = words[2]
            @info @sprintf """ Time integrator:       %s""" tintegrator
        end
        if uppercase(strip(words[1])) == "TINTEGRATORTYPE"
            tintegratortype = words[2]
            @info @sprintf """ Time integrator type:  %s""" tintegratortype
        end

        
        #BDY Conditions
        if uppercase(strip(words[1])) == "LEFT"
            bdy_left = words[2]
            @info @sprintf """ B.C. Left wall:        %s""" bdy_left            
        end
        if uppercase(strip(words[1])) == "RIGHT"
            bdy_right = words[2] 
            @info @sprintf """ B.C. Right wall:       %s""" bdy_right         
        end
        if uppercase(strip(words[1])) == "TOP"
            bdy_top = words[2]
            @info @sprintf """ B.C. Top wall:         %s""" bdy_top
        end
        if uppercase(strip(words[1])) == "BOTTOM"
            bdy_bott = words[2]
            @info @sprintf """ B.C. Bottom wall:      %s""" bdy_bott
        end
        
    end
    
    close(filein)
end

read_configuration_file()

end
