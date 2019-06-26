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
    #
    # DEFINE HERE ALL QUANTITIES THAT WILL BE INPUT BY THE USER:
    #   
    local DInt
    local DFloat
    local nsd
    local geometry 
    local npoly    
    local Δx       
    local Δy       
    local Δz       
    local xmin     
    local xmax     
    local ymin     
    local ymax     
    local zmin     
    local zmax     
    local tfinal
    local dt    
    local tintegrator
    local tintegratortype
    local sgs
    local left
    local right
    local top  
    local bottom
    local sponge
    local sponge_type   
    local problem
    local iinput_entry
    local dict_entry
    local dict_user_input = Dict()
    
    #local ... more here if adding more
    
    @info @sprintf """ ----------------------------------------------------"""
    @info @sprintf """   ______ _      _____ __  ________                  """     
    @info @sprintf """  |  ____| |    |_   _|  ...  |  __  |               """  
    @info @sprintf """  | |    | |      | | |   .   | |  | |               """ 
    @info @sprintf """  | |    | |      | | | |   | | |__| |               """
    @info @sprintf """  | |____| |____ _| |_| |   | | |  | |               """
    @info @sprintf """  | _____|______|_____|_|   |_|_|  |_|               """
    @info @sprintf """                                                     """
    @info @sprintf """ ----------------------------------------------------"""
    
    split_chars = ["=", ",", ";", "->", "_"]
    filein      = open(joinpath(@__DIR__, "./../../src/InputOutput/configuration.ini"))
    
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
    iinput_entry = 0
    for line in eachline(filein)

        r_split = Regex( join(split_chars, "|") ) 
        words  = split( line, r_split )

        if uppercase(strip(words[1])) == "PROBLEM"
            problem = words[2]
            if (!(haskey(dict_user_input, "problem"))) dict_user_input["problem"]=problem end;#if
            iinput_entry += 1
            
            @info @sprintf """ Problem:               %s""" problem            
        end
        if uppercase(strip(words[1])) == "GEOMETRY"
            geometry = words[2]
            if (!(haskey(dict_user_input, "geometry"))) dict_user_input["geometry"]=geometry end;#if
                        
            iinput_entry += 1
                
            @info @sprintf """ Geometry:              %s""" geometry           
        end       
        if uppercase(strip(words[1])) == "NSD"
            nsd = parse(DInt, words[2])
            if (!(haskey(dict_user_input, "nsd"))) dict_user_input["nsd"]=nsd end;#if
            
            iinput_entry += 1
                
            @info @sprintf """ Space dimensions:       %d""" nsd
        end  
        if uppercase(strip(words[1])) == "NPOLY"
            npoly = parse(DInt, words[2])  
            if (!(haskey(dict_user_input, "Npoly"))) dict_user_input["Npoly"]=npoly end;#if
            iinput_entry += 1
                
            @info @sprintf """ Polynomila order:       %d""" npoly
        end
        if uppercase(strip(words[1])) == "SGS"
            sgs = words[2]  
            if (!(haskey(dict_user_input, "sgs"))) dict_user_input["sgs"]=sgs end;#if
            iinput_entry += 1
                
            cs  = parse(DFloat, words[3]) 
            if (!(haskey(dict_user_input, "Cs"))) dict_user_input["Cs"]=cs end;#if 
            iinput_entry += 1

            @info @sprintf """ Stabilization model:   %s""" sgs
            @info @sprintf """ Coefficient:            %.6f""" cs
        end
        if uppercase(strip(words[1])) == "ΔX"
            Δx = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "Δx"))) dict_user_input["Δx"]=Δx end;#if
            iinput_entry += 1
            
            @info @sprintf """ Δx:                     %.6f""" Δx
        end        
        if uppercase(strip(words[1])) == "ΔY"
            Δy = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "Δy"))) dict_user_input["Δy"]=Δy end;#if
            iinput_entry += 1
            
            @info @sprintf """ Δy:                     %.6f""" Δy
        end
        if uppercase(strip(words[1])) == "ΔZ"
            Δz = parse(DFloat, words[2])
            if (!(haskey(dict_user_input, "Δz"))) dict_user_input["Δz"]=Δz end;#if
            iinput_entry += 1
            
            @info @sprintf """ Δz:                     %.6f""" Δz
        end
        #
        if uppercase(strip(words[1])) == "XMIN"
            xmin = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "xmin"))) dict_user_input["xmin"]=xmin end;#if
            iinput_entry += 1
            
            @info @sprintf """ xmin:                   %.6f""" xmin
        end     
        if uppercase(strip(words[1])) == "XMAX"
            xmax = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "xmax"))) dict_user_input["xmax"]=xmax end;#if
            iinput_entry += 1
            
            @info @sprintf """ xmax:                   %.6f""" xmax
        end
        #
        if uppercase(strip(words[1])) == "YMIN"
            ymin = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "ymin"))) dict_user_input["ymin"]=ymin end;#if
            iinput_entry += 1
            
            @info @sprintf """ ymin:                   %.6f""" ymin
        end     
        if uppercase(strip(words[1])) == "YMAX"
            ymax = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "ymax"))) dict_user_input["ymax"]=ymax end;#if           
            @info @sprintf """ ymax:                   %.6f""" ymax
        end 
        #
        if uppercase(strip(words[1])) == "ZMIN"
            zmin = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "zmin"))) dict_user_input["zmin"]=zmin end;#if
            iinput_entry += 1
            
            @info @sprintf """ zmin:                   %.6f""" zmin
        end     
        if uppercase(strip(words[1])) == "ZMAX"
            zmax = parse(DFloat, words[2]) 
            if (!(haskey(dict_user_input, "zmax"))) dict_user_input["zmax"]=zmax end;#if
            iinput_entry += 1
            
            @info @sprintf """ zmax:                   %.6f""" zmax
        end 

        if uppercase(strip(words[1])) == "TFINAL"
            tfinal = parse(DFloat, words[2])
            if (!(haskey(dict_user_input, "tfinal"))) dict_user_input["tfinal"]=tfinal end;#if
            iinput_entry += 1
            
            @info @sprintf """ tfinal:                 %.2f""" tfinal
        end
        if uppercase(strip(words[1])) == "DT"
            dt = parse(DFloat, words[2])
            if (!(haskey(dict_user_input, "dt"))) dict_user_input["dt"]=dt end;#if
            iinput_entry += 1
            
            @info @sprintf """ Explicit time step:     %.6f""" dt
        end
        if uppercase(strip(words[1])) == "TINTEGRATOR"
            tintegrator = words[2]
            if (!(haskey(dict_user_input, "tintegrator"))) dict_user_input["tintegrator"]=tintegrator end;#if
            iinput_entry += 1
            
            @info @sprintf """ Time integrator:       %s""" tintegrator
        end
        if uppercase(strip(words[1])) == "TINTEGRATORTYPE"
            tintegratortype = words[2]
            if (!(haskey(dict_user_input, "tintegratortype"))) dict_user_input["tintegratortype"]=tintegratortype end;#if
            iinput_entry += 1
            
            @info @sprintf """ Time integrator type:  %s""" tintegratortype
        end
      
        #BDY Conditions
        if uppercase(strip(words[1])) == "LEFT"
            bdy_left = words[2]
            if (!(haskey(dict_user_input, "left"))) dict_user_input["left"]=bdy_left end;#if
            iinput_entry += 1
            
            @info @sprintf """ B.C. Left wall:        %s""" bdy_left            
        end
        if uppercase(strip(words[1])) == "RIGHT"
            bdy_right = words[2] 
            if (!(haskey(dict_user_input, "right"))) dict_user_input["right"]=bdy_right end;#if
            iinput_entry += 1
            
            @info @sprintf """ B.C. Right wall:       %s""" bdy_right         
        end
        if uppercase(strip(words[1])) == "TOP"
            bdy_top = words[2]
            if (!(haskey(dict_user_input, "top"))) dict_user_input["top"]=bdy_top end;#if
            iinput_entry += 1
            
            @info @sprintf """ B.C. Top wall:         %s""" bdy_top
        end
        if uppercase(strip(words[1])) == "BOTTOM"
            bdy_bott = words[2]
            if (!(haskey(dict_user_input, "bottom"))) dict_user_input["bottom"]=bdy_bott end;#if
            iinput_entry += 1
            
            @info @sprintf """ B.C. Bottom wall:      %s""" bdy_bott
        end
     end
     #Print dictionary keys and values:
     #for key in dict_user_input
     #    @info @sprintf """ %s %s """ (key[1]) (key[2])
    #end
    close(filein)
    return dict_user_input
end
#}}}

     #dict_user_input = read_configuration_file()

     ##Print dictionary keys and values:
     #for key in dict_user_input
     #    @info @sprintf """ %s %s """ (key[1]) (key[2])
     #end

end
