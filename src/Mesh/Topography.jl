module Topography

using DocStringExtensions
using Printf

export TopographyReadExternal

import Canary
using MPI

# {{{ 
"""
       READTOPOtxt_header(txt_inputfile,nlon, nlat, deltaLon, deltaLat)

       reads the header file containing the details 
       of the topography from a NOAA text file
    """
function TopographyReadExternal(file_type, header_file_in, body_file_in)

    if file_type == "NOAA" || file_type == "noaa" || file_type == "txt"
        
        (nlon, nlat, lonmin, lonmax, latmin, latmax, dlon, dlat) = ReadExternalHeader(header_file_in)
        
        ReadExternalTxtCoordinates(body_file_in, topoBathy_flg, Nex, Ney)
    else
        error( " Mesh/Topography.jl:
                         ONLY NOAA txt files can be read in at this time. 
                         Feel free to add your own reader. ") 
    end
    
end


# {{{ READTOPOtxt_header
"""
       READTOPOtxt_header(txt_inputfile,nlon, nlat, deltaLon, deltaLat)

       reads the header file containing the details 
       of the topography from a NOAA text file
    """
function ReadExternalHeader(header_file_in)
    
    @info @sprintf """ Topography header file %s ... DONE""" header_file_in
    
    ftopo_header = open(header_file_in)
    @info @sprintf """ Grids.jl: Opening topography header file ... DONE"""
    
    topo_header = readdlm(ftopo_header)
    @info @sprintf """ Grids.jl: Reading topography header file ... DONE""" 

    #Check that the file is not empty
    (nrows, _) = size(topo_header)    

    close(ftopo_header)           
    @info @sprintf """ Grids.jl: Closing topography header file ... DONE"""

    
    if nrows == 0
        error(" ERROR Grids.jl --> READTOPOtxt_header: grid header file is empty!")
        return
    end
    
    nlon      =   Int64(topo_header[1,2])
    nlat      =   Int64(topo_header[2,2])
    lonmin    = Float64(topo_header[3,2])
    latmin    = Float64(topo_header[4,2])
    deltacell = Float64(topo_header[5,2])
    lonmax    = lonmin + nlon*deltacell
    latmax    = latmin + nlat*deltacell
    
    dlon  = lonmax - lonmin
    dlat  = latmax - latmin

    @info @sprintf """ File type: NOAA Text header:"""
    @info @sprintf """     %s %s""" topo_header[7,1] topo_header[7,2]  #units
    @info @sprintf """     %s %d""" topo_header[1,1] nlon              #ncols (LON)
    @info @sprintf """     %s %d""" topo_header[2,1] nlat              #nrows (LAT)
    @info @sprintf """     Lon_min, Lon_max: %f, %f""" lonmin lonmax #
    @info @sprintf """     Lat_min, Lat_max: %f, %f""" latmin latmax #
    @info @sprintf """     Δlon:    %f""" dlon #
    @info @sprintf """     Δlat:    %f""" dlat #
    
    return (nlon, nlat, lonmin, lonmax, latmin, latmax, dlon, dlat)
    
end
# }}}


# {{{ READTOPOtxt_header
"""
                   READTOPOtxt_file(txt_inputfile,nlon, nlat, deltaLon, deltaLat)

                   Reads the topography from a NOAA text file of shape [1:nnodes][3]
                   where the first and second column are the ordered lat-lon coordinates
                   and the third column is the height of topography at that specific
                   coordinate point.
                 
                  1) XYZ files from NOAA
                   
                  READTOPOtxt_header() reads the parameters from the header file (*.hdr)
                  READTOPOtxt_file()   reads the actual file of coordinates      (*.xyz)
                     
                  2) DEM files
                  
                  READTOPO_DEM()       reads a DEM file from NOAA page (file extension: *.asc)
                 
                    """
function ReadExternalTxtCoordinates(body_file_in, topoBathy_flg, Nex, Ney)

    @info @sprintf """ Topography file: %s""" body_file_in

    DFloat = Float64 #CHANGE THIS TO GENERAL
    
    ftopo_body = open(body_file_in)
    @info @sprintf """ Grids.jl: Opening topography ... DONE"""
    
    topo_body = readdlm(ftopo_body)

    # Create array on the device
    nnodesx, nnodesy = Nex + 1, Ney + 1 #Linear grid
    LonLatZ = Array{DFloat, 2}(undef, nnodesx, nnodesy)
    
    k = 0
    for j = nnodesy:-1:1
        for i = 1:1:nnodesx
            k = k + 1
            LonLatZ[i, j] = topo_body[k,3]
            @show  LonLatZ[i, j]
        end
    end
    
    @info @sprintf """ Grids.jl: Reading topography file ... DONE""" 

    #Check that the file is not empty
    (nrows, _) = size(topo_body)
    if nrows == 0
        error(" ERROR Grids.jl --> READTOPOtxt_xyz: grid file is empty!")
        return
    end
    
    close(ftopo_body)
    @info @sprintf """ Grids.jl: Closing topography ... DONE"""
    
end
#}}}


end
