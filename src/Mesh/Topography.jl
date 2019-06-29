module Topography
"""
 module Topography
    
"""

using Printf
using DelimitedFiles
using Dierckx

export TopographyReadExternal, ReadExternalHeader, ReadExternalTxtCoordinates

import Canary
using MPI


# {{{ READTOPOtxt_header
"""
   READTOPOtxt_header(txt_inputfile,nlon, nlat, deltaLon, deltaLat)

   reads the header file containing the details of the topography from a NOAA text file
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
                     
    returns xTopo1d, yTopo1d, zTopo2d

 """
function ReadExternalTxtCoordinates(body_file_in, TopoBathy_flg, nlon, nlat)

    @info @sprintf """ Topography file: %s""" body_file_in

    DFloat  = Float64 #NOTICE: I recommend the grid is always build in double precision even if the code is run in single 
     
    ftopo_body = open(body_file_in)
    @info @sprintf """ Grids.jl: Opening topography file ... DONE"""
   
    topo_body = readdlm(ftopo_body)
    (npoin_linear_grid, _) = size(topo_body)
    if npoin_linear_grid == 0
        error(" ERROR Grids.jl --> READTOPOtxt_xyz: grid file is empty!")
        return
    end
    #
    # Create array
    #
    nnodes_lon, nnodes_lat, nnodes = nlon, nlat, nlon*nlat #Linear grid
    zTopo2d = Array{DFloat, 2}(undef, nnodes_lon, nnodes_lat)
    
    xTopo1d = Array{DFloat, 1}(undef, nnodes_lon)
    yTopo1d = Array{DFloat, 1}(undef, nnodes_lat)
    
    k = 0
    for j = 1:nnodes_lat
        for i = 1:nnodes_lon
            k = k + 1
            zTopo2d[i,j] = topo_body[k,3]
        end
    end
    npoin = k
    
    #
    # Store 1D X array in the x direction: size -> nnodes_lat
    # and order in increasing order
    for i = 1:nnodes_lon     
        xTopo1d[i] = topo_body[i,1]    
        #@info @sprintf """ X[%d]: %.16e""" i xTopo1d[i]
    end
    if minimum(xTopo1d) < 0
        xTopo1d =  xTopo1d .- minimum(xTopo1d)
    end
    if xTopo1d[1] > xTopo1d[end]
        xTopo1d = reverse(xTopo1d)
    end
  
    #
    # Store 1D X array in the x direction: size -> nnodes_lat
    # and order in increasing order
    l = 0
    interval = nnodes_lon
    for k = 1:interval:nnodes
        l = l + 1
        yTopo1d[l] = topo_body[k,2]
        #@info @sprintf """ Y[%d]: %.16e""" l yTopo1d[l]
    end 
    
    #Shift if negative
    if minimum(yTopo1d) < 0
        yTopo1d =  yTopo1d .- minimum(yTopo1d)
    end   
    if yTopo1d[1] > yTopo1d[end]
        yTopo1d = reverse(yTopo1d)
    end
    
    close(ftopo_body)
    
    @info @sprintf """ Grids.jl: Closing topography file ... DONE"""
    
    return xTopo1d, yTopo1d, zTopo2d
 
end
#}}}


end
