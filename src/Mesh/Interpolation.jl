module Interpolation

using MPI
import GaussQuadrature
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.Mesh.Elements
using LinearAlgebra
using StaticArrays

export InterpolationBrick, interpolate_brick!, InterpolationCubedSphere, invert_trilear_mapping_hex, interpolate_cubed_sphere!

#--------------------------------------------------------
"""
    InterpolationBrick(grid::DiscontinuousSpectralElementGrid, xres, ::Type{FT}) where FT <: AbstractFloat

This interpolation structure and the corresponding functions works for a brick, where stretching/compression happens only along the x, y & z axis.
Here x1 = X1(ξ1); x2 = X2(ξ2); x3 = X3(ξ3)

# input for the inner constructor
 - `grid` DiscontinousSpectralElementGrid
 - `xres` Resolution of the interpolation grid in x1, x2 and x3 directions
"""
struct InterpolationBrick{FT <:AbstractFloat}
    realelems::UnitRange{Int64}
    poly_order::Int

    xbnd::Array{FT,2}      # domain bounds, [2(min/max),ndim]
    xres::Array{FT,1}      # resolutions in x1, x2, x3 directions for the uniform grid

    ξ1::Vector{Vector{FT}} # unique ξ1 coordinates of interpolation points within each element 
    ξ2::Vector{Vector{FT}} # unique ξ2 coordinates of interpolation points within each element 
    ξ3::Vector{Vector{FT}} # unique ξ3 coordinates of interpolation points within each element 
  
    x::Vector{Array{FT,2}} # x[elno][3,npts] -> (x1,x2,x3) coordinates of portion of interpolation grid embedded within each element 

    V::Vector{Vector{FT}}  # interpolated variable within each element
#--------------------------------------------------------
    function InterpolationBrick(grid::DiscontinuousSpectralElementGrid{FT}, xres) where FT <: AbstractFloat
        T = Int
        poly_order = polynomialorder(grid)
        ndim = 3
        xbnd = zeros(FT, 2, ndim) # domain bounds (min,max) in each dimension
        for dim in 1:ndim 
            xbnd[1,dim], xbnd[2,dim] = extrema(grid.topology.elemtocoord[dim,:,:])
        end
        x1g = range(xbnd[1,1], xbnd[2,1], step=xres[1])
        x2g = range(xbnd[1,2], xbnd[2,2], step=xres[2])
        x3g = range(xbnd[1,3], xbnd[2,3], step=xres[3]) 
        #-----------------------------------------------------------------------------------
        realelems =  grid.topology.realelems # Element (numbers) on the local processor
        Nel       = length(realelems)
        n123      = zeros(T,    ndim)     # # of unique ξ1, ξ2, ξ3 points in each cell
        xsten     = zeros(T, 2, ndim)     # x1, x2, x3 start and end for each brick element
        xbndl     = zeros(T, 2, ndim)     # location of x1,x2,x3 limits (min,max) for each brick element

        ξ1 = map( i -> zeros(FT,i), zeros(T,Nel))
        ξ2 = map( i -> zeros(FT,i), zeros(T,Nel))
        ξ3 = map( i -> zeros(FT,i), zeros(T,Nel))
        V  = map( i -> zeros(FT,i), zeros(T,Nel))
        x  = map( i -> zeros(FT,ndim,i), zeros(T,Nel)) # interpolation grid points embedded in each cell 
        #-----------------------------------------------------------------------------------
        for el in 1:Nel
            for (ξ,xg,dim) in zip((ξ1,ξ2,ξ3), (x1g, x2g, x3g), 1:ndim)
                xbndl[1,dim], xbndl[2,dim] = extrema(grid.topology.elemtocoord[dim,:,el])
                xsten[1,dim], xsten[2,dim] = findfirst( temp -> temp ≥ xbndl[1,dim], xg ), findlast( temp -> temp ≤ xbndl[2,dim], xg )
                n123[dim] = xsten[2,dim] - xsten[1,dim]
                ξ[el] = [ 2 * ( xg[ xsten[1,dim] + i - 1] - xbndl[1,dim] ) / (xbndl[2,dim]-xbndl[1,dim]) -  1 for i in 1:n123[dim] ]
            end

            x_el  = zeros(FT,ndim,prod(n123))
            V[el] = zeros(FT,prod(n123))

            ctr = 1

            for k in 1:n123[3], j in 1:n123[2], i in 1:n123[1]
                x_el[1,ctr] = x1g[ xsten[1,1] + i - 1 ]
                x_el[2,ctr] = x2g[ xsten[1,2] + j - 1 ]
                x_el[3,ctr] = x3g[ xsten[1,3] + k - 1 ]
                ctr += 1
            end
            x[el] = x_el
        end # el loop
        #-----------------------------------------------------------------------------------
        return new{FT}(realelems, poly_order, xbnd, xres, ξ1, ξ2, ξ3, x, V)
    end
#--------------------------------------------------------
end # struct InterpolationBrick 
#--------------------------------------------------------
"""
    interpolate_brick!(intrp_brck::InterpolationBrick, sv::AbstractArray{FT}, st_idx::T, poly_order::T) where {T <: Integer, FT <: AbstractFloat}

This interpolation function works for a brick, where stretching/compression happens only along the x, y & z axis.
Here x1 = X1(ξ1); x2 = X2(ξ2); x3 = X3(ξ3)

# input
 - `intrp_brck` Initialized InterpolationBrick structure
 - `sv` State vector
 - `st_idx` # of state vector variable to be interpolated
 - `poly_order` polynomial order for the simulation
"""
function interpolate_brick!(intrp_brck::InterpolationBrick, sv::AbstractArray{FT}, st_idx::T) where {T <: Integer, FT <: AbstractFloat}
    qm1 = intrp_brck.poly_order + 1
    Nel = length(intrp_brck.realelems)
    m1_r, m1_w = GaussQuadrature.legendre(FT,qm1,GaussQuadrature.both)
    wb = Elements.baryweights(m1_r)

    vout    = FT(0)
    vout_ii = FT(0)
    vout_ij = FT(0)

    phit = Vector{FT}(undef,qm1)

    for el in 1:Nel #-----for each element elno 
        if length(intrp_brck.ξ1[el]) > 0
            l1 = length(intrp_brck.ξ1[el]); l2 = length(intrp_brck.ξ2[el]); l3 = length(intrp_brck.ξ3[el]) 
            lag    = @view sv[:,st_idx,el]

            phir = Elements.interpolationmatrix(m1_r, intrp_brck.ξ1[el], wb)
            phis = Elements.interpolationmatrix(m1_r, intrp_brck.ξ2[el], wb)

            for k in 1:l3 # interpolating point-by-point
                interpolationvector_1pt!(m1_r,intrp_brck.ξ3[el][k],wb,phit)

                for j in 1:l2, i in 1:l1
                    vout = 0            
                    for ik in 1:qm1
                        v_ij = 0
                        for ij in 1:qm1
                            v_ii = 0
                            for ii in 1:qm1
                                @inbounds v_ii += lag[ii + (ij-1)*qm1 + (ik-1)*qm1*qm1] * phir[i,ii]
                            end # ii loop

                            @inbounds v_ij += v_ii * phis[j,ij]
                        end # ij loop
                        @inbounds vout += v_ij * phit[ik]
                    end # ik loop
                    @inbounds intrp_brck.V[el][i + (j-1)*l1 + (k-1)*l1*l2] = vout 
                end # j, i loop
            end # k loop
        end
  #--------------------
  end
end
#--------------------------------------------------------
function interpolate_brick_matrix!(intrp_brck::InterpolationBrick, sv::AbstractArray{FT}, st_no::T, poly_order::T) where {T <: Integer, FT <: AbstractFloat}
    qm1 = poly_order + 1
    Nel = length(intrp_brck.realelems)
    m1_r, m1_w = GaussQuadrature.legendre(FT,qm1,GaussQuadrature.both)
    wb = Elements.baryweights(m1_r)
    Nel = length(intrp_brck.realelems)
    alpha = FT(1)
    beta  = FT(0)
    #-----for each element elno 
    for el in 1:Nel
        if length(intrp_brck.ξ1[el]) > 0
            lag    = @view sv[:,st_no,el]
            phir = Elements.interpolationmatrix(m1_r, intrp_brck.ξ1[el], wb)
            phis = Elements.interpolationmatrix(m1_r, intrp_brck.ξ2[el], wb)
            phit = Elements.interpolationmatrix(m1_r, intrp_brck.ξ3[el], wb)

            sr,si = size(phir); ss,sj = size(phis); st,sk = size(phit)
            #---------------------------------------------------------------
            h1 = zeros(FT, sj*sk, sr)  # this is for testing only
            h2 = zeros(FT, sk*sr, ss)  # memory allocations within the function will be removed
            #---------------------------------------------------------------
            LinearAlgebra.BLAS.gemm!('T', 'T', alpha, reshape(lag,si, sj*sk), phir, beta, h1)
            LinearAlgebra.BLAS.gemm!('T', 'T', alpha, reshape(h1, sj, sk*sr), phis, beta, h2)
            LinearAlgebra.BLAS.gemm!('T', 'T', alpha, reshape(h2, sk, sr*ss), phit, beta, reshape(intrp_brck.V[el], sr*ss, st))
            #---------------------------------------------------------------
        end
  #--------------------
  end
end
#--------------------------------------------------------
"""
    interpolationvector_1pt!(rsrc::Vector{FT}, rdst::FT,
                             wbsrc::Vector{FT}, phi::Vector{FT}) where FT <: AbstractFloat

returns the polynomial interpolation matrix for interpolating between the points
`rsrc` (with associated barycentric weights `wbsrc`) and a single point rdst. This 
function writes the result to a vector since the basis functions are calculated at a single point

Reference:
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
@inline function interpolationvector_1pt!(rsrc::AbstractVector{FT}, rdst::FT,
                             wbsrc::AbstractVector{FT}, phi::AbstractVector{FT}) where FT <: AbstractFloat
    qm1 = length(rsrc)
    @assert length(phi) == qm1

    for ib in 1:qm1
        if rdst==rsrc[ib]
            phi .= FT(0)
            @inbounds phi[ib] = FT(1) 
            break
        else
            @inbounds phi[ib] = wbsrc[ib] / (rdst-rsrc[ib])
        end
    end
    d = sum(phi)
    phi ./= d 
    return nothing
end
#--------------------------------------------------------
"""
    InterpolationCubedSphere(grid::DiscontinuousSpectralElementGrid, vert_range::AbstractArray{FT}, lat_res::FT, long_res::FT, rad_res::FT) where {FT <: AbstractFloat}

This interpolation structure and the corresponding functions works for a cubed sphere topology. The data is interpolated along a lat/long/rad grid.

# input for the inner constructor
 - `grid` DiscontinousSpectralElementGrid
 - `vert_range` vertex range along the radial coordinate 
 - `lat_res` Resolution of the interpolation grid along the latitude coordinate in radians 
 - `long_res` Resolution of the interpolation grid along the longitude coordinate in radians 
 - `rad_res` Resolution of the interpolation grid along the radial coordinate 
"""
struct InterpolationCubedSphere{T <: Integer, FT <: AbstractFloat}

    realelems::UnitRange{Int64}
    poly_order::Int

    lat_min::FT;  long_min::FT;  rad_min::FT; # domain bounds, min
    lat_max::FT;  long_max::FT;  rad_max::FT; # domain bounds, max
    lat_res::FT;  long_res::FT;  rad_res::FT; # respective resolutions for the uniform grid

    n_lat::T; n_long::T; n_rad::T;            # # of lat, long & rad grid locations

    ξ1::Vector{Vector{FT}} # ξ1 coordinates of interpolation points within each element 
    ξ2::Vector{Vector{FT}} # ξ2 coordinates of interpolation points within each element 
    ξ3::Vector{Vector{FT}} # ξ3 coordinates of interpolation points within each element 

    radc::Vector{Vector{FT}}  # rad coordinates of interpolation points within each element
    latc::Vector{Vector{FT}}  # lat coordinates of interpolation points within each element
    longc::Vector{Vector{FT}} # long coordinates of interpolation points within each element

    V::Vector{Vector{FT}}  # interpolated variable within each element
  #--------------------------------------------------------
    function InterpolationCubedSphere(grid::DiscontinuousSpectralElementGrid, vert_range::AbstractArray{FT}, nhor::Int, lat_res::FT, long_res::FT, rad_res::FT) where {FT <: AbstractFloat}
        poly_order = polynomialorder(grid)
        toler1 = eps(FT) * vert_range[1] * 2.0 # tolerance for unwarp function
        #toler1 = eps(FT) * 2.0 # tolerance for unwarp function
        toler2 = eps(FT) * 4.0                 # tolerance 
        toler3 = eps(FT) * vert_range[1] * 4.0 # tolerance for Newton-Raphson 

        lat_min,   lat_max = FT(0.0), FT(π)                 # inclination/zeinth angle range
        long_min, long_max = FT(0.0), FT(2*π)  			    # azimuthal angle range
        rad_min,   rad_max = vert_range[1], vert_range[end] # radius range

        realelems = grid.topology.realelems # Element (numbers) on the local processor
        Nel = length(realelems)

        nvert = length(vert_range) - 1              # # of elements in vertical direction
        Nel_glob = nvert * nhor * nhor * 6

        nblck = nhor * nhor * nvert
        Δh = 2 / nhor                               # horizontal grid spacing in unwarped grid

        lat_grd, long_grd, rad_grd = range(lat_min, lat_max, step=lat_res), range(long_min, long_max, step=long_res), range(rad_min, rad_max, step=rad_res) 

        n_lat, n_long, n_rad = Int(length(lat_grd)), Int(length(long_grd)), Int(length(rad_grd))

        uw_grd = zeros(FT, 3, 1)
        #---------------------------------------------- 
        glob_ord = grid.topology.origsendorder # to account for reordering of elements after the partitioning process 

        glob_elem_no = zeros(Int, nvert*length(glob_ord))

        for i in 1:length(glob_ord), j in 1:nvert
            glob_elem_no[j + (i-1)*nvert] = (glob_ord[i] - 1)*nvert + j 
        end

        ξ1, ξ2, ξ3        = map( i -> zeros(FT,i), zeros(Int,Nel)), map( i -> zeros(FT,i), zeros(Int,Nel)), map( i -> zeros(FT,i), zeros(Int,Nel))
        radc, latc, longc = map( i -> zeros(FT,i), zeros(Int,Nel)), map( i -> zeros(FT,i), zeros(Int,Nel)), map( i -> zeros(FT,i), zeros(Int,Nel))

        for k in 1:n_long, j in 1:n_lat, i in 1:n_rad 
            x1_grd = rad_grd[i] * sin(lat_grd[j]) * cos(long_grd[k]) # inclination -> latitude; azimuthal -> longitude.
            x2_grd = rad_grd[i] * sin(lat_grd[j]) * sin(long_grd[k]) # inclination -> latitude; azimuthal -> longitude.
            x3_grd = rad_grd[i] * cos(lat_grd[j])   

            uw_grd[1], uw_grd[2], uw_grd[3] = Topologies.cubedshellunwarp(x1_grd, x2_grd, x3_grd) # unwarping from sphere to cubed shell
            rad = FT(maximum(abs.(uw_grd)))
            #--------------------------------
            x1_uw2_grd = uw_grd[1] / rad # unwrapping cubed shell on to a 2D grid (in 3D space, -1 to 1 cube)
            x2_uw2_grd = uw_grd[2] / rad
            x3_uw2_grd = uw_grd[3] / rad
            #--------------------------------
            if rad ≤ vert_range[1]       # accounting for minor rounding errors from unwarp function at boundaries 
                vert_range[1] - rad < toler1 ? l_nrm = 1 :  error("fatal error, rad lower than inner radius: ", vert_range[1] - rad," $x1_grd /// $x2_grd //// $x3_grd" )
            elseif rad ≥ vert_range[end] # accounting for minor rounding errors from unwarp function at boundaries 
                rad - vert_range[end] < toler1 ? l_nrm = nvert : error("fatal error, rad greater than outer radius")
            else                         # normal scenario
                l_nrm = findfirst( X -> X .- rad .> 0.0, vert_range ) - 1 # identify stack bin 
            end
            #--------------------------------
            if     abs(x1_uw2_grd + 1) < toler2 # face 1 (x1 == -1 plane)
	    	    l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
    		    l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                el_glob = Int(l_nrm + (nhor-l2)*nvert + (l3-1)*nvert*nhor)
            elseif abs(x2_uw2_grd + 1) < toler2 # face 2 (x2 == -1 plane)
		        l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
    		    l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                el_glob = Int(l_nrm + (l1-1)*nvert + (l3-1)*nvert*nhor + nblck*1)
            elseif abs(x1_uw2_grd - 1) < toler2 # face 3 (x1 == +1 plane)
		        l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
    		    l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                el_glob = Int(l_nrm + (l2-1)*nvert + (l3-1)*nvert*nhor + nblck*2 )
            elseif abs(x3_uw2_grd - 1) < toler2 # face 4 (x3 == +1 plane)
      		    l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
	    	    l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
                el_glob = Int(l_nrm + (l1-1)*nvert + (l2-1)*nvert*nhor + nblck*3)
            elseif abs(x2_uw2_grd - 1) < toler2 # face 5 (x2 == +1 plane)
	            l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
		        l3 = min(div(x3_uw2_grd + 1, Δh) + 1, nhor)
                el_glob = Int(l_nrm + (l1-1)*nvert + (nhor-l3)*nvert*nhor + nblck*4 )
            elseif abs(x3_uw2_grd + 1) < toler2 # face 6 (x3 == -1 plane)
	    	    l1 = min(div(x1_uw2_grd + 1, Δh) + 1, nhor)
		        l2 = min(div(x2_uw2_grd + 1, Δh) + 1, nhor)
                el_glob = Int(l_nrm + (l1-1)*nvert + (nhor-l2)*nvert*nhor + nblck*5)
            else
                error("error: unwrapped grid does on lie on any of the 6 faces")
            end
            #--------------------------------
            el_loc = findfirst(X -> X-el_glob == 0, glob_elem_no)
            if ( el_loc ≠ nothing ) # computing inner coordinates for local elements
                ξ = invert_trilear_mapping_hex(grid.topology.elemtocoord[1,:,el_loc], 
                                               grid.topology.elemtocoord[2,:,el_loc], 
                                               grid.topology.elemtocoord[3,:,el_loc], uw_grd, toler3)
                push!(ξ1[el_loc],ξ[1]) 
                push!(ξ2[el_loc],ξ[2]) 
                push!(ξ3[el_loc],ξ[3]) 
                push!(radc[el_loc],  rad_grd[i])
                push!(latc[el_loc],  lat_grd[j]) 
                push!(longc[el_loc],long_grd[k])
            end
            #--------------------------------
        end
 
        V = [ Vector{FT}(undef, length(radc[el])) for el in 1:Nel ] # Allocating storage for interpolation variable

        return new{Int, FT}(realelems, poly_order, lat_min, long_min, rad_min, lat_max, long_max, rad_max, lat_res, long_res, rad_res, 
                    n_lat, n_long, n_rad, ξ1, ξ2, ξ3, radc, latc, longc, V)
    #-----------------------------------------------------------------------------------
    end # Inner constructor function InterpolationCubedSphere
#-----------------------------------------------------------------------------------
end # structure InterpolationCubedSphere
#--------------------------------------------------------
"""
    invert_trilear_mapping_hex(X1::Array{FT}, X2::Array{FT}, X3::Array{FT}, x::Array{FT}, tol::FT) where FT <: AbstractFloat 

This function computes (ξ1,ξ2,ξ3) given (x1,x2,x3) and the (8) vertex coordinates of a Hexahedron. Newton-Raphson method is used
# input
 - `X1` X1 coordinates of the (8) vertices of the hexahedron
 - `X2` X2 coordinates of the (8) vertices of the hexahedron
 - `X3` X3 coordinates of the (8) vertices of the hexahedron
 - `x` (x1,x2,x3) coordinates of the 
"""
function invert_trilear_mapping_hex(X1::Array{FT}, X2::Array{FT}, X3::Array{FT}, x::Array{FT}, tol::FT) where FT <: AbstractFloat 
    max_it = 10            # maximum # of iterations
    ξ      = zeros(FT,3,1) # initial guess => cell centroid

    d   = trilinear_map(ξ, X1, X2, X3) - x
    err = norm(d)
    ctr = 0 
    #---Newton-Raphson iterations---------------------------
    while err > tol
        trilinear_map_IJac_x_vec!(ξ, X1, X2, X3, d)
        ξ .-= d
        d = trilinear_map(ξ, X1, X2, X3) - x
        err = norm(d)
        ctr += 1
        if ctr > max_it
            error("invert_trilinear_mapping_hex: Newton-Raphson not converging to desired tolerance after max_it = ", max_it," iterations; err = ", err,"; toler = ", tol)
        end
    end
    #-------------------------------------------------------
    return ξ
end
#--------------------------------------------------------
function trilinear_map(ξ::Array{FT}, x1v::Array{FT}, x2v::Array{FT}, x3v::Array{FT}) where FT <: AbstractFloat
    x = Array{FT}(undef,3)
    for (vert,dim) = zip((x1v,x2v,x3v),1:3)
        x[dim] = ((1 - ξ[1]) * (1 - ξ[2]) * (1 - ξ[3]) * vert[1] + (1 + ξ[1]) * (1 - ξ[2]) * (1 - ξ[3]) * vert[2] +
                  (1 - ξ[1]) * (1 + ξ[2]) * (1 - ξ[3]) * vert[3] + (1 + ξ[1]) * (1 + ξ[2]) * (1 - ξ[3]) * vert[4] +
                  (1 - ξ[1]) * (1 - ξ[2]) * (1 + ξ[3]) * vert[5] + (1 + ξ[1]) * (1 - ξ[2]) * (1 + ξ[3]) * vert[6] +
                  (1 - ξ[1]) * (1 + ξ[2]) * (1 + ξ[3]) * vert[7] + (1 + ξ[1]) * (1 + ξ[2]) * (1 + ξ[3]) * vert[8] )/ 8.0
    end
    return x
end
#--------------------------------------------------------
function trilinear_map_IJac_x_vec!(ξ::Array{FT}, x1v::Array{FT}, x2v::Array{FT}, x3v::Array{FT}, v::Array{FT}) where FT <: AbstractFloat
    Jac = MMatrix{3,3,FT,9}(undef)
    for (vert,dim) = zip((x1v,x2v,x3v),1:3)
        Jac[dim,1] = ((-1) * (1 - ξ[2]) * (1 - ξ[3]) * vert[1] + ( 1) * (1 - ξ[2]) * (1 - ξ[3]) * vert[2] +
                      (-1) * (1 + ξ[2]) * (1 - ξ[3]) * vert[3] + (+1) * (1 + ξ[2]) * (1 - ξ[3]) * vert[4] +
                      (-1) * (1 - ξ[2]) * (1 + ξ[3]) * vert[5] + (+1) * (1 - ξ[2]) * (1 + ξ[3]) * vert[6] +
                      (-1) * (1 + ξ[2]) * (1 + ξ[3]) * vert[7] + (+1) * (1 + ξ[2]) * (1 + ξ[3]) * vert[8] )/ 8.0

        Jac[dim,2] = ((1 - ξ[1]) * (-1) * (1 - ξ[3]) * vert[1] + (1 + ξ[1]) * (-1) * (1 - ξ[3]) * vert[2] +
                      (1 - ξ[1]) * (+1) * (1 - ξ[3]) * vert[3] + (1 + ξ[1]) * (+1) * (1 - ξ[3]) * vert[4] +
                      (1 - ξ[1]) * (-1) * (1 + ξ[3]) * vert[5] + (1 + ξ[1]) * (-1) * (1 + ξ[3]) * vert[6] +
                      (1 - ξ[1]) * (+1) * (1 + ξ[3]) * vert[7] + (1 + ξ[1]) * (+1) * (1 + ξ[3]) * vert[8] )/ 8.0

        Jac[dim,3] = ((1 - ξ[1]) * (1 - ξ[2]) * (-1) * vert[1] + (1 + ξ[1]) * (1 - ξ[2]) * (-1) * vert[2] +
                      (1 - ξ[1]) * (1 + ξ[2]) * (-1) * vert[3] + (1 + ξ[1]) * (1 + ξ[2]) * (-1) * vert[4] +
                      (1 - ξ[1]) * (1 - ξ[2]) * (+1) * vert[5] + (1 + ξ[1]) * (1 - ξ[2]) * (+1) * vert[6] +
                      (1 - ξ[1]) * (1 + ξ[2]) * (+1) * vert[7] + (1 + ξ[1]) * (1 + ξ[2]) * (+1) * vert[8] )/ 8.0
    end
    LinearAlgebra.LAPACK.gesv!(Jac,v)

    return nothing 
end
#--------------------------------------------------------
function interpolate_cubed_sphere!(intrp_cs::InterpolationCubedSphere, sv::AbstractArray{FT}, st_no::T) where {T <: Integer, FT <: AbstractFloat}
    qm1 = intrp_cs.poly_order + 1
    Nel = length(intrp_cs.realelems)
    m1_r, m1_w = GaussQuadrature.legendre(FT,qm1,GaussQuadrature.both)
    wb = Elements.baryweights(m1_r)
    phir = MVector{qm1,FT}(undef)
    phis = MVector{qm1,FT}(undef)
    phit = MVector{qm1,FT}(undef)

    vout    = FT(0)
    vout_ii = FT(0)
    vout_ij = FT(0)

    for el in 1:Nel #-----for each element elno 
        np = length(intrp_cs.ξ1[el])
        lag    = @view sv[:,st_no,el]
        for i in 1:np # interpolating point-by-point
            interpolationvector_1pt!(m1_r,intrp_cs.ξ1[el][i],wb,phir)
            interpolationvector_1pt!(m1_r,intrp_cs.ξ2[el][i],wb,phis)
            interpolationvector_1pt!(m1_r,intrp_cs.ξ3[el][i],wb,phit)
            vout = 0.0
            for ik in 1:qm1
                vout_ij = 0.0
                for ij in 1:qm1 
                    vout_ii = 0.0 
                    for ii in 1:qm1
                        @inbounds vout_ii += lag[ii + (ij-1)*qm1 + (ik-1)*qm1*qm1] * phir[ii]
                    end
                    @inbounds vout_ij += vout_ii * phis[ij]
                end
                @inbounds vout += vout_ij * phit[ik]
            end
            @inbounds intrp_cs.V[el][i] = vout 
        end
  #--------------------
  end
  #--------------------
end
#--------------------------------------------------------
end # module interploation
