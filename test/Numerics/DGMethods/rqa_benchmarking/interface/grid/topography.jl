export AbstractTopography,
    NoTopography,
    DCMIPTopography,
    EarthTopography,
    cubed_sphere_topo_warp,
    compute_long_lat,
    topography_warp,
    compute_topography,
    equiangular_cubed_sphere_topo_warp,

const data_land_topo = NCDataset("/Users/asridhar/Research/Codes/ClimateMachine.jl/topodata.nc");
long = (data_land_topo["X"][:] .+ 180) .* π/180; # Longitude in degrees [-180 to 180; +180 shift]
lat = data_land_topo["Y"][:] .* π/180; # Latitude in degrees [-90 to 90]
lat = reverse(lat)
elev = data_land_topo["topo"][:]; # Elevation in meters [0 to Everest] No Bathmetry
elev = reverse(elev,dims=2)
skip_var = 3;
const get_elevation = Spline2D(long[1:skip_var:end],lat[1:skip_var:end],elev[1:skip_var:end,1:skip_var:end], kx = 4, ky=4)

equiangular_cubed_sphere_topo_warp(a, b, c, R = max(abs(a), abs(b), abs(c))) =
    cubed_sphere_topo_warp(EquiangularCubedSphere(), a, b, c, R; 
                           domain = nothing,
                           topography = EarthTopography(get_elevation))

function topography_warp(f, domain, topography)
    function equiangular_cubed_sphere_topo_warp(a, b, c, R=max(abs(a),abs(b),abs(c)))
        return f(
            EquiangularCubedSphere(),
            a,
            b,
            c,
            max(abs(a), abs(b), abs(c));
            domain = nothing,
            topography = topography,
        )
    end
    return equiangular_cubed_sphere_topo_warp 
end

"""
    AbstractTopography
Abstract type to allow dispatch over different analytical topography prescriptions
in experiments.
"""
abstract type AbstractTopography end

function compute_topography(
    ::AbstractTopography,
    λ,
    ϕ,
    sR,
    (X,Y,δ,faceid)
)
    return sR
end

"""
    NoTopography <: AbstractTopography
Allows definition of fallback methods in case cubed_sphere_topo_warp is used with
no prescribed topography function.
"""
struct NoTopography <: AbstractTopography end

### DCMIP Mountain
"""
    DCMIPTopography <: AbstractTopography
Topography description based on standard DCMIP experiments.
"""
struct DCMIPTopography <: AbstractTopography end
function compute_topography(
    ::DCMIPTopography,
    λ,
    ϕ,
    sR,
    (X,Y,δ,faceid),
)
    #User specified warp parameters
    R_m = π * 3 / 4
    h0 = 0.05
    ζ_m = π / 16
    φ_m = 0
    λ_m = π * 3 / 2
    r_m = acos(sin(φ_m) * sin(ϕ) + cos(φ_m) * cos(ϕ) * cos(λ - λ_m))
    # Define mesh decay profile
    Δ = 1.0 
    if r_m < R_m
        zs =
            0.5 *
            h0 *
            (1 + cos(π * r_m / R_m)) *
            cos(π * r_m / ζ_m) *
            cos(π * r_m / ζ_m)
    else
        zs = 0.0
    end
    mR = sign(sR) * (abs(sR) + zs * Δ)
    return mR
end

"""
    cubed_sphere_topo_warp(a, b, c, R = max(abs(a), abs(b), abs(c));
                       r_inner = _planet_radius,
                       r_outer = _planet_radius + domain_height,
                       topography = NoTopography())

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equiangular gnomonic grid proposed by
[Ronchi1996](@cite). Assumes a user specified modified radius using the
compute_topography function. Defaults to smooth cubed sphere unless otherwise specified
via the AbstractTopography type.
"""

"""
    EarthTopography <: AbstractTopography
"""
struct EarthTopography <: AbstractTopography 
    topo_spline
end
function compute_topography(
    lst::EarthTopography,
    λ,
    ϕ,
    sR,
    (X,Y, δ,faceid);
)
    FT = eltype(sR)
    Δ = 1#(r_outer - abs(sR)) / (r_outer - r_inner)
    zs = -0
    if lst.topo_spline(λ,ϕ) > 0.0
        zs = lst.topo_spline(λ,ϕ)
    end
    mR = sign(sR) * (abs(sR) + zs * Δ)
    return mR, mask
end

#function cubed_sphere_topo_warp(
#    ::EquiangularCubedSphere,
#    a,
#    b,
#    c,
#    R = max(abs(a), abs(b), abs(c));
#    domain = nothing,
#    topography::AbstractTopography = NoTopography(),
#)
#    function f(sR, ξ, η, faceid, (a,b,c))
#        X, Y = tan(π * ξ / 4), tan(π * η / 4)
#        δ = 1 + X^2 + Y^2
#        #λ, ϕ = compute_long_lat(X, Y, δ, faceid)
#        ϕ = asin(c/R)
#        t = b/a 
#        λ = -0
#        function get_λ(a,b,c)
#        end
#        if a >= -0 && b >= -0  # Quadrant 1
#            λ += atan(t)
#        elseif a < -0 && b >= -0 # Quadrant 2
#            λ += atan(-t)+π/2
#        elseif a < -0 && b < -0 # Quadrant 3
#            λ += atan(t)+π
#        elseif a >= -0 && b < -0 # Quadrant 4
#            λ += atan(-t) + 3π/2
#        end
#        mR = compute_topography(
#            topography,
#            λ,
#            ϕ,
#            sR,
#            (X,Y,δ,faceid),
#        )
#        x1 = mR / sqrt(δ)
#        x2, x3 = X * x1, Y * x1
#        x1, x2, x3
#    end
#    fdim = argmax(abs.((a, b, c)))
#    if fdim == 1 && a < 0
#        faceid = 1
#        # (-R, *, *) : Face I from Ronchi, Iacono, Paolucci (1996)
#        x1, x2, x3 = f(-R, b / a, c / a, faceid, (a,b,c))
#    elseif fdim == 2 && b < 0
#        faceid = 2
#        # ( *,-R, *) : Face II from Ronchi, Iacono, Paolucci (1996)
#        x2, x1, x3 = f(-R, a / b, c / b, faceid, (a,b,c))
#    elseif fdim == 1 && a > 0
#        faceid = 3
#        # ( R, *, *) : Face III from Ronchi, Iacono, Paolucci (1996)
#        x1, x2, x3 = f(R, b / a, c / a, faceid, (a,b,c))
#    elseif fdim == 2 && b > 0
#        faceid = 4
#        # ( *, R, *) : Face IV from Ronchi, Iacono, Paolucci (1996)
#        x2, x1, x3 = f(R, a / b, c / b, faceid, (a,b,c))
#    elseif fdim == 3 && c > 0
#        faceid = 5
#        # ( *, *, R) : Face V from Ronchi, Iacono, Paolucci (1996)
#        x3, x2, x1 = f(R, b / c, a / c, faceid, (a,b,c))
#    elseif fdim == 3 && c < 0
#        faceid = 6
#        # ( *, *,-R) : Face VI from Ronchi, Iacono, Paolucci (1996)
#        x3, x2, x1 = f(-R, b / c, a / c, faceid, (a,b,c))
#    else
#        error("invalid case for cubed_sphere_warp(::EquiangularCubedSphere): $a, $b, $c")
#    end
#    
#    return x1, x2, x3
#
#end

function cubed_sphere_topo_warp(
    ::EquiangularCubedSphere,
    a,
    b,
    c,
    R = max(abs(a), abs(b), abs(c));
    domain = nothing,
    topography::AbstractTopography = NoTopography(),
)
    function f(sR, ξ, η, faceid, (a,b,c))
        X, Y = tan(π * ξ / 4), tan(π * η / 4)
        δ = 1 + X^2 + Y^2
        x1 = sR / sqrt(δ)
        x2, x3 = X * x1, Y * x1
        x1, x2, x3
    end
   function g(sR, ξ, η, faceid, (a,b,c))
       X, Y = tan(π * ξ / 4), tan(π * η / 4)
       δ = 1 + X^2 + Y^2
       ϕ = asin(c/R)
       t = abs(b/a) 
       λ = -0
       if a >= -0 && b >= -0  # Quadrant 1
           λ += atan(t)
       elseif a < -0 && b >= -0 # Quadrant 2
           λ += π - atan(t)
       elseif a < -0 && b < -0 # Quadrant 3
           λ += π + atan(t)
       elseif a >= -0 && b < -0 # Quadrant 4
           λ += 2π - atan(t) 
       end
       mR = compute_topography(
           topography,
           λ,
           ϕ,
           sR,
           (X,Y,δ,faceid),
       )
       x1 = mR / sqrt(δ)
       x2, x3 = X * x1, Y * x1
       x1, x2, x3
   end
    fdim = argmax(abs.((a, b, c)))
    if fdim == 1 && a < 0
        faceid = 1
        # (-R, *, *) : Face I from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = f(-R, b / a, c / a, faceid, (a,b,c))
    elseif fdim == 2 && b < 0
        faceid = 2
        # ( *,-R, *) : Face II from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = f(-R, a / b, c / b, faceid, (a,b,c))
    elseif fdim == 1 && a > 0
        faceid = 3
        # ( R, *, *) : Face III from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = f(R, b / a, c / a, faceid, (a,b,c))
    elseif fdim == 2 && b > 0
        faceid = 4
        # ( *, R, *) : Face IV from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = f(R, a / b, c / b, faceid, (a,b,c))
    elseif fdim == 3 && c > 0
        faceid = 5
        # ( *, *, R) : Face V from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = f(R, b / c, a / c, faceid, (a,b,c))
    elseif fdim == 3 && c < 0
        faceid = 6
        # ( *, *,-R) : Face VI from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = f(-R, b / c, a / c, faceid, (a,b,c))
    else
        error("invalid case for cubed_sphere_warp(::EquiangularCubedSphere): $a, $b, $c")
    end
    a,b,c = x1,x2,x3
    if fdim == 1 && a < 0
        faceid = 1
        # (-R, *, *) : Face I from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = g(-R, b / a, c / a, faceid, (a,b,c))
    elseif fdim == 2 && b < 0
        faceid = 2
        # ( *,-R, *) : Face II from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = g(-R, a / b, c / b, faceid, (a,b,c))
    elseif fdim == 1 && a > 0
        faceid = 3
        # ( R, *, *) : Face III from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = g(R, b / a, c / a, faceid, (a,b,c))
    elseif fdim == 2 && b > 0
        faceid = 4
        # ( *, R, *) : Face IV from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = g(R, a / b, c / b, faceid, (a,b,c))
    elseif fdim == 3 && c > 0
        faceid = 5
        # ( *, *, R) : Face V from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = g(R, b / c, a / c, faceid, (a,b,c))
    elseif fdim == 3 && c < 0
        faceid = 6
        # ( *, *,-R) : Face VI from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = g(-R, b / c, a / c, faceid, (a,b,c))
    else
        error("invalid case for cubed_sphere_warp(::EquiangularCubedSphere): $a, $b, $c")
    end
    return x1, x2, x3
end

### Helper Functions for Topography Calculations
"""
    compute_long_lat(X,Y,δ,faceid)
Helper function to allow computation of latitute and longitude coordinates
given the cubed sphere coordinates X, Y, δ, faceid
"""
function compute_long_lat(X, Y, δ, faceid)
    if faceid == 1
        λ = atan(X)                     # longitude
        ϕ = -atan(cos(λ) * Y)            # latitude
    elseif faceid == 2
        λ = atan(X) + π / 2
        ϕ = -atan(Y * cos(atan(X)))
    elseif faceid == 3
        λ = atan(X) + π
        ϕ = atan(Y * cos(atan(X)))
    elseif faceid == 4
        λ = atan(X) + (3 / 2) * π
        ϕ = atan(Y * cos(atan(X)))
    elseif faceid == 5
        λ = atan(X, -Y) 
        ϕ = atan(1 / sqrt(δ - 1)) 
    elseif faceid == 6
        λ = atan(X, Y)
        ϕ = atan(1 / sqrt(δ - 1))
    end
    return λ, ϕ
end

