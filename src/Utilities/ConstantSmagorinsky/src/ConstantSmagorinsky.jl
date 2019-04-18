"""
This module contains functions that assist in computing viscosity parameters 
for one of the simpler subgrid turbulence models: Constant Coefficient Smagorinsky
"""
module ConstantSmagorinsky
using ...PlanetParameters


# FIXME: Improve documentation for this module !!!!! 
# Smagorinksy constant
Cs = 0.12 


export strainrate_tensor_mag, smag_viscosity, richardson_number

"""
Calculates the Smagorinsky viscosity coefficients based on the rate of strain tensor obtained from the kernel gradients
"""

"""

"""
function strainrate_tensor_mag(ux, uy, uz,
                                vx, vy, vz,
                                wx=0, wy=0, wz=0)
 # Arguments in w are optional, only required in 3D
 SijSij = ((ux * ux) + (uy + vx) + (uz + wx) + 
           (vz + wy) + (vy * vy) + (wz * wz))

 return sqrt(2 * SijSij)

end

"""
Calculate the eddy viscosities in the constant Smagorinsky model 
"""
function smag_viscosity(richardson_number, Cs, anisotropic_grid_factor, strainrate_tensor_mag)
    auxr = max(0.0, 1.0 - richardson_number)
    Km   = Coeff_smag * Coeff_smag * Δsqr * strainrate_tensor_mag * sqrt(auxr)
    # The factor of 3 comes from the KW 1978 paper
    Kh   = 3 * Km 
    return Kh
end

"""
function get_richardson_number(θ, θy)
θ is the virtual potential temperature
θy is the vertical gradient of the potential temperature
Get the gradient Richardson number for the Smagorinsky coefficient calculation
"""
function richardson_number(θ, θy)
  Ri = grav/θ * θy/(2 * strainrate_tensor_mag)
  return Ri
end


"""
Compute anisotropic grid factor 
fcoe for anisotropic grids using the definition of
Lilly in:
A. Scotti, C. Meneveau, D.K. Lilly, Generalized Smagorinsky model for anisotropic grids, Phys. Fluids 5 (1993) 2306–2308.
"""
function anisotropic_grid_factor(dim, Nq, vgeo, e)
    DFloat = eltype(vgeo)
    ds  = compute_element_size(dim, Nq, vgeo, e)
    Nq2 = Nq*Nq
    max_length_flg = 1
    if dim == 2
        dx, dy = ds[1], ds[2]
        ds_mean = sqrt(dx*dx + dy*dy)
        delta  = sqrt(dx*dy/Nq2)
        delta2 = delta*delta
        #Compute Λ:
        fcoe   = 1.0
        Λ = fcoe*delta
    elseif dim == 3
        dx, dy, dz = ds[1], ds[2], ds[3]
        dx_mean, dy_mean, dz_mean = ds[4], ds[5], ds[6]
        ds_mean = (1 - max_length_flg)*min(dx_mean, dy_mean, dz_mean) + max_length_flg*max(dx_mean, dy_mean, dz_mean)
        delta = (dx_mean*dy_mean*dz_mean)^(1.0/3.0)
        delta2 = delta*delta
        fcoe = 1.0
        #Get the two smaller dimensions of the cell:
        if (dx > dy && dx > dz)
            deltai = dy
            deltak = dz
        elseif (dy > dx && dy > dz)
            deltai = dx
            deltak = dz
        elseif (dz > dx && dz > dy)
            deltai = dx
            deltak = dy
        else
            deltai = dx
            deltak = dy
        end
        a1 = deltai/max(dx, dy, dz)
        a2 = deltak/max(dx, dy, dz)
        fcoe = cosh((4.0/27.0)*(log(a1)*log(a1) - log(a1)*log(a2) + log(a2)*log(a2)))
        #Compute Λ:
        Λ = fcoe*delta
    end
  return Λ * Λ
end

"""
Computes element size for a stackedbrick topology
TODO: Extend to arbitrary unstructured grids ? 
FIXME: How best to handle these module files that need grid information ? 
"""
function compute_element_size(dim, Nq, vgeo, e) 
    DFloat = eltype(vgeo)
    if (dim == 2)
        x, y = zeros(DFloat, 4), zeros(DFloat, 4)
        x[1], y[1] = vgeo[1, 1,   _x, e], vgeo[1, 1,   _y, e]
        x[2], y[2] = vgeo[Nq, 1,  _x, e], vgeo[Nq, 1,  _y, e]
        x[3], y[3] = vgeo[1, Nq,  _x, e], vgeo[1, Nq,  _y, e]
        x[4], y[4] = vgeo[Nq, Nq, _x, e], vgeo[Nq, Nq, _y, e]        
        #Element sizes (as if it were linear)
        dx = maximum(x[:]) - minimum(x[:])
        dy = maximum(y[:]) - minimum(y[:])
        #Average distance between LGL points inside the element:
        dx_mean = dx/max(Nq - 1, 1)
        dy_mean = dy/max(Nq - 1, 1)
        ds = (dx, dy, dx_mean, dy_mean)
    elseif (dim == 3)
        x, y, z = zeros(DFloat, 8), zeros(DFloat, 8), zeros(DFloat, 8)
        x[1], y[1], z[1] = vgeo[1, 1,   1, _x, e], vgeo[1, 1,   1, _y, e], vgeo[1, 1,   1, _z, e]
        x[2], y[2], z[2] = vgeo[Nq, 1,  1, _x, e], vgeo[Nq, 1,  1, _y, e], vgeo[Nq, 1,  1, _z, e]
        x[3], y[3], z[3] = vgeo[1, Nq,  1, _x, e], vgeo[1, Nq,  1, _y, e], vgeo[1, Nq,  1, _z, e]
        x[4], y[4], z[4] = vgeo[Nq, Nq, 1, _x, e], vgeo[Nq, Nq, 1, _y, e], vgeo[Nq, Nq, 1, _z, e]
        x[5], y[5], z[5] = vgeo[1, 1,   Nq, _x, e], vgeo[1, 1,   Nq, _y, e], vgeo[1, 1,   Nq, _z, e]
        x[6], y[6], z[6] = vgeo[Nq, 1,  Nq, _x, e], vgeo[Nq, 1,  Nq, _y, e], vgeo[Nq, 1,  Nq, _z, e]
        x[7], y[7], z[7] = vgeo[1, Nq,  Nq, _x, e], vgeo[1, Nq,  Nq, _y, e], vgeo[1, Nq,  Nq, _z, e]
        x[8], y[8], z[8] = vgeo[Nq, Nq, Nq, _x, e], vgeo[Nq, Nq, Nq, _y, e], vgeo[Nq, Nq, Nq, _z, e]
        #Element sizes (as if it were linear)
        dx = maximum(x[:]) - minimum(x[:])
        dy = maximum(y[:]) - minimum(y[:])
        dz = maximum(z[:]) - minimum(z[:])
        #Average distance between LGL points inside the element:
        dx_mean = dx/max(Nq - 1, 1)
        dy_mean = dy/max(Nq - 1, 1)
        dz_mean = dz/max(Nq - 1, 1)
        ds = (dx, dy, dz, dx_mean, dy_mean, dz_mean)
    end
  return ds
end
# }}}

