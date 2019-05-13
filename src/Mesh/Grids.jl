module Grids
using ..Topologies

export DiscontinuousSpectralElementGrid, AbstractGrid
export dofs_per_element, arraytype, dimensionality, polynomialorder
import Canary

abstract type AbstractGrid{FloatType, dim, polynomialorder, numberofDOFs,
                         DeviceArray} end

dofs_per_element(::AbstractGrid{T, D, N, Np}) where {T, D, N, Np} = Np

polynomialorder(::AbstractGrid{T, dim, N}) where {T, dim, N} = N

dimensionality(::AbstractGrid{T, dim}) where {T, dim} = dim

arraytype(::AbstractGrid{T, D, N, Np, DA}) where {T, D, N, Np, DA} = DA

# {{{
const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _M, _MI,
       _x, _y, _z = 1:_nvgeo
const vgeoid = (ξxid = _ξx, ηxid = _ηx, ζxid = _ζx,
                ξyid = _ξy, ηyid = _ηy, ζyid = _ζy,
                ξzid = _ξz, ηzid = _ηz, ζzid = _ζz,
                Mid  = _M , MIid = _MI,
                xid  = _x , yid  = _y , zid  = _z)

const _nsgeo = 5
const _nx, _ny, _nz, _sM, _vMI = 1:_nsgeo
const sgeoid = (nxid = _nx, nyid = _ny, nzid = _nz, sMid = _sM,
                vMIid = _vMI)
# }}}

"""
    DiscontinuousSpectralElementGrid(topology; FloatType, DeviceArray,
                                     polynomialorder,
                                     meshwarp = (x...)->identity(x))

Generate a discontinuous spectral element (tensor product,
Legendre-Gauss-Lobatto) grid/mesh from a `topology`, where the order of the
elements is given by `polynomialorder`. `DeviceArray` gives the array type used
to store the data (`CuArray` or `Array`), and the coordinate points will be of
`FloatType`.

The optional `meshwarp` function allows the coordinate points to be warped after
the mesh is created; the mesh degrees of freedom are orginally assigned using a
trilinear blend of the element corner locations.
"""
struct DiscontinuousSpectralElementGrid{T, dim, N, Np, DA,
                                        DAT2, DAT3, DAT4, DAI1, DAI2, DAI3,
                                        TOP
                                       } <: AbstractGrid{T, dim, N, Np, DA }
  "mesh topology"
  topology::TOP

  "volume metric terms"
  vgeo::DAT3

  "surface metric terms"
  sgeo::DAT4

  "element to boundary condition map"
  elemtobndy::DAI2

  "volume DOF to element minus side map"
  vmapM::DAI3

  "volume DOF to element plus side map"
  vmapP::DAI3

  "list of elements that need to be communicated (in neighbors order)"
  sendelems::DAI1

  "1-D derivative operator on the device"
  D::DAT2

  function DiscontinuousSpectralElementGrid(topology::AbstractTopology{dim};
                                            FloatType = nothing,
                                            DeviceArray = nothing,
                                            polynomialorder = nothing,
                                            meshwarp::Function =
                                            (x...)->identity(x)) where dim
    @assert FloatType != nothing
    @assert DeviceArray != nothing
    @assert polynomialorder != nothing

    N = polynomialorder
    (ξ, ω) = Canary.lglpoints(FloatType, N)
    D = Canary.spectralderivative(ξ)

    (vmapM, vmapP) = mappings(N, topology.elemtoelem, topology.elemtoface,
                              topology.elemtoordr)

    (vgeo, sgeo) = computegeometry(topology, D, ξ, ω, meshwarp, vmapM)
    Np = (N+1)^dim
    @assert Np == size(vgeo, 1)

     # Create arrays on the device
     vgeo = DeviceArray(vgeo)
     sgeo = DeviceArray(sgeo)
     elemtobndy = DeviceArray(topology.elemtobndy)
     vmapM = DeviceArray(vmapM)
     vmapP = DeviceArray(vmapP)
     sendelems = DeviceArray(topology.sendelems)
     D = DeviceArray(D)

     # FIXME: There has got to be a better way!
     DAT2 = typeof(D)
     DAT3 = typeof(vgeo)
     DAT4 = typeof(sgeo)
     DAI1 = typeof(sendelems)
     DAI2 = typeof(elemtobndy)
     DAI3 = typeof(vmapM)
     TOP = typeof(topology)

    new{FloatType, dim, N, Np, DeviceArray, DAT2, DAT3, DAT4, DAI1, DAI2, DAI3,
        TOP
       }(topology, vgeo, sgeo, elemtobndy, vmapM, vmapP, sendelems, D)
  end
end

function Base.getproperty(G::DiscontinuousSpectralElementGrid, s::Symbol)
  if s ∈ keys(vgeoid)
    vgeoid[s]
  elseif s ∈ keys(sgeoid)
    sgeoid[s]
  else
    getfield(G, s)
  end
end

function Base.propertynames(G::DiscontinuousSpectralElementGrid)
  (fieldnames(DiscontinuousSpectralElementGrid)...,
   keys(vgeoid)..., keys(sgeoid)...)
end

# {{{ mappings
"""
    mappings(N, elemtoelem, elemtoface, elemtoordr)

This function takes in a polynomial order `N` and parts of a topology (as
returned from `connectmesh`) and returns index mappings for the element surface
flux computation.  The returned `Tuple` contains:

 - `vmapM` an array of linear indices into the volume degrees of freedom where
   `vmapM[:,f,e]` are the degrees of freedom indices for face `f` of element
    `e`.

 - `vmapP` an array of linear indices into the volume degrees of freedom where
   `vmapP[:,f,e]` are the degrees of freedom indices for the face neighboring
   face `f` of element `e`.
"""
function mappings(N, elemtoelem, elemtoface, elemtoordr)
  nface, nelem = size(elemtoelem)

  d = div(nface, 2)
  Np, Nfp = (N+1)^d, (N+1)^(d-1)

  p = reshape(1:Np, ntuple(j->N+1, d))
  fd(f) =   div(f-1,2)+1
  fe(f) = N*mod(f-1,2)+1
  fmask = hcat((p[ntuple(j->(j==fd(f)) ? (fe(f):fe(f)) : (:), d)...][:]
                for f=1:nface)...)
  inds = LinearIndices(ntuple(j->N+1, d-1))

  vmapM = similar(elemtoelem, Nfp, nface, nelem)
  vmapP = similar(elemtoelem, Nfp, nface, nelem)

  for e1 = 1:nelem, f1 = 1:nface
    e2 = elemtoelem[f1,e1]
    f2 = elemtoface[f1,e1]
    o2 = elemtoordr[f1,e1]

    vmapM[:,f1,e1] .= Np*(e1-1) .+ fmask[:,f1]

    if o2 == 1
      vmapP[:,f1,e1] .= Np*(e2-1) .+ fmask[:,f2]
    elseif d == 3 && o2 == 3
      n = 1
      @inbounds for j = 1:N+1, i = N+1:-1:1
        vmapP[n,f1,e1] = Np*(e2-1) + fmask[inds[i,j],f2]
        n+=1
      end
    else
      error("Orientation '$o2' with dim '$d' not supported yet")
    end
  end

  (vmapM, vmapP)
end
# }}}

# {{{ compute geometry
function computegeometry(topology::AbstractTopology{dim}, D, ξ, ω, meshwarp,
                         vmapM) where {dim}
  # Compute metric terms
  Nq = size(D, 1)
  DFloat = eltype(D)

  (nface, nelem) = size(topology.elemtoelem)

  # crd = creategrid(Val(dim), elemtocoord(topology), ξ)

  vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
  sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)

  (ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, MJ, MJI, x, y, z) =
      ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x)
  (nx, ny, nz, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
  Canary.creategrid!(X..., topology.elemtocoord, ξ)

  @inbounds for j = 1:length(x)
    (x[j], y[j], z[j]) = meshwarp(x[j], y[j], z[j])
  end

  # Compute the metric terms
  if dim == 1
    Canary.computemetric!(x, J, ξx, sJ, nx, D)
  elseif dim == 2
    Canary.computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  elseif dim == 3
    Canary.computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)
  end

  M = kron(1, ntuple(j->ω, dim)...)
  MJ .= M .* J
  MJI .= 1 ./ MJ
  vMJI .= MJI[vmapM]

  sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
  sMJ .= sM .* sJ

  (vgeo, sgeo)
end
# }}}


# {{{ compute element size for box domain 
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


# {{{ Computes `fcoe` for anisotropic grids using the definition of
#     Lilly in:
#     A. Scotti, C. Meneveau, D.K. Lilly, Generalized Smagorinsky model for anisotropic grids, Phys. Fluids 5 (1993) 2306–2308.
#
function compute_anisotropic_grid_factor(dim, Nq, vgeo, e)

    DFloat = eltype(vgeo)
    
    ds  = compute_element_size(dim, Nq, vgeo, e)
       
    Nq2 = Nq*Nq

    max_length_flg = 1 #WARNING should the user decide this?
    
    if dim == 2
        
        dx, dy = ds[1], ds[2]
        ds_mean = sqrt(dx*dx + dy*dy)

        delta  = sqrt(dx*dy/Nq2)
        delta2 = delta*delta
        
        #Compute Lambda:
        fcoe   = 1.0
        Lambda = fcoe*delta
        
    elseif dim == 3

        dx, dy, dz = ds[1], ds[2], ds[3]
        dx_mean, dy_mean, dz_mean = ds[4], ds[5], ds[6]
        
        ds_mean = (1 - max_length_flg)*min(dx_mean, dy_mean, dz_mean) + max_length_flg*max(dx_mean, dy_mean, dz_mean)
        
        delta = (dx_mean*dy_mean*dz_mean)^(1.0/3.0)
        #delta = (dx*dy*dz)^(1.0/3.0)
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
        
        #Compute Lambda:
        Lambda = fcoe*delta
    end
    
  return Lambda
end
# }}}



end
