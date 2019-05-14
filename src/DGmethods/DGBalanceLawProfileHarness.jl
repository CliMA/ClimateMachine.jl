using GPUifyLoops
include("DGBalanceLawDiscretizations_kernels.jl")
using Random
using StaticArrays

# From src/Mesh/Grids.jl
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

function DGProfiler(ArrayType, DFloat, dim, nelem, N, nstate, flux!,
                    numerical_flux!;
                    nauxstate = 0, source! = nothing,
                    numerical_boundary_flux! = nothing,
                    nviscstate = 0,
                    ngradstate = 0,
                    states_grad = (),
                    viscous_transform! = nothing,
                    gradient_transform! = nothing,
                    viscous_penalty! = nothing,
                    viscous_boundary_penalty! = nothing,
                    t = zero(DFloat),
                    rnd = MersenneTwister(0),
                    numerical_boundary_faces = 0,
                    stateoffset = ())

  # Generate a random mapping
  nface = 2dim

  elemtoelem = zeros(Int, nface, nelem)
  elemtobndy = zeros(Int, nface, nelem)
  elemtoface = zeros(Int, nface, nelem)
  elemtoordr = ones(Int, nface, nelem)

  Faces = Set(1:nelem*nface)

  for b = 1:numerical_boundary_faces
    gf = rand(Faces)
    pop!(Faces, gf)
    e = div(gf-1, nface) + 1
    f = ((gf-1) % nface) + 1

    @assert elemtoelem[f, e] == 0
    @assert elemtoface[f, e] == 0
    @assert elemtobndy[f, e] == 0

    elemtoelem[f, e] = e
    elemtoface[f, e] = f
    elemtobndy[f, e] = 1
  end

  for gf1 in Faces
    pop!(Faces, gf1)
    e1 = div(gf1-1, nface) + 1
    f1 = ((gf1-1) % nface) + 1

    @assert elemtoelem[f1, e1] == 0
    @assert elemtoface[f1, e1] == 0
    @assert elemtobndy[f1, e1] == 0

    gf2 = rand(rnd, Faces)
    pop!(Faces, gf2)
    e2 = div(gf2-1, nface) + 1
    f2 = ((gf2-1) % nface) + 1

    @assert elemtoelem[f2, e2] == 0
    @assert elemtoface[f2, e2] == 0
    @assert elemtobndy[f2, e2] == 0

    elemtoelem[f1, e1], elemtoelem[f2, e2] = e2, e1
    elemtoface[f1, e1], elemtoface[f2, e2] = f2, f1
  end
  vmapM, vmapP = mappings(N, elemtoelem, elemtoface, elemtoordr)

  # Generate random geometry terms and solutions
  Nq = N + 1
  Nqk = dim == 3 ? N + 1 : 1
  Np = Nq * Nq * Nqk
  Q = rand(rnd, DFloat, Np, nstate, nelem)
  for (s, offset) in stateoffset
    Q[:, s, :] .+= offset
  end
  Qvisc = rand(rnd, DFloat, Np, nviscstate, nelem)
  auxstate = rand(rnd, DFloat, Np, nauxstate, nelem)
  vgeo = rand(rnd, DFloat, Np, _nvgeo, nelem)
  sgeo = rand(rnd, DFloat, _nsgeo, Nq^(dim-1), nface, nelem)
  D = rand(rnd, DFloat, Nq, Nq)
  rhs = similar(Q)
  rhs .= 0


  # Make sure the entries of the mass matrix satisfy the inverse relation
  vgeo[:, _MJ, :] .+= 3
  vgeo[:, _MJI, :] .= 1 ./ vgeo[:, _MJ, :]

  (D                         = ArrayType(D),
   Q                         = ArrayType(Q),
   Qvisc                     = ArrayType(Qvisc),
   auxstate                  = ArrayType(auxstate),
   elemtobndy                = ArrayType(elemtobndy),
   rhs                       = ArrayType(rhs),
   sgeo                      = ArrayType(sgeo),
   vgeo                      = ArrayType(vgeo),
   vmapM                     = ArrayType(vmapM),
   vmapP                     = ArrayType(vmapP),
   N                         = N,
   dim                       = dim,
   elems                     = 1:nelem,
   flux!                     = flux!,
   gradient_transform!       = gradient_transform!,
   nauxstate                 = nauxstate,
   ngradstate                = ngradstate,
   nstate                    = nstate,
   numerical_boundary_flux!  = numerical_boundary_flux!,
   numerical_flux!           = numerical_flux!,
   nviscstate                = nviscstate,
   source!                   = source!,
   states_grad               = states_grad,
   t                         = t,
   viscous_boundary_penalty! = viscous_boundary_penalty!,
   viscous_penalty!          = viscous_penalty!,
   viscous_transform!        = viscous_transform!,
   device                    = ArrayType == Array ? CPU() : CUDA())
end

function volumerhs!(dg)
  Nq = dg.N+1
  Nqk = dg.dim == 2 ? 1 : Nq
  DEV = dg.device
  nelem = length(dg.elems)

  @launch(DEV, threads=(Nq, Nq, Nqk), blocks=nelem,
          volumerhs!(Val(dg.dim), Val(dg.N), Val(dg.nstate), Val(dg.nviscstate),
                     Val(dg.nauxstate), dg.flux!, dg.source!, dg.rhs, dg.Q,
                     dg.Qvisc, dg.auxstate, dg.vgeo, dg.t, dg.D, dg.elems))
end

function facerhs!(dg)
  DEV = dg.device
  Nq = dg.N+1
  Nqk = dg.dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nelem = length(dg.elems)

  @launch(DEV, threads=Nfp, blocks=nelem,
          facerhs!(Val(dg.dim), Val(dg.N), Val(dg.nstate), Val(dg.nviscstate),
                   Val(dg.nauxstate), dg.numerical_flux!,
                   dg.numerical_boundary_flux!, dg.rhs, dg.Q, dg.Qvisc,
                   dg.auxstate, dg.vgeo, dg.sgeo, dg.t, dg.vmapM, dg.vmapP,
                   dg.elemtobndy, dg.elems))
end

function volumeviscterms!(dg)
  Nq = dg.N+1
  Nqk = dg.dim == 2 ? 1 : Nq
  DEV = dg.device
  nelem = length(dg.elems)

  @launch(DEV, threads=(Nq, Nq, Nqk), blocks=nelem,
          volumeviscterms!(Val(dg.dim), Val(dg.N), Val(dg.nstate),
                           Val(dg.states_grad), Val(dg.ngradstate),
                           Val(dg.nviscstate), Val(dg.nauxstate),
                           dg.viscous_transform!, dg.gradient_transform!, dg.Q,
                           dg.Qvisc, dg.auxstate, dg.vgeo, dg.t, dg.D,
                           dg.elems))
end

function faceviscterms!(dg)
  DEV = dg.device
  Nq = dg.N+1
  Nqk = dg.dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nelem = length(dg.elems)

  @launch(DEV, threads=Nfp, blocks=nelem,
          faceviscterms!(Val(dg.dim), Val(dg.N), Val(dg.nstate),
                         Val(dg.states_grad), Val(dg.ngradstate),
                         Val(dg.nviscstate), Val(dg.nauxstate),
                         dg.viscous_penalty!, dg.viscous_boundary_penalty!,
                         dg.gradient_transform!, dg.Q, dg.Qvisc, dg.auxstate,
                         dg.vgeo, dg.sgeo, dg.t, dg.vmapM, dg.vmapP,
                         dg.elemtobndy, dg.elems))
end

function runall!(dg)
  volumerhs!(dg)
  facerhs!(dg)
  if dg.nviscstate > 0
    volumeviscterms!(dg)
    faceviscterms!(dg)
  end
end
