module Courant

using Dates
using FileIO
using JLD2
using MPI
using StaticArrays

using ..Atmos
using ..DGmethods: num_state, vars_state, num_aux, vars_aux, vars_diffusive, num_diffusive
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using ..MPIStateArrays
using ..PlanetParameters
using ..VariableTemplates

export gather_Courant








function gather_Courant(mpicomm, dg, Q,
                            xmax, ymax, out_dir,Dx,Dy,Dz,dt)
    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)

    # extract grid information
    bl = dg.balancelaw
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get the state, auxiliary and geo variables onto the host if needed
    if Array ∈ typeof(Q).parameters
        localQ    = Q.realdata
        localaux  = dg.auxstate.realdata
        localvgeo = grid.vgeo
        localdiff = dg.diffstate.realdata
    else
        localQ    = Array(Q.realdata)
        localaux  = Array(dg.auxstate.realdata)
        localvgeo = Array(grid.vgeo)
        localdiff = Array(dg.diffstate.realdata)
    end
    FT = eltype(localQ)


    zvals = zeros(Nqk, nvertelem)
    thermoQ = zeros(Nq*Nq*Nqk,1,nrealelem)
    ss = 0
    sau = 0
    sav = 0
    saw = 0
    sdiff = 0
    for eh in 1:nhorzelem
            for ev in 1:nvertelem
                e = ev + (eh - 1) * nvertelem
                for k in 1:Nqk
                    for j in 1:Nq
                        for i in 1:Nq
                         ijk = i + Nq * ((j-1) + Nq * (k-1))
                          u̅ = localQ[ijk,2,e] / localQ[ijk,1,e]
                          v̅ = localQ[ijk,3,e] / localQ[ijk,1,e]
                          w̅ = localQ[ijk,4,e] / localQ[ijk,1,e]
                          e̅_tot = localQ[ijk,5,e] / localQ[ijk,1,e]
                          q̅_tot = localQ[ijk,6,e] / localQ[ijk,1,e]

                          e_int = e̅_tot - 1//2 * (u̅^2 + v̅^2 + w̅^2) - grav * localvgeo[ijk,grid.x3id,e]

                          ts = PhaseEquil(convert(FT, e_int), localQ[ijk,1,e],q̅_tot)
                          thermoQ[ijk,1,e]=soundspeed_air(ts)
                          if ss <= thermoQ[ijk,1,e]
                          ss = abs(thermoQ[ijk,1,e])
                          end
                          if abs(sau) <= abs(localQ[ijk,2,e]) / localQ[ijk,1,e]
                          sau = abs(localQ[ijk,2,e]) / localQ[ijk,1,e]
                          end
                          if abs(sav) <= abs(localQ[ijk,2,e]) / localQ[ijk,1,e]
                          sav = abs(localQ[ijk,2,e]) / localQ[ijk,1,e]
                          end
                          if abs(saw) <= abs(localQ[ijk,4,e]) / localQ[ijk,1,e]
                          saw = abs(localQ[ijk,4,e]) / localQ[ijk,1,e]
                          end
                          if abs(sdiff) <= abs(localdiff[ijk,3,e]) / localQ[ijk,1,e]
                          sdiff = abs(localdiff[ijk,3,e]) / localQ[ijk,1,e]
                          end
                        end
                    end
                end
             end
     end
Courssx = dt*MPI.Allreduce(ss,MPI.MAX,mpicomm)/Dx
Courssy = dt*MPI.Allreduce(ss,MPI.MAX,mpicomm)/Dy
Courssz = dt*MPI.Allreduce(ss,MPI.MAX,mpicomm)/Dz
Couru = dt*MPI.Allreduce(sau,MPI.MAX,mpicomm)/Dx
Courv = dt*MPI.Allreduce(sav,MPI.MAX,mpicomm)/Dy
Courw = dt*MPI.Allreduce(saw,MPI.MAX,mpicomm)/Dz
Courdx = dt*MPI.Allreduce(sdiff,MPI.MAX,mpicomm)/Dx^2
Courdy = dt*MPI.Allreduce(sdiff,MPI.MAX,mpicomm)/Dy^2
Courdz = dt*MPI.Allreduce(sdiff,MPI.MAX,mpicomm)/Dz^2

@info thermoQ[50,1,50]
@info "courant sound + advection x =",Courssx+Couru
@info "courant sound + advection y =",Courssy+Courv
@info "courant sound + advection z =",Courssz+Courw
@info "courant diffusive x =",Courdx
@info "courant diffusive y =",Courdy
@info "courant diffusive z =",Courdz


    return nothing
end # function gather_Courant

end # module Courant
