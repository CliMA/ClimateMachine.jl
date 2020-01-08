module Courant

using Dates
using FileIO
using JLD2
using MPI
using StaticArrays
using Printf

using ..Atmos
using ..DGmethods: num_state, vars_state, num_aux, vars_aux, vars_diffusive, num_diffusive, DGModel
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using ..MPIStateArrays
using ..PlanetParameters
using ..VariableTemplates

export gather_Courant

"""
    gather_Courant(mpicomm::MPI.Comm,
                   dg::DGModel,
                   Q::MPIStateArray,
                   xmax::FT,
                   ymax::FT,
                   CFLmax_theoretical::FT,
                   out_dir::AbstractString,
                   Dx::FT,
                   Dy::FT,
                   Dz::FT,
                   dt_inout) where {FT}


"""
function gather_Courant(mpicomm::MPI.Comm,
                        dg::DGModel,
                        Q::MPIStateArray{FT},
                        xmax::FT,
                        ymax::FT,
                        CFLmax_theoretical::FT,
                        out_dir::AbstractString,
                        Dx::FT,
                        Dy::FT,
                        Dz::FT,
                        dt_inout) where {FT}
    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)

    dt = dt_inout[]

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
    max_sound_speed = MPI.Allreduce(ss,MPI.MAX,mpicomm)
    Cour_sound_x = dt*max_sound_speed/Dx
    Cour_sound_y = dt*max_sound_speed/Dy
    Cour_sound_z = dt*max_sound_speed/Dz
    Couru = dt*MPI.Allreduce(sau,MPI.MAX,mpicomm)/Dx
    Courv = dt*MPI.Allreduce(sav,MPI.MAX,mpicomm)/Dy
    Courw = dt*MPI.Allreduce(saw,MPI.MAX,mpicomm)/Dz
    max_sdiff = MPI.Allreduce(sdiff,MPI.MAX,mpicomm)
    Courdx = dt*max_sdiff/Dx^2
    Courdy = dt*max_sdiff/Dy^2
    Courdz = dt*max_sdiff/Dz^2

    cfl_h  = max(Cour_sound_x, Cour_sound_y)
    cfl_v  = Cour_sound_z
    cflu_h = max(Couru, Courv)
    cflu_v = Courw

    cfldiff_h = max(Courdx, Courdy)
    cfldiff_v = Courdz

    CFLh = max(cfl_h, cflu_h, cfldiff_h)
    CFLv = max(cfl_v, cflu_v, cfldiff_v)

    CFLh = min(CFLh, CFLmax_theoretical)
    CFLv = min(CFLv, CFLmax_theoretical)

    dt_exp = CFLh * Dz          / max_sound_speed
    dt_imp = CFLv * min(Dx, Dy) / max_sound_speed

    @info @sprintf """Courant numbers:
           dt_exp           = %.5e
           dt_imp           = %.5e
           CFL_h            = %.5f
           CFL_v            = %.5f
           CFLadv_h         = %.5f
           CFLadv_v         = %.5f
           CFLdiff_h        = %.5f
           CFLdiff_v        = %.5f
""" dt_exp dt_imp cfl_h cfl_v cflu_h cflu_v cfldiff_h cfldiff_v

    dt_inout[] = dt_imp

    return nothing

end

end #module
