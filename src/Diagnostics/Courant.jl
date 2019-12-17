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

include("diagnostic_vars.jl")

 #, vars_diffusive, num_diffusive
function extract_state(dg, localQ, ijk, e)
    bl = dg.balancelaw
    FT = eltype(localQ)
    nstate = num_state(bl, FT)
    l_Q = MArray{Tuple{nstate},FT}(undef)
    for s in 1:nstate
        l_Q[s] = localQ[ijk,s,e]
    end
    return Vars{vars_state(bl, FT)}(l_Q)
end

function extract_diffusion(dg, localdiff, ijk, e)
    bl = dg.balancelaw
    #bl = dg.diffstate
    FT = eltype(localdiff)
    ndiff = num_diffusive(bl, FT)
    l_diff = MArray{Tuple{ndiff},FT}(undef)
    for s in 1:ndiff
        l_diff[s] = localdiff[ijk,s,e]
    end
    return Vars{vars_diffusive(bl, FT)}(l_diff)
end

# thermodynamic variables of interest
function vars_thermo(FT)
  @vars begin
    sound::FT
  end
end
num_thermo(FT) = varsize(vars_thermo(FT))
thermo_vars(array) = Vars{vars_thermo(eltype(array))}(array)

function compute_thermo!(FT, state, diffusive_flx, i, j, k, ijk, ev, eh, e,
                         x, y, z, zvals, thermoQ)
    zvals[k,ev] = z

    u̅ = state.ρu[1] / state.ρ
    v̅ = state.ρu[2] / state.ρ
    w̅ = state.ρu[3] / state.ρ
    e̅_tot = state.ρe / state.ρ
    q̅_tot = state.moisture.ρq_tot / state.ρ

    e_int = e̅_tot - 1//2 * (u̅^2 + v̅^2 + w̅^2) - grav * z

    ts = PhaseEquil(convert(FT, e_int), q̅_tot, state.ρ)
    th = thermo_vars(thermoQ[ijk,e])
    th.sound = soundspeed_air(ts)
    
end

# horizontal averages
function vars_horzavg(FT)
  @vars begin
    ρ::FT
    ρu::FT
    ρv::FT
    ρw::FT
    e_tot::FT
    ρq_tot::FT
    q_liq::FT
    q_vap::FT
    θ_liq_ice::FT
    θ_dry::FT
    θ_v::FT
    e_int::FT
    h_m::FT
    h_t::FT
    qt_sgs::FT
    ht_sgs::FT      
  end
end
num_horzavg(FT) = varsize(vars_horzavg(FT))
horzavg_vars(array) = Vars{vars_horzavg(eltype(array))}(array)

function compute_horzsums!(FT, state, diffusive_flx, i, j, k, ijk, ev, eh, e, x, y, z, MH,
                           Nq, xmax, ymax, Nqk, nvertelem, localaux,
                           LWP, thermoQ, horzsums, repdvsr)
    #rep = node_adjustment(i, j, Nq, x, xmax, y, ymax)
    th = thermo_vars(thermoQ[ijk,e])
    hs = horzavg_vars(horzsums[k,ev])
    repdvsr[Nqk * (ev - 1) + k] += MH

    # liquid water path
    # this condition is also going to be used to get the number of points that
    # exist on a horizontal plane provided all planes have the same number of
    # points
    # TODO adjust for possibility of non equivalent horizontal slabs
    if ev == floor(nvertelem/2) && k == floor(Nqk/2)
        # TODO: uncomment the line below after rewriting the LWP assignment below using aux.∫dz...?
        # aux = extract_aux(dg, localaux, ijk, e)
        LWP[1] += MH * (localaux[ijk,1,e] + localaux[ijk,2,e])
        #repdvsr[1] += MH # number of points to be divided by
    end
end


function horz_average_all(FT, mpicomm, num, (Nqk, nvertelem), sums, repdvsr)
    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)
    avgs = [zeros(FT, num) for _ in 1:Nqk, _ in 1:nvertelem]
    for ev in 1:nvertelem
        for k in 1:Nqk
            for n in 1:num
                avgs[k,ev][n] = MPI.Reduce(sums[k,ev][n], +, 0, mpicomm)
                if mpirank == 0
                    avgs[k,ev][n] /= repdvsr[Nqk *(ev - 1) + k] 
                end
            end
        end
    end
    return avgs
end

"""
    gather_diagnostics(mpicomm, dg, Q, current_time_string, xmax, ymax, out_dir)
Compute various diagnostic variables and write them to JLD2 files in `out_dir`,
indexed by `current_time_string`.
"""
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

    nstate    = num_state(bl, FT)
    nauxstate = num_aux(bl, FT)
    ndiff     = num_diffusive(bl, FT)

    # traverse the grid, running each of `funs` on each node
    function visitQ(FT, funs::Vector{Function})
        for eh in 1:nhorzelem
            for ev in 1:nvertelem
                e = ev + (eh - 1) * nvertelem
                for k in 1:Nqk
                    for j in 1:Nq
                        for i in 1:Nq
                            
                            ijk = i + Nq * ((j-1) + Nq * (k-1))
                            
                            state         = extract_state(dg, localQ, ijk, e)
                            diffusive_flx = extract_diffusion(dg, localdiff, ijk, e)
                            x = localvgeo[ijk,grid.x1id,e]
                            y = localvgeo[ijk,grid.x2id,e]
                            z = localvgeo[ijk,grid.x3id,e]
                            MH = localvgeo[ijk,grid.MHid,e]
                            #state = cat![qstate, diffusive_flx]
                            for f in funs
                                f(FT, state, diffusive_flx, i, j, k, ijk, ev, eh, e, x, y, z, MH)
                            end
                        end
                    end
                end
            end
        end
    end

    # record the vertical coordinates and compute thermo variables
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
                          if abs(sau) <= abs(localQ[ijk,2,e])
                          sau = abs(localQ[ijk,2,e])
                          end
                          if abs(sav) <= abs(localQ[ijk,2,e])
                          sav = abs(localQ[ijk,2,e])
                          end
                          if abs(saw) <= abs(localQ[ijk,4,e])
                          saw = abs(localQ[ijk,4,e])
                          end
                          if abs(sdiff) <= abs(localdiff[ijk,3,e])
                          sdiff = abs(localdiff[ijk,3,e])
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
@info "Courant numbers in the following order: Sound in x, in y and in z then advection in x, in y and in z then diffusion in x, in y and in z"
@info "" Courssx,Courssy,Courssz, Couru, Courv, Courw, Courdx,Courdy,Courdz

    return nothing
end # function gather_diagnostics

end # module Diagnostics
