# TODO: figure out a better interface for this.
# at the moment we can just pass a function, but we should do something better
# need to figure out how subcomponents will interact.
function atmos_boundarycondition!(f::Function, m::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  f(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

# lookup boundary condition by face
function atmos_boundarycondition!(bctup::Tuple, m::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)  
  atmos_boundarycondition!(bctup[bctype], m, stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end


abstract type BoundaryCondition
end

"""
    NoFluxBC <: BoundaryCondition

Set the momentum at the boundary to be zero.
"""
struct NoFluxBC <: BoundaryCondition
end

function atmos_boundarycondition!(bc::NoFluxBC, m::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) 
    DF = eltype(stateM)
    stateP.ρ = stateM.ρ
    stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
    diffP.ρτ = -zero(eltype(diffM.ρτ))
    diffP.moisture.ρd_q_tot = -zero(eltype(diffM.moisture.ρd_q_tot))
    diffP.moisture.ρd_h_tot = -zero(eltype(diffM.moisture.ρd_h_tot))
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is mainly useful for cases where the problem has an explicit solution.
"""
struct InitStateBC <: BoundaryCondition
end
function atmos_boundarycondition!(bc::InitStateBC, m::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) 
  init_state!(m, stateP, auxP, auxP.coord, t)
end
