

module PySDMCallbacks

export PySDMCallback

include("PySDMCall.jl")
#include("src/PySDMCall/PySDMCall.jl")

using .PySDMCall

using CUDA
using Dates
using FileIO
using JLD2
using KernelAbstractions
using MPI
using OrderedCollections
using Printf
using StaticArrays
import KernelAbstractions: CPU

using ClimateMachine.BalanceLaws
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.VariableTemplates

using CLIMAParameters

import ClimateMachine.GenericCallbacks





mutable struct PySDMCallback
    name::String
    dg
    interpol
    mpicomm::MPI.Comm
    pysdm::PySDMCall.PySDM
end


function PySDMCallback(name, dg, interpol, mpicomm, pysdmconf)
    PySDMCallback(
        name,
        dg,
        interpol,
        mpicomm,
        PySDM(pysdmconf, nothing, nothing, nothing)
    )
end    


function GenericCallbacks.init!(cb::PySDMCallback, solver, Q, param, t)
    println("Pysdm CALLBACK init")
    println(cb.name)

    interpol = cb.interpol
    mpicomm = cb.mpicomm
    dg = cb.dg
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    istate = similar(Q.data, interpol.Npl, number_states(bl, Prognostic()))
    
    interpolate_local!(interpol, Q.data, istate)

    if interpol isa InterpolationCubedSphere
        # TODO: get indices here without hard-coding them
        _ρu, _ρv, _ρw = 2, 3, 4
        project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    end

    iaux = similar(
        dg.state_auxiliary.data,
        interpol.Npl,
        number_states(bl, Auxiliary()),
    )

    interpolate_local!(interpol, dg.state_auxiliary.data, iaux)

    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)

    pysdm_vars = ["ρ", "ρu[1]", "ρu[3]", "moisture.θ_v", "ρe", "ρq_tot", "T", "q_vap"] # no "ρu[2]"


    if mpirank == 0
        statenames = flattenednames(vars_state(bl, Prognostic(), FT))
        auxnames = flattenednames(vars_state(bl, Auxiliary(), FT))
        
        println("CUSTOM CALLBACK PYSDM STATENAMES")
        println(statenames)

        println("CUSTOM CALLBACK PYSDM AUXNAMES")
        println(auxnames)

        varvals = OrderedDict()
        for (vari, varname) in enumerate(statenames)
            if varname in pysdm_vars
                println(varname)
                varvals[varname] = all_state_data[:, :, :, vari]
                println(typeof(all_state_data[:, :, :, vari]))

                println(size(all_state_data[:, :, :, vari]))
            end    
        end

        for (vari, varname) in enumerate(auxnames)
            if varname in pysdm_vars
                println(varname)
                varvals[varname] = all_aux_data[:, :, :, vari]
                println(typeof(all_aux_data[:, :, :, vari]))


                println(size(all_aux_data[:, :, :, vari]))
            end  
        end

        
        #cb.pysdm = PySDMCall.pysdm_init1(varvals, cb.dt, cb.dx, cb.simtime) # probably varvals should be converted to an array
        println(typeof(cb.pysdm))
        
        #println("check size of ρe")
                
        #println(size(varvals["ρe"]))

        PySDMCall.pysdm_init!(cb.pysdm, varvals) # probably varvals should be converted to an array
        
        println("CLIMA ")
        #println(repr(UInt64(pointer_from_objref(cb.pysdm))))

    end

    MPI.Barrier(mpicomm)

    return nothing
end

function GenericCallbacks.call!(cb::PySDMCallback, solver, Q, param, t)
    println()
    println("================Simulation pysdm from clima ==============")
    println("Call")
    println(t)
    println()

    vals = vals_interpol!(cb, Q)

    update_pysdm_fields!(cb, vals)

    cb.pysdm.core.env.sync()
    cb.pysdm.core.run(1)

    #env.sync() # take data from CliMA
    #cb.pysdm.run(1) # dynamic in dynamics
    # upd CliMa state vars
    return nothing
end
function GenericCallbacks.fini!(cb::PySDMCallback, solver, Q, param, t)
    return nothing
end


function update_pysdm_fields!(cb::PySDMCallback, vals)

    println("T, q_vap")

    pysdm_th = vals["T"][:, 1, :]
    @assert size(pysdm_th) == (76, 76)
    pysdm_th = [ (pysdm_th[y, x-1] + pysdm_th[y, x]) / 2 for y in 1:size(pysdm_th)[1], x in 2:size(pysdm_th)[2]]
    @assert size(pysdm_th) == (76, 75)
    pysdm_th = [ (pysdm_th[y-1, x] + pysdm_th[y, x]) / 2 for y in 2:size(pysdm_th)[1], x in 1:size(pysdm_th)[2]]
    @assert size(pysdm_th) == (75, 75)

    pysdm_qv = vals["q_vap"][:, 1, :]
    @assert size(pysdm_qv) == (76, 76)
    pysdm_qv = [ (pysdm_qv[y, x-1] + pysdm_qv[y, x]) / 2 for y in 1:size(pysdm_qv)[1], x in 2:size(pysdm_qv)[2]]
    @assert size(pysdm_qv) == (76, 75)
    pysdm_qv = [ (pysdm_qv[y-1, x] + pysdm_qv[y, x]) / 2 for y in 2:size(pysdm_qv)[1], x in 1:size(pysdm_qv)[2]]
    @assert size(pysdm_qv) == (75, 75)



    cb.pysdm.core.dynamics["ClimateMachine"].set_qv(pysdm_qv)
    cb.pysdm.core.dynamics["ClimateMachine"].set_th(pysdm_th)
    
    return nothing
end



function vals_interpol!(cb::PySDMCallback, Q)

    interpol = cb.interpol
    mpicomm = cb.mpicomm
    dg = cb.dg
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    istate = similar(Q.data, interpol.Npl, number_states(bl, Prognostic()))
    
    interpolate_local!(interpol, Q.data, istate)

    if interpol isa InterpolationCubedSphere
        # TODO: get indices here without hard-coding them
        _ρu, _ρv, _ρw = 2, 3, 4
        project_cubed_sphere!(interpol, istate, (_ρu, _ρv, _ρw))
    end

    iaux = similar(
        dg.state_auxiliary.data,
        interpol.Npl,
        number_states(bl, Auxiliary()),
    )

    interpolate_local!(interpol, dg.state_auxiliary.data, iaux)

    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)

    pysdm_vars = ["ρ", "ρu[1]", "ρu[3]", "T", "q_vap"] # no "ρu[2]", "moisture.θ_v", "ρe", "ρq_tot",


    varvals = nothing

    if mpirank == 0
        statenames = flattenednames(vars_state(bl, Prognostic(), FT))
        auxnames = flattenednames(vars_state(bl, Auxiliary(), FT))
        
        println("CUSTOM CALLBACK PYSDM STATENAMES")
        println(statenames)

        println("CUSTOM CALLBACK PYSDM AUXNAMES")
        println(auxnames)

        varvals = OrderedDict()
        for (vari, varname) in enumerate(statenames)
            if varname in pysdm_vars
                println(varname)
                varvals[varname] = all_state_data[:, :, :, vari]
                println(typeof(all_state_data[:, :, :, vari]))

                println(size(all_state_data[:, :, :, vari]))
            end    
        end

        for (vari, varname) in enumerate(auxnames)
            if varname in pysdm_vars
                println(varname)
                varvals[varname] = all_aux_data[:, :, :, vari]
                println(typeof(all_aux_data[:, :, :, vari]))


                println(size(all_aux_data[:, :, :, vari]))
            end  
        end

        
        #cb.pysdm = PySDMCall.pysdm_init1(varvals, cb.dt, cb.dx, cb.simtime) # probably varvals should be converted to an array
        println(typeof(cb.pysdm))
        
        #println("check size of ρe")
                
        #println(size(varvals["ρe"]))

        #PySDMCall.pysdm_init!(cb.pysdm, varvals) # probably varvals should be converted to an array
        
        println("CLIMA ")
        #println(repr(UInt64(pointer_from_objref(cb.pysdm))))

    end

    MPI.Barrier(mpicomm)

    return varvals
end


  
"""
example of use:
testcb = GenericCallbacks.EveryXSimulationSteps(PySDMCallback("PySDMCallback", solver_config.dg, interpol, mpicomm), 1)


mutable struct MyCallback
    initialized::Bool
    calls::Int
    finished::Bool
end
MyCallback() = MyCallback(false, 0, false)

GenericCallbacks.init!(cb::MyCallback, _...) = cb.initialized = true
GenericCallbacks.call!(cb::MyCallback, _...) = (cb.calls += 1; nothing)
GenericCallbacks.fini!(cb::MyCallback, _...) = cb.finished = true
"""

end