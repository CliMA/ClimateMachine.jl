

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
using ClimateMachine.DGMethods: SpaceDiscretization

using CLIMAParameters

import ClimateMachine.GenericCallbacks





mutable struct PySDMCallback
    name::String
    dg::SpaceDiscretization
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
    println()
    println("PySDMCallback init")
    println(cb.name)
    println(t)
    println()

    varvals = vals_interpol(cb, Q)
 

    PySDMCall.pysdm_init!(cb.pysdm, varvals)


    return nothing
end

function GenericCallbacks.call!(cb::PySDMCallback, solver, Q, param, t)
    println()
    println("PySDMCallback call")
    println(t)
    println()

    vals = vals_interpol(cb, Q)

    update_pysdm_fields!(cb, vals)

    cb.pysdm.core.env.sync()
    cb.pysdm.core.run(1)

    #env.sync() # take data from CliMA
    #cb.pysdm.run(1) # dynamic in dynamics
    # upd CliMa state vars
    return nothing
end
function GenericCallbacks.fini!(cb::PySDMCallback, solver, Q, param, t)
    println()
    println("PySDMCallback fini")
    println(t)
    println()
    return nothing
end


function update_pysdm_fields!(cb::PySDMCallback, vals)

    println("theta_dry, q_vap")

    pysdm_th = vals["theta_dry"][:, 1, :]
    @assert size(pysdm_th) == (76, 76)
    pysdm_th = bilinear_interpol(pysdm_th)
    @assert size(pysdm_th) == (75, 75)

    pysdm_qv = vals["q_vap"][:, 1, :]
    @assert size(pysdm_qv) == (76, 76)
    pysdm_qv = bilinear_interpol(pysdm_qv)
    @assert size(pysdm_qv) == (75, 75)



    cb.pysdm.core.dynamics["ClimateMachine"].set_th(pysdm_th)
    cb.pysdm.core.dynamics["ClimateMachine"].set_qv(pysdm_qv)
    
    return nothing
end



function vals_interpol(cb::PySDMCallback, Q)

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

    pysdm_vars = ["ρ", "ρu[1]", "ρu[3]", "q_vap", "theta_dry"]


    varvals = nothing

    if mpirank == 0
        statenames = flattenednames(vars_state(bl, Prognostic(), FT))
        auxnames = flattenednames(vars_state(bl, Auxiliary(), FT))
        
        #println("CUSTOM CALLBACK PYSDM STATENAMES")
        #println(statenames)

        #println("CUSTOM CALLBACK PYSDM AUXNAMES")
        #println(auxnames)

        varvals = OrderedDict()
        for (vari, varname) in enumerate(statenames)
            if varname in pysdm_vars

                varvals[varname] = all_state_data[:, :, :, vari]
            
            end    
        end

        for (vari, varname) in enumerate(auxnames)
            if varname in pysdm_vars
                
                varvals[varname] = all_aux_data[:, :, :, vari]
                
            end  
        end

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