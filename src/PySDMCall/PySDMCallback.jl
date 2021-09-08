

module PySDMCallbacks

export PySDMCallback

include("PySDMCall.jl")
#include("src/PySDMCall/PySDMCall.jl")
include("../../test/Atmos/Parameterizations/Microphysics/KinematicModel.jl") #param_set

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
import Thermodynamics
const THDS = Thermodynamics

using CLIMAParameters

import ClimateMachine.GenericCallbacks

using PyCall



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
        PySDM(pysdmconf, nothing, nothing, nothing, nothing)
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

    export_particles_to_vtk(cb.pysdm)

    vals = vals_interpol(cb, Q)

    println(keys(cb.pysdm.core.dynamics))
    dynamics = cb.pysdm.core.dynamics

    #run Displacement
    #TODO: add Displacement 2 times: 1 for Condensation and 1 for Advection
    dynamics["Displacement"]()
    #delete!(dynamics, "Displacement")

    update_pysdm_fields!(cb, vals, t)

    cb.pysdm.core.env.sync()
    #cb.pysdm.core.run(1)

    dynamics["ClimateMachine"]()
    dynamics["Condensation"]()

"""
    for (key, value) in dynamics
        # TODO: insert if in here
        println(key)
        value()
    end
"""
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


function update_pysdm_fields!(cb::PySDMCallback, vals, t)

    println("theta_dry, q_vap")
"""
    pysdm_th = vals["theta_dry"][:, 1, :]
    @assert size(pysdm_th) == (76, 76)
    pysdm_th = bilinear_interpol(pysdm_th)
    @assert size(pysdm_th) == (75, 75)

    pysdm_qv = vals["q_vap"][:, 1, :]
    @assert size(pysdm_qv) == (76, 76)
    pysdm_qv = bilinear_interpol(pysdm_qv)
    @assert size(pysdm_qv) == (75, 75)
"""

    # set frequency of plotting
    n_steps = 10

    n_simtime = n_steps * 10 # dt = 10, simtime 100 = steps 10

    # water_mixing_ratio = get product 3 moment objentosci kropel (get water mixing ratio product)
    #liquid_water_mixing_ratio = pysdm.get_product(water_mixing_ratio)
    liquid_water_mixing_ratio = cb.pysdm.core.products["qc"].get()
    println(typeof(liquid_water_mixing_ratio))
    println(size(liquid_water_mixing_ratio))

    if t % n_simtime == 0
        export_plt(liquid_water_mixing_ratio, "liquid_water_mixing_ratio", t)
    end

    #liquid_water_specific_humidity = some_f(liquid_water_mixing_ratio) # Sylwester podesli na Slacku
    liquid_water_specific_humidity = liquid_water_mixing_ratio

    #q = THDS.PhasePartition(q_tot, liquid_water_mixing_ratio, .0) # instead of liquid_water_mixing_ratio should be liquid_water_specific_humidity
    q_tot = vals["q_tot"][:, 1, :]
    q_tot = bilinear_interpol(q_tot)

    if t % n_simtime == 0
        export_plt(q_tot, "q_tot", t)
    end

    q = THDS.PhasePartition.(q_tot, liquid_water_specific_humidity, .0)

    println("q")
    println(size(q))
    println(typeof(q))

    # q is Matrix of PhasePartition objects, thus not plottable

    #qv = THDS.vapor_specific_humidity(q)
    qv = THDS.vapor_specific_humidity.(q)

    if t % n_simtime == 0
        export_plt(qv, "qv", t)
    end

    println(typeof(qv))

    #T = THDS.air_temperature(param_set, e_int, q) # CLIMAParameters: param_set
    e_int = vals["e_int"][:, 1, :]
    e_int = bilinear_interpol(e_int)

    if t % n_simtime == 0
        export_plt(e_int, "e_int", t)
    end

    # TODO - AJ shouldnt we compute new e_int and new T based on new pp?
    T = THDS.air_temperature.(param_set, e_int, q)

    if t % n_simtime == 0
        export_plt(T, "T", t)
    end

    #thd = THDS.dry_pottemp(param_set, T, ρ) # rho has to be rhod (dry)
    ρ = cb.pysdm.rhod
    thd = THDS.dry_pottemp.(param_set, T, ρ)

    if t % n_simtime == 0
        export_plt(thd, "thd", t)
    end

    RH_machine = THDS.supersaturation.(param_set, q, ρ, T, THDS.Liquid()) .+ 1.0

    println(cb.pysdm.core.products.keys)

    RH_pysdm = cb.pysdm.core.products["RH_env"].get() ./ 100.0

    if t % n_simtime == 0
        export_plt(RH_machine, "RH_machine", t)
        export_plt(RH_pysdm, "RH_pysdm", t)
        export_plt((RH_pysdm .- RH_machine)./RH_machine, "RH_rel_diff", t)
    end

    println("new qv and thd")
    println(size(qv))
    println(size(thd))

    pysdm_th = thd
    pysdm_qv = qv

    cb.pysdm.core.dynamics["ClimateMachine"].set_th(pysdm_th)
    cb.pysdm.core.dynamics["ClimateMachine"].set_qv(pysdm_qv)

    #run dynamics (except Displacement)
    # passing effective_radius to ClimateMachine (Auxiliary)
    return nothing
end



function export_plt(var, title, t)
    py"""
    from matplotlib.pyplot import cm
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_vars(A, title=None):
        # Contour Plot
        X, Y = np.mgrid[0:A.shape[0], 0:A.shape[1]]
        Z = A
        cp = plt.contourf(X, Y, Z)
        cb = plt.colorbar(cp)
        if title:
            plt.title(title)

        plt.show()
        return plt
    """

    println(string(title, "plot"))
    plt = py"plot_vars($var, title=$title)"
    plt.savefig(string(title, t, ".png"))
    plt.clf()
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

    pysdm_vars = ["ρ", "ρu[1]", "ρu[3]", "q_vap", "theta_dry", "q_tot", "e_int"]


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
