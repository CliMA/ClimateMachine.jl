
module PySDMCall

using PyCall
import Thermodynamics
const THDS = Thermodynamics

export PySDM, PySDMConfig, PySDMCallWrapper, __init__

"""
    PySDMConfig

Holds a set of parameters that configure simulation in PySDM.
"""
mutable struct PySDMConfig
    grid::Tuple{Int64, Int64}
    size::Tuple{Int64, Int64}
    dxdz::Tuple{Float64, Float64}
    simtime::Float64
    dt::Float64
    n_sd::Int64
    kappa::Int64 # hygroscopicity
    kernel::PyCall.PyObject # from PySDM
    spectrum_per_mass_of_dry_air::PyCall.PyObject
end

function PySDMConfig(
    size::Tuple{Int64, Int64},
    dxdz::Tuple{Float64, Float64},
    simtime::Float64,
    dt::Float64,
    n_sd_per_gridbox::Int64,
    kappa::Int64,
    kernel::Any,
    spectrum_per_mass_of_dry_air::Any,
)
    grid = (Int(size[1] / dxdz[1]), Int(size[2] / dxdz[2]))

    n_sd = grid[1] * grid[2] * n_sd_per_gridbox

    PySDMConfig(
        grid,
        size,
        dxdz,
        simtime,
        dt,
        n_sd,
        kappa,
        kernel,
        spectrum_per_mass_of_dry_air,
    )
end

"""
    PySDM

Represents PySDM. particulator, rhod, exporter are set during initialization of simulation.

particulator - used to manage the system state and control the simulation.
rhod - dry density matrix.
exporter - PySDM's exporter. (e.g. VTKExporter)
"""
mutable struct PySDM
    config::PySDMConfig
    particulator::Any
    rhod::Any
    exporter::Any
end

"""
    PySDMCallWrapper

Packs together PySDM with functions that manage the course of PySDM's simulation.

init!(::PySDM, varvals) - initializes simulation.
do_step!(::PySDM, varvals, t) - performs step on PySDM's side.
fini!(::PySDM, varvals, t) - run at the end of simulation.

varvals - interpolated OrderedDict of ClimateMachine's variables.
"""
mutable struct PySDMCallWrapper
    pysdm::PySDM
    init!::Any
    do_step!::Any
    fini!::Any

    function PySDMCallWrapper(pysdm_conf::PySDMConfig, init!, do_step!, fini!)

        return new(
            PySDM(pysdm_conf, nothing, nothing, nothing),
            init!,
            do_step!,
            fini!,
        )
    end
end


"""
    __init__()

Adds directories to the Python search path.
"""
function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), "")
    pushfirst!(PyVector(pyimport("sys")."path"), "test/PySDMCall/")
    pushfirst!(
        PyVector(pyimport("sys")."path"),
        "test/Atmos/Parameterizations/Microphysics/KM_PySDMachine/test/PySDMCall/",
    )
end

end # module PySDMCall
