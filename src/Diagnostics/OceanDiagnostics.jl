using ..HydrostaticBoussinesqModel

function init(model::HydrostaticBoussinesqModel,
              mpicomm::MPI.Comm,
              dg::DGModel,
              Q::MPIStateArray,
              starttime::String)
    @warn "No diagnostics implemented for HydrostaticBoussinesqModel"
end

"""
    collect(bl, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function collect(model::HydrostaticBoussinesqModel, currtime)
    return nothing
end
