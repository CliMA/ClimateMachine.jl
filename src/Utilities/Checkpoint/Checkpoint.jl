module Checkpoint

export write_checkpoint, rm_checkpoint, read_checkpoint

using JLD2
using MPI
using Printf
import KernelAbstractions: CPU

using ..ODESolvers
using ..ODESolvers: AbstractODESolver
using ..MPIStateArrays
import ..MPIStateArrays: array_device

"""
    write_checkpoint(solver_config, checkpoint_dir, name, mpicomm, num)

Read in the state and auxiliary arrays as well as the simulation time
stored in the checkpoint file for `name` and `num`.
"""
function write_checkpoint(
    solver_config,
    checkpoint_dir::String,
    name::String,
    mpicomm::MPI.Comm,
    num::Int,
)
    Q = solver_config.Q
    A = solver_config.dg.state_auxiliary
    odesolver = solver_config.solver

    write_checkpoint(Q, A, odesolver, checkpoint_dir, name, mpicomm, num)

    return nothing
end

function write_checkpoint(
    Q::MPIStateArray,
    A::MPIStateArray,
    odesolver::AbstractODESolver,
    checkpoint_dir::String,
    name::String,
    mpicomm::MPI.Comm,
    num::Int,
)
    nm = replace(name, " " => "_")
    cname = @sprintf(
        "%s_checkpoint_mpirank%04d_num%04d.jld2",
        nm,
        MPI.Comm_rank(mpicomm),
        num,
    )
    cfull = joinpath(checkpoint_dir, cname)
    @info @sprintf(
        """
Checkpoint
    saving to %s""",
        cfull
    )

    if array_device(Q) isa CPU
        h_Q = Q.realdata
        h_aux = A.realdata
    else
        h_Q = Array(Q.realdata)
        h_aux = Array(A.realdata)
    end
    t = ODESolvers.gettime(odesolver)
    @save cfull h_Q h_aux t

    return nothing
end

"""
    rm_checkpoint(checkpoint_dir, name, mpicomm, num)

Remove the checkpoint file identified by `solver_config.name` and `num`.
"""
function rm_checkpoint(
    checkpoint_dir::String,
    name::String,
    mpicomm::MPI.Comm,
    num::Int,
)
    nm = replace(name, " " => "_")
    cname = @sprintf(
        "%s_checkpoint_mpirank%04d_num%04d.jld2",
        nm,
        MPI.Comm_rank(mpicomm),
        num,
    )
    rm(joinpath(checkpoint_dir, cname), force = true)

    return nothing
end

"""
    read_checkpoint(checkpoint_dir, name, mpicomm, num)

Read in the state and auxiliary arrays as well as the simulation time
stored in the checkpoint file for `name` and `num`.
"""
function read_checkpoint(
    checkpoint_dir::String,
    name::String,
    array_type,
    mpicomm::MPI.Comm,
    num::Int,
)
    nm = replace(name, " " => "_")
    cname = @sprintf(
        "%s_checkpoint_mpirank%04d_num%04d.jld2",
        nm,
        MPI.Comm_rank(mpicomm),
        num,
    )
    cfull = joinpath(checkpoint_dir, cname)
    if !isfile(cfull)
        error("Cannot restore from checkpoint in $(cfull), file not found")
    end

    @load cfull h_Q h_aux t

    return (array_type(h_Q), array_type(h_aux), t)
end

end # module
