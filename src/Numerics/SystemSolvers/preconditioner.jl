export ColumnwiseLUPreconditioner, preconditioner_update!, preconditioner_solve!, preconditioner_counter_update!


"""
Do nothing, when there is no preconditioner, preconditioner = Nothing
"""
function preconditioner_update!(op, dg, preconditioner::Nothing, args...)
end

"""
Do nothing, when there is no preconditioner, preconditioner = Nothing
"""
function preconditioner_solve!(preconditioner::Nothing, Q)
end

"""
Do nothing, when there is no preconditioner, preconditioner = Nothing
"""
function preconditioner_counter_update!(preconditioner::Nothing)
end

"""
mutable struct ColumnwiseLUPreconditioner{AT}
    A::DGColumnBandedMatrix
    Q::AT
    PQ::AT
    counter::Int64
    update_freq::Int64
end

A: the lu factor of the precondition (approximated Jacobian), in the DGColumnBandedMatrix format
Q: MPIArray container, used to update A 
PQ: MPIArray container, used to update A 
counter: count the number of Newton, when counter > update_freq or counter < 0, update precondition
update_freq: preconditioner update frequency

"""
mutable struct ColumnwiseLUPreconditioner{AT}
    A::DGColumnBandedMatrix
    Q::AT
    PQ::AT
    counter::Int64
    update_freq::Int64
end



"""
ColumnwiseLUPreconditioner constructor
build an empty ColumnwiseLUPreconditioner

dg: DG model, use only the grid information
Q0: MPIArray, use only its size
counter = -1, which indicates the preconditioner is empty
update_freq: preconditioner update frequency

"""
function ColumnwiseLUPreconditioner(dg, Q0, update_freq=100)
    single_column = false
    Q = similar(Q0)
    PQ = similar(Q0)

    A = empty_banded_matrix(
        dg,
        Q;
        single_column = single_column,
    )

    # counter = -1, which indicates the preconditioner is empty
    ColumnwiseLUPreconditioner(A, Q, PQ, -1, update_freq)
end

"""
update the DGColumnBandedMatrix by the finite difference approximation

op: operator used to compute the finte difference information
dg: the DG model, use only the grid information
"""
function preconditioner_update!(op, dg, preconditioner::ColumnwiseLUPreconditioner, args...)
    
    # preconditioner.counter < 0, means newly constructed empty preconditioner
    if preconditioner.counter >= 0 && (preconditioner.counter < preconditioner.update_freq)
        return
    end

    A = preconditioner.A
    Q = preconditioner.Q
    PQ = preconditioner.PQ

    update_banded_matrix!(
        A,
        op,
        dg,
        Q,
        PQ,
        args...
    )
    band_lu!(A)

    preconditioner.counter = 0
end

"""
Inplace applying the preconditioner 

Q = P⁻¹ * Q
"""
function preconditioner_solve!(preconditioner::ColumnwiseLUPreconditioner, Q)
    A = preconditioner.A
    band_forward!(Q, A)
    band_back!(Q, A)

end


"""
Update the preconditioner counter, after each Newton iteration
"""
function preconditioner_counter_update!(preconditioner::ColumnwiseLUPreconditioner)
    preconditioner.counter += 1
end
