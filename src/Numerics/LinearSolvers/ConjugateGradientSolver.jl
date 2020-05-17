module ConjugateGradientSolver

export ConjugateGradient

using ..LinearSolvers
const LS = LinearSolvers
using ClimateMachine.MPIStateArrays
using LinearAlgebra
using LazyArrays
using StaticArrays

struct ConjugateGradient{AT1, AT2, FT, RD, RT, IT} <:
       LS.AbstractIterativeLinearSolver
    # tolerances (2)
    rtol::FT
    atol::FT
    # arrays of size reshape_tuple (7)
    r0::AT1
    z0::AT1
    p0::AT1
    r1::AT1
    z1::AT1
    p1::AT1
    Lp::AT1
    # arrays of size(MPIStateArray) which are aliased to two of the previous dimensions (2)
    alias_p0::AT2
    alias_Lp::AT2
    # reduction dimension (1)
    dims::RD
    # reshape dimension (1)
    reshape_tuple::RT
    # maximum number of iterations (1)
    max_iter::IT

end

# Define the outer constructor for the ConjugateGradient struct
"""
# ConjugateGradient
function ConjugateGradient(Q::AT; rtol = eps(eltype(Q)), atol = eps(eltype(Q)), dims = :) where AT

# Description
- Outer constructor for the ConjugateGradient struct

# Arguments
- `Q`:(array). The kind of object that linearoperator! acts on.

# Keyword Arguments
- `rtol`: (float). relative tolerance
- `atol`: (float). absolute tolerance
- `dims`: (tuple or : ). the dimensions to reduce over
- `reshape_tuple`: (tuple). the dimensions that the conjugate gradient solver operators over

# Comment
- The reshape tuple is necessary in case the linearoperator! is defined over vectors of a different size as compared to what plays nicely with the dimension reduction in the ConjugateGradient. It also allows the user to define preconditioners over arrays that are more convenienently shaped.

# Return
- ConjugateGradient struct
"""
function ConjugateGradient(
    Q::AT;
    rtol = eps(eltype(Q)),
    atol = eps(eltype(Q)),
    max_iter = length(Q),
    dims = :,
    reshape_tuple = size(Q),
) where {AT}

    # allocate arrays (5)
    r0 = reshape(similar(Q), reshape_tuple)
    z0 = reshape(similar(Q), reshape_tuple)
    r1 = reshape(similar(Q), reshape_tuple)
    z1 = reshape(similar(Q), reshape_tuple)
    p1 = reshape(similar(Q), reshape_tuple)
    # allocate array of different shape (2)
    alias_p0 = similar(Q)
    alias_Lp = similar(Q)
    # allocate create aliased arrays (2)
    p0 = reshape(alias_p0, reshape_tuple)
    Lp = reshape(alias_Lp, reshape_tuple)

    container = [
        rtol,
        atol,
        r0,
        z0,
        p0,
        r1,
        z1,
        p1,
        Lp,
        alias_p0,
        alias_Lp,
        dims,
        reshape_tuple,
        max_iter,
    ]
    # create struct instance by splatting the container into the default constructor
    return ConjugateGradient{
        typeof(z0),
        typeof(Q),
        eltype(Q),
        typeof(dims),
        typeof(reshape_tuple),
        typeof(max_iter),
    }(container...)
end

# Define the outer constructor for the ConjugateGradient struct
"""
function ConjugateGradient(Q::MPIStateArray; rtol = eps(eltype(Q)), atol = eps(eltype(Q)), dims = :)

# Description
Outer constructor for the ConjugateGradient struct with MPIStateArrays. THIS IS A HACK DUE TO RESHAPE FUNCTIONALITY ON MPISTATEARRAYS.

# Arguments
- `Q`:(array). The kind of object that linearoperator! acts on.

# Keyword Arguments
- `rtol`: (float). relative tolerance
- `atol`: (float). absolute tolerance
- `dims`: (tuple or : ). the dimensions to reduce over
- `reshape_tuple`: (tuple). the dimensions that the conjugate gradient solver operators over

# Comment
- The reshape tuple is necessary in case the linearoperator! is defined over vectors of a different size as compared to what plays nicely with the dimension reduction in the ConjugateGradient. It also allows the user to define preconditioners over arrays that are more convenienently shaped.

# Return
- ConjugateGradient struct
"""
function ConjugateGradient(
    Q::MPIStateArray;
    rtol = eps(eltype(Q)),
    atol = eps(eltype(Q)),
    max_iter = length(Q),
    dims = :,
    reshape_tuple = size(Q),
)

    # create empty container for pushing struct objects
    # allocate arrays (5)
    r0 = reshape(similar(Q.data), reshape_tuple)
    z0 = reshape(similar(Q.data), reshape_tuple)
    r1 = reshape(similar(Q.data), reshape_tuple)
    z1 = reshape(similar(Q.data), reshape_tuple)
    p1 = reshape(similar(Q.data), reshape_tuple)
    # allocate array of different shape (2)
    alias_p0 = similar(Q)
    alias_Lp = similar(Q)
    # allocate create aliased arrays (2)
    p0 = reshape(alias_p0.data, reshape_tuple)
    Lp = reshape(alias_Lp.data, reshape_tuple)
    container = [
        rtol,
        atol,
        r0,
        z0,
        p0,
        r1,
        z1,
        p1,
        Lp,
        alias_p0,
        alias_Lp,
        dims,
        reshape_tuple,
        max_iter,
    ]
    # create struct instance by splatting the container into the default constructor
    return ConjugateGradient{
        typeof(z0),
        typeof(Q),
        eltype(Q),
        typeof(dims),
        typeof(reshape_tuple),
        typeof(max_iter),
    }(container...)
end


"""
LS.initialize!(linearoperator!, Q, Qrhs, solver::ColumnwisePreconditionedConjugateGradient, args...)

# Description

- This function initializes the iterative solver. It is called as part of the AbstractIterativeLinearSolver routine. SEE CODEREF for documentation on AbstractIterativeLinearSolver

# Arguments

- `linearoperator!`: (function). This applies the predefined linear operator on an array. Applies a linear operator to object "y" and overwrites object "z". The function argument i s linearoperator!(z,y, args...) and it returns nothing.
- `Q`: (array). This is an object that linearoperator! outputs
- `Qrhs`: (array). This is an object that linearoperator! acts on
- `solver`: (struct). This is a scruct for dispatch, in this case for ColumnwisePreconditionedConjugateGradient
- `args...`: (arbitrary). This is optional arguments that can be passed into linearoperator! function.

# Keyword Arguments

- There are no keyword arguments

# Return
- `converged`: (bool). A boolean to say whether or not the iterative solver has converged.
- `threshold`: (float). The value of the residual for the first timestep

# Comment
- This function does nothing for conjugate gradient

"""
function LS.initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::ConjugateGradient,
    args...,
)

    return false, Inf
end


"""
LS.doiteration!(linearoperator!, Q, Qrhs, solver::ColumnwisePreconditionedConjugateGradient, threshold, args...; applyPC!)

# Description

- This function enacts the iterative solver. It is called as part of the AbstractIterativeLinearSolver routine. SEE CODEREF for documentation on AbstractIterativeLinearSolver

# Arguments

- `linearoperator!`: (function). This applies the predefined linear operator on an array. Applies a linear operator to object "y" and overwrites object "z". It is a function with arguments linearoperator!(z,y, args...), where "z" gets overwritten by "y" and "args..." are additional arguments passed to the linear operator. The linear operator is assumed to return nothing.
- `Q`: (array). This is an object that linearoperator! overwrites
- `Qrhs`: (array). This is an object that linearoperator! acts on. This is the rhs to the linear system
- `solver`: (struct). This is a scruct for dispatch, in this case for ConjugateGradient
- `threshold`: (float). Either an absolute or relative tolerance
- `applyPC!`: (function). Applies a preconditioner to objecy "y" and overwrites object "z". applyPC!(z,y)
- `args...`: (arbitrary). This is necessary for the linearoperator! function which has a signature linearoperator!(b, x, args....)

# Keyword Arguments

- There are no keyword arguments

# Return
- `converged`: (bool). A boolean to say whether or not the iterative solver has converged.
- `iteration`: (int). Iteration number for the iterative solver
- `threshold`: (float). The value of the residual for the first timestep

# Comment
- This function does conjugate gradient

"""
function LS.doiteration!(
    linearoperator!,
    Q,
    Qrhs,
    solver::ConjugateGradient,
    threshold,
    args...;
    applyPC! = (x, y) -> x .= y,
)


    # unroll names for convenience

    rtol = solver.rtol
    atol = solver.atol
    residual_norm = typemax(eltype(Q))
    dims = solver.dims
    converged = false

    max_iter = solver.max_iter
    r0 = solver.r0
    z0 = solver.z0
    p0 = solver.p0
    r1 = solver.r1
    z1 = solver.z1
    p1 = solver.p1
    Lp = solver.Lp
    alias_p0 = solver.alias_p0
    alias_Lp = solver.alias_Lp
    alias_Q = reshape(Q, solver.reshape_tuple)

    # Smack residual by linear operator
    linearoperator!(alias_Lp, Q, args...)
    # make sure that arrays are of the appropriate size
    alias_p0 .= Qrhs
    r0 .= p0 - Lp
    # apply the preconditioner
    applyPC!(z0, r0)
    # update p0
    p0 .= z0

    # TODO: FIX THIS
    absolute_residual = maximum(sqrt.(sum(r0 .* r0, dims = dims)))
    relative_residual =
        absolute_residual / maximum(sqrt.(sum(Qrhs .* Qrhs, dims = :)))
    # TODO: FIX THIS
    if (absolute_residual <= atol) || (relative_residual <= rtol)
        # wow! what a great guess
        converged = true
        return converged, 1, absolute_residual
    end

    for j in 1:max_iter
        linearoperator!(alias_Lp, alias_p0, args...)

        α = sum(r0 .* z0, dims = dims) ./ sum(p0 .* Lp, dims = dims)

        # Update along preconditioned direction, (note that broadcast will indeed work as expected)
        @. alias_Q += α * p0

        @. r1 = r0 - α * Lp

        # TODO: FIX THIS
        absolute_residual = maximum(sqrt.(sum(r1 .* r1, dims = dims)))
        relative_residual =
            absolute_residual / maximum(sqrt.(sum(Qrhs .* Qrhs, dims = :)))
        # TODO: FIX THIS
        converged = false
        if (absolute_residual <= atol) || (relative_residual <= rtol)
            converged = true
            return converged, j, absolute_residual
        end

        applyPC!(z1, r1)

        β = sum(z1 .* r1, dims = dims) ./ sum(z0 .* r0, dims = dims)

        # Update
        @. p0 = z1 + β * p0
        @. z0 = z1
        @. r0 = r1

    end

    # TODO: FIX THIS
    converged = true
    return converged, max_iter, absolute_residual
end

"""
LS.doiteration!(linearoperator!, Q::MPIStateArray, Qrhs::MPIStateArray, solver::ColumnwisePreconditionedConjugateGradient, threshold, args...; applyPC!)

# Description

This function enacts the iterative solver. It is called as part of the AbstractIterativeLinearSolver routine. SEE CODEREF for documentation on AbstractIterativeLinearSolver. THIS IS A HACK TO WORK WITH MPISTATEARRAYS. THE ISSUE IS WITH RESHAPE.

# Arguments

- `linearoperator!`: (function). This applies the predefined linear operator on an array. Applies a linear operator to object "y" and overwrites object "z". It is a function with arguments linearoperator!(z,y, args...), where "z" gets overwritten by "y" and "args..." are additional arguments passed to the linear operator. The linear operator is assumed to return nothing.
- `Q`: (array). This is an object that linearoperator! overwrites
- `Qrhs`: (array). This is an object that linearoperator! acts on. This is the rhs to the linear system
- `solver`: (struct). This is a scruct for dispatch, in this case for ConjugateGradient
- `threshold`: (float). Either an absolute or relative tolerance
- `applyPC!`: (function). Applies a preconditioner to objecy "y" and overwrites object "z". applyPC!(z,y)
- `args...`: (arbitrary). This is necessary for the linearoperator! function which has a signature linearoperator!(b, x, args....)

# Keyword Arguments

- There are no keyword arguments

# Return
- `converged`: (bool). A boolean to say whether or not the iterative solver has converged.
- `iteration`: (int). Iteration number for the iterative solver
- `threshold`: (float). The value of the residual for the first timestep

# Comment
- This function does conjugate gradient

"""
function LS.doiteration!(
    linearoperator!,
    Q::MPIStateArray,
    Qrhs::MPIStateArray,
    solver::ConjugateGradient,
    threshold,
    args...;
    applyPC! = (x, y) -> x .= y,
)


    # unroll names for convenience

    rtol = solver.rtol
    atol = solver.atol
    residual_norm = typemax(eltype(Q))
    dims = solver.dims
    converged = false

    max_iter = solver.max_iter
    r0 = solver.r0
    z0 = solver.z0
    p0 = solver.p0
    r1 = solver.r1
    z1 = solver.z1
    p1 = solver.p1
    Lp = solver.Lp
    alias_p0 = solver.alias_p0
    alias_Lp = solver.alias_Lp
    alias_Q = reshape(Q.data, solver.reshape_tuple)

    # Smack residual by linear operator
    linearoperator!(alias_Lp, Q, args...)
    # make sure that arrays are of the appropriate size
    alias_p0 .= Qrhs.data
    r0 .= p0 - Lp
    # apply the preconditioner
    applyPC!(z0, r0)
    # update p0
    p0 .= z0

    # TODO: FIX THIS
    absolute_residual = maximum(sqrt.(sum(r0 .* r0, dims = dims)))
    relative_residual =
        absolute_residual / maximum(sqrt.(sum(Qrhs .* Qrhs, dims = :)))
    # TODO: FIX THIS
    if (absolute_residual <= atol) || (relative_residual <= rtol)
        # wow! what a great guess
        converged = true
        return converged, 1, absolute_residual
    end

    for j in 1:max_iter
        linearoperator!(alias_Lp, alias_p0, args...)

        α = sum(r0 .* z0, dims = dims) ./ sum(p0 .* Lp, dims = dims)

        # Update along preconditioned direction, (note that broadcast will indeed work as expected)
        @. alias_Q += α * p0

        @. r1 = r0 - α * Lp

        # TODO: FIX THIS
        absolute_residual = maximum(sqrt.(sum(r1 .* r1, dims = dims)))
        relative_residual =
            absolute_residual / maximum(sqrt.(sum(Qrhs .* Qrhs, dims = :)))
        # TODO: FIX THIS
        converged = false
        if (absolute_residual <= atol) || (relative_residual <= rtol)
            converged = true
            return converged, j, absolute_residual
        end

        applyPC!(z1, r1)

        β = sum(z1 .* r1, dims = dims) ./ sum(z0 .* r0, dims = dims)

        # Update
        @. p0 = z1 + β * p0
        @. z0 = z1
        @. r0 = r1

    end

    # TODO: FIX THIS
    converged = true
    return converged, max_iter, absolute_residual
end


end #module
