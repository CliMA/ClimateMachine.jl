# # Conjugate Gradient
# In this tutorial we describe the basics of using the conjugate gradient iterative solvers
# At the end you should be able to
# 1. Use Conjugate Gradient to solve a linear system
# 2. Know when to not use it
# 3. Contruct a column-wise linear solver with Conjugate Gradient

# ## What is it?
# Conjugate Gradient is an iterative method for solving special kinds of linear systems:
# ```math
#  Ax = b
# ```
# via iterative methods.
# !!! warning
#     The linear operator need to be symmetric positive definite and the preconditioner must be symmetric.
# See the [wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method) for more details.

# ## Basic Example
# First we must load a few things
using ClimateMachine,
    ClimateMachine.LinearSolvers, ClimateMachine.ConjugateGradientSolver
using LinearAlgebra, Random

# Next we define a 3x3 symmetric positive definite linear system. (In the ClimateMachine code a symmetric positive definite system could arise from treating diffusion implicitly.)
A = [
    2.0 -1.0 0.0
    -1.0 2.0 -1.0
    0.0 -1.0 2.0
];
# We define the matrix `A` here as a global variable for convenience later.

# We can see that it is symmetric. We can check that it is positive definite by checking the spectrum
eigvals(A)

# The linear operators that are passed into the abstract iterative solvers need to be defined as functions that act on vectors. Let us do that with our matrix. We are using function closures for type stability.
function closure_linear_operator!(A)
    function linear_operator!(x, y)
        mul!(x, A, y)
    end
    return linear_operator!
end;

# We now define our linear operator using the function closure

linear_operator! = closure_linear_operator!(A)

# We now define our `b` in the linear system
b = ones(typeof(1.0), 3);

# The exact solution to the system `Ax = b` is
x_exact = [1.5, 2.0, 1.5];

# Now we can set up the ConjugateGradient struct
linearsolver = ConjugateGradient(b);
# and an initial guess for the iterative solver.
x = ones(typeof(1.0), 3);
# To solve the linear system we just need to pass to the linearsolve! function
iters = linearsolve!(linear_operator!, linearsolver, x, b)
# The variable `x` gets overwitten during the linear solve
# The norm of the error is
norm(x - x_exact) / norm(x_exact)
# The relative norm of the residual is
norm(A * x - b) / norm(b)
# The number of iterations is
iters
# Conjugate Gradient is guaranteed to converge in 3 iterations with perfect arithmetic in this case.

# ## Non-Example

# Conjugate Gradient is not guaranteed to converge with nonsymmetric matrices. Consider
A = [
    2.0 -1.0 0.0
    0.0 2.0 -1.0
    0.0 0.0 2.0
];
# We define the matrix `A` here as a global variable for convenience later.

# We can see that it is not symmetric, but it does have all positive eigenvalues
eigvals(A)

# The linear operators that are passed into the abstract iterative solvers need to be defined as functions that act on vectors. Let us do that with our matrix. We are using function closures for type stability.
function closure_linear_operator!(A)
    function linear_operator!(x, y)
        mul!(x, A, y)
    end
    return linear_operator!
end;

# We define the linear operator using our function closure

linear_operator! = closure_linear_operator!(A)

# We now define our `b` in the linear system
b = ones(typeof(1.0), 3);

# The exact solution to the system `Ax = b` is
x_exact = [0.875, 0.75, 0.5];

# Now we can set up the ConjugateGradient struct
linearsolver = ConjugateGradient(b, max_iter = 100);
# We also passed in the keyword argument "max_iter" for the maximum number of iterations of the iterative solver. By default it is assumed to be the size of the vector.
# As before we need to define an initial guess
x = ones(typeof(1.0), 3);
# To (not) solve the linear system we just need to pass to the linearsolve! function
iters = linearsolve!(linear_operator!, linearsolver, x, b)
# The variable `x` gets overwitten during the linear solve
# The norm of the error is
norm(x - x_exact) / norm(x_exact)
# The relative norm of the residual is
norm(A * x - b) / norm(b)
# The number of iterations is
iters
# Conjugate Gradient is guaranteed to converge in 3 iterations with perfect arithmetic for a symmetric positive definite matrix. Here we see that the matrix is not symmetric and it didn't converge even after 100 iterations.


# ## More Complex Example
# Here we show how to construct a column-wise iterative solver similar to what is is in the ClimateMachine code. The following is not for the faint of heart.
# We must first define a linear operator that acts like one in the ClimateMachine
function closure_linear_operator!(A, tup)
    function linear_operator!(y, x)
        alias_x = reshape(x, tup)
        alias_y = reshape(y, tup)
        for i6 in 1:tup[6]
            for i4 in 1:tup[4]
                for i2 in 1:tup[2]
                    for i1 in 1:tup[1]
                        tmp = alias_x[i1, i2, :, i4, :, i6][:]
                        tmp2 = A[i1, i2, i4, i6] * tmp
                        alias_y[i1, i2, :, i4, :, i6] .=
                            reshape(tmp2, (tup[3], tup[5]))
                    end
                end
            end
        end
    end
end;

# Now that we have this function, we can define a linear system that we will solve columnwise
# First we define the structure of our array as ```tup``` in a manner that is similar to a stacked brick topology
tup = (3, 4, 7, 2, 20, 2);
# where
# 1. tup[1] is the number of Gauss–Lobatto points in the x-direction
# 2. tup[2] is the number of Gauss–Lobatto points in the y-direction
# 3. tup[3] is the number of Gauss–Lobatto points in the z-direction
# 4. tup[4] is the number of states
# 6. tup[5] is the number of elements in the vertical direction
# 7. tup[6] is the number of elements in the other directions

# Now we define our linear operator as a random matrix.
Random.seed!(1235);
B = [
    randn(tup[3] * tup[5], tup[3] * tup[5])
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];
columnwise_A = [
    B[i1, i2, i4, i6] * B[i1, i2, i4, i6]' + 10I
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];
columnwise_inv_A = [
    inv(columnwise_A[i1, i2, i4, i6])
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];
columnwise_linear_operator! = closure_linear_operator!(columnwise_A, tup);
columnwise_inverse_linear_operator! =
    closure_linear_operator!(columnwise_inv_A, tup);

# We define our `x` and `b` with matrix structures similar to an MPIStateArray
mpi_tup = (tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6]);
b = randn(mpi_tup);
x = randn(mpi_tup);

# Now we solve the linear system columnwise
linearsolver = ConjugateGradient(
    x,
    max_iter = tup[3] * tup[5],
    dims = (3, 5),
    reshape_tuple = tup,
);
# The keyword arguments dims is the reduction dimension for the linear solver. In this case dims = (3,5) are the ones associated with a column. The reshape_tuple argument is to convert the shapes of the array `x` in the a form that is more easily usable for reductions in the linear solver

# Now we can solve it
iters = linearsolve!(columnwise_linear_operator!, linearsolver, x, b);
x_exact = copy(x);
columnwise_inverse_linear_operator!(x_exact, b);
# The norm of the error is
norm(x - x_exact) / norm(x_exact)
# The number of iterations is
iters
# The algorithm converges within `tup[3]*tup[5] = 140` iterations

# ## Tips
# 1. The convergence criteria should be changed, machine precision is too small and the maximum iterations is often too large
# 2. Use a preconditioner if possible
# 3. Make sure that the linear system really is symmetric and positive-definite 
