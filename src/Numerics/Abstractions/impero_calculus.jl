using MPI
using ClimateMachine.MPIStateArrays

export curl
export vector_grad_closure, curl_closure, div_closure, impero_operators

function impero_operators(grid)
    ∇op = vector_grad_closure(grid)
    ∇ = Operator(nothing, OperatorMetaData(∇op, "∇"))
    curlop = curl_closure(grid)
    curl = Operator(nothing, OperatorMetaData(curlop, "∇×"))
    divop = div_closure(grid)
    div = Operator(nothing, OperatorMetaData(divop, "∇⋅"))

    ×(o::Operator, a::AbstractExpression) = curl(a)
    ⋅(o::Operator, a::AbstractExpression) = div(a)

    return div, ∇, curl, ×, ⋅
end


function curl(grid, Q)
    d1 = Diagnostics.VectorGradient(grid, Q[1], 1)
    d2 = Diagnostics.VectorGradient(grid, Q[2], 1)
    d3 = Diagnostics.VectorGradient(grid, Q[3], 1)
    vgrad = Diagnostics.VectorGradients(d1, d2, d3)
    vort = Diagnostics.Vorticity(grid, vgrad)
    return vort.data
end


function vector_grad_closure(grid)
    function tmp(Q::MPIStateArray)
        return Diagnostics.VectorGradient(grid, Q, 1)
    end
    function tmp(Q::Tuple)
        d1 = Diagnostics.VectorGradient(grid, Q[1], 1)
        d2 = Diagnostics.VectorGradient(grid, Q[2], 1)
        d3 = Diagnostics.VectorGradient(grid, Q[3], 1)
        return (d1,d2,d3)
    end
    return tmp
end

function curl_closure(grid)
    function tmp(Q::Tuple)
        d1 = Diagnostics.VectorGradient(grid, Q[1], 1)
        d2 = Diagnostics.VectorGradient(grid, Q[2], 1)
        d3 = Diagnostics.VectorGradient(grid, Q[3], 1)
        vgrad = Diagnostics.VectorGradients(d1, d2, d3)
        vort = Diagnostics.Vorticity(grid, vgrad)
        return vort
    end
    return tmp
end

function div_closure(grid)
    function tmp(Q::Tuple)
        d1 = Diagnostics.VectorGradient(grid, Q[1], 1)
        d2 = Diagnostics.VectorGradient(grid, Q[2], 1)
        d3 = Diagnostics.VectorGradient(grid, Q[3], 1)
        return d1[:,1,:] + d2[:,2,:] + d3[:,3,:]
    end
    return tmp
end

#=
∇op = vector_grad_closure(grid)
∇ = Operator(nothing, OperatorMetaData(∇op, "∇"))
curlop = curl_closure(grid)
curl = Operator(nothing, OperatorMetaData(curlop, "∇×"))
divop = div_closure(grid)
div = Operator(nothing, OperatorMetaData(divop, "∇⋅"))

# TODO: this is hacky and to do properly
#  requires the notion of a field object
×(o::Operator, a::AbstractExpression) = curl(a)
⋅(o::Operator, a::AbstractExpression) = div(a)
=#
