using Base.Threads, LinearAlgebra
import Base: getindex, materialize!, broadcasted

abstract type AbstractField end
struct ScalarField{S, T} <: AbstractField
    data::S
    grid::T
end

function (ϕ::ScalarField)(x::Tuple)
    return getvalue(ϕ.data, x, ϕ.grid)
end

function (ϕ::ScalarField)(x::Number, y::Number, z::Number)
    return getvalue(ϕ.data, (x, y, z), ϕ.grid)
end

function (ϕ::ScalarField)(x::Number, y::Number)
    return getvalue(ϕ.data, (x, y), ϕ.grid)
end

getindex(ϕ::ScalarField, i::Int) = ϕ.data[i]

materialize!(ϕ::ScalarField, f::Base.Broadcast.Broadcasted) =
    materialize!(ϕ.data, f)
broadcasted(identity, ϕ::ScalarField) = broadcasted(Base.identity, ϕ.data)

function (ϕ::ScalarField)(
    xlist::StepRangeLen,
    ylist::StepRangeLen,
    zlist::StepRangeLen;
    threads = false,
)
    newfield = zeros(length(xlist), length(ylist), length(zlist))
    if threads
        @threads for k in eachindex(zlist)
            for j in eachindex(ylist)
                for i in eachindex(xlist)
                    newfield[i, j, k] =
                        getvalue(ϕ.data, (xlist[i], ylist[j], zlist[k]), ϕ.grid)
                end
            end
        end
    else
        for k in eachindex(zlist)
            for j in eachindex(ylist)
                for i in eachindex(xlist)
                    newfield[i, j, k] =
                        getvalue(ϕ.data, (xlist[i], ylist[j], zlist[k]), ϕ.grid)
                end
            end
        end
    end
    return newfield
end

function (ϕ::ScalarField)(
    xlist::StepRangeLen,
    ylist::StepRangeLen;
    threads = false,
)
    newfield = zeros(length(xlist), length(ylist))
    if threads
        @threads for j in eachindex(ylist)
            for i in eachindex(xlist)
                newfield[i, j] = getvalue(ϕ.data, (xlist[i], ylist[j]), ϕ.grid)
            end
        end
    else
        for j in eachindex(ylist)
            for i in eachindex(xlist)
                newfield[i, j] = getvalue(ϕ.data, (xlist[i], ylist[j]), ϕ.grid)
            end
        end
    end
    return newfield
end
