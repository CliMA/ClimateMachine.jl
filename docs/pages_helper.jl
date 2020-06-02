"""
    get_second

Gets `second` in nested array of `Pair`s
and filters empty entries.
"""
function get_second end

get_second(x::String) = x
get_second(x::Pair{String, String}) = x.second
get_second(x::Pair{String, T}) where {T} = get_second(x.second)
get_second(A::Array{T}) where {T} =
    filter(y -> !isempty(y), [get_second(x) for x in A if !isempty(x)])

"""
    flatten_to_array_of_strings(A)

Recursive `flatten` of nested array of strings.
"""
function flatten_to_array_of_strings(A)
    V = String[]
    for a in A
        if a isa String
            push!(V, a)
        else
            push!(V, flatten_to_array_of_strings(a)...)
        end
    end
    return V
end

"""
    transform_second

Transform `second` in nested array of `Pair`s.
"""
function transform_second end

transform_second(transform, x::String) = transform(x)
transform_second(transform, x::Pair{String, String}) =
    Pair(x.first, transform_second(transform, x.second))
transform_second(transform, x::Pair{String, T}) where {T} =
    Pair(x.first, transform_second(transform, x.second))
transform_second(transform, A::Array{T}) where {T} =
    Any[transform_second(transform, x) for x in A]
