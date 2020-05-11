export flattenednames

flattenednames(::Type{NamedTuple{(), Tuple{}}}; prefix = "") = ()
flattenednames(::Type{T}; prefix = "") where {T <: Real} = (prefix,)
flattenednames(::Type{T}; prefix = "") where {T <: SVector} =
    ntuple(i -> "$prefix[$i]", length(T))
function flattenednames(::Type{T}; prefix = "") where {T <: SHermitianCompact}
    N = size(T, 1)
    [["$prefix[$i,$j]" for i in j:N] for j in 1:N] |>
    Iterators.flatten |>
    collect
end
function flattenednames(::Type{T}; prefix = "") where {T <: NamedTuple}
    map(1:fieldcount(T)) do i
        Ti = fieldtype(T, i)
        name = fieldname(T, i)
        flattenednames(
            Ti,
            prefix = prefix == "" ? string(name) : string(prefix, '.', name),
        )
    end |>
    Iterators.flatten |>
    collect
end
