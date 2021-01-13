module FieldArrays

export FieldArray, localfieldarray
import ..MPIStateArrays: MPIStateArray, vars

struct FieldArray{V, A}
    data::A
end
FieldArray{V}(data::A) where {V, A} = FieldArray{V, A}(data)
vars(::FieldArray{V}) where {V} = V

localfieldarray(M::MPIStateArray{FT, V}) where {FT, V} = FieldArray{V}(M.data)
Base.eltype(F::FieldArray) = eltype(F.data)

end
