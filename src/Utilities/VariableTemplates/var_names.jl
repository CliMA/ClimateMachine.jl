export var_names, flatten_state

function var_names(model, DT, state::Function=vars_state)
  return string.(flatten_state(model, DT, state))
end

function flatten_state(model, DT, state::Function=vars_state)
  vn_all = Symbol[]
  vs = state(model, DT)
  expand_sizes_from_var_names!(vn_all, vs, Symbol())
  rflatten_state!(vn_all, state, model, typeof(model), Symbol(), DT)
  return vn_all
end

function rflatten_state!(vn_all, state, model, ::Type{T}, prefix, DT, i::Int=0; recursive::Bool=true) where T
    fields = fieldnames(T)
    if !isempty(fields)
        for field in fields
            sub_model = getfield(model, field)
            prefix = Symbol(prefix, Symbol(field, "."))
            try
              vs = state(sub_model, DT)
              expand_sizes_from_var_names!(vn_all, vs, Symbol(field, "."))
            catch
            end
            if recursive
                rflatten_state!(vn_all, state, sub_model, fieldtype(T, field), prefix, DT, i+1, recursive=true)
            end
        end
    end
end

function expand_sizes_from_var_names!(vn_all, vs, prefix)
  for (ft, fn) in zip(fieldtypes(vs), fieldnames(vs))
    if !(ft <: NamedTuple)
      X = ft <: SArray ? range(1,stop=length(ft)) : (Symbol(),)
      push!(vn_all, [Symbol(prefix, fn, x) for x in X]...)
    end
  end
end
