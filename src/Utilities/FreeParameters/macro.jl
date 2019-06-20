macro parameters(expr)
  expr = macroexpand(__module__, expr) # to expand @static
  expr isa Expr && expr.head == :struct || error("Invalid usage of @kwdef")
  T = expr.args[2]
  if T isa Expr && T.head == :<:
    T = T.args[1]
  end

  params_ex = Expr(:parameters)
  call_args = Any[]
  domains_vals = Any[]

  _parameters!(expr.args[3], params_ex.args, call_args, domains_vals)
  # Only define a constructor if the type has fields, otherwise we'll get a stack
  # overflow on construction
  if !isempty(params_ex.args)
    if T isa Symbol
      kwdefs = quote
        ($(esc(T)))($params_ex) = ($(esc(T)))($(call_args...))
        FreeParameters.domains(::Type{$(esc(T))}) =
          ($(domains_vals...),)
      end
    elseif T isa Expr && T.head == :curly
      # if T == S{A<:AA,B<:BB}, define two methods
      #   S(...) = ...
      #   S{A,B}(...) where {A<:AA,B<:BB} = ...
      S = T.args[1]
      P = T.args[2:end]
      Q = [U isa Expr && U.head == :<: ? U.args[1] : U for U in P]
      SQ = :($S{$(Q...)})
      kwdefs = quote
        ($(esc(S)))($params_ex) =($(esc(S)))($(call_args...))
        ($(esc(SQ)))($params_ex) where {$(esc.(P)...)} =
          ($(esc(SQ)))($(call_args...))
        FreeParameters.domains(::Type{$(esc(SQ))}) where {$(esc.(P)...)} =
          ($(domains_vals...),)
      end
    else
      error("Invalid usage of @parameters")
    end
  else
    kwdefs = nothing
  end
  quote
    Base.@__doc__($(esc(expr)))
    $kwdefs
  end
end

# @kwdef helper function
# mutates arguments inplace
function _parameters!(blk, params_args, call_args, domains_vals)
  for i in eachindex(blk.args)
    ei = blk.args[i]
    if ei isa Symbol
      #  var
      push!(params_args, ei)
      push!(call_args, ei)
      push!(domains_vals, Any)
    elseif ei isa Expr
      if ei.head == :(=)
        lhs = ei.args[1]
        if lhs isa Symbol
          #  var = defexpr [∈ domexpr]
          var = lhs
          T = Any
        elseif lhs isa Expr && lhs.head == :(::) && lhs.args[1] isa Symbol
          #  var::T = defexpr [∈ domexpr]
          var = lhs.args[1]
          T = lhs.args[2]
        else
          # something else, e.g. inline inner constructor
          #   F(...) = ...
          continue
        end
        rhs = ei.args[2]  # defexpr

        if rhs isa Expr && rhs.head == :call && rhs.args[1] in (:∈, :in)
          defexpr = rhs.args[2]
          domexpr = rhs.args[3]
        else
          defexpr = rhs
          domexpr = T
        end

        if domexpr isa Expr && domexpr.head == :tuple
          domexpr = :(RealDomain($(esc(domexpr.args[1])), $(esc(domexpr.args[1]))))
        else
          domexpr = esc(domexpr)
        end

        push!(params_args, Expr(:kw, var, esc(defexpr)))
        push!(call_args, var)
        push!(domains_vals, domexpr)

        blk.args[i] = lhs

      elseif ei.head == :call && ei.args[1] in (:∈, :in)
        lhs = ei.args[2]
        if lhs isa Symbol
          #  var ∈ domexpr
          var = lhs
        elseif lhs isa Expr && lhs.head == :(::) && lhs.args[1] isa Symbol
          #  var::T ∈ domexpr
          var = lhs.args[1]
        else
          # something else, e.g. inline inner constructor
          #   F(...)
          continue
        end
        domexpr = ei.args[3]

        if domexpr isa Expr && domexpr.head == :tuple
          domexpr = :(RealDomain($(esc(domexpr.args[1])), $(esc(domexpr.args[1]))))
        else
          domexpr = esc(domexpr)
        end

        push!(params_args, var)
        push!(call_args, var)
        push!(domains_vals, domexpr)

        blk.args[i] = lhs

      elseif ei.head == :(::) && ei.args[1] isa Symbol
        # var::Typ
        var = ei.args[1]
        push!(params_args, var)
        push!(call_args, var)
        push!(domains_vals, Any)
      elseif ei.head == :block
        # can arise with use of @static inside type decl
        _parameters!(ei, params_args, call_args, domains_vals)
      end
    end
  end
  blk
end
