using MacroTools
using MacroTools: postwalk, @capture, @expand, prettify

export @uaware

"""
  Duplicate structure definitions, maintaining a common constructor, to enable optional
  unit annotation.

  This macro is most useful when defining structures or methods that are unit aware, however
  may still be invoked on drivers which do not support units.
"""
macro uaware(ex)
  if @capture(ex, struct sig_ attrs__ end)

    # Remove docstrings from attrs
    attrs = filter(x->!(x isa String), attrs)

    # Rename the struct
    local sig_unitless, sig_unitful, name, union, p1, p2
    if @capture(sig, name_{p1__} <: sname_{p2__})
      lname = Symbol(:L, name)
      uname = Symbol(:U, name)
      sig_unitless = :($lname{$(p1...)} <: $sname{$(p2...)})
      sig_unitful  = :($uname{$(p1...)} <: $sname{$(p2...)})
      union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
    elseif @capture(sig, name_{p1__} <: sname_)
      p2 = Any[]
      lname = Symbol(:L, name)
      uname = Symbol(:U, name)
      sig_unitless = :($lname{$(p1...)} <: $sname)
      sig_unitful  = :($uname{$(p1...)} <: $sname)
      union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
    elseif @capture(sig, name_{p1__})
      lname = Symbol(:L, name)
      uname = Symbol(:U, name)
      sig_unitless = :($lname{$(p1...)})
      sig_unitful  = :($uname{$(p1...)})
      union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
    elseif @capture(sig, name_)
      lname = Symbol(:L, name)
      uname = Symbol(:U, name)
      p1, p2 = Any[], Any[]
      sig_unitless = lname
      sig_unitful  = uname
      union = :($name = Union{$sig_unitless, $sig_unitful})
    end
    @assert sig_unitful !== sig_unitless

    attrs_unitless, attrs_unitful = split_U(attrs)

    # First just remove all units
    unitless = quote
      struct $sig_unitless
        $(attrs_unitless...)
      end
    end

    # Now make a version with unit annotations
    unitful = quote
      struct $sig_unitful
        $(attrs_unitful...)
      end
    end

    params_unitless = filter(x->!(x isa String), attrs_unitless)
    params_unitful  = filter(x->!(x isa String), attrs_unitful)

    # Lastly provide the constructors
    constr_unitless = quote
      function ($name{$(p1...)})($(params_unitless...)) where {$(p1...)}
        $lname{$(p1...)}($(attrs_unitless...))
      end
      function $name($(params_unitless...)) where {$(p1...)}
        $lname{$(p1...)}($(attrs_unitless...))
      end
    end
    constr_unitful = quote
      function ($name{$(p1...)})($(params_unitful... )) where {$(p1...)}
        $uname{$(p1...)}($(attrs_unitful...))
      end
      function $name($(params_unitful... )) where {$(p1...)}
        $uname{$(p1...)}($(attrs_unitful...))
      end
    end

    return quote
                    $unitless
      Base.@__doc__ $unitful
                    $union
                    $constr_unitless
                    $constr_unitful
    end |> esc
  end

  error("Expected a structure or function definition for annotation.")
end

"""
  From an array of expressions, replace all U(FT,:unitsymbol) statements with their
  corresponding unitless, unitful variations, as a tuple of arrays.
"""
function split_U(exprs)
  unitless(ex) = postwalk(ex) do x
    @capture(x, U(FT_, usym_)) && (return FT)
    x
  end
  unitful(ex)  = postwalk(ex) do x
    @capture(x, U(FT_, usym_)) && (return :(units($FT, $usym)))
    x
  end
  (map(unitless, exprs), map(unitful, exprs))
end
