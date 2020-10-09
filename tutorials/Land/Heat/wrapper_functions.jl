
function Base.show(io::IO, field::Wrapper{T, S}) where {T <: Char, S}
    print(io, field.data)
end
function Base.show(io::IO, field::Wrapper{T, S}) where {T, S <: WrapperMetaData}
    print(io, field.meta_data.io_name)
end

struct Wrapper{T, S} <: AbstractExpression
    data::T
    meta_data::S
end

# Struct for MetaData
struct WrapperMetaData{T}
    io_name::T
end

compute(a::Wrapper) = a.data

macro wrapper(expr)
    rewritten_expr = _wrapper(expr)
    return rewritten_expr
end

function _wrapper(expr::Expr)
    symb = expr.args[1]
    val  = expr.args[2]
    if expr.head != :(=)
        println( "@wrapper macro not in proper form")
        println( "must be ")
        println( "@wrapper a=1 b=2 c=3")
        return error()
    end
    string_symb = String(symb)
    new_expr = :($(esc(symb)) =  Wrapper($val, WrapperMetaData($string_symb)))
    return new_expr
end

macro wrapper(exprs...)
    rewritten_exprs = [_wrapper(expr) for expr in exprs]
    return Expr(:block, rewritten_exprs...)
end