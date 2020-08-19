# Quick Structs for checking calculations
import Impero:compute
struct Wrapper{T, S} <: AbstractExpression
    data::T
    meta_data::S
end
# Struct for MetaData
struct WrapperMetaData{T}
    io_name::T
end

function Base.show(io::IO, field::Wrapper{T, S}) where {T <: Char, S}
    color = 230
    printstyled(io, field.data, color = color)
end
function Base.show(io::IO, field::Wrapper{T, S}) where {T, S <: WrapperMetaData}
    color = 230
    printstyled(io, field.meta_data.io_name, color = color)
end

compute(a::Wrapper) = a.data

macro wrapper(expr)
    rewritten_expr = _wrapper(expr)
    return rewritten_expr
end

function _wrapper(expr::Expr)
    symb = expr.args[1]
    val  = expr.args[2]
    string_symb = String(symb)
    new_expr = :($(esc(symb)) =  Wrapper($val, WrapperMetaData($string_symb)))
    return new_expr
end

macro wrapper(exprs...)
    rewritten_exprs = [_wrapper(expr) for expr in exprs]
    return Expr(:block, rewritten_exprs...)
end

## Add directional derivatives
struct DirectionalDerivative{𝒮} <: AbstractExpression
    direction::𝒮
end
struct GradientMetaData{𝒮}
    direction::𝒮
end
function (p::DirectionalDerivative)(expr::AbstractExpression)
    return Gradient(expr, GradientMetaData(p.direction))
end
function Base.show(io::IO, p::DirectionalDerivative{S}) where S <: String
    print(io, Char(0x02202) * p.direction)
end
function Base.show(io::IO, p::Gradient{S, T}) where {S, T <: GradientMetaData{String}}
    printstyled(io, Char(0x02202) * p.metadata.direction, "(", color = 165)
    print(io, p.operand)
    printstyled(io, ")", color = 165)
end

∂x = DirectionalDerivative("x")
∂y = DirectionalDerivative("y")
∂z = DirectionalDerivative("z")
∂t = DirectionalDerivative("t")