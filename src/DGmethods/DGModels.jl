module DGModels
export DGModel, @model

abstract type DGModel{Dim, N} end

macro model(expr)
    @assert expr.head === :<:
    def = Expr(:struct, false, expr, Expr(:block))
    methods = (
        :flux!,
        :numerical_flux!,
        :numerical_boundary_flux!,
        :gradient_transform!,
        :viscous_transform!,
        :viscous_penalty!,
        :viscous_boundary_penalty!,
        :source!,
        :hasbctype,
        :nstate,
        :nviscstate,
        :ngradstate,
        :nauxstate,
        :states_grad)
    imports = (Expr(:., m) for m in methods)

    Expr(:toplevel,
        Expr(:import, Expr(Symbol(":"), Expr(:., fullname(@__MODULE__)...), imports...)),
        def)
end

flux!(::DGModel, args...) = error("")
numerical_flux!(::DGModel, args...) = error("")
numerical_boundary_flux!(::DGModel, args...) = error("")
gradient_transform!(::DGModel, args...) = nothing
viscous_transform!(::DGModel, args...) = nothing
viscous_penalty!(::DGModel, args...) = nothing
viscous_boundary_penalty!(::DGModel, args...) = nothing
source!(::DGModel, args...)  = nothing
hasbctype(::DGModel) = error()

export nstate, nviscstate, ngradstate, nauxstate, states_grad
function nstate(::DGModel) end
function nviscstate(::DGModel) end
function ngradstate(::DGModel) end
function nauxstate(::DGModel) end
function states_grad(::DGModel) end

end# module
