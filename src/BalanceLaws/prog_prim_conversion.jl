# Add `Primitive` type to BalanceLaws, AtmosModel

# Vars wrapper
function prognostic_to_primitive!(
    bl::BalanceLaw,
    prim::AbstractArray,
    prog::AbstractArray,
    aux::AbstractArray,
)
    FT = eltype(prim)
    prognostic_to_primitive!(
        bl,
        Vars{vars_state(bl, Primitive(), FT)}(prim),
        Vars{vars_state(bl, Prognostic(), FT)}(prog),
        Vars{vars_state(bl, Auxiliary(), FT)}(aux),
    )
end
function primitive_to_prognostic!(
    bl::BalanceLaw,
    prog::AbstractArray,
    prim::AbstractArray,
    aux::AbstractArray,
)
    FT = eltype(prog)
    primitive_to_prognostic!(
        bl,
        Vars{vars_state(bl, Prognostic(), FT)}(prog),
        Vars{vars_state(bl, Primitive(), FT)}(prim),
        Vars{vars_state(bl, Auxiliary(), FT)}(aux),
    )
end

# By default the primitive is the prognostic
vars_state(bl::BalanceLaw, ::Primitive, FT) = vars_state(bl, Prognostic(), FT)

function prognostic_to_primitive!(bl, prim::Vars, prog::Vars, aux)
    prim_arr = parent(prim)
    prog_arr = parent(prog)
    prim_arr .= prog_arr
end
function primitive_to_prognostic!(bl, prog::Vars, prim::Vars, aux)
    prim_arr = parent(prim)
    prog_arr = parent(prog)
    prog_arr .= prim_arr
end


"""
    construct_face_auxiliary_state!(bl::AtmosModel, aux_face::AbstractArray, aux_cell::AbstractArray, Δz::FT)
Default constructor
 - `bl` balance law
 - `aux_face` face auxiliary variables to be constructed 
 - `aux_cell` cell center auxiliary variable
 - `Δz` cell vertical size 
"""
function construct_face_auxiliary_state!(
    bl,
    aux_face::AbstractArray,
    aux_cell::AbstractArray,
    Δz::FT,
) where {FT}
    aux_face .= aux_cell
end
