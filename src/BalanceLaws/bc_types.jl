##### Boundary condition types

export DefaultBC, set_bcs!

"""
    BCDef

Subtypes are used for specifying
each boundary condition.
"""
abstract type BCDef{PV <: PrognosticVariable} end

"""
    DefaultBC

The default boundary condition
"""
struct DefaultBC{PV} <: BCDef{PV} end

"""
    bc_val(::BCDef{PV}, bl, nf, args)

Return the value of the boundary condition, given
 - `bcd::BCDef` the boundary condition definition type
 - `bl` the balance law
 - `nf` the numerical flux
 - `args` top-level arguments, packed into a NamedTuple
"""
bc_val(::BCDef{PV}, bl, nf, args) where {PV} = DefaultBC{PV}()

"""
    set_bcs!(state⁺, bl, nf, bc, ntargs, prog_vars = prognostic_vars(bl))

A convenience method for setting `state⁺` such that numerical fluxes,
computed in `flux_first_order!`, enforces boundary conditions.

This method is to be called in `boundary_state!`.

Arguments:
 - `state⁺` the exterior state
 - `bl` the balance law
 - `nf::Union{NumericalFluxFirstOrder,NumericalFluxGradient}` the numerical flux
 - `ntargs` the top-level arguments
 - `prog_vars` (optional) the balance law's prognostic variables
"""
function set_bcs!(state⁺, bl, nf, bc, ntargs, prog_vars = prognostic_vars(bl))
    state⁻ = ntargs.state
    map(prog_vars) do prog
        var⁺, name = get_prog_state(state⁺, prog)
        var⁻, name = get_prog_state(state⁻, prog)
        var_bcs = bcs_per_prog_var(bc.tup, prog)
        bcvals = map(var_bcs) do bc_pv
            bc_val(bc_pv, bl, nf, ntargs)
        end
        set_bc!(var⁺, name, bcvals)
    end
end

#####
##### Internal methods for setting/diagonalizing boundary conditions
#####

diag_bc(::PVA, ::BCDef{PVB}) where {PVA, PVB} = nothing
diag_bc(::PV, bcd::BCDef{PV}) where {PV} = bcd

filter_bcs(t::Tuple) = filter(x -> !(x == nothing), t)
diag_bc(tup, pv) = filter_bcs(map(bc -> diag_bc(pv, bc), tup))

prog_var_bcs(::Tuple{}, ::PV) where {PV} = (DefaultBC{PV}(),)
prog_var_bcs(t::Tuple, ::PV) where {PV} = t

bcs_per_prog_var(tup, pv::PrognosticVariable) =
    prog_var_bcs(diag_bc(tup, pv), pv)

set_bc!(var⁺, name, bcvals::Tuple{DefaultBC}) = nothing
set_bc!(var⁺, name, bcvals) = setproperty!(var⁺, name, sum(bcvals))
