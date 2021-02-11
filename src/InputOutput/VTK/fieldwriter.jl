import KernelAbstractions: CPU

using MPI
using Printf

using ..BalanceLaws
using ..MPIStateArrays
using ..DGMethods: SpaceDiscretization
using ..VariableTemplates

"""
    VTKFieldWriter(
        name::String,
        FT::DataType,
        fields::Vector{<:Tuple{String, <:Function}};
        path_prefix = ".",
        number_sample_points = 0,
    )

Construct a callable type that computes the specified `fields` and
writes them to a VTK file in `path_prefix`/. `number_sample_points`
are passed to [`writevtk`](@ref).

Intended for use in a callback passed to a running simulation; files
are suffixed with a running number beginning with 0.

# Example
```julia
    function pres_fun(atmos::AtmosModel, prognostic::Vars, auxiliary::Vars)
        ts = recover_thermo_state(atmos, prognostic, auxiliary)
        air_pressure(ts)
    end
    fwriter = VTKFieldWriter(solver_config.name, [("pressure", pres_fun)])
    cbfw = GenericCallbacks.EveryXSimulationTime(60) do
        fwriter(solver_config.dg, solver_config.Q)
    end
```
"""
mutable struct VTKFieldWriter
    path_prefix::String
    name::String
    number_sample_points::Int
    nfields::Int
    field_names::Vector{String}
    field_funs::Vector{<:Function}
    vars_type::DataType
    num::Int

    function VTKFieldWriter(
        name::String,
        FT::DataType,
        fields::Vector{<:Tuple{String, <:Function}};
        path_prefix = ".",
        number_sample_points = 0,
    )
        nfields = length(fields)
        field_names = [name for (name, _) in fields]
        field_funs = [fun for (_, fun) in fields]
        vars_type = NamedTuple{
            tuple(Symbol.(field_names)...),
            Tuple{[FT for _ in 1:length(fields)]...},
        }
        new(
            path_prefix,
            name * "_fields",
            number_sample_points,
            nfields,
            field_names,
            field_funs,
            vars_type,
            0,
        )
    end
end
function (vfw::VTKFieldWriter)(dg::SpaceDiscretization, Q::MPIStateArray)
    bl = dg.balance_law
    fQ = similar(Q, Array; vars = vfw.vars_type, nstate = vfw.nfields)

    if array_device(Q) isa CPU
        prognostic_array = Q.realdata
        auxiliary_array = dg.state_auxiliary.realdata
    else
        prognostic_array = Array(Q.realdata)
        auxiliary_array = Array(dg.state_auxiliary.realdata)
    end
    FT = eltype(prognostic_array)

    for e in 1:size(prognostic_array, 3)
        for n in 1:size(prognostic_array, 1)
            prognostic = Vars{vars_state(bl, Prognostic(), FT)}(view(
                prognostic_array,
                n,
                :,
                e,
            ),)
            auxiliary = Vars{vars_state(bl, Auxiliary(), FT)}(view(
                auxiliary_array,
                n,
                :,
                e,
            ),)
            for i in 1:(vfw.nfields)
                fQ[n, i, e] = vfw.field_funs[i](bl, prognostic, auxiliary)
            end
        end
    end

    mpirank = MPI.Comm_rank(fQ.mpicomm)
    vprefix = @sprintf("%s_mpirank%04d_num%04d", vfw.name, mpirank, vfw.num,)
    outprefix = joinpath(vfw.path_prefix, vprefix)
    writevtk(outprefix, fQ, dg, number_sample_points = vfw.number_sample_points)

    # Generate the pvtu file for these vtk files
    if mpirank == 0
        pprefix = @sprintf("%s_num%04d", vfw.name, vfw.num)
        pvtuprefix = joinpath(vfw.path_prefix, pprefix)

        prefixes = ntuple(MPI.Comm_size(fQ.mpicomm)) do i
            @sprintf("%s_mpirank%04d_num%04d", vfw.name, i - 1, vfw.num,)
        end
        writepvtu(pvtuprefix, prefixes, tuple(vfw.field_names...), eltype(fQ))
    end

    vfw.num += 1
    nothing
end
