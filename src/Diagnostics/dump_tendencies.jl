"""
    setup_dump_tendencies_diagnostics(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create the "DumpTendencies" `DiagnosticsGroup` which contains all the
tendencies for the simulation's `BalanceLaw`. These are collected on
an interpolated grid (`interpol` _must_ be specified).

!!! warn
    This diagnostics group can produce a lot of output and is
    very expensive.

!!! warn
    These diagnostics are intended for physics-debugging, and not numerics debugging, because
    `flux` and `source` are called on interpolated data, which may not exactly match the
    non-interpolated evaluation of fluxes and sources that the solver uses.
"""
function setup_dump_tendencies_diagnostics(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "DumpTendencies",
        Diagnostics.dump_tendencies_init,
        Diagnostics.dump_tendencies_fini,
        Diagnostics.dump_tendencies_collect,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

# helpers for PV and tendency names
params(::T) where {T} = T.parameters
unval(::Val{i}) where {i} = i
param_suffix(p, ::Val{n}) where {n} = string("_", unval(p[1]))
param_suffix(p, ::Val{0}) = ""
param_suffix(p) = param_suffix(p, Val(length(p)))
prog_name(pv::PV) where {PV} = string(nameof(PV), param_suffix(params(pv)))
tend_name(t) = String(nameof(typeof(t)))

function precompute_args(
    bl,
    state,
    aux,
    diffusive,
    hyperdiffusive,
    t,
    direction,
)
    _args_fx1 = (; state, aux, t, direction)
    _args_fx2 = (; state, aux, t, diffusive, hyperdiffusive)
    _args_src = (; state, aux, t, direction, diffusive)

    cache_fx1 = precompute(bl, _args_fx1, Flux{FirstOrder}())
    cache_fx2 = precompute(bl, _args_fx2, Flux{SecondOrder}())
    cache_src = precompute(bl, _args_src, Source())

    args_fx1 = merge(_args_fx1, (; precomputed = cache_fx1))
    args_fx2 = merge(_args_fx2, (; precomputed = cache_fx2))
    args_src = merge(_args_src, (; precomputed = cache_src))

    return (args_fx1, args_fx2, args_src)
end

function dump_tendencies_init(dgngrp, t)
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q)
    bl = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if mpirank == 0
        if array_device(Q) isa CPU
            prognostic_data = Q.realdata
            auxiliary_data = dg.state_auxiliary.realdata
            gradient_flux_data = dg.state_gradient_flux.realdata
            hyperdiffusive_data = dg.states_higher_order[2].realdata
        else
            prognostic_data = Array(Q.realdata)
            auxiliary_data = Array(dg.state_auxiliary.realdata)
            gradient_flux_data = Array(dg.state_gradient_flux.realdata)
            hyperdiffusive_data = Array(dg.states_higher_order[2].realdata)
        end

        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)
        dim_names = tuple(collect(keys(dims))...)

        state, aux, diffusive, hyperdiffusive = (
            extract_state(dg, prognostic_data, 1, 1, Prognostic()),
            extract_state(dg, auxiliary_data, 1, 1, Auxiliary()),
            extract_state(dg, gradient_flux_data, 1, 1, GradientFlux()),
            extract_state(dg, hyperdiffusive_data, 1, 1, Hyperdiffusive()),
        )
        direction = dg.direction
        args_fx1, args_fx2, args_src = precompute_args(
            bl,
            state,
            aux,
            diffusive,
            hyperdiffusive,
            t,
            direction,
        )

        # set up the variables we're going to be writing
        vars = OrderedDict()
        for (tend_fun, tend_type, tend_args) in (
            (flux, Flux{FirstOrder}(), args_fx1),
            (flux, Flux{SecondOrder}(), args_fx2),
            (source, Source(), args_src),
        )
            for pv in prognostic_vars(bl)
                progname = prog_name(pv)
                for tend in eq_tends(pv, bl, tend_type)
                    flat_vals = flattened_tuple(
                        FlattenArr(),
                        tend_fun(pv, tend, bl, tend_args),
                    )
                    for i in 1:length(flat_vals)
                        varname = "$(progname)_$(tend_name(tend))_$(i)"
                        vars[varname] = (dim_names, FT, Dict())
                    end
                end
            end
        end

        dprefix = @sprintf("%s_%s", dgngrp.out_prefix, dgngrp.name)
        dfilename = joinpath(Settings.output_dir, dprefix)
        noov = Settings.no_overwrite
        init_data(dgngrp.writer, dfilename, noov, dims, vars)
    end

    return nothing
end

function dump_tendencies_collect(dgngrp, t)
    interpol = dgngrp.interpol
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    istate = similar(Q.data, interpol.Npl, number_states(bl, Prognostic()))
    interpolate_local!(interpol, Q.data, istate)
    iaux = similar(Q.data, interpol.Npl, number_states(bl, Auxiliary()))
    interpolate_local!(interpol, dg.state_auxiliary.data, iaux)
    igf = similar(Q.data, interpol.Npl, number_states(bl, GradientFlux()))
    interpolate_local!(interpol, dg.state_gradient_flux.data, igf)
    ihd = similar(Q.data, interpol.Npl, number_states(bl, Hyperdiffusive()))
    interpolate_local!(interpol, dg.states_higher_order[2].data, ihd)

    if interpol isa InterpolationCubedSphere
        i_ρu = varsindex(vars_state(bl, Prognostic(), FT), :ρu)
        project_cubed_sphere!(interpol, istate, tuple(collect(i_ρu)...))
    end

    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)
    all_gf_data = accumulate_interpolated_data(mpicomm, interpol, igf)
    all_hd_data = accumulate_interpolated_data(mpicomm, interpol, ihd)

    if mpirank == 0
        varvals = OrderedDict()
        dims = dimensions(dgngrp.interpol)
        direction = dg.direction
        nx, ny, nz = [length(dim[1]) for dim in values(dims)]
        @traverse_interpolated_grid nx ny nz begin
            vars_view(st, data) =
                Vars{vars_state(bl, st, FT)}(view(data, lo, la, le, :))
            statei = vars_view(Prognostic(), all_state_data)
            auxi = vars_view(Auxiliary(), all_aux_data)
            diffusivei = vars_view(GradientFlux(), all_gf_data)
            hyperdiffusivei = vars_view(Hyperdiffusive(), all_hd_data)

            args_fx1, args_fx2, args_src = precompute_args(
                bl,
                statei,
                auxi,
                diffusivei,
                hyperdiffusivei,
                t,
                direction,
            )

            for (tend_fun, tend_type, tend_args) in (
                (flux, Flux{FirstOrder}(), args_fx1),
                (flux, Flux{SecondOrder}(), args_fx2),
                (source, Source(), args_src),
            )
                for pv in prognostic_vars(bl)
                    progname = prog_name(pv)
                    for tend in eq_tends(pv, bl, tend_type)
                        flat_vals = flattened_tuple(
                            FlattenArr(),
                            tend_fun(pv, tend, bl, tend_args),
                        )
                        for i in 1:length(flat_vals)
                            varname = "$(progname)_$(tend_name(tend))_$(i)"
                            vals = get!(
                                varvals,
                                varname,
                                Array{FT}(undef, nx, ny, nz),
                            )
                            vals[lo, la, le] = flat_vals[i]
                        end
                    end
                end
            end

        end

        append_data(dgngrp.writer, varvals, t)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function dump_tendencies_fini(dgngrp, t) end
