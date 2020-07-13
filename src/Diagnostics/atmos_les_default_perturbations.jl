# AtmosLESDefaultPerturbations
#
# Computes perturbations from the horizontal averages for various
# fields.

using ..Atmos
using ..Mesh.Topologies
using ..Mesh.Grids
using ..Thermodynamics

# Compute sums for density-averaged horizontal averages
#
# These are trimmed down versions of `atmos_les_default_simple_sums!()`:
# - operate on interpolated grid (no `MH` scaling)
# - skip `w_ht_sgs` and `w_qt_sgs`
function atmos_les_default_perturbations_sums!(
    atmos::AtmosModel,
    state,
    thermo,
    sums,
)
    sums.u += state.ρu[1]
    sums.v += state.ρu[2]
    sums.w += state.ρu[3]
    sums.avg_rho += state.ρ
    sums.rho += state.ρ * state.ρ
    sums.temp += thermo.temp * state.ρ
    sums.pres += thermo.pres * state.ρ
    sums.thd += thermo.θ_dry * state.ρ
    sums.et += state.ρe
    sums.ei += thermo.e_int * state.ρ
    sums.ht += thermo.h_tot * state.ρ
    sums.hi += thermo.h_int * state.ρ

    atmos_les_default_perturbations_sums!(atmos.moisture, state, thermo, sums)

    return nothing
end
function atmos_les_default_perturbations_sums!(
    ::MoistureModel,
    state,
    thermo,
    sums,
)
    return nothing
end
function atmos_les_default_perturbations_sums!(
    moist::EquilMoist,
    state,
    thermo,
    sums,
)
    sums.moisture.qt += state.moisture.ρq_tot
    sums.moisture.ql += thermo.moisture.q_liq * state.ρ
    sums.moisture.qv += thermo.moisture.q_vap * state.ρ
    sums.moisture.thv += thermo.moisture.θ_vir * state.ρ
    sums.moisture.thl += thermo.moisture.θ_liq_ice * state.ρ

    return nothing
end

# Perturbations from horizontal averages
function vars_atmos_les_default_perturbations(m::AtmosModel, FT)
    @vars begin
        u_prime::FT
        v_prime::FT
        w_prime::FT
        avg_rho_prime::FT             # ρ
        temp_prime::FT
        pres_prime::FT
        thd_prime::FT                 # θ_dry
        et_prime::FT                  # e_tot
        ei_prime::FT                  # e_int
        ht_prime::FT
        hi_prime::FT

        moisture::vars_atmos_les_default_perturbations(m.moisture, FT)
    end
end
vars_atmos_les_default_perturbations(::MoistureModel, FT) = @vars()
function vars_atmos_les_default_perturbations(m::EquilMoist, FT)
    @vars begin
        qt_prime::FT                  # q_tot
        ql_prime::FT                  # q_liq
        qv_prime::FT                  # q_vap
        thv_prime::FT                 # θ_vir
        thl_prime::FT                 # θ_liq
    end
end
num_atmos_les_default_perturbation_vars(m, FT) =
    varsize(vars_atmos_les_default_perturbations(m, FT))
atmos_les_default_perturbation_vars(m, array) =
    Vars{vars_atmos_les_default_perturbations(m, eltype(array))}(array)

# Compute the perturbations from horizontal averages
function atmos_les_default_perturbations!(
    atmos::AtmosModel,
    state,
    thermo,
    ha,
    vars,
)
    u = state.ρu[1] / state.ρ
    vars.u_prime = u - ha.u
    v = state.ρu[2] / state.ρ
    vars.v_prime = v - ha.v
    w = state.ρu[3] / state.ρ
    vars.w_prime = w - ha.w
    vars.avg_rho_prime = state.ρ - ha.avg_rho
    vars.temp_prime = thermo.temp - ha.temp
    vars.pres_prime = thermo.pres - ha.pres
    vars.thd_prime = thermo.θ_dry - ha.thd
    et = state.ρe / state.ρ
    vars.et_prime = et - ha.et
    vars.ei_prime = thermo.e_int - ha.ei
    vars.ht_prime = thermo.h_tot - ha.ht
    vars.hi_prime = thermo.h_int - ha.hi

    atmos_les_default_perturbations!(
        atmos.moisture,
        atmos,
        state,
        thermo,
        ha,
        vars,
    )

    return nothing
end
function atmos_les_default_perturbations!(
    ::MoistureModel,
    ::AtmosModel,
    state,
    thermo,
    ha,
    vars,
)
    return nothing
end
function atmos_les_default_perturbations!(
    m::EquilMoist,
    atmos::AtmosModel,
    state,
    thermo,
    ha,
    vars,
)
    qt = state.moisture.ρq_tot / state.ρ
    vars.moisture.qt_prime = qt - ha.moisture.qt
    vars.moisture.ql_prime = thermo.moisture.q_liq - ha.moisture.ql
    vars.moisture.qv_prime = thermo.moisture.q_vap - ha.moisture.qv
    vars.moisture.thv_prime = thermo.moisture.θ_vir - ha.moisture.thv
    vars.moisture.thl_prime = thermo.moisture.θ_liq_ice - ha.moisture.thl

    return nothing
end

function atmos_les_default_perturbations_init(
    dgngrp::DiagnosticsGroup,
    currtime,
)
    FT = eltype(Settings.Q)
    atmos = Settings.dg.balance_law
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if !(dgngrp.interpol isa InterpolationBrick)
        @warn """
            Diagnostics $(dgngrp.name): requires `InterpolationBrick`!
            """
        return nothing
    end

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(dgngrp.interpol)

        # set up the variables we're going to be writing
        vars = OrderedDict()
        varnames = map(
            s -> startswith(s, "moisture.") ? s[10:end] : s,
            flattenednames(vars_atmos_les_default_perturbations(atmos, FT)),
        )
        for varname in varnames
            vars[varname] = (tuple(collect(keys(dims))...), FT, Dict())
        end

        # create the output file
        dprefix = @sprintf(
            "%s_%s_%s_rank%04d",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
            mpirank,
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        init_data(dgngrp.writer, dfilename, dims, vars)
    end

    return nothing
end

"""
    atmos_les_default_perturbations_collect(dgngrp, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function atmos_les_default_perturbations_collect(
    dgngrp::DiagnosticsGroup,
    currtime,
)
    interpol = dgngrp.interpol
    if !(interpol isa InterpolationBrick)
        @warn """
            Diagnostics $(dgngrp.name): requires `InterpolationBrick`!
            """
        return nothing
    end

    dg = Settings.dg
    atmos = dg.balance_law
    Q = Settings.Q
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get needed arrays onto the CPU
    if array_device(Q) isa CPU
        ArrayType = Array
        state_data = Q.realdata
        aux_data = dg.state_auxiliary.realdata
    else
        ArrayType = CuArray
        state_data = Array(Q.realdata)
        aux_data = Array(dg.state_auxiliary.realdata)
    end
    FT = eltype(state_data)

    # Compute thermo variables
    thermo_array = Array{FT}(undef, npoints, num_thermo(atmos, FT), nrealelem)
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state_conservative(dg, state_data, ijk, e)
        aux = extract_state_auxiliary(dg, aux_data, ijk, e)

        thermo = thermo_vars(atmos, view(thermo_array, ijk, :, e))
        compute_thermo!(atmos, state, aux, thermo)
    end

    # Interpolate the state and thermo variables.
    interpol = dgngrp.interpol
    istate =
        ArrayType{FT}(undef, interpol.Npl, number_state_conservative(atmos, FT))
    interpolate_local!(interpol, Q.realdata, istate)

    ithermo = ArrayType{FT}(undef, interpol.Npl, num_thermo(atmos, FT))
    interpolate_local!(interpol, ArrayType(thermo_array), ithermo)

    # FIXME: accumulating to rank 0 is not scalable
    all_state_data = accumulate_interpolated_data(mpicomm, interpol, istate)
    all_thermo_data = accumulate_interpolated_data(mpicomm, interpol, ithermo)

    if mpirank == 0
        # get dimensions for the interpolated grid
        dims = dimensions(interpol)

        nx = length(dims["x"][1])
        ny = length(dims["y"][1])
        nz = length(dims["z"][1])

        # collect horizontal sums
        simple_sums = [
            zeros(FT, num_atmos_les_default_simple_vars(atmos, FT))
            for _ in 1:nz
        ]
        @visitI nx ny nz begin
            statei = Vars{vars_state_conservative(atmos, FT)}(view(
                all_state_data,
                lo,
                la,
                le,
                :,
            ))
            thermoi = thermo_vars(atmos, view(all_thermo_data, lo, la, le, :))
            simple = atmos_les_default_simple_vars(atmos, simple_sums[le])
            atmos_les_default_perturbations_sums!(
                atmos,
                statei,
                thermoi,
                simple,
            )
        end

        # compute horizontal averages
        simple_avgs = [
            zeros(FT, num_atmos_les_default_simple_vars(atmos, FT))
            for _ in 1:nz
        ]
        for le in 1:nz
            simple_avgs[le] .= simple_sums[le] ./ (nx * ny)
        end

        # complete density averaging
        simple_varnames = map(
            s -> startswith(s, "moisture.") ? s[10:end] : s,
            flattenednames(vars_atmos_les_default_simple(atmos, FT)),
        )
        for vari in 1:length(simple_varnames)
            for le in 1:nz
                ha = atmos_les_default_simple_vars(atmos, simple_avgs[le])
                avg_rho = ha.avg_rho
                if simple_varnames[vari] != "avg_rho"
                    simple_avgs[le][vari] /= avg_rho
                end
            end
        end

        # now compute the perturbations from the horizontal averages
        perturbations_array = Array{FT}(
            undef,
            nx,
            ny,
            nz,
            num_atmos_les_default_perturbation_vars(atmos, FT),
        )
        @visitI nx ny nz begin
            statei = Vars{vars_state_conservative(atmos, FT)}(view(
                all_state_data,
                lo,
                la,
                le,
                :,
            ))
            thermoi = thermo_vars(atmos, view(all_thermo_data, lo, la, le, :))
            ha = atmos_les_default_simple_vars(atmos, simple_avgs[le])
            perturbations = atmos_les_default_perturbation_vars(
                atmos,
                view(perturbations_array, lo, la, le, :),
            )
            atmos_les_default_perturbations!(
                atmos,
                statei,
                thermoi,
                ha,
                perturbations,
            )
        end

        # prepare and write out the perturbations
        varvals = OrderedDict()
        varnames = map(
            s -> startswith(s, "moisture.") ? s[10:end] : s,
            flattenednames(vars_atmos_les_default_perturbations(atmos, FT)),
        )
        for (vari, varname) in enumerate(varnames)
            varvals[varname] = perturbations_array[:, :, :, vari]
        end

        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_les_default_perturbations_fini(
    dgngrp::DiagnosticsGroup,
    currtime,
) end
