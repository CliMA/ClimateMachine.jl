"""
    StateCheck

Module with a minimal set of functions for getting statistics
and basic I/O from ClimateMachine DG state arrays (`MPIStateArray` type).
Created for regression testing and code change tracking and debugging.
StateCheck functions iterate over named variables in an `MPIStateArray`,
calculate and report their statistics and/or write values for all or
some subset of points at a fixed frequency.

# Functions

 - [`sccreate`](@ref) Create a StateCheck callback variable.
 - [`scdocheck`](@ref) Check StateCheck variable values against reference values.
 - [`scprintref`](@ref) Print StateCheck variable in format for creating reference values.
"""
module StateCheck

# Imports from standard Julia packages
using Formatting
using MPI
using ..MPIStateArrays: vars
using PrettyTables
using Printf
using StaticArrays
using Statistics
using ..VariableTemplates:
    flattenednames, flattened_tup_chain, RetainArr, varsindex, varsize

# Imports from ClimateMachine core
import ..GenericCallbacks: EveryXSimulationSteps
import ..MPIStateArrays: MPIStateArray

"""
    VStat

Type for returning variable statistics.
"""
struct VStat
    max::Any
    min::Any
    mean::Any
    std::Any
end

# Global functions to expose
export sccreate # Create a state checker callback
export scdocheck
export scprintref

const nt_freq_def = 10 # default frequency (in time steps) for output.
const prec_def = 15    # default precision used for formatted output table

# TODO: this should use the new callback interface

"""
    sccreate(
        io::IO,
        fields::Array{<:Tuple{<:MPIStateArray, String}},
        nt_freq::Int = $nt_freq_def;
        prec = $prec_def
    )

Create a "state check" call-back for one or more `MPIStateArray` variables
that will report basic statistics for the fields in the array.

  -  `io` an IO stream to use for printed output
  -  `fields` a required first argument that is an array of one or more
                        `MPIStateArray` variable and label string pair tuples.
                        State array statistics will be reported for the named symbols
                        in each `MPIStateArray` labeled with the label string.
  -  `nt_freq` an optional second argument with default value of
                        $nt_freq_def that sets how frequently (in time-step counts) the
                        statistics are reported.
  -  `prec` a named argument that sets number of decimal places to print for
                        statistics, defaults to $prec_def.

# Examples
```julia
using ClimateMachine.VariableTemplates
using StaticArrays
using ClimateMachine.MPIStateArrays
using MPI
MPI.Init()
FT=Float64
F1=@vars begin; ν∇u::SMatrix{3, 2, FT, 6}; κ∇θ::SVector{3, FT}; end
F2=@vars begin; u::SVector{2, FT}; θ::SVector{1, FT}; end
Q1=MPIStateArray{Float32,F1}(MPI.COMM_WORLD,ClimateMachine.array_type(),4,9,8);
Q2=MPIStateArray{Float64,F2}(MPI.COMM_WORLD,ClimateMachine.array_type(),4,6,8);
cb=ClimateMachine.StateCheck.sccreate([(Q1,"My gradients"),(Q2,"My fields")],1; prec=$prec_def);
```
"""
function sccreate(
    io::IO,
    fields::Array{<:Tuple{<:MPIStateArray, String}},
    nt_freq::Int = nt_freq_def;
    prec = prec_def,
    print_head = true,
)
    mpicomm = first(first(fields)).mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    mpirank == 0 && println(io, "# SC Start: creating state check callback")

    ####
    # Print fields that the callback created by this call will query
    ####
    for (Q, lab) in fields
        V = vars(Q)
        ftc = flattened_tup_chain(V, RetainArr())
        nss = length(flattenednames(V))
        if length(flattenednames(V)) == 0 && mpirank == 0
            println(
                io,
                "# SC  MPIStateArray labeled \"$lab\" has no named symbols.",
            )
        elseif mpirank == 0
            println(
                io,
                "# SC Creating state check callback labeled \"$lab\" for symbols",
            )
            for s in join.(ftc, "")
                println(io, "# SC ", s)
            end
        end
    end

    ###
    # Initialize total calls counter for the callback
    ###
    n_cb_calls = 0

    ###
    # Create holder for most recent stats
    ###
    nvars = sum([length(flattenednames(vars(Q))) for (Q, lab) in fields])
    cur_stats_flat = Array{Any}(undef, nvars)

    # Save io pointer
    iosave = io

    ######
    # Create the callback
    ######
    cb = EveryXSimulationSteps(nt_freq) do
        # Track which timestep this is
        n_cb_calls = n_cb_calls + 1
        n_step = (n_cb_calls - 1) * nt_freq + 1
        ns_str = @sprintf("%7.7d", n_step - 1)
        io = iosave
        # Obscure trick to do with running in cells in notebook that close Base.stdout
        # between different stages. This affects parsing Literate/Documenter examples.
        if !isopen(io)
            io = Base.stdout
        end

        ## Print header
        nprec = min(max(1, prec), 20)
        if mpirank == 0
            println(
                io,
                "# SC +++++++++++ClimateMachine StateCheck call-back start+++++++++++++++++",
            )
            println(io, "# SC Step = $ns_str")
        end

        labs = vcat([
            map(x -> lab, flattenednames(vars(Q))) for (Q, lab) in fields
        ]...)

        varnames = vcat([
            map(x -> x, flattenednames(vars(Q))) for (Q, lab) in fields
        ]...)

        header = ["Label" "Field" "min()" "max()" "mean()" "std()"]

        stats_all = vcat([
            [scstats(Q, i, nprec)[5] for i in 1:varsize(vars(Q))] for (Q, lab) in fields
        ]...)

        cur_stats_flat .= vcat([
            [
                begin
                    scstats_i = scstats(Q, i, nprec)[5]
                    [
                        lab,
                        fn,
                        scstats_i.max,
                        scstats_i.min,
                        scstats_i.mean,
                        scstats_i.std,
                    ]
                end for (i, fn) in enumerate(flattenednames(vars(Q)))
            ] for (Q, lab) in fields
        ]...)

        # TODO: Verify, not sure why we are swapping these columns:
        # data_min = map(x->x.min, stats_all)
        # data_max = map(x->x.max, stats_all)

        data_max = map(x -> x.min, stats_all)
        data_min = map(x -> x.max, stats_all)

        data_mean = map(x -> x.mean, stats_all)
        data_std = map(x -> x.std, stats_all)
        data = hcat(labs, varnames, data_min, data_max, data_mean, data_std)
        pretty_table(
            io,
            data,
            header;
            formatters = ft_printf("%.16e", 3:6),
            header_crayon = crayon"yellow bold",
            subheader_crayon = crayon"green bold",
            crop = :none,
        )

        if mpirank == 0
            println(
                io,
                "# SC +++++++++++ClimateMachine StateCheck call-back end+++++++++++++++++++",
            )
        end
    end

    if mpirank == 0
        println(io, "# SC Finish: creating state check callback")
    end

    return cb
end

function scstats(Q, ivar, nprec)

    # Get number of MPI procs
    mpicomm = Q.mpicomm
    nproc = MPI.Comm_size(mpicomm)

    npr = nprec
    fmt = @sprintf("%%%d.%de", npr + 8, npr)

    getByField(Q, ivar) = (Q.realdata[:, ivar, :])

    # Get local and global field sizes (degrees of freedom).
    n_size_loc = length(getByField(Q, ivar))
    n_size_tot = MPI.Allreduce(n_size_loc, +, mpicomm)

    # Min
    phi_loc = minimum(getByField(Q, ivar))
    phi_min = MPI.Allreduce(phi_loc, min, mpicomm)
    phi = phi_min
    # minVstr=@sprintf("%23.15e",phi)
    min_v_str = sprintf1(fmt, phi)

    # Max
    phi_loc = maximum(getByField(Q, ivar))
    phi_max = MPI.Allreduce(phi_loc, max, mpicomm)
    phi = phi_max
    # maxVstr=@sprintf("%23.15e",phi)
    max_v_str = sprintf1(fmt, phi)

    # Ave
    phi_loc = mean(getByField(Q, ivar))
    phi_loc = phi_loc * n_size_loc / n_size_tot
    phi_sum = MPI.Allreduce(phi_loc, +, mpicomm)
    phi_mean = phi_sum
    phi = phi_mean
    # aveVstr=@sprintf("%23.15e",phi)
    ave_v_str = sprintf1(fmt, phi)

    # Std
    phi_loc = (getByField(Q, ivar) .- phi_mean) .^ 2
    phi_loc = sum(phi_loc)  # Sum local data explicitly since GPU Allreduce
    # does not take arrays yet.
    phi_sum = MPI.Allreduce(phi_loc, +, mpicomm)
    n_val_sum = n_size_tot
    phi_std = sqrt(sum(phi_sum) / (n_val_sum - 1))
    phi = phi_std
    # stdVstr=@sprintf("%23.15e",phi)
    std_v_str = sprintf1(fmt, phi)

    vals = VStat(phi_min, phi_max, phi_mean, phi_std)

    return min_v_str, max_v_str, ave_v_str, std_v_str, vals
end

function sccreate(
    fields::Array{<:Tuple{<:MPIStateArray, String}},
    nt_freq::Int = nt_freq_def;
    prec = prec_def,
)
    return sccreate(Base.stdout, fields, nt_freq; prec = prec)
end


"""
    scprintref(cb)

Print out a "state check" call-back table of values in a format
suitable for use as a set of reference numbers for CI comparison.

 - `cb` callback variable of type ClimateMachine.GenericCallbacks.Every*
"""
function scprintref(cb)
    sc = cb.callback
    io = sc.iosave
    # Obscure trick to do with running in cells in notebook
    if !isopen(io)
        io = Base.stdout
    end
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpirank == 0 || return
    # Get print format lengths for cols 1 and 2 so they are aligned
    # for readability.
    phi = sc.cur_stats_flat
    a1l = maximum(length.(map(x -> x[1], phi)))
    a2l = maximum(length.(String.(map(x -> x[2], phi))))
    fmt1 = @sprintf("%%%d.%ds", a1l, a1l) # Column 1
    fmt2 = @sprintf("%%%d.%ds", a2l, a2l) # Column 2
    fmt3 = @sprintf("%%28.20e")         # All numbers at full precision
    # Create an string of spaces to be used for formatting
    sp = repeat(" ", 75)

    # Write header
    println(io, "# BEGIN SCPRINT")
    println(io, "# varr - reference values (from reference run)    ")
    println(io, "# parr - digits match precision (hand edit as needed) ")
    println(io, "#")
    println(io, "# [")
    println(
        io,
        "#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],",
    )
    println(
        io,
        "#  [         :                :          :        :      :          :           ],",
    )
    println(io, "# ]")
    #
    # Write tables
    #  Reference value and precision match tables are separate since it is more
    #  common to update reference values occasionally while precision values are
    #  typically changed rarely and the precision values are hand edited from experience.
    #
    # Write table of reference values
    println(io, "varr = [")
    for (s1, s2, s3′, s4′, s5′, s6′) in sc.cur_stats_flat
        s1 = sp[1:(a1l - length(s1))] * "\"" * s1 * "\""
        s2 = sp[1:(a2l - length(s2))] * "\"" * s2 * "\""
        s3 = sprintf1(fmt3, s3′)
        s4 = sprintf1(fmt3, s4′)
        s5 = sprintf1(fmt3, s5′)
        s6 = sprintf1(fmt3, s6′)
        srow = (s3, s4, s5, s6)
        println(io, " [ ", s1, ", ", s2, join(srow, ","), " ],")
    end
    println(io, "]")

    # Write table of reference match precisions using default precision that
    # can be hand updated.
    println(io, "parr = [")
    for (s1, s2, s3′, s4′, s5′, s6′) in sc.cur_stats_flat
        s1 = sp[1:(a1l - length(s1))] * "\"" * s1 * "\""
        s2 = sp[1:(a2l - length(s2))] * "\"" * s2 * "\""
        println(io, " [ ", s1, ", ", s2, ",", "    16,    16,    16,    16 ],")
    end
    println(io, "]")
    println(io, "# END SCPRINT")
end

# TODO: write unit test for process_stat
function process_stat(row)
    ## Debugging
    # println(row)
    # println(ref_dat_val)
    # println(ref_dat_prec)
    row_col_pass = 0
    row_col_na = 0
    row_col_fail = 0

    ## Make array copy for reporting

    ## Check MPIStateArrayName
    cval = row.lab.cur
    rval = row.lab.ref_val
    if cval != rval
        row_col_fail += 1
        lab = "N" * "(" * rval * ")"
    else
        lab = cval
        row_col_pass += 1
        lab = rval
    end

    ## Check term name
    cval = row.varname.cur
    rval = row.varname.ref_val
    if cval != rval
        varname = "N" * "(" * rval * ")"
        row_col_fail += 1
    else
        varname = cval
        row_col_pass += 1
    end

    res_dat = Dict()
    # Check numeric values
    for nv in (:min, :max, :mean, :std)
        fmt = @sprintf("%%28.20e")
        lfld = 28
        cval = getproperty(row, nv).cur
        cvalc = sprintf1(fmt, cval)
        rval = getproperty(row, nv).ref_val
        rvalc = sprintf1(fmt, rval)
        pcmp = getproperty(row, nv).ref_prec

        # Skip if compare digits set to 0
        if pcmp > 0

            # Check exponent part
            ep1 = cvalc[(lfld - 3):lfld]
            ep2 = rvalc[(lfld - 3):lfld]
            if ep1 != ep2
                nmatch = 0
            else
                # Now check individual digits left to right
                dp1 = cvalc[2:(3 + pcmp + 1)]
                dp2 = rvalc[2:(3 + pcmp + 1)]
                nmatch = 0
                imatch = 1
                for c in dp1
                    if c == dp2[imatch]
                        nmatch = imatch
                    else
                        break
                    end
                    imatch += 1
                end
            end
            # Check for trailing exact 0s number change numerically
            if nmatch < pcmp
                if rval != 0
                    e_diffl10 = round(log10(abs((rval - cval) / rval)))
                else
                    e_diffl10 = round(log10(abs(rval - cval)))
                end
                if e_diffl10 < -pcmp
                    nmatch = Int(-e_diffl10)
                end
            end
            if nmatch < pcmp
                res_dat[nv] = "N(" * string(nmatch) * ")"
                row_col_fail += 1
            else
                res_dat[nv] = string(nmatch)
                row_col_pass += 1
            end
        else
            res_dat[nv] = "0"
            row_col_na += 1
        end
    end
    na_count = row_col_na
    fail_count = row_col_fail
    pass_count = row_col_pass
    return (; lab, varname, res_dat..., pass_count, fail_count, na_count)
end

"""
    scdocheck(cb, ref_dat)

Compare a current State check call-back set of values with a
reference set and match precision table pair.

 - `cb` `StateCheck` call-back variables
 - `ref_dat` an array of reference values and precision to match tables.
"""
function scdocheck(cb, ref_dat)
    sc = cb.callback
    io = sc.iosave
    # Obscure trick to do with running in cells in notebook
    if !isopen(io)
        io = Base.stdout
    end
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpirank == 0 || return true
    println(
        io,
        "# SC +++++++++++ClimateMachine StateCheck ref val check start+++++++++++++++++",
    )
    println(
        io,
        "# SC \"N( )\" bracketing indicates field failed to match      ",
    )

    get_row(cur, ref_prec, ref_val, i) =
        (; cur = cur[i], ref_prec = ref_prec[i], ref_val = ref_val[i])

    Z = (sc.cur_stats_flat, ref_dat...)
    stats_all = [
        begin
            (;
                lab = get_row(row, row_ref_prec, row_ref_val, 1),
                varname = get_row(row, row_ref_prec, row_ref_val, 2),
                min = get_row(row, row_ref_prec, row_ref_val, 3),
                max = get_row(row, row_ref_prec, row_ref_val, 4),
                mean = get_row(row, row_ref_prec, row_ref_val, 5),
                std = get_row(row, row_ref_prec, row_ref_val, 6),
            )
        end for (row, row_ref_val, row_ref_prec) in zip(Z...)
    ]

    data_all = map(row -> process_stat(row), stats_all)

    labs = map(stats -> stats.lab, data_all)
    varnames = map(stats -> stats.varname, data_all)
    data_min = map(stats -> stats.min, data_all)
    data_max = map(stats -> stats.max, data_all)
    data_mean = map(stats -> stats.mean, data_all)
    data_std = map(stats -> stats.std, data_all)
    pass_count = map(stats -> stats.pass_count, data_all)
    fail_count = map(stats -> stats.fail_count, data_all)
    na_count = map(stats -> stats.na_count, data_all)
    all_pass = sum(fail_count) == 0

    data = hcat(
        labs,
        varnames,
        data_min,
        data_max,
        data_mean,
        data_std,
        pass_count,
        fail_count,
        na_count,
    )
    header = [
        "Label" "Field" "min()" "max()" "mean()" "std()" "Pass" "Fail" "Not checked"
        "" "" "" "" "" "" "count" "count" "count"
    ]

    pretty_table(
        io,
        data,
        header;
        header_crayon = crayon"yellow bold",
        subheader_crayon = crayon"green bold",
        crop = :none,
    )

    println(
        io,
        "# SC +++++++++++ClimateMachine StateCheck ref val check end+++++++++++++++++",
    )
    return all_pass

end

end # module
