"""
    StateCheck

Module with a minimal set of functions for getting statistics
and basic I/O from ClimateMachine DG state arrays (`MPIStateArray` type).
Created for regression testing and code change tracking and debugging.
StateCheck functions iterate over named variables in an `MPIStateArray`,
calculate and report their statistics and/or write values for all or
some subset of points at a fixed frequency.

# Functions

 - [`sccreate`](@ref) Create a StateCheck call back variable.
 - [`scdocheck`](@ref) Check StateCheck variable values against reference values.
 - [`scprintref`](@ref) Print StateCheck variable in format for creating reference values.
"""
module StateCheck

# Imports from standard Julia packages
using Formatting
using MPI
using Printf
using StaticArrays
using Statistics

# Imports from ClimateMachine core
import ..GenericCallbacks: EveryXSimulationSteps
import ..MPIStateArrays: MPIStateArray
import ..VariableTemplates: flattenednames

####
# For testing put a new function signature here!
# Needs to go in src/Utilities/VariableTemplates/var_names.jl
# This handles SMatrix case
flattenednames(::Type{T}; prefix = "") where {T <: SArray} =
    ntuple(i -> "$prefix[$i]", length(T))
####

"""
    VStat

Type for returning variable statistics.
"""
struct VStat
    max
    min
    mean
    std
end

# Global functions to expose
export sccreate # Create a state checker call back
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
)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(io, "# SC Start: creating state check callback")
    end

    ####
    # Print fields that the call back created by this call will query
    ####
    nr = 0
    for f in fields
        print_head = true
        Q = f[1]
        lab = f[2]
        V = typeof(Q).parameters[2]
        slist = typeof(Q).parameters[2].names
        l_s = length(slist)
        nss = 0
        if l_s == 0
            println(
                io,
                "# SC  MPIStateArray labeled \"$lab\" has no named symbols.",
            )
        else
            ns = 0
            for s in slist
                ns = ns + 1
                if MPI.Comm_rank(MPI.COMM_WORLD) == 0
                    if print_head
                        println(
                            io,
                            "# SC Creating state check callback labeled \"$lab\" for symbols",
                        )
                    end
                    print_head = false
                    println(io, "# SC ", s)
                end
                for n in
                    flattenednames(fieldtype(V, ns), prefix = fieldname(V, ns))
                    nss = nss + 1
                end
            end
        end
        nr = nr + nss
    end

    ###
    # Initialize total calls counter for the call back
    ###
    n_cb_calls = 0

    ###
    # Create holder for most recent stats
    ###
    cur_stats_dict = Dict()
    cur_stats_flat = Array{Any}(undef, nr)

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
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println(
                io,
                "# SC +++++++++++ClimateMachine StateCheck call-back start+++++++++++++++++",
            )
            println(
                io,
                "# SC  Step  |   Label    |  Field   |                            Stats                       ",
            )
        end
        h_var_fmt = "%" * sprintf1("%d", nprec + 8) * "s"
        min_str = sprintf1(h_var_fmt, " min() ")
        max_str = sprintf1(h_var_fmt, " max() ")
        ave_str = sprintf1(h_var_fmt, " mean() ")
        std_str = sprintf1(h_var_fmt, " std() ")
        fmt_str = [" min() ", " max() ", " mean() ", " std() "]
        fmt_str = sprintf1.(Ref(h_var_fmt), fmt_str)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            print(io, "# SC -------|------------|----------|")
            println(io, join(fmt_str, "|"), "|")
        end

        ## Iterate over the set of MPIStateArrays for this callback
        nr = 0
        for f in fields
            olabel = f[2]
            ol_str = @sprintf("%12.12s", olabel)
            m_array = f[1]

            # Get descriptor for MPIStateArray

            V = typeof(m_array).parameters[2]

            ## Iterate over fields in each MPIStateArray
            #  (use ivar to index individual arrays within the MPIStateArray)
            ivar = 0
            stats_val_dict = Dict()
            for i in 1:length(V.names)
                for n in flattenednames(fieldtype(V, i), prefix = fieldname(V, i))
                    nr = nr + 1
                    ivar = ivar + 1
                    n_str = @sprintf("%9.9s", n)
                    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
                        print(io, "# SC ", ns_str, "|", ol_str, "|", n_str, " |")
                    end
                    stats_string = scstats(m_array, ivar, nprec)
                    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
                        println(io, join(stats_string[1:4], "|"), "|")
                    end
                    stats_val_dict[n] = stats_string[5]
                    cur_stats_flat[nr] = [
                        olabel,
                        n,
                        stats_string[5].max,
                        stats_string[5].min,
                        stats_string[5].mean,
                        stats_string[5].std,
                    ]
                end
            end
            cur_stats_dict[olabel] = stats_val_dict
        end
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println(
                io,
                "# SC +++++++++++ClimateMachine StateCheck call-back end+++++++++++++++++++",
            )
        end
    end

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(io, "# SC Finish: creating state check callback")
    end

    return cb
end

function scstats(V, ivar, nprec)

    # Get number of MPI procs
    nproc = MPI.Comm_size(V.mpicomm)

    npr = nprec
    fmt = @sprintf("%%%d.%de", npr + 8, npr)

    getByField(V, ivar) = (V.realdata[:, ivar, :])
    Vmcomm = V.mpicomm

    # Get local and global field sizes (degrees of freedom).
    n_size_loc = length(getByField(V, ivar))
    n_size_tot = MPI.Allreduce(n_size_loc, +, Vmcomm)

    # Min
    phi_loc = minimum(getByField(V, ivar))
    phi_min = MPI.Allreduce(phi_loc, min, Vmcomm)
    phi = phi_min
    # minVstr=@sprintf("%23.15e",phi)
    min_v_str = sprintf1(fmt, phi)

    # Max
    phi_loc = maximum(getByField(V, ivar))
    phi_max = MPI.Allreduce(phi_loc, max, Vmcomm)
    phi = phi_max
    # maxVstr=@sprintf("%23.15e",phi)
    max_v_str = sprintf1(fmt, phi)

    # Ave
    phi_loc = mean(getByField(V, ivar))
    phi_loc = phi_loc * n_size_loc / n_size_tot
    phi_sum = MPI.Allreduce(phi_loc, +, Vmcomm)
    phi_mean = phi_sum
    phi = phi_mean
    # aveVstr=@sprintf("%23.15e",phi)
    ave_v_str = sprintf1(fmt, phi)

    # Std
    phi_loc = (getByField(V, ivar) .- phi_mean) .^ 2
    phi_loc = sum(phi_loc)  # Sum local data explicitly since GPU Allreduce
    # does not take arrays yet.
    phi_sum = MPI.Allreduce(phi_loc, +, Vmcomm)
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
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        # Get print format lengths for cols 1 and 2 so they are aligned
        # for readability.
        phi = sc.cur_stats_flat
        f = 1
        a1l = maximum(length.(map(
            i -> (phi[i])[f],
            range(1, length = length(phi)),
        )))
        f = 2
        a2l = maximum(length.(String.((map(
            i -> (phi[i])[f],
            range(1, length = length(phi)),
        )))))
        fmt1 = @sprintf("%%%d.%ds", a1l, a1l) # Column 1
        fmt2 = @sprintf("%%%d.%ds", a2l, a2l) # Column 2
        fmt3 = @sprintf("%%28.20e")         # All numbers at full precision
        # Create an string of spaces to be used for formatting
        sp = "                                                                           "

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
        for lv in sc.cur_stats_flat
            s1 = lv[1]
            l1 = length(s1)
            s1 = sp[1:(a1l - l1)] * "\"" * s1 * "\""
            s2 = lv[2]
            if typeof(s2) == String
                l2 = length(s2)
                s2 = sp[1:(a2l - l2)] * "\"" * s2 * "\""
                s22 = ""
            end
            if typeof(s2) == Symbol
                s22 = s2
                l2 = length(String(s2))
                s2 = sp[1:(a2l - l2 + 1)] * ":"
            end
            s3 = sprintf1(fmt3, lv[3])
            s4 = sprintf1(fmt3, lv[4])
            s5 = sprintf1(fmt3, lv[5])
            s6 = sprintf1(fmt3, lv[6])
            println(
                io,
                " [ ",
                s1,
                ", ",
                s2,
                s22,
                ",",
                s3,
                ",",
                s4,
                ",",
                s5,
                ",",
                s6,
                " ],",
            )
        end
        println(io, "]")

        # Write table of reference match precisions using default precision that
        # can be hand updated.
        println(io, "parr = [")
        for lv in sc.cur_stats_flat
            s1 = lv[1]
            l1 = length(s1)
            s1 = sp[1:(a1l - l1)] * "\"" * s1 * "\""
            s2 = lv[2]
            if typeof(s2) == String
                l2 = length(s2)
                s2 = sp[1:(a2l - l2)] * "\"" * s2 * "\""
                s22 = ""
            end
            if typeof(s2) == Symbol
                s22 = s2
                l2 = length(String(s2))
                s2 = sp[1:(a2l - l2 + 1)] * ":"
            end
            s3 = sprintf1(fmt3, lv[3])
            s4 = sprintf1(fmt3, lv[4])
            s5 = sprintf1(fmt3, lv[5])
            s6 = sprintf1(fmt3, lv[6])
            println(
                io,
                " [ ",
                s1,
                ", ",
                s2,
                s22,
                ",",
                "    16,    16,    16,    16 ],",
            )
        end
        println(io, "]")
        println(io, "# END SCPRINT")
    end
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
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 || return true
    println(
        io,
        "# SC +++++++++++ClimateMachine StateCheck ref val check start+++++++++++++++++",
    )
    println(
        io,
        "# SC \"N( )\" bracketing indicates field failed to match      ",
    )
    println(io, "# SC \"P=\"  row pass count      ")
    println(io, "# SC \"F=\"  row pass count      ")
    println(io, "# SC \"NA=\" row not checked count      ")
    println(io, "# SC ")
    println(
        io,
        "# SC        Label         Field      min()      max()     mean()      std() ",
    )
    irow = 1
    i_val = 1
    i_prec = 2
    all_pass = true

    for row in sc.cur_stats_flat
        ## Debugging
        # println(row)
        # println(ref_dat[i_val][irow])
        # println(ref_dat[i_prec][irow])
        row_pass = true
        row_col_pass = 0
        row_col_na = 0
        row_col_fail = 0

        ## Make array copy for reporting
        res_dat = copy(ref_dat[i_prec][irow])

        ## Check MPIStateArrayName
        cval = row[1]
        rval = ref_dat[i_val][irow][1]
        if cval != rval
            all_pass = false
            row_pass = false
            row_col_fail += 1
            res_dat[1] = "N" * "(" * rval * ")"
        else
            res_dat[1] = cval
            row_col_pass += 1
            res_dat[1] = rval
        end

        ## Check term name
        cval = row[2]
        rval = ref_dat[i_val][irow][2]
        if cval != rval
            all_pass = false
            row_pass = false
            if typeof(rval) == String
                res_dat[2] = "N" * "(" * rval * ")"
            else
                res_dat[2] = "N" * "(" * string(rval) * ")"
            end
            row_col_fail += 1
        else
            res_dat[2] = cval
            row_col_pass += 1
        end

        # Check numeric values
        nv = 3
        for nv in [3, 4, 5, 6]
            fmt = @sprintf("%%28.20e")
            lfld = 28
            ndig = 20
            cval = row[nv]
            cvalc = sprintf1(fmt, cval)
            rval = ref_dat[i_val][irow][nv]
            rvalc = sprintf1(fmt, rval)
            pcmp = ref_dat[i_prec][irow][nv]

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
                        imatch = imatch + 1
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
                    all_pass = false
                    row_pass = false
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


        #
        # println(resDat)
        @printf(
            io,
            "# SC %12.12s, %12.12s, %9.9s, %9.9s, %9.9s, %9.9s",
            res_dat[1],
            res_dat[2],
            res_dat[3],
            res_dat[4],
            res_dat[5],
            res_dat[6]
        )
        @printf(
            io,
            " :: P=%d, F=%d, NA=%d\n",
            row_col_pass,
            row_col_fail,
            row_col_na
        )
        # Next row
        irow = irow + 1
    end
    println(
        io,
        "# SC +++++++++++ClimateMachine StateCheck ref val check end+++++++++++++++++",
    )
    return all_pass

end

end # module
