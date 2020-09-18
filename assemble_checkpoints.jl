# assemble_checkpoints.jl
#
# Script to assemble ClimateMachine checkpoint files from multiple ranks
# into a single checkpoint file that can be restarted from by one rank.
#
# This is a temporary hack for until ClimateMachine unifies checkpoint
# files across multiple ranks itself.

using JLD2
using Printf

function get_checkpoint_filename(checkpoint_dir, exp_name, rank, num)
    cname = @sprintf(
        "%s_checkpoint_mpirank%04d_num%04d.jld2",
        exp_name,
        rank,
        num,
    )
    return joinpath(checkpoint_dir, cname)
end


function assemble(checkpoint_dir, exp_name, nranks)
    fnames = filter(x -> occursin(exp_name, x), readdir( checkpoint_dir ) );
    file = checkpoint_dir*'/'*fnames[1]
    @load file h_Q h_aux t
    println("Extracted checkpoint data for rank 0")
    fullQ = h_Q
    fullaux = h_aux
    fullt = t
    for r in 1:(nranks - 1)
        file = checkpoint_dir*'/'*fnames[r + 1]
        @load file h_Q h_aux t
        println("Extracted checkpoint data for rank $r from $file")
        fullQ = cat(fullQ, h_Q; dims = 3)
        fullaux = cat(fullaux, h_aux; dims = 3)
    end
    h_Q = fullQ
    h_aux = fullaux
    t = fullt
    out_dir = checkpoint_dir
    mkpath(out_dir)
    file = get_checkpoint_filename( checkpoint_dir, exp_name, 0, 9999)
    print("Writing to $file... ")
    @save file h_Q h_aux t
    println("done.")
end

if length(ARGS) != 4
    println(
        """
        Usage:
            assemble_checkpoints.jl <checkpoint_dir> <exp_name> <num_ranks> <del_ranks>""",
    )
    exit()
end

####
# this deletes the 9999 restart and any previous restarts, ARGS[4] is the rank no whose files to delete
fnames = filter(x -> occursin( @sprintf("num%04d",9999), x), readdir( ARGS[1] )); 
fname = ARGS[1]*'/'*fnames[1]
rm(fname)

del_rankno = parse(Int, ARGS[4])
if del_rankno > -0.1 
  fnames = filter(x -> occursin( @sprintf("num%04d",del_rankno), x), readdir( ARGS[1] ) );
  nranks = parse(Int, ARGS[3])
  for r in 1:(nranks)
    fname = ARGS[1]*'/'*fnames[r]
    rm(fname)
  end
end
####

assemble(ARGS[1], ARGS[2], parse(Int, ARGS[3]))
