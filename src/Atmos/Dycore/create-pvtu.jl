#V1.0 of tool to allow VTU stitching
#FIXME: Is there a way to directly do this through the WriteVTK module 

#USAGE: 
# To use this file place it in the same folder as your .vtu output files.
# Manually define the number of time steps and number of partitions available
# below. Then $> julia create-pvtu.jl from terminal will return a series of time
# stamped pvtu files which you can load into visit or paraview. Note that the pvtu
# files are simply "pointers" to the main field data contained in the .vtu files

#User defined step and core count
#Assuming starting timestep = 0
nsteps = 220
ncores = 16

# Note: The initial condition is step0000
# Note: The first core is mpirank0000

## For intermediate timestamped data
# Here we assume all timesteps use the same output format
# So the header data and offset values are the same as
# output by the WriteVTK module

samplefile = "RTB_2D_step0000"*"_mpirank0000.vtu"

for step = 1:nsteps
  stepid = lpad(step - 1,4,'0')
  open(samplefile) do f
    filename = open("RTB_2D_step$stepid.pvtu","w")
    line = 1 
    while line < 24 #TODO: fix this so its not hardcoded but derives value from XML file
        x = readline(f)
        if (occursin("Piece", x)==true)
          x = ""
          line += 1
        else
          x=replace(x, "Points>" => "PPoints>")
          x=replace(x, "Cells>" => "PCells>")
          x=replace(x, "PointData>" => "PPointData>")
          x=replace(x, "UnstructuredGrid" => "PUnstructuredGrid")
          write(filename,x*"\n")
          line += 1
        end
    end
    for rank=1:ncores
      rankid = lpad(rank - 1,4,'0')
      write(filename,"<Piece Source=\"RTB_2D_step"*"$stepid"*"_mpirank$rankid.vtu\"/>\n")
    end
      write(filename, "</PUnstructuredGrid>\n")
      write(filename, "</VTKFile>")
  close(filename)
end
end
