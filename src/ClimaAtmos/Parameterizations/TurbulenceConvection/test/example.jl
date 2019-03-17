using Test

function main()
  if Sys.isunix()
    path_separator = "/"
  elseif Sys.iswindows()
    path_separator = "\\"
  else
    error("path primitives for this OS need to be defined")
  end
  println("Testing ",split(@__FILE__, path_separator)[end],"...")

  a = 1
  b = 1

  @testset begin
    @test a==b
    @test a==b
    @test a==b
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
