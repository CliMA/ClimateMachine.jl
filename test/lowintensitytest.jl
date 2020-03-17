ENV["CLIMA_GPU"] = "false"
ENV["intensity"] = "low"

println(" ******************************************************* ")
println(" ******************************************************* ")
println(" ******************************************************* ")
println(" ********* TESTING CLIMA IN LOW INTENSITY MODE ********* ")
println(" ******************************************************* ")
println(" ******************************************************* ")
println(" ******************************************************* ")
println("")
@warn " Not all of the code is tested, nor is it
  guaranteed to pass in normal intensity if low intensity mode passes "
println("")

using Pkg
Pkg.test()
