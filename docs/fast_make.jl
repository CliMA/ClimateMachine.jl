Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625 
using ClimateMachine, Documenter, Literate                                      
ENV["GKSwstype"] = "100" # https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
generated_dir = joinpath(@__DIR__, "src", "generated") # generated files directory
rm(generated_dir, force = true, recursive = true)                               
mkpath(generated_dir)                                                           
pages = Any[  
    "Microph" => "Microphysics.md",                                             
]                                                                               
mathengine = MathJax(Dict(                                                      
    :TeX => Dict(                                                               
        :equationNumbers => Dict(:autoNumber => "AMS"),                         
        :Macros => Dict(),                                                      
    ),                                                                          
))                                                                              
format = Documenter.HTML(                                                       
    prettyurls = get(ENV, "CI", "") == "true",                                  
    mathengine = mathengine,                                                    
    collapselevel = 1,                                                          
    # prettyurls = !("local" in ARGS),                                          
    # canonical = "https://CliMA.github.io/ClimateMachine.jl/stable/",          
)                                                                               
makedocs(                                                                       
    sitename = "ClimateMachine",                                                
    doctest = false,                                                            
    strict = false,                                                             
    source="src/tmp/",                                                          
    build="build/generated/Theory/Atmos_fast",                                  
    linkcheck = false,                                                          
    format = format,                                                            
    checkdocs = :none,                                                          
    # checkdocs = :exports,                                                     
    # checkdocs = :all,                                                         
    clean = true,                                                               
    modules = [ClimateMachine],                                                 
    pages = pages,                                                              
)                                                                               
include("clean_build_folder.jl")                                                
deploydocs(                                                                     
    repo = "github.com/CliMA/ClimateMachine.jl.git",                            
    target = "build",                                                           
    push_preview = true,                                                        
)