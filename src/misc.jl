using Pkg

export haspkg

"""
    haspkg(pkgname::String)::Bool

Determines if the package `pkgname` available in the current environment.
"""
haspkg(pkgname::String) = haskey(Pkg.installed(), pkgname)
  
