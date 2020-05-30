"""
Contains stubs for advective, diffusive, and nondiffusive courant number
calculations to be used in [`DGMethods.courant`](@ref). Models should provide
concrete implementations if they wish to use these functions.
"""
module Courant

export advective_courant, diffusive_courant, nondiffusive_courant

function advective_courant end

function nondiffusive_courant end

function diffusive_courant end

function viscous_courant end

end
