#### Utilities

export @unpack

macro unpack(stuff, syms...)
  thunk = Expr(:block)
  for sym in syms
    push!(thunk.args, :($(esc(sym)) = $(esc(stuff))[$(QuoteNode(sym))]))
  end
  push!(thunk.args, nothing)
  return thunk
end
