# RHS evaluation sequence

Stepping a balance law in time involves calling a "right hand side" or "tendencies" function. This "tendencies" function calls several other functions[^1] which, collectively, account for the sum of all tendencies corresponding to an instant in time.

## Flow chart

Below is a flow chart to provide a more detailed picture of the order these functions are called:


```@example tendencies_flow_chart
using TikzGraphs
using LightGraphs
using TikzPictures # this is required for saving

flow_chart = [
  "update\\_aux!",
  "flux\\_diffusive!",
  "flux\\_nondiffusive!",
  "source!",
]
n_nodes = length(flow_chart)
g = DiGraph(n_nodes)
for i in 1:n_nodes-1
  add_edge!(g, i, i+1)
end

t = TikzGraphs.plot(g, flow_chart, node_style="draw, rounded corners, fill=blue!10", options="scale=2, font=\\huge\\sf")
TikzPictures.save(SVG(joinpath(@__DIR__,"tendencies_flow_chart")), t)

nothing # hide
```
![](tendencies_flow_chart.svg)


[^1] [How to build a balance law]()