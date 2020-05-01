# CLIMA

```@meta
CurrentModule = CLIMA
```

`CLIMA` is a software package that models the evolution of the Earth system over weeks to centuries.
`CLIMA` solves three-dimensional partial differential equations for the distributions of water, momentum, energy,
and tracers such as carbon in the atmosphere, oceans, and on land.

`CLIMA` will harness a wide range of Earth observations and data generated computationally to predict the evolution of Earth’s climate and features such as
droughts, rainfall extremes, and high-impact storms.

## Subcomponents

`CLIMA` currently consists of three models for the subcomponents of the Earth system:
 
* `CLIMA.Atmos`: A model of the fluid mechanics of the atmosphere and its interaction with solar radiation and phase changes of water that occur, for example, in clouds.
* `CLIMA.Ocean`: A model for the fluid mechanics of the ocean and its distributions of heat, salinity, carbon, and other tracers.
* `CLIMA.Land`: A model for the flow of energy and water in soils and on the land surface, for the biophysics of vegetation on land, and for the transfer and storage of carbon in the land biosphere.

The subcomponents will be coupled by exchanging water, momentum, energy, and tracers such as carbon dioxide across their boundaries.

## Dynamical core

A dynamical core based on discontinuous Galerkin numerical methods is used to discretize the physical conservation laws that underlie each of `CLIMA`'s subcomponents.

## Authors

`CLIMA` is being developed by [the Climate Modeling Alliance](https://clima.caltech.edu).

