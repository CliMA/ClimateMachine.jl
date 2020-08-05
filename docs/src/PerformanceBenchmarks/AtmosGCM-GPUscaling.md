# Test the scalability of GCM to multi-GPUs

## baroclinic wave

Test on the [dry baroclinic wave experiment](https://github.com/CliMA/ClimateMachine.jl/blob/master/experiments/AtmosGCM/baroclinic-wave.jl) with the following modifications on the experiment

```
-    poly_order = 3                           # discontinuous Galerkin polynomia
-    n_horz = 20                              # horizontal element number
-    n_vert = 5                               # vertical element number
-    n_days = 20
-    CFL::FT = 0.1
+    poly_order = 4                          # discontinuous Galerkin polynomial
+    n_horz = 32                              # horizontal element number
+    n_vert = 7                               # vertical element number
+    n_days::FT = 0.1
+    CFL::FT = 0.08
```

### strong scaling
Problem size: `n_horz = 32, n_vert = 7, poly_order = 4`. It scales quite well to 2 GPUs but the problem size is too small and simulations with more GPUs are limited by the overheads.
```
   GPU              |   wall time (sec) per step   |  efficienty
    1               |       0.3501                 |     --
    2               |       0.1776                 |     0.9856
    3               |       0.1710                 |     0.6824
    4               |       0.1297                 |     0.6748
    8 (2nodes)      |       0.1099                 |     0.3982
    12 (3nodes)     |       0.0837                 |     0.3485
    16 (4nodes)     |       0.0973                 |     0.2248
```

### weaking scaling
Problem size: `n_horz = 32, n_vert = 7, poly_order = 4` on 1 GPU. To increase the problem size and the number of GPUs simultaneously shows a good scalibility for weak scaling.
```
n_horz     |  GPU      |  min(Δ_horz)    |    Δt   |   wall time (sec) per step
    32     |    1      |   38184.92 m    |   8.65  |       0.3501
 x2(45)    |    2      |   27156.29 m    |   6.15  |       0.3702
 x3(55)    |    3      |   22218.01 m    |   5.03  |       0.3858
 x4(64)    |    4      |   19092.31 m    |   4.33  |       0.3686
 x16(128)  |    16     |    9546.14 m    |   2.16  |       0.4059
```