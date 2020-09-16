# RHS method sequence

 - `update_auxiliary_state!` on `realelems`

 - `volume_gradients!` on `realelems`
   - calls
   - `compute_gradient_argument!`
   - `compute_gradient_flux!`

 - `interface_gradients!` on `interiorelems`
   - calls
   - `compute_gradient_argument!`
   - `compute_gradient_flux!`
   - `numerical_flux_gradient!`
   - `numerical_boundary_flux_gradient!`
     - calls
     - `boundary_state!`
     - `compute_gradient_argument!`

 - `update_auxiliary_state!` on `ghostelems`

 - `interface_gradients!` on `exteriorelems`
   - calls
   - `compute_gradient_argument!`
   - `compute_gradient_flux!`
   - `numerical_flux_gradient!`
   - `numerical_boundary_flux_gradient!`
     - calls
     - `boundary_state!`
     - `compute_gradient_argument!`

 - `update_auxiliary_state_gradient!` on `realelems`

 - `volume_divergence_of_gradients!` on `realelems` (no front-end calls)

 - `interface_divergence_of_gradients!` on `interiorelems`
   - calls
   - `numerical_flux_divergence!`
   - `numerical_boundary_flux_divergence!`
     - calls
     - `boundary_state!`
     - `numerical_flux_divergence!`

 - `interface_divergence_of_gradients!` on `exteriorelems`
   - calls
   - `numerical_flux_divergence!`
   - `numerical_boundary_flux_divergence!`
     - calls
     - `boundary_state!`
     - `numerical_flux_divergence!`

 - `volume_gradients_of_laplacians!` on `realelems`
   - calls
   - `transform_post_gradient_laplacian!`

 - `interface_gradients_of_laplacians!` on `interiorelems`
   - calls
   - `numerical_flux_higher_order!`
   - `numerical_boundary_flux_higher_order!`

 - `interface_gradients_of_laplacians!` on `exteriorelems`
   - calls
   - `numerical_flux_higher_order!`
   - `numerical_boundary_flux_higher_order!`

 - `volume_tendency!` on `realelems`
   - calls
   - `flux_first_order!` (direction)
   - `flux_second_order!` (direction)
   - `flux_first_order!` (HorizontalDirection)
   - `flux_first_order!` (VerticalDirection)
   - `source!`

 - `interface_tendency!` on `interiorelems`
   - calls
   - `numerical_flux_first_order!`
   - `numerical_flux_second_order!`
   - `numerical_boundary_flux_first_order!`
     - calls
     - `boundary_state!`
     - `numerical_flux_first_order!`
   - `numerical_boundary_flux_second_order!`

 - `update_auxiliary_state_gradient!` on `ghostelems`

 - `interface_tendency!` on `exteriorelems`
   - calls
   - `numerical_flux_first_order!`
   - `numerical_flux_second_order!`
   - `numerical_boundary_flux_first_order!`
     - calls
     - `boundary_state!`
     - `numerical_flux_first_order!`
   - `numerical_boundary_flux_second_order!`

