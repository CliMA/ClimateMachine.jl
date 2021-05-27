abstract type AbstractSimulation end

struct Simulation{𝒯,𝒰,𝒱,𝒲,𝒳,𝒴,L} <: AbstractSimulation
    model::𝒯
    grid::L
    timestepper::𝒰
    time::𝒱
    callbacks::𝒲
    rhs::𝒳
    state::𝒴
end

function Simulation(model::ModelSetup; grid, timestepper, time, callbacks)
    rhs = DGModel(
        model, 
        grid.numerical,
        model.numerics.flux,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    ) 

    FT = eltype(rhs.grid.vgeo)
    state = init_ode_state(rhs, FT(0); init_on_cpu = true)
    
    return Simulation(model, grid, timestepper, time, callbacks, rhs, state)
end

function Simulation(model::DryAtmosModel; grid, timestepper, time, callbacks)
    rhs = ESDGModel(
        model, 
        grid.numerical,
        surface_numerical_flux_first_order = model.numerics.flux,
        volume_numerical_flux_first_order = KGVolumeFlux(),
    ) 

    FT = eltype(rhs.grid.vgeo)
    state = init_ode_state(rhs, FT(0); init_on_cpu = true)
    
    return Simulation(model, grid, timestepper, time, callbacks, rhs, state)
end

# prep for timestepping changes  {AbstractRate, AbstractRate}
# Alternatively, define method Explicit(::DryAtmosModel) so tmp = Explicit(item.model)
function Simulation(model::Tuple; grid, timestepper, time, callbacks)
    println("Constructing DG Models")
    rhs = []
    for item in model
        if item isa Explicit
            println("constructing explicit model") 
            if item.model isa DryAtmosModel
                tmp = Explicit(ESDGModel(
                    item.model,
                    grid.numerical,
                    surface_numerical_flux_first_order = item.model.numerics.flux,
                    volume_numerical_flux_first_order = KGVolumeFlux(),
                ))
            elseif item.model isa ModelSetup
                tmp = Explicit(DGModel(
                    item.model,
                    grid.numerical,
                    item.model.numerics.flux,
                    CentralNumericalFluxSecondOrder(),
                    CentralNumericalFluxGradient();
                    direction = item.model.numerics.direction,
                ))
            else
                println("what are you doing!?!??!")
            end
            push!(rhs, tmp)
        elseif item isa Implicit
            println("constructing implicit models")
            if item.model isa DryAtmosModel 
                tmp = Implicit(VESDGModel(
                    item.model,
                    grid.numerical,
                    surface_numerical_flux_first_order = item.model.numerics.flux,
                    volume_numerical_flux_first_order = LinearKGVolumeFlux(),
                ))
            elseif item.model isa ModelSetup
                tmp = Implicit(DGModel(
                    item.model,
                    grid.numerical,
                    item.model.numerics.flux,
                    CentralNumericalFluxSecondOrder(),
                    CentralNumericalFluxGradient();
                    direction = item.model.numerics.direction,
                ))
            else
                println("what are you doing!?!??!")
            end
            push!(rhs, tmp)
        end
    end

    rhs = Tuple(rhs)

    implicit_rhs = implicit(rhs)
    explicit_rhs = explicit(rhs)
    number_implicit = length(implicit_rhs)
    number_explicit = length(explicit_rhs) 
    if (number_implicit != 1) | (number_explicit != 1)
        error_string = "IMEX methods require 1 implicit model and one explicit model"
        error_string *= "\n for example, a tuple of the form  (Explicit(model), Implicit(linear_model),)"
        @error(error_string)
        return nothing
    end

    FT = eltype(rhs[1].model.grid.vgeo)
    println("constructing initial state")
    state = init_ode_state(rhs[1].model, FT(0); init_on_cpu = true)
    println("done")

    return Simulation(model, grid, timestepper, time, callbacks, rhs, state)
end

function initialize!(simulation::Simulation; overwrite = false)
    if overwrite
        simulation = Simulation(
            model = simulation.model, 
            timestepper = simulation.timestepper, 
            time = simulation.time, 
            callbacks = simulation.callbacks,
        )
    end

    return nothing
end

function evolve!(simulation::Simulation; refDat = ())
    # Unpack everything we need in this routine here
    timestepper = simulation.timestepper
    state = simulation.state
    rhs   = simulation.rhs
    grid  = simulation.grid

    npoly = convention(grid.resolution.polynomial_order, Val(ndims(grid.domain)))

    t0 = simulation.time.start
    tend = simulation.time.finish
    Δt = timestepper.timestep

    # Set up overintegration and polynomial order-based spatial staggering
    if haskey(grid.resolution, :overintegration_order)
        nover = convention(grid.resolution.overintegration_order, Val(ndims(grid.domain)))
    else
        nover = (0, 0, 0)
    end
    staggering = get(simulation.model.numerics, :staggering, false)

    # Perform overintegration filtering (only works if nover > 0)
    overintegration_filter!(state, rhs, npoly, nover)

    # Make right-hand side function with built-in filters
    rhs = rhs_closure(rhs, npoly, nover, staggering = staggering)

    # Instantiate time stepping method
    odesolver = timestepper.method(rhs, state, dt = Δt, t0 = t0)

    # Make callbacks from callbacks tuple
    cbvector = create_callbacks(simulation, odesolver)

    # Perform evolution of simulations
    if isempty(cbvector)
        solve!(state, odesolver; timeend = tend)
    else
        solve!(
            state,
            odesolver;
            timeend = tend,
            callbacks = cbvector,
        )
    end

    # Check results against reference if StateCheck callback is used
    # TODO: TB: I don't think this should live within this function
    if any(typeof.(simulation.callbacks) .<: StateCheck)
      check_inds = findall(typeof.(simulation.callbacks) .<: StateCheck)
      @assert length(check_inds) == 1 "Only use one StateCheck in callbacks!"

      ClimateMachine.StateCheck.scprintref(cbvector[check_inds[1]])
      if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[check_inds[1]], refDat)
      end
    end

    return nothing
end


function evolve!(simulation::Nothing)
    println(" you dun messed up")
    return nothing
end

function evolve!(simulation::Simulation{<:Tuple}; refDat = ())
    # Unpack everything we need in this routine here
    state         = simulation.state
    rhs           = simulation.rhs
    timestepper   = simulation.timestepper
    t0            = simulation.time.start
    tend          = simulation.time.finish
    Δt            = timestepper.timestep
    
    # Instantiate time stepping method    
    odesolver = construct_odesolver(timestepper.method, rhs, state, Δt, t0 = t0) 
    # Make callbacks from callbacks tuple
    cbvector = create_callbacks(simulation, odesolver)

    # Perform evolution of simulations
    if isempty(cbvector)
        solve!(state, odesolver; timeend = tend, adjustfinalstep = false)
    else
        solve!(
            state,
            odesolver;
            timeend = tend,
            callbacks = cbvector,
            adjustfinalstep = false,
        )
    end

    # Check results against reference if StateCheck callback is used
    # TODO: TB: I don't think this should live within this function
    if any(typeof.(simulation.callbacks) .<: StateCheck)
      check_inds = findall(typeof.(simulation.callbacks) .<: StateCheck)
      @assert length(check_inds) == 1 "Only use one StateCheck in callbacks!"

      ClimateMachine.StateCheck.scprintref(cbvector[check_inds[1]])
      if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[check_inds[1]], refDat)
      end
    end

    return nothing
end

function overintegration_filter!(state_array, rhs, npoly, nover)
    if sum(nover) > 0
        cutoff_order = npoly .+ 1 # yes this is confusing
        cutoff = MassPreservingCutoffFilter(rhs.grid, cutoff_order)
        num_state_prognostic = number_states(rhs.balance_law, Prognostic())
        filterstates = 1:num_state_prognostic
        ClimateMachine.Mesh.Filters.apply!(
            state_array,
            filterstates,
            rhs.grid,
            cutoff,
        ) 
    end

    return nothing
end

function staggering_filter!(state_array, rhs, npoly, nover)
    if sum(nover) > 0
        cutoff_order = npoly .+ 0 # one polynomial order less than scalars
        cutoff = MassPreservingCutoffFilter(rhs.grid, cutoff_order)
        num_state_prognostic = number_states(rhs.balance_law, Prognostic())
        filterstates = 2:4
        ClimateMachine.Mesh.Filters.apply!(
            state_array,
            filterstates,
            rhs.grid,
            cutoff,
        )
       
    end

    return nothing
end

function rhs_closure(rhs, npoly, nover; staggering = false)
    if staggering
        function rhs_staggered(state_array, args...; kwargs...)
            rhs(state_array, args...; kwargs...)
            overintegration_filter!(state_array, rhs, npoly, nover)
            staggering_filter!(state_array, rhs, npoly, nover)
            return nothing
        end

        rhs_filtered = rhs_staggered
    else
        function rhs_unstaggered(state_array, args...; kwargs...)
            rhs(state_array, args...; kwargs...)
            overintegration_filter!(state_array, rhs, npoly, nover)
            return nothing
        end
        
        rhs_filtered = rhs_unstaggered
    end

    return rhs_filtered 
end # returns a closure