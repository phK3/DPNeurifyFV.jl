# just the general_priority_optimization method from ZoPE (https://github.com/sisl/NeuralPriorityOptimizer.jl/blob/main/src/optimization_core.jl)
# but with our PriorityOptimizerParameters type (s.t. we can set a timeout and use our print_progress methods)


"""
    general_priority_optimization(start_cell::Hyperrectangle, approximate_optimize_cell, achievable_value, params::PriorityOptimizerParameters)

Use a priority based approach to split your space and optimize an objective function. We assume we are maximizing our objective.
General to any objective function passed in as well as an evaluate objective
The function overestimate_cell takes in a cell and returns an overestimate of the objective value.
The function achievable_value takes in the input cell and returns an input and the value it achieves in that cell.
This optimization strategy then uses these functions to provide bounds on the maximum objective

If we ever get an upper bound on our objective that's lower than the upper_bound_threshold then we return

This function returns the best input found, a lower bound on the optimal value, an upper bound on the optimal value, and the number of steps taken.
"""
function NPO.general_priority_optimization(start_cell::Hyperrectangle, overestimate_cell, achievable_value, params::PriorityOptimizerParameters, lower_bound_threshold, upper_bound_threshold, split)
    # set timer
    t_start = time()

    initial_cells = split_multiple_times(start_cell, params.initial_splits)
    # Create your queue, then add your original new_cells
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first
    [enqueue!(cells, cell, overestimate_cell(cell)) for cell in initial_cells] # add with priority
    best_lower_bound = -Inf
    best_x = nothing
    # For n_steps dequeue a cell, split it, and then
    for i = 1:params.max_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        @assert value + TOL[] >= best_lower_bound string("Our highest upper bound must be greater than the highest achieved value. Upper bound: ", value, " achieved value: ", best_lower_bound)
        dequeue!(cells)

        # We've passed some threshold on our upper bound that is satisfactory, so we return
        if value < upper_bound_threshold
            println("Returning early because of upper bound threshold")
            return best_x, best_lower_bound, value, i
        end

        # Early stopping
        if params.early_stop
            t_now = time()
            elapsed_time = t_now - t_start

            if i % params.stop_frequency == 0
                input_in_cell, lower_bound = achievable_value(cell)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = input_in_cell
                end

                if ((value .- lower_bound) <= params.stop_gap
                    || lower_bound > lower_bound_threshold
                    || elapsed_time >= params.timeout)

                    print_progress(params.verbosity, i, lower_bound, best_lower_bound,
                                    lower_bound_threshold, value, cell, elapsed_time)

                    return best_x, best_lower_bound, value, i
                end
            end

            if params.verbosity >= 1 && i % params.print_frequency == 0
                print_progress(params.verbosity, i, lower_bound, best_lower_bound,
                                lower_bound_threshold, value, cell, elapsed_time)
            end

            if params.plotting && i % params.plot_frequency == 0
                push!(params.history_vals, value)
                push!(params.history_lbs, best_lower_bound)
                push!(params.history_ts, elapsed_time)
            end
        end

        new_cells = split(cell)
        # Enqueue each of the new cells
        for new_cell in new_cells
            # If you've made the max objective cell tiny
            # break (otherwise we end up with zero radius cells)
            if radius(new_cell) < NeuralVerification.TOL[]
                # Return a concrete value and the upper bound from the parent cell
                # that was just dequeued, as it must have higher value than all other cells
                # that were on the queue, and they constitute a tiling of the space
                input_in_cell, lower_bound = achievable_value(cell)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = input_in_cell
                end
                return best_x, best_lower_bound, value, i
            end
            new_value = overestimate_cell(new_cell)
            enqueue!(cells, new_cell, new_value)
        end
    end
    # The largest value in our queue is the approximate optimum
    cell, value = peek(cells)
    input_in_cell, lower_bound = achievable_value(cell)
    if lower_bound > best_lower_bound
        best_lower_bound = lower_bound
        best_x = input_in_cell
    end
    return best_x, best_lower_bound, value, params.max_steps
end