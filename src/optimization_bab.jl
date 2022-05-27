
# modified from ZoPE implementation (https://github.com/sisl/NeuralPriorityOptimizer.jl)


"""
    struct PriorityOptimizerParameters
Define a struct which holds all the parameters for the priority optimizer
  steps: the maximum number of steps to take before returning the best bounds so far
  early_stop: whether to use evaluate_objective occasionally to narrow your optimality gap and potentially return early.
  stop_frequency: how often to check if you should return early
  stop_gap: optimality gap that you need to be beneath in order to return early
  initial_splits: a number of times to split the original hyperrectangle before doing any analysis.
"""
@with_kw struct PriorityOptimizerParameters
    max_steps::Int = 1000
    early_stop::Bool = true
    stop_frequency::Int = 200
    stop_gap::Float64 = 1e-4
    initial_splits::Int = 0
    verbosity::Int = 0
    print_frequency::Int = 100
    # timeout in seconds
    timeout::Float64 = 60
    plotting::Bool = false
    plot_frequency::Int = 100
    # history of (lb, ub) at time t for plotting
    history_vals::Vector{Float64} = Float64[]
    history_lbs::Vector{Float64} = Float64[]
    history_ts::Vector{Float64} = Float64[]
end


function print_progress(verbosity::Int, step, lower_bound, best_lower_bound,
    lower_bound_threshold, value, cell, elapsed_time)
    if verbosity == 1
        println("i: ", step)
        println("lower bound: ", lower_bound)
        println("Interval: ", [best_lower_bound, value])
        println("max radius: ", max(radius(cell)))

        # should print the lower and upper bounds of the input hyperrectangle
        println("Cell low: ", low(domain(cell)))
        println("Cell high: ", high(domain(cell)))
        println("lower_bound: ", lower_bound)
        println("lower bound threshold: ", lower_bound_threshold)
    elseif verbosity == 2
        println("i: ", step, " - ", [best_lower_bound, value], ", ", elapsed_time, " sec")
    end
end


"""
overestimate_cell : cell -> (val, cell_out) returns overestimate of value as well as cell propagated through the nn
"""
function NPO.general_priority_optimization(start_cell, overestimate_cell, achievable_value, params::PriorityOptimizerParameters, lower_bound_threshold, upper_bound_threshold, split)   
    # set timer
    t_start = time()

    initial_cells = split_multiple_times(start_cell, params.initial_splits)
    # Create your queue, then add your original new_cells
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first

    # add with priority
    for cell in initial_cells
        res = overestimate_cell(cell)
        #println(res)
        val, sym_out = res
        enqueue!(cells, sym_out, val)
    end

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

            new_value, new_sym_out = overestimate_cell(new_cell)
            enqueue!(cells, new_sym_out, new_value)
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


function NPO.general_priority_optimization(start_cell, relaxed_optimize_cell,
                                       evaluate_objective, params::PriorityOptimizerParameters,
                                       maximize; bound_threshold_realizable=(maximize ? Inf : -Inf),
                                       bound_threshold_approximate=(maximize ? -Inf : Inf),
                                       split=split_largest_interval)
    if maximize
        return general_priority_optimization(start_cell, relaxed_optimize_cell, evaluate_objective,
                                             params, bound_threshold_realizable,
                                             bound_threshold_approximate, split)
    else
        # overestimate_cell = cell -> -relaxed_optimize_cell(cell)
        overestimate_cell = cell -> begin
            val, out_cell = relaxed_optimize_cell(cell)
            return -val, out_cell
        end
        neg_evaluate_objective = cell -> begin
            input, result = evaluate_objective(cell)
            return input, -result
        end
        x, lower, upper, steps = general_priority_optimization(start_cell, overestimate_cell,
                                                               neg_evaluate_objective, params,
                                                               -bound_threshold_realizable,
                                                               -bound_threshold_approximate, split)
        return x, -upper, -lower, steps
    end
end



function get_initial_symbolic_interval(network, input_set, solver::DPNFV)
    return init_symbolic_interval_fvheur(network, input_set, max_vars=solver.max_vars)
end


function optimize_linear_deep_poly(network, input_set, coeffs, params::PriorityOptimizerParameters; maximize=true, solver=DPNFV(),
                              split=split_largest_interval, concrete_sample=:Center)
    min_sign_flip = maximize ? 1.0 : -1.0

    # should we include the preprocessing in the timout???
    network = merge_into_network(network, min_sign_flip .* coeffs)
    network = NetworkNegPosIdx(network)

    initial_sym = get_initial_symbolic_interval(network, input_set, solver)

    function approximate_optimize_cell(cell)
        out_cell = forward_network(solver, network, cell)
        # val = min_sign_flip * ρ(min_sign_flip .* coeffs, out_cell)
        val = min_sign_flip * out_cell.ubs[end][1]
        return val, out_cell
    end

    if concrete_sample == :Center
        # since we used min_sign_flip, when merging the coeffs into the NN, we need to
        # add min_sign_flip here too
        achievable_value = cell -> (domain(cell).center, min_sign_flip * compute_output(network, domain(cell).center)[1])
    else concrete_sample == :BoundsMaximizer
        function achievable_value(cell)
            x_star = maximizer(cell)
            x_center = domain(cell).center

            # merged NN should only have one output
            y_star = min_sign_flip * compute_output(network, x_star)[1]
            y_center = min_sign_flip * compute_output(network, x_center)[1]

            if y_star > y_center
                y = y_star
                x = x_star
            else
                y = y_center
                x = x_center
            end

            return x, y
        end
    end

    return general_priority_optimization(initial_sym, approximate_optimize_cell, achievable_value, params, maximize, split=split)
end


"""
Maximizes the violation of the polytope's constraints.

Used to prove that the NN's output is contained in the polytope.

If the upper bound on the maximum is < 0 -> contained in polytope.
If we find a point that violates one constraint of the polytope -> not contained in polytope.
"""
function contained_within_polytope_deep_poly(network, input_set, polytope, params::PriorityOptimizerParameters; solver=DPNeurifyFV(),
                                            split=split_largest_interval, concrete_sample=:Center)
    min_sign_flip = 1.  # we want to maximize violation of polytope to prove containment
    A, b = tosimplehrep(polytope)
    network = merge_into_network(network, A, b)
    network = NetworkNegPosIdx(network)

    initial_sym = get_initial_symbolic_interval(network, input_set, solver)

    function approximate_optimize_cell(cell)
        out_cell = forward_network(solver, network, cell)
        violations = min_sign_flip * out_cell.ubs[end] # note: upper bounds
        # after merging, if all constraints are satisifed, we have Ax - b ≤ 0
        max_violation = maximum(violations)
        return max_violation, out_cell
    end

    if concrete_sample == :Center
        achievable_value = cell -> (domain(cell).center, maximum(compute_output(network, domain(cell).center)))
    elseif concrete_sample == :BoundsMaximizer
        function achievable_value(cell)
            x_star = maximizer(cell)

            if length(cell.ubs[end]) == 1
                # there only is one constraint
                y_star = compute_output(network, x_star)
            else
                # if there are multiple constraints, take maximizer of the one with the most potential
                # i.e. the one with the largest upper bound
                max_vio_idx = argmax(cell.ubs[end])
                y_star = compute_output(network, x_star[max_vio_idx, :])
            end

            return x_star, maximum(y_star)
        end
    else
        throw(ArgumentError("keyword $(concrete_sample) does not exist!"))
    end

    # maximize = true,
    # bound_threshold_realizable = 0 (-> if we find some concrete input x with NN(x) > 0, then we are not in the polytope)
    # bound_threshold_approximate = 0 (-> if we can prove that NN(x) < 0 for all x, then we are guaranteed to be in the polytope)
    return general_priority_optimization(initial_sym, approximate_optimize_cell, achievable_value, params, true,
                                             split=split, bound_threshold_realizable=0., bound_threshold_approximate=0.)
end


"""
Minimizes the violation of the polytope's constraints.

Used to prove that NN's output is disjoint with the polytope.

If the lower bound on the minimal violation is > 0 -> guaranteed to be outside polytope.
If we find a point that satisfies all of the polytope's constraints -> intersects with polytope
"""
function reaches_polytope_deep_poly(network, input_set, polytope, params::PriorityOptimizerParameters; solver=DPNeurifyFV(),
                                    split=split_largest_interval, concrete_sample=:Center)
    min_sign_flip = -1.  # we want to minimize violation of polytope to prove that we stay outside
    A, b = tosimplehrep(polytope)
    network = merge_into_network(network, A, b)
    network = NetworkNegPosIdx(network)

    initial_sym = get_initial_symbolic_interval(network, input_set, solver)

    function approximate_optimize_cell(cell)
        out_cell = forward_network(solver, network, cell)
        violations = out_cell.lbs[end]  # note: lower bounds

        max_violation = maximum(violations)
        return max_violation, out_cell
    end

    if concrete_sample == :Center
        achievable_value = cell -> (domain(cell).center, maximum(compute_output(network, domain(cell).center)))
    elseif concrete_sample == :BoundsMaximizer
        function achievable_value(cell)
            x_star = minimizer(cell)

            if length(cell.ubs[end]) == 1
                # there only is one constraint
                y_star = compute_output(network, x_star)
            else
                # if there are multiple constraints, take minimizer of the one that is closest to leaving the polytope
                # i.e. the one with the largest lower bound
                max_vio_idx = argmax(cell.lbs[end])
                y_star = compute_output(network, x_star[max_vio_idx, :])
            end

            return x_star, maximum(y_star)
        end
    else
        throw(ArgumentError("keyword $(concrete_sample) does not exist!"))
    end

    return general_priority_optimization(initial_sym, approximate_optimize_cell, achievable_value, params, false,
                                             split=split, bound_threshold_realizable=0., bound_threshold_approximate=0.)
end


function NPO.split_largest_interval(s::SymbolicIntervalFVHeur)
    largest_dimension = argmax(high(domain(s)) - low(domain(s)))
    return split_symbolic_interval_fv_heur(s, largest_dimension)
end


function NPO.split_multiple_times(cell::SymbolicIntervalFVHeur, n; split=split_largest_interval)
    q = Queue{SymbolicIntervalFVHeur}()
    enqueue!(q, cell)
    for i in 1:n
        new_cells = split(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end
