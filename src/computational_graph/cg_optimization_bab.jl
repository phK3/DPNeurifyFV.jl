

"""
Appends violation of polytope specification Ax ≤ b as linear layer Ax - b to the end of the network.

If Ax - b ≤ 0, the specification is fulfilled, if one dimension is > 0, we have a violation.
"""
function merge_into_network(nn::CompGraph, A::AbstractMatrix, b::AbstractVector)
    spec_node = Linear(nn.out_node.outputs, ["spec_out"], "spec_node", A, .-b)
    
    nodes = copy(nn.nodes)
    nodes[spec_node.name] = spec_node

    in_node = nn.in_node
    out_node = spec_node

    out_dict = copy(nn.out_dict)
    out_dict[spec_node.outputs[1]] = spec_node

    in_shape = nn.input_shape
    out_shape = (size(b), 1)  # with batch dim

    return CompGraph(nodes, in_node, out_node, out_dict, in_shape, out_shape)
end



function reaches_polytope(nn::CompGraph, input_set::AbstractHyperrectangle, polytope, params::PriorityOptimizerParameters;
                          solver=DPNFV(), split=split_largest_interval, concrete_sample=:Center)
    A, b = tosimplehrep(polytope)
    nn_spec = merge_into_network(nn, A, b)

    in_shape = map(x -> ifelse(isa(x, Integer), x, 1), nn_spec.input_shape)
    s = init_symbolic_interval_graph(nn_spec, input_set, in_shape)

    function approximate_optimize_cell(cell)
        out_cell = propagate(solver, nn_spec, cell)
        violations, _ = bounds(out_cell)
        max_violation = maximum(violations)
        return max_violation, out_cell
    end

    if concrete_sample == :Center
        achievable_value = cell -> (domain(cell).center, maximum(propagate(nn_spec, domain(cell).center)))
    elseif concrete_sample == :BoundsMaximizer
        function achievable_value(cell)
            x_star = minimizer(cell)
            violations, _ = bounds(cell)

            if length(violations) == 1
                # there only is one constraint
                y_star = propagate(nn_spec, x_star)
            else
                # if there are multiple constraints, take minimizer of the one that is closest to leaving the polytope
                # i.e. the one with the largest lower bound
                max_vio_idx = argmax(violations)
                # TODO: maybe reshape x_star?
                y_star = propagate(nn_spec, x_star[max_vio_idx, :])
                x_star = x_star[max_vio_idx, :]
            end

            return x_star, maximum(y_star)
        end
    else
        throw(ArgumentError("keyword $(concrete_sample) does not exist!"))
    end


    return general_priority_optimization(s, approximate_optimize_cell, achievable_value, params, false,
                        split=split, bound_threshold_realizable=0., bound_threshold_approximate=0.)
end


function contained_within_polytope(nn::CompGraph, input_set::AbstractHyperrectangle, polytope, params::PriorityOptimizerParameters;
    solver=DPNFV(), split=split_largest_interval, concrete_sample=:Center)
    A, b = tosimplehrep(polytope)
    nn_spec = merge_into_network(nn, A, b)

    in_shape = map(x -> ifelse(isa(x, Integer), x, 1), nn_spec.input_shape)
    s = init_symbolic_interval_graph(nn_spec, input_set, in_shape)

    function approximate_optimize_cell(cell)
        out_cell = propagate(solver, nn_spec, cell)
        _, violations = bounds(out_cell)
        max_violation = maximum(violations)
        return max_violation, out_cell
    end

    if concrete_sample == :Center
        achievable_value = cell -> (domain(cell).center, maximum(propagate(nn_spec, domain(cell).center)))
    elseif concrete_sample == :BoundsMaximizer
        function achievable_value(cell)
            x_star = maximizer(cell)
            _, violations = bounds(cell)

            if length(violations) == 1
                # there only is one constraint
                y_star = propagate(nn_spec, x_star)
            else
                # if there are multiple constraints, take maximizer of the one with the most potential
                # i.e. the one with the largest upper bound
                max_vio_idx = argmax(violations)
                # TODO: do we need to reshape x_star?
                y_star = propagate(nn_spec, x_star[max_vio_idx, :])
                x_star = x_star[max_vio_idx, :]
            end

            return x_star, maximum(y_star)
        end
    else
        throw(ArgumentError("keyword $concrete_sample doesn't exist!"))
    end

    return general_priority_optimization(s, approximate_optimize_cell, achievable_value, params, true,
            split=split, bound_threshold_realizable=0., bound_threshold_approximate=0.)
end


"""
Verifies network for given vnnlib specification.

args:
    solver - solver instance to use for verification
    network - network instance
    vnnlib_file - location of vnnlib specification file 
    params - parameters for solver

kwargs:
    split - method to use for input splitting
    concrete_sample - method to use for counterexample generation
    printing - (bool) whether to print results

returns:
    counterexample - or nothing, if no counterexample could be found
    all_steps - number of steps performed by verifier
    result - (String) SAT, UNSAT or inconclusive
"""
function verify_vnnlib(solver::DPNFV, network::CompGraph, vnnlib_file::String, params::PriorityOptimizerParameters; 
                        split=split_important_interval, concrete_sample=:BoundsMaximizer, printing=true, eager=nothing)
    # DPNeurifyFV doesn't use eager 

    speclist = PyVnnlib.generate_specs(vnnlib_file, dtype=Float64)
    specs = generate_specs(speclist)
    
    x_star = nothing
    result = "inconclusive"
    all_steps = 0
    # if length(specs) > 1, we are dealing with a disjunction of constraints -> can abort, if we found one SAT
    for (input_set, output_set) in specs
    	GC.gc()
    	# is this better with memory management?
    	s_intern = DPNFV(method=solver.method, max_vars=solver.max_vars, var_frac=solver.var_frac, get_fresh_var_idxs=solver.get_fresh_var_idxs)
        
        if output_set isa AbstractPolytope
            println("Checking if contained within polytope")
            
            # contained_within_polytope maximizes violation of polytope's constraints
            x_star, lower_bound, upper_bound, steps = contained_within_polytope(network, input_set, output_set, params; solver=s_intern,
                                                                split=split, concrete_sample=concrete_sample)
            
            result = get_sat(:contained_within_polytope, lower_bound, upper_bound, params.stop_gap) 
        elseif output_set isa Complement{<:Number, <:AbstractPolytope}
            println("Checking if polytope can be reached")
            
            # reaches_polytope minimizes distance to polytope
            x_star, lower_bound, upper_bound, steps = reaches_polytope(network, input_set, output_set.X, params; solver=s_intern,
                                                                split=split, concrete_sample=concrete_sample)
            result = get_sat(:reaches_polytope, lower_bound, upper_bound, params.stop_gap)
        else
            @assert false "No implementation for output_set = $(output_set)"
        end

        all_steps += steps

        printing && println("Steps: ", steps, " - ", [lower_bound, upper_bound], " -> ", result)
        result == "SAT" && break  # can terminate loop, if one term of the disjunction is true
    end

    if isnothing(x_star)
        # when we can prove property UNSAT in first step, there is no counterexample to try
        y_star = nothing
    else
        y_star = propagate(network, x_star)
    end

    return x_star, y_star, all_steps, result
end


function verify_vnnlib(solver::DPNFV, onnx_file::String, vnnlib_file::String, params::PriorityOptimizerParameters;
                       split=split_important_interval, concrete_sample=:BoundsMaximizer, printing=true, eager=nothing)
    nn = NNL.load_network_dict(CGType, onnx_file)
    return verify_vnnlib(solver, nn, vnnlib_file, params, split=split, concrete_sample=concrete_sample, printing=printing, eager=eager)
end


# max_properties is maximum number of properties we want to verify in this run (useful for debugging and testing)
"""
Verifies properties for network in directory with instances.csv file.
params:
    solver - solver instance to use for verification
    dir - directory containing instances.csv file with combinations of onnx networks and vnnlib properties to test
    params - parameters for solver

kwargs:
    logfile - where to store verification results 
    max_properties - maximum number of instances to verify (useful for debugging, so we don't have to run all the tasks)
    split - splitting heuristic for DPNeurifyFV
    concrete_sample - sampling for concrete solutions for DPNeurifyFV
    eager - use eager Bounds checking in ZoPE

returns:
    counterexample - or nothing, if no counterexample could be found
    all_steps - number of steps performed by verifier
    result - (String) SAT, UNSAT or inconclusive
"""
function verify_vnnlib_directory(solver::DPNFV, dir::String, params::PriorityOptimizerParameters; logfile=nothing, max_properties=Inf, 
                        split=split_important_interval, concrete_sample=:BoundsMaximizer, eager=false)
    f = CSV.File(string(dir, "/instances.csv"), header=false)

    n = length(f)
    networks = String[]
    properties = String[]
    results = String[]
    all_steps = zeros(Integer, n)
    times = zeros(n)

    old_netpath = nothing
    net = nothing

    for (i, instance) in enumerate(f)
        netpath, propertypath, time_limit = instance

        if netpath != old_netpath
            println("-- loading network ", netpath)
            net = NNL.load_network_dict(CGType, string(dir, "/", netpath))
            old_netpath = netpath
        end

        # TODO: maybe include keyword arguments for ZoPE and DPNeurifyFV?
        time = @elapsed x_star, y_star, steps, result = verify_vnnlib(solver, net, string(dir, "/", propertypath), params, 
                                                                split=split, concrete_sample=concrete_sample, eager=eager)

        push!(networks, netpath)
        push!(properties, propertypath)
        push!(results, result)
        all_steps[i] = steps
        times[i] = time

        if i >= max_properties
            break
        end
    end

    if !isnothing(logfile)
        open(logfile, "w") do f
            println(f, "network,property,result,time,steps")
            [println(f, string(network, ", ", property, ", ", result, ", ", time, ", ", steps)) 
                    for (network, property, result, time, steps) in zip(networks, properties, results, times, all_steps)]
        end
    end

    return properties, results, times, all_steps
end