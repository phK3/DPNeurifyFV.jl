

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

    usage_map = copy(nn.usage_map)
    if haskey(usage_map, spec_node.inputs[1])
        usage_map[spec_node.inputs[1]] += 1
    else
        usage_map[spec_node.inputs[1]] = 1
    end

    in_shape = nn.input_shape
    out_shape = (size(b), 1)  # with batch dim

    return CompGraph(nodes, in_node, out_node, out_dict, usage_map, in_shape, out_shape)
end



function test_counterexample(x, y, in_shape, in_name, out_name, onnx_file)
    # test if onnx execution matches own execution
    # need to propagate again as we might have merged output spec with last layer of nn
    model = OX.load_inference(onnx_file)
    outs = model(Dict(in_name => reshape(Float32.(x), reverse(in_shape))))
    y_onnx = outs[out_name]

    res = "SAT"
    if any(abs.(y .- reversedims(y_onnx)) .> 1e-4)
        println("Numberical precision error! Changing result to inconclusive")
        res = "inconclusive"
    end

    return res, x, y 
end


function reaches_polytope(nn::CompGraph, input_set::AbstractHyperrectangle, polytope, params::PriorityOptimizerParameters;
                          solver=DPNFV(), split=split_largest_interval, concrete_sample=:Center)
    A, b = tosimplehrep(polytope)
    nn_spec = merge_into_network(nn, A, b)

    #in_shape = map(x -> ifelse(isa(x, Integer), x, 1), nn_spec.input_shape)
    in_shape = nn_spec.input_shape
    s = init_symbolic_interval_graph(nn_spec, input_set, Tuple(in_shape))

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

    #in_shape = map(x -> ifelse(isa(x, Integer), x, 1), nn_spec.input_shape)
    in_shape = nn_spec.input_shape
    s = init_symbolic_interval_graph(nn_spec, input_set, Tuple(in_shape))

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


# TODO: put this somewhere else 
#       together with other concretization functions, get rid of the concrete_sample keyword and accept a function instead
function concrete_input_valid(s::SymbolicIntervalGraph{<:Hyperrectangle}; h=2, w=2)
    lx = low(domain(s))
    ux = high(domain(s))
    width = ux .- lx

    l = lx[1:h*w] .+ width[1:h*w] .* rand(h*w)
    l = accumulate(max, l)

    u = lx[h*w+1:2*h*w] .+ width[h*w+1:2*h*w] .* rand(h*w)
    u = max.(l, u)
    i = argmax(u)
    u[i] = 1.

    a = lx[2*h*w+1:end] .+ width[2*h*w+1:end] .* rand(h*w)

    return [l; u; a]    
end


function optimize_linear(nn::CompGraph, input_set::AbstractHyperrectangle, coeffs::AbstractVector, params::PriorityOptimizerParameters;
    solver=DPNFV(), split=split_important_interval, concrete_sample=:Center, maximize=true)
    min_sign_flip = maximize ? 1. : -1.
    nn_spec = merge_into_network(nn, min_sign_flip .* coeffs', zeros(1))

    #in_shape = map(x -> ifelse(isa(x, Integer), x, 1), nn_spec.input_shape)
    in_shape = nn_spec.input_shape
    s = init_symbolic_interval_graph(nn_spec, input_set, Tuple(in_shape))

    function approximate_optimize_cell(cell)
        out_cell = propagate(solver, nn_spec, cell)
        lbs, ubs = bounds(out_cell)
        # after merging into network, we only have a single output
        val = min_sign_flip * ubs[1]
        return val, out_cell
    end

    if concrete_sample == :Center
        achievable_value = cell -> (domain(cell).center, min_sign_flip * propagate(nn_spec, domain(cell).center)[1])
    elseif concrete_sample == :BoundsMaximizer
        achievable_value = cell -> begin
            x_star = maximizer(cell)
            x_center = domain(cell).center

            y_star = propagate(nn_spec, x_star)[1]
            y_center = propagate(nn_spec, x_center)[1]

            if y_star > y_center
                y = y_star
                x = x_star
            else
                y = y_center
                x = x_center
            end

            return x, y
        end
    elseif concrete_sample == :valid 
        achievable_value = cell -> begin
            x = concrete_input_valid(cell)
            y = propagate(nn_spec, x)[1]

            return x, y
        end
    else
        throw(ArgumentError("keyword $concrete_sample doesn't exist!"))
    end

    return general_priority_optimization(s, approximate_optimize_cell, achievable_value, params, maximize, split=split)
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
                        split=split_important_interval, concrete_sample=:BoundsMaximizer, printing=true, eager=nothing,
                        check_onnx=false)
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
                       split=split_important_interval, concrete_sample=:BoundsMaximizer, printing=true, eager=nothing,
                       check_onnx=false, convert2linear=true)
    nn = try 
        NNL.load_network_dict(CGType, onnx_file)
    catch err
        println("Failed to load: ", netpath)
        println(err)
        return zeros(5), zeros(5), 0, "inconclusive"
    end

    # need to store here in case we convert2linear, which changes shapes and may change input node
    in_shape = nn.input_shape
    in_name = get_inputs(nn.in_node)[1]
    out_name = get_outputs(nn.out_node)[1]

    if convert2linear
        nn = cg2dense(nn, double_precision=true)
        nn = summarize_linear_layers(nn, double_precision=true)
    end

    x_star, y_star, all_steps, result = verify_vnnlib(solver, nn, vnnlib_file, params, split=split, concrete_sample=concrete_sample, printing=printing, eager=eager)

    if result == "SAT" && check_onnx
        x_star = reshape(x_star, in_shape)
        result, x_star, y_star = test_counterexample(x_star, y_star, in_shape, in_name, out_name, onnx_file)
    end

    return x_star, y_star, all_steps, result
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
            net = try
                NNL.load_network_dict(CGType, string(dir, "/", netpath))
            catch e
                println("Error encountered: $e")
                println("skipping network!")
                push!(networks, netpath)
                push!(properties, propertypath)
                push!(results, "error")
                all_steps[i] = 0
                times[i] = 0
                continue
            end
            
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