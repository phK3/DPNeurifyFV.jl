
## Generate specification for NeuralPriorityOptimizer

"""
Generates specification for NeuralPriorityOptimizer from result value from vnnlib parser.

Returns a list [(input_set, output_set)], where input_set is a Hyperrectangle and
output_set is either of
- HPolytope (for a disjunctive constraint like adversarial robustness)
    -> want to use contained_within_polytope(), if it is violated, there is an adversarial example
- Complement(HPolytope) (for a conjunctive constraint)
    -> want to use reaches_polytope(), if it is satisfied, we have reached the unsafe state

Mixed disjunctive and conjunctive constraints are not fully supported!

"""
function generate_specs(rv)
    # returns list of specs
    # (input_set, output_set)
    specs = []
    
    if length(rv) > 1
        println("WARNING: No efficient verification for disjunction of conjunctions of input spaces implemented yet!\n
                     Creating multiple sub-problems.")
    end

    for rv_tuple in rv
        l, u, output_specs = rv_tuple

        input_set = Hyperrectangle(low=l, high=u)

        if length(output_specs) == 1
            # a single polytope
            A, b = output_specs[1]

            # spec is SAT, if we reach the polytope
            # we want to use
            output_set = Complement(HPolytope(A, b))
            push!(specs, (input_set, output_set))
        elseif all([length(b) for (A, b) in output_specs] .== ones(Integer, length(output_specs)))
            # disjunction of halfspaces
            A₁, b₁ = output_specs[1]
            Â = zeros(length(output_specs), length(A₁))
            b̂ = zeros(length(output_specs))

            for (i, (A, b)) in enumerate(output_specs)
                Â[i, :] .= vec(A)
                b̂[i] = b[1]
            end

            # we want to use contained within polytope
            # spec is SAT, if we are not contained -> violation is greater than 0
            output_set = HPolytope(-Â, -b̂)

            push!(specs, (input_set, output_set))
        else
            # disjunction of conjunction of halfspaces
            println("WARNING: No efficient verification for disjunction of conjunctions of halfspaces implemented yet!\n
                     Creating multiple sub-problems.")

            for (A, b) in output_specs
                output_set = Complement(HPolytope(A, b))
                push!(specs, (input_set, output_set))
            end
        end

    end

    return specs
end


function get_sat(type, lower_bound, upper_bound, stop_gap)
    if type == :contained_within_polytope
        if lower_bound > 0
            # at least one constraint of the polytope is violated! Have found a counterexample
            return "SAT"
        elseif upper_bound <= stop_gap
            return "UNSAT"
        else
            return "inconclusive"
        end
    elseif type == :reaches_polytope 
        if upper_bound <= stop_gap
            # distance to polytope is smaller than allowed! Have found a counterexample
            # TODO: could return x_star as counterexample
            return "SAT"
        elseif lower_bound > 0
            # distance to polytope is larger than 0 (and stop_gap), property is proven
            return "UNSAT"
        else
            return "inconclusive"
        end
    else
        throw(ArgumentError("Unknown verification type $type !"))
    end
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
function verify_vnnlib(solver::DPNFV, network::Network, vnnlib_file, params::PriorityOptimizerParameters; 
                        split=split_important_interval, concrete_sample=:BoundsMaximizer, printing=true, eager=nothing)
    # DPNeurifyFV doesn't use eager 

    n_in = size(network.layers[1].weights, 2)
    n_out = NV.n_nodes(network.layers[end])
    
    rv = read_vnnlib_simple(vnnlib_file, n_in, n_out)
    specs = generate_specs(rv)
    
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
            x_star, lower_bound, upper_bound, steps = contained_within_polytope_deep_poly(network, input_set, output_set, params; solver=s_intern,
                                                                split=split, concrete_sample=concrete_sample)
            
            result = get_sat(:contained_within_polytope, lower_bound, upper_bound, params.stop_gap) 
        elseif output_set isa Complement{<:Number, <:AbstractPolytope}
            println("Checking if polytope can be reached")
            
            # reaches_polytope minimizes distance to polytope
            x_star, lower_bound, upper_bound, steps = reaches_polytope_deep_poly(network, input_set, output_set.X, params; solver=s_intern,
                                                                split=split, concrete_sample=concrete_sample)
            result = get_sat(:reaches_polytope, lower_bound, upper_bound, params.stop_gap)
        else
            @assert false "No implementation for output_set = $(output_set)"
        end

        all_steps += steps

        printing && println("Steps: ", steps, " - ", [lower_bound, upper_bound], " -> ", result)
        result == "SAT" && break  # can terminate loop, if one term of the disjunction is true
    end

    return x_star, all_steps, result
end


"""
Verifies network for given vnnlib specification.

params:
    solver - solver instance to use for verification
    network - network instance
    vnnlib_file - location of vnnlib specification file 
    params - parameters for solver

returns:
    counterexample - or nothing, if no counterexample could be found
    all_steps - number of steps performed by verifier
    result - (String) SAT, UNSAT or inconclusive
"""
function verify_vnnlib(solver::Ai2z, network::Network, vnnlib_file, params::PriorityOptimizerParameters; eager=false, printing=true, 
                        split=nothing, concrete_sample=nothing)  
    # ZoPE doesn't use split and concrete_sample

    n_in = size(network.layers[1].weights, 2)
    n_out = NV.n_nodes(network.layers[end])
    
    rv = read_vnnlib_simple(vnnlib_file, n_in, n_out)
    specs = generate_specs(rv)
    
    x_star = nothing
    result = "inconclusive"
    all_steps = 0
    # if length(specs) > 1, we are dealing with a disjunction of constraints -> can abort, if we found one SAT
    for (input_set, output_set) in specs
        
        if output_set isa AbstractPolytope
            println("Checking if contained within polytope")
            
            # contained_within_polytope maximizes violation of polytope's constraints
            x_star, lower_bound, upper_bound, steps = contained_within_polytope(network, input_set, output_set, params; solver=solver)
            
            result = get_sat(:contained_within_polytope, lower_bound, upper_bound, params.stop_gap)    
        elseif output_set isa Complement{<:Number, <:AbstractPolytope}
            println("Checking if polytope can be reached")
            
            # reaches_polytope minimizes distance to polytope
            x_star, lower_bound, upper_bound, steps = reaches_polytope_binary(network, input_set, output_set.X, params; solver=solver,
                                                                eager=eager)
            result = get_sat(:reaches_polytope, lower_bound, upper_bound, params.stop_gap)
        else
            @assert false "No implementation for output_set = $(output_set)"
        end

        all_steps += steps

        printing && println("Steps: ", steps, " - ", [lower_bound, upper_bound], " -> ", result)
        result == "SAT" && break  # can terminate loop, if one term of the disjunction is true
    end
    
    return x_star, all_steps, result
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
function verify_vnnlib(solver, dir, params::PriorityOptimizerParameters; logfile=nothing, max_properties=Inf, 
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
            net = read_onnx_network(string(dir, "/", netpath), dtype=Float64)
            old_netpath = netpath
        end

        # TODO: maybe include keyword arguments for ZoPE and DPNeurifyFV?
        time = @elapsed x_star, steps, result = verify_vnnlib(solver, net, string(dir, "/", propertypath), params, 
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

        
