
# taken from https://github.com/sisl/NeuralPriorityOptimizer.jl/blob/main/examples/acas_example.jl
# Parameters were adjusted to the parameters in given in the ZoPE paper: https://arxiv.org/pdf/2106.05325.pdf

###
# This script demonstrates how to check whether the output of a network is contained within a polytope
# we use ACASXu property 1: see https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/properties/acas_property_1.txt
###

# Include the optimizer as well as supporting packages
using DPNeurifyFV
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets
using LinearAlgebra

@assert Threads.nthreads()==1 "for benchmarking threads must be 1"
LinearAlgebra.BLAS.set_num_threads(1)


# need to redefine from NeuralPriorityOptimizer, as PolytopeComplement is not defined, need just Complement now.
function NeuralPriorityOptimizer.get_acas_sets(property_number)
    if property_number == 1
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = HalfSpace([1.0, 0.0, 0.0, 0.0, 0.0], 3.9911256459)
    elseif property_number == 2
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = Complement(HPolytope([-1.0 1.0 0.0 0.0 0.0; -1.0 0.0 1.0 0.0 0.0; -1.0 0.0 0.0 1.0 0.0; -1.0 0.0 0.0 0.0 1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 3
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.4933803236, 0.3, 0.3], high=[-0.2985528119, 0.0095492966, 0.5, 0.5, 0.5])
        output_set = Complement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 4
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.0, 0.3181818182, 0.0833333333], high=[-0.2985528119, 0.0095492966, 0.0, 0.5, 0.1666666667])
        output_set = Complement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    else
        @assert false "Unsupported property number"
    end

    return input_set, output_set
end


"""
    test_acas_network(index1, index2, property_index, params; p=1)

Test ACAS network index1-index2 on property property_index with solver parameters given by params.
p sets the norm to use for properties 2, 3, and 4 which project onto a polytope under that norm.
"""
function test_acas_network(index1, index2, property_index, params; p=1)
    network_name = string("ACASXU_experimental_v2a_", index1, "_", index2, ".nnet")
    # Read in the network. Named CAS so as not to confuse with the official ACAS Xu tables.
    network_file = string(@__DIR__, "/../networks/", network_name)
    network = read_nnet(network_file)

    # Define your input and output sets
    input_set, output_set = get_acas_sets(property_index)

    # Solve the problem
    if output_set isa HalfSpace || output_set isa AbstractPolytope
        println("Checking if contained within polytope")
        time = @elapsed x_star, lower_bound, upper_bound, steps = contained_within_polytope(network, input_set, output_set, params)
    elseif output_set isa Complement{<:Number, <:AbstractPolytope}
        time = @elapsed x_star, lower_bound, upper_bound, steps = reaches_polytope_binary(network, input_set, output_set.X, params)
    else
        @assert false "Haven't implemented reach polytope yet"
    end
    # Print your results
    println("Elapsed time: ", time)
    println("Interval: ", [lower_bound, upper_bound])
    println("Steps: ", steps)

    return lower_bound, upper_bound, time, steps
end

function print_results(lower_bounds, upper_bounds, times, steps, max_properties, max_index_1, max_index_2, stop_gap)
    # Nicely formatted printout of the tests
    for property_index = 1:max_properties
        for i = 1:max_index_1
            for j = 1:max_index_2
                println("Property: ", property_index, "   Network: ", (i, j))
                println("    bounds: ", (lower_bounds[property_index, i, j], upper_bounds[property_index, i, j]), "  time: ", times[property_index, i, j])
                println("    steps: ", steps[property_index, i, j])
                println("    ", get_sat(property_index, lower_bounds[property_index, i, j], upper_bounds[property_index, i, j], stop_gap))
            end
        end
    end
end

function get_sat(property_index, lower_bound, upper_bound, stop_gap)
    # Here we are seeing if we are contained within a polytope given by the complement of the
    # property's single linear output constraint. If the maximum
    # violation is > 0 then we can exit the polytope, meaning we can satisfy the original constraint.
    if property_index == 1
        if lower_bound > 0
            return "SAT"
        elseif upper_bound <= stop_gap
            return "UNSAT"
        else
            return "Inconclusive"
        end
    end
    # For properties 2, 3, and 4 we're using the function which tests if a polytope is reachable.
    # The polytope is given by the output constraints in the original property.
    # this finds the distance of the projection onto this polytope. If this is 0 then it is reachable.
    # so if we find something <= stop_gap we return SAT.
    if property_index in [2, 3, 4]
        if upper_bound <= stop_gap
            return "SAT"
        elseif lower_bound > 0
            return "UNSAT"
        else
            return "inconclusive"
        end
    end
end

# Each line will look like:
# property_number, network_index_1-network_index_2, SAT UNSAT or Inconclusive, lower_bound, upper_bound, time, steps
# this will overwrite a file if one already exists
function write_results(filename, lower_bounds, upper_bounds, times, steps, max_properties, max_index_1, max_index_2, stop_gap)
    open(filename, "w") do f
        println(f, "property,network,result,lower_bound,upper_bound,time,steps")
        # k, j, i at the end of the line to iterate like a for loop with the outermost i
        [writeline(f, i, j, k, lower_bounds[i, j, k], upper_bounds[i, j, k], times[i, j, k], steps[i, j, k], stop_gap) for k=1:max_index_2, j=1:max_index_1, i=1:max_properties]
    end
end

# Write an individual line
function writeline(file, property_index, index_1, index_2, lower_bound, upper_bound, time, steps, stop_gap)
    sat_string = get_sat(property_index, lower_bound, upper_bound, stop_gap)
    println(file, string(property_index, ",", index_1, "-", index_2, ",", sat_string, ",", lower_bound, ",", upper_bound, ",", time, ",", steps))
end

###
# Setup your parameters and then run the tests
###
filename=string(@__DIR__, "/acas_fullrun_onethread_zope.csv")
# was commented out before, but printing doesn't work without it
max_steps = 200000  # hard coded below now to be different for the properties
timeout = 60.
properties_to_test = 4
max_index_1 = 5
max_index_2 = 9
#timeout = 15.
# max_index_1 = 2
# max_index_2 = 2
p = Inf  # added by me
stop_frequency = 1
# stop_frequency = 100 # was default in GitHub

#### just for precompilation
println("precompilation ...")
params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=5, timeout=10., stop_frequency=stop_frequency)
test_acas_network(1, 1, 1, params, p=p)
test_acas_network(1, 1, 2, params, p=p)

params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=200000, verbosity=0, timeout=timeout, stop_frequency=stop_frequency) # added by me


full_time = @elapsed begin
    lower_bounds = Array{Float64, 3}(undef, 4, 5, 9)
    upper_bounds = Array{Float64, 3}(undef, 4, 5, 9)
    times = Array{Float64, 3}(undef, 4, 5, 9)
    steps = Array{Integer, 3}(undef, 4, 5, 9)
    for property_index = 1:properties_to_test
	    params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=max_steps, stop_frequency=stop_frequency, verbosity=0, timeout=timeout)
	    if property_index > 1
		    params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=max_steps, stop_frequency=stop_frequency, verbosity=0, timeout=timeout)
	    end
        for i = 1:max_index_1
            for j = 1:max_index_2
                println("Property ", property_index, " Network ", i, "-", j)
                lower_bounds[property_index, i, j], upper_bounds[property_index, i, j], times[property_index, i, j], steps[property_index, i, j] = test_acas_network(i, j, property_index, params; p=p)
                println()
            end
        end
    end
end

println("p norm: ", p)
println("Max steps: ", max_steps)
println("Full time: ", full_time)

print_results(lower_bounds, upper_bounds, times, steps, properties_to_test, max_index_1, max_index_2, params.stop_gap)
write_results(filename, lower_bounds, upper_bounds, times, steps, properties_to_test, max_index_1, max_index_2, params.stop_gap)
