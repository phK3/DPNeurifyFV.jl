
using DPNeurifyFV, VNNLib, LazySets, CSV, PyVnnlib
const DP = DPNeurifyFV
const NNL = VNNLib.NNLoader


"""
Compares values obtained from concrete execution to bounds obtained from symbolic values stored in symdict.

If a violation is found, the error-inducing input is added to the `errs` list and the output name, where the error occured 
together with the index of the error-inducing input is stored in the `err_dict`

args:
    nn - computational graph
    input_set - Hyperrectangular input set to check for inputs violating the symbolic bounds

kwargs:
    n_test - number of inputs sampled from hyperrectangular input set to check bounds

returns:
    errs, err_dict - list of inputs that caused a bounds violation, dict mapping output specifier to index of input in errs that caused the violation
"""
function random_testing(nn, input_set::Hyperrectangle, symdict::Dict{A,B}; n_test=1) where {A,B}    
    concrete_dict = Dict{A, NTuple{2,Vector{Float64}}}()
    for (k, s) in symdict
        l, u = DP.bounds(s)
        concrete_dict[k] = (l, u)
    end

    l = low(input_set)
    u = high(input_set)

    max_err = 0.
    errs = []
    err_dict = Dict{A, Vector{Int}}()
    for i in 1:n_test
        err_occurred = false
        x = (u .- l) .* rand(dim(input_set)) .+ l
        y_dict = DP.propagate(DP.ConcreteExecution(), nn, x, return_dict=true)

        for (k, y) in y_dict
            l_layer, u_layer = concrete_dict[k]
            
            low_vio = l_layer .- y # want < 0
            up_vio  = y .- u_layer # want < 0

            if maximum(low_vio) > 0 || maximum(up_vio) > 0
                println("Error at $k:\n\tlow_vio: ", maximum(low_vio), "\n\tup_vio : ", maximum(up_vio))

                max_err = max(max_err, maximum(low_vio), maximum(up_vio))

                if haskey(err_dict, k)
                    push!(err_dict[k], i)
                else
                    err_dict[k] = [i]
                end

                err_occurred = true
            end
        end

        if err_occurred
            push!(errs, x)
        end
    end

    return max_err, errs, err_dict
end


function random_testing(solver::DP.DPNFV, nn::DP.CompGraph, input_set::Hyperrectangle; n_test=1)
    s = DP.init_symbolic_interval_graph(nn, input_set, max_vars=solver.max_vars)
    symdict = DP.propagate(solver, nn, s, return_dict=true)

    return random_testing(nn, input_set, symdict, n_test=n_test)
end


function random_testing(solver, nn::DP.CompGraph; n_test=1, widths=nothing, verbosity=0)
    widths = isnothing(widths) ? [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100] : widths

    err_width_dict = Dict()
    for i in eachindex(widths)
        verbosity > 0 && println("=== width = ", widths[i], " ===")
        lo = randn(prod(nn.input_shape))
        hi = lo .+ widths[i]

        input_set = Hyperrectangle(low=lo, high=hi)
        max_err, err_inputs, err_dict = random_testing(solver, nn, input_set, n_test=n_test)

        if length(err_inputs) > 0   
            err_width_dict[widths[i]] = (max_err, input_set, err_inputs, err_dict)
        end
    end

    return err_width_dict
end


function test_property(solver, nn::DP.CompGraph, property_path; n_test=1, verbosity=0)
    compressed = false
    if !isfile(property_path)
        compressed = true
        run(`gunzip $(property_path).gz`)
    end


    speclist = PyVnnlib.generate_specs(property_path, dtype=Float64)
    specs = DP.generate_specs(speclist)
    
    # all specs in one speclist should have the same input set
    # they might differ in the output constraint, if the original specification is a disjunction of output constraints.
    input_set, _ = specs[1]

    max_err, errs, err_dict = random_testing(solver, nn, input_set; n_test=n_test)

    if compressed
        # compress if it was compressed before
        run(`gzip $(property_path)`)
    end

    return max_err, errs, err_dict    
end



"""
Tests if symbolic bounds are indeed an overapproximation to concrete values for a vnncomp benchmark set.

args:
    solver - solver used to obtain symbolic bounds
    benchmark_dir - directory containing `instances.csv` with paths to the benchmark's networks and properties

kwargs:
    tests_per_network - number of properties to check for each network (but can at most test all of the properties for that NN) 
    n_test - number of concrete inputs to sample for each check
    linearize - (bool) true iff networks theoretically linear layers should be converted to dense layers
    summarize - (bool) true iff consecutive linear layers should be summarized into one linear layer
    verbosity - print info

returns:
    errdict - dict mapping networkpath to either an error message, if the network could not be loaded or a list of concrete inputs violating the symbolic bounds
"""
function test_benchmark(solver, benchmark_dir; tests_per_network=2, n_test=1, linearize=true, summarize=true, double_precision=true, verbosity=0)
    f = CSV.File(joinpath(benchmark_dir, "instances.csv"), header=false)

    network_dict = Dict{String, Vector{Int}}()
    for (i, instance) in enumerate(f)
        netpath, propertypath, time_limit = instance
        if haskey(network_dict, netpath)
            push!(network_dict[netpath], i)
        else
            network_dict[netpath] = [i]
        end
    end

    errdict = Dict()
    for netpath in keys(network_dict)
        verbosity > 0 && println("== testing ", netpath)

        full_netpath = joinpath(benchmark_dir, netpath)
        compressed = false
        if !isfile(full_netpath) && !occursin("transformer", full_netpath)  # this is a hack for cgan!
            compressed = true
            run(`gunzip $(full_netpath).gz`)
        end

        # load NN
        nn = try 
            nn = NNL.load_network_dict(DP.CGType, full_netpath)

            if linearize
                nn = DP.cg2dense(nn, double_precision=double_precision)
            end

            if summarize
                nn = DP.summarize_linear_layers(nn, double_precision=double_precision)
            end

            nn
        catch err
            println("Failed to load: ", netpath)
            errdict[netpath] = ("failed to load", err)

            if compressed
                run(`gzip $(full_netpath)`)
            end

            # skip remainder of tests for this network
            continue
        end

        # test properties
        for i in 1:min(tests_per_network, length(network_dict[netpath]))
            csv_idx = network_dict[netpath][i]
            _, propertypath, _ = f[csv_idx]
            res = test_property(solver, nn, joinpath(benchmark_dir, propertypath), n_test=n_test, verbosity=verbosity)
            errdict[netpath] = res
        end

        if compressed
            run(`gzip $(full_netpath)`)
        end
    end

    return errdict
end
