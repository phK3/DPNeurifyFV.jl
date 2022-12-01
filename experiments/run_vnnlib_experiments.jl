# Include the optimizer as well as supporting packages
using DPNeurifyFV
using NeuralVerification
using NeuralPriorityOptimizer
using LinearAlgebra

const NV = NeuralVerification

@assert Threads.nthreads()==1 "for benchmarking threads must be 1"
LinearAlgebra.BLAS.set_num_threads(1)


# Setup your parameters and then run the tests
max_steps = 5
#max_steps = 200000  
timeout = 60.
result_dir = string(@__DIR__, "/results")  # just use this directory for now

benchmark_dir = "../vnncomp22/"
# benchmarks have to be passed without '/' at the end, directories have to contain the instances.csv!
benchmarks = ["acasxu", "reach_prob_density", "rl_benchmarks", "tllverifybench"]
# benchmarks = ["acasxu"]


# different solver configurations to test
configs = [
    Dict("name" => "dpnfv", "max_vars" => 20, "method" => :DeepPolyRelax, "get_fresh_var_idxs" => DPNeurifyFV.fresh_var_range_non_zero,
        "split" => DPNeurifyFV.split_important_interval, "concrete_sample" => :BoundsMaximizer),
    # max_vars=0 initially for subsequent forward passes, max_vars is calculated as ∑_l N_l / l
    Dict("name" => "neurodiff", "max_vars" => 0, "method" => :DeepPolyRelax, "get_fresh_var_idxs" => DPNeurifyFV.fresh_var_earliest,
        "split" => DPNeurifyFV.split_important_interval_neurodiff, "concrete_sample" => :BoundsMaximizer),
    Dict("name" => "center", "max_vars" => 20, "method" => :DeepPolyRelax, "get_fresh_var_idxs" => DPNeurifyFV.fresh_var_range_non_zero,
        "split" => DPNeurifyFV.split_important_interval, "concrete_sample" => :Center),
    Dict("name" => "no_heur", "max_vars" => 20, "method" => :DeepPolyRelax, "get_fresh_var_idxs" => DPNeurifyFV.fresh_var_range_non_zero,
        "split" => NeuralPriorityOptimizer.split_largest_interval, "concrete_sample" => :BoundsMaximizer),
    Dict("name" => "no_vars", "max_vars" => 0, "method" => :DeepPolyRelax, "get_fresh_var_idxs" => DPNeurifyFV.fresh_var_range_non_zero,
        "split" => DPNeurifyFV.split_important_interval, "concrete_sample" => :BoundsMaximizer),
]


for cfg in configs
    solver = DPNFV(method=cfg["method"], max_vars=cfg["max_vars"], get_fresh_var_idxs=cfg["get_fresh_var_idxs"])

    println("precompilation ...")
    params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=5, timeout=10.)
    verify_vnnlib(solver, string(benchmark_dir, benchmarks[1]), params, max_properties=1)

    println("\nstarting verification ...")
    params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=max_steps, timeout=timeout, stop_frequency=1, verbosity=2)
    split = cfg["split"]
    concrete_sample = cfg["concrete_sample"]

    for b in benchmarks
        verify_vnnlib(solver, string(benchmark_dir, b), params, logfile=string(result_dir, "/", b, "_", cfg["name"], ".csv"))
    end

end

# solver = DPNFV(method=:DeepPolyRelax, max_vars=20)

# println("precompilation ...")
# params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=5, timeout=10.)
# time = @elapsed x_star, steps, result = verify_vnnlib(solver, string(benchmark_dir, benchmarks[1]), params, max_properties=1)

# println("\nstarting verification ...")
# params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=5, print_frequency=100, stop_frequency=1, verbosity=2)
# for b in benchmarks
#     verify_vnnlib(solver, string(benchmark_dir, b), params, logfile=string(result_dir, "/", b, ".csv"))
# end
