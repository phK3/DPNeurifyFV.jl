# load basic functions for execution of benchmark
include(string(@__DIR__, "/dpnfv_experiments_basis.jl"))


# Setup your parameters and then run the tests
# was commented out before, but printing doesn't work without it
max_steps = 200000  # hard coded below now to be different for the properties
# timeout = 60.
# properties_to_test = 4
# max_index_1 = 5
# max_index_2 = 9
timeout = 5.
properties_to_test = 4
max_index_1 = 2
max_index_2 = 2
result_dir = string(@__DIR__, "/")  # just use this directory for now


solver = DPNFV(max_vars=20, method=:DeepPolyRelax, get_fresh_var_idxs=DPNeurifyFV.fresh_var_range_non_zero)
split = DPNeurifyFV.split_important_interval
concrete_sample = :BoundsMaximizer
filename=string(result_dir, "acas_fullrun_onethread_dnfv.csv")
run_experiment(solver, split, concrete_sample, filename; max_steps=max_steps, timeout=timeout, 
                properties_to_test=properties_to_test, max_index_1=max_index_1, max_index_2=max_index_2)

# max_vars=0 initially for subsequent forward passes, max_vars is calculated as ∑_l N_l / l
solver = DPNFV(max_vars=0, var_frac=1., method=:DeepPolyRelax, get_fresh_var_idxs=DPNeurifyFV.fresh_var_earliest)
split = DPNeurifyFV.split_important_interval_neurodiff
concrete_sample = :BoundsMaximizer
filename=string(result_dir, "acas_fullrun_onethread_neurodiff.csv")
run_experiment(solver, split, concrete_sample, filename; max_steps=max_steps, timeout=timeout, 
                properties_to_test=properties_to_test, max_index_1=max_index_1, max_index_2=max_index_2)

solver = DPNFV(max_vars=20, method=:DeepPolyRelax, get_fresh_var_idxs=DPNeurifyFV.fresh_var_range_non_zero)
split = DPNeurifyFV.split_important_interval
concrete_sample = :Center
filename=string(result_dir, "acas_fullrun_onethread_center.csv")
run_experiment(solver, split, concrete_sample, filename; max_steps=max_steps, timeout=timeout, 
                properties_to_test=properties_to_test, max_index_1=max_index_1, max_index_2=max_index_2)

solver = DPNFV(max_vars=20, method=:DeepPolyRelax, get_fresh_var_idxs=DPNeurifyFV.fresh_var_range_non_zero)
split = NeuralPriorityOptimizer.split_largest_interval
concrete_sample = :BoundsMaximizer
filename=string(result_dir, "acas_fullrun_onethread_no_heur.csv")
run_experiment(solver, split, concrete_sample, filename; max_steps=max_steps, timeout=timeout, 
                properties_to_test=properties_to_test, max_index_1=max_index_1, max_index_2=max_index_2)

solver = DPNFV(max_vars=0, method=:DeepPolyRelax, get_fresh_var_idxs=DPNeurifyFV.fresh_var_range_non_zero)
split = DPNeurifyFV.split_important_interval
concrete_sample = :BoundsMaximizer
filename=string(result_dir, "acas_fullrun_onethread_no_vars.csv")
run_experiment(solver, split, concrete_sample, filename; max_steps=max_steps, timeout=timeout, 
                properties_to_test=properties_to_test, max_index_1=max_index_1, max_index_2=max_index_2)
