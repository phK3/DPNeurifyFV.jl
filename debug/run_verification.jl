
using DPNeurifyFV, VNNLib, LazySets
const DP = DPNeurifyFV
const NNL = VNNLib.NNLoader

params = DP.PriorityOptimizerParameters(max_steps=4, print_frequency=1, stop_frequency=1, verbosity=2)
solver = DP.DPNFV(method=:DeepPolyRelax)

onnx_file = joinpath(@__DIR__, "../networks/precompile_nns/ACASXU_run2a_1_1_batch_2000.onnx")
vnnlib_file = joinpath(@__DIR__, "../networks/precompile_nns/prop_1.vnnlib")

DP.verify_vnnlib(solver, onnx_file, vnnlib_file, params, printing=true)



## Single forward pass 

nn = NNL.load_network_dict(DP.CGType, onnx_file)
input_set, output_set = DP.get_acas_sets(1)

s = DP.init_symbolic_interval_graph(nn, input_set, max_vars=0)
ŝ = DP.propagate(solver, nn, s)

s = DP.init_symbolic_interval_graph(nn, input_set, max_vars=10)
ŝ = DP.propagate(solver, nn, s)


## more complicated network

nn = NNL.load_network_dict(DP.CGType, "../vnncomp2023_benchmarks/benchmarks/nn4sys/onnx/pensieve_big_simple.onnx")
nn_lin = DP.cg2dense(nn)
nn_summ = DP.summarize_linear_layers(nn_lin)

input_set = Hyperrectangle(low=-0.05 .* ones(nn_summ.input_shape), high=0.05 .* ones(nn_summ.input_shape))
s = DP.init_symbolic_interval_graph(nn_summ, input_set, max_vars=10)
ŝ = DP.propagate(solver, nn_summ, s)

s = DP.init_symbolic_interval_graph(nn_summ, input_set, max_vars=10)
symdict = DP.propagate(solver, nn_summ, s, return_dict=true)